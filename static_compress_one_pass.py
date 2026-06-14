#!/usr/bin/env python3
"""

python static_compress_one_pass.py   --teacher-path Qwen/Qwen3-4B   --output-dir Qwen3-4B-L0-L35-Half   --temp-dir ./temp_activations_l0_35   --start-layer 0   --num-layers 36   --num-samples 960   --batch-size 32   --steps 96000   --lr 1e-3   --enable-thinking   --gen-len 512

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import gc
import os
import shutil
import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# ==========================================
# 1. 模型定义
# ==========================================
class StudentMLP(nn.Module):
    def __init__(self, emb_dim=2560, hidden_dim=4864):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ==========================================
# 2. 全局收集器 (只跑一次)
# ==========================================
class GlobalCollector:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = args.temp_dir
        
        # 创建临时目录结构
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        
        # 为每一层创建一个子目录
        self.target_layers = range(args.start_layer, args.start_layer + args.num_layers)
        for i in self.target_layers:
            os.makedirs(os.path.join(self.temp_dir, f"layer_{i}"))

        print(f"🔄 Loading Teacher: {args.teacher_path}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_path, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device,
            attn_implementation="sdpa"
        )
        self.teacher.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("📚 Loading Prompts...")
        path = hf_hub_download(
            repo_id='HuggingFaceH4/ultrachat_200k',
            filename='data/train_sft-00000-of-00003-a3ecf92756993583.parquet',
            repo_type='dataset'
        )
        df = pd.read_parquet(path)
        self.prompts = df['prompt'].dropna().tolist()[:args.num_samples]

    def run_collection(self):
        print(f"\n🚀 Starting ONE-PASS Collection for {len(self.target_layers)} layers...")
        print(f"📂 Data will be streamed to {self.temp_dir}/")

        # 1. 注册所有的 Hooks
        # 我们需要一个字典来暂存当前 Batch 所有层的数据
        # structure: batch_buffer[layer_idx] = {'in': [], 'out': []}
        batch_buffer = {l: {'in': [], 'out': []} for l in self.target_layers}

        def get_hook(layer_idx):
            def hook(module, input, output):
                # input[0]: (B, Seq, Dim)
                batch_buffer[layer_idx]['in'].append(input[0].detach().cpu())
                batch_buffer[layer_idx]['out'].append(output.detach().cpu())
            return hook

        handles = []
        for layer_idx in self.target_layers:
            mlp = self.teacher.model.layers[layer_idx].mlp
            handles.append(mlp.register_forward_hook(get_hook(layer_idx)))

        # 2. 运行推理循环
        try:
            with torch.no_grad():
                for batch_idx, i in enumerate(tqdm(range(0, len(self.prompts), self.args.batch_size), desc="Global Generation")):
                    # 清空 Buffer
                    for l in self.target_layers:
                        batch_buffer[l]['in'] = []
                        batch_buffer[l]['out'] = []

                    batch = self.prompts[i:i+self.args.batch_size]
                    
                    # A. Format Prompts
                    texts = []
                    for p in batch:
                        messages = [{"role": "user", "content": p}]
                        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if self.args.enable_thinking:
                            text += "<think>\n"
                        texts.append(text)

                    encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
                    original_seq_len = encodings["input_ids"].shape[1]

                    # B. Generate (Hooks will fire here)
                    generated_ids = self.teacher.generate(
                        **encodings, 
                        max_new_tokens=self.args.gen_len,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    # C. Compute Mask (GPU Vectorized)
                    full_attention_mask = torch.ones_like(generated_ids)
                    
                    # Mask Left Padding
                    original_pad_mask = (encodings["input_ids"] == self.tokenizer.pad_token_id)
                    full_attention_mask[:, :original_seq_len] = (~original_pad_mask).long()
                    
                    # Mask Right Padding
                    is_pad_gen = (generated_ids[:, original_seq_len:] == self.tokenizer.pad_token_id)
                    pad_cumsum = is_pad_gen.cumsum(dim=1)
                    gen_mask = (pad_cumsum <= 1)
                    full_attention_mask[:, original_seq_len:] = gen_mask.long()

                    # D. Process and Save EACH layer to disk immediately
                    for layer_idx in self.target_layers:
                        # Concat captured steps (Prefill + Decode steps)
                        full_in = torch.cat(batch_buffer[layer_idx]['in'], dim=1) # (B, Seq, Dim)
                        full_out = torch.cat(batch_buffer[layer_idx]['out'], dim=1)
                        
                        # Align Mask
                        S = full_in.shape[1]
                        aligned_mask = full_attention_mask[:, :S]
                        filter_mask = (aligned_mask.reshape(-1) == 1).cpu()

                        # Filter and Flatten
                        saved_in = full_in.view(-1, 2560)[filter_mask]
                        saved_out = full_out.view(-1, 2560)[filter_mask]

                        # Save to disk (Shard per batch)
                        # Format: temp_data/layer_0/batch_001.pt
                        save_path = os.path.join(self.temp_dir, f"layer_{layer_idx}", f"batch_{batch_idx:05d}.pt")
                        torch.save({"in": saved_in.clone(), "out": saved_out.clone()}, save_path)

        finally:
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()
            
        print("✅ Collection Complete. Teacher can now be discarded.")

# ==========================================
# 3. 训练器
# ==========================================
class StaticTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trained_students = {}

    def load_layer_dataset(self, layer_idx):
        """
        从很多小文件中读取数据并合并
        """
        layer_dir = os.path.join(self.args.temp_dir, f"layer_{layer_idx}")
        files = glob.glob(os.path.join(layer_dir, "*.pt"))
        
        if not files:
            raise ValueError(f"No data found for layer {layer_idx}")

        print(f"   📂 Loading {len(files)} shards for Layer {layer_idx}...")
        
        inputs = []
        outputs = []
        for f in tqdm(files, desc="Loading Shards", leave=False):
            data = torch.load(f)
            inputs.append(data["in"])
            outputs.append(data["out"])
            
        x_data = torch.cat(inputs, dim=0)
        y_data = torch.cat(outputs, dim=0)
        
        print(f"   📊 Total Layer {layer_idx} Samples: {x_data.shape[0]}")
        return x_data, y_data

    def train_layer(self, layer_idx):
        print(f"\n🧪 [Layer {layer_idx}] Training Student...")
        
        # 加载数据 (消耗内存，但只加载这一层)
        x_data, y_data = self.load_layer_dataset(layer_idx)
        
        # 转移到 BF16
        x_data = x_data.to(torch.bfloat16)
        y_data = y_data.to(torch.bfloat16)

        student = StudentMLP().to(self.device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(student.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.steps)
        
        n_samples = x_data.shape[0]
        batch_size = self.args.train_batch_size
        
        # Training Loop
        pbar = tqdm(range(self.args.steps), desc=f"Distilling L{layer_idx}")
        for step in pbar:
            idx = torch.randint(0, n_samples, (batch_size,))
            x = x_data[idx].to(self.device)
            y_true = y_data[idx].to(self.device)
            
            y_pred = student(x)
            
            mse_loss = F.mse_loss(y_pred.float(), y_true.float())
            cos_loss = F.cosine_embedding_loss(
                y_pred.float(), y_true.float(), 
                torch.ones(y_pred.size(0), device=self.device)
            )
            
            loss = mse_loss + self.args.lambda_cos * cos_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                pbar.set_postfix({"mse": f"{mse_loss.item():.5f}", "cos": f"{cos_loss.item():.5f}"})

        # Save to memory
        self.trained_students[layer_idx] = student.cpu()
        
        # 清理数据释放内存
        del x_data, y_data
        gc.collect()

    def save_final_model(self):
        print(f"\n🔧 Assembling Final Model...")
        # 为了组装，我们需要重新加载 Teacher 的结构 (不需要权重，只要结构，但加载 pretrained 最方便)
        # 这里可以使用 CPU 加载以节省显存
        print("   Loading base model structure...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu" 
        )
        tokenizer = AutoTokenizer.from_pretrained(self.args.teacher_path)
        
        for layer_idx, student in self.trained_students.items():
            print(f"   - Replacing Layer {layer_idx} MLP...")
            base_model.model.layers[layer_idx].mlp = student
            
        output_path = f"{self.args.output_dir}_StaticOnePass"
        print(f"💾 Saving to {output_path}...")
        base_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # 清理临时文件
        if not self.args.keep_data:
            print("🧹 Cleaning up temporary data...")
            shutil.rmtree(self.args.temp_dir)
            
        print("✨ Done!")

# ==========================================
# 4. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", type=str, default="Qwen3-4B-Compressed")
    parser.add_argument("--temp-dir", type=str, default="./temp_activations")
    parser.add_argument("--keep-data", action="store_true", help="Don't delete temp data after finish")
    
    # 范围
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=32)
    
    # 收集参数
    parser.add_argument("--num-samples", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    
    # 训练参数
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--lambda-cos", type=float, default=1.0)
    
    args = parser.parse_args()

    # Step 1: Collect (One Pass)
    collector = GlobalCollector(args)
    collector.run_collection()
    
    # 释放显存，销毁 Teacher，为训练腾出空间
    del collector
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 2: Train (Layer by Layer)
    trainer = StaticTrainer(args)
    end_layer = args.start_layer + args.num_layers
    
    for layer_idx in range(args.start_layer, end_layer):
        trainer.train_layer(layer_idx)
        
    # Step 3: Assemble
    trainer.save_final_model()

if __name__ == "__main__":
    main()
