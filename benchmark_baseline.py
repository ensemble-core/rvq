import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _print_cuda_memory(tag: str):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[cuda-memory] {tag}: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct") # Adjust to your exact local uncompressed model path if needed
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    device = "cuda"
    
    print(f"Loading {args.model}...")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda", 
        trust_remote_code=True
    )
    _print_cuda_memory("after load")

    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    print("Starting inference...\n")
    
    t0 = time.perf_counter()
    # Prefill
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _print_cuda_memory("after prefill")
    
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
    generated = torch.cat([inputs["input_ids"], next_token], dim=-1)
    
    # Decode
    with torch.inference_mode():
        for step in range(args.max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=-1)
            if step == 0:
                _print_cuda_memory("after first decode step")
            if next_token.item() == tokenizer.eos_token_id:
                break
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    _print_cuda_memory("after generation")

    ttft = t1 - t0
    tokens_generated = generated.shape[1] - inputs["input_ids"].shape[1]
    tbt = (t2 - t1) / tokens_generated if tokens_generated > 0 else 0
    
    print(f"\n[Performance] TTFT: {ttft*1000:.1f}ms | Decode: {1/tbt if tbt > 0 else 0:.1f} tokens/sec")
    print(f"[generation-time] time_to_first_token={ttft*1000:.1f} ms time_between_tokens={tbt*1000:.1f} ms/token generated_tokens={tokens_generated}")
    
    raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse:\n{raw}")

if __name__ == "__main__":
    main()
