#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _print_cuda_memory(tag: str):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    print(
        f"[cuda-memory] {tag}: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB"
    )


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def _dict_bytes(d: Dict) -> int:
    total = 0
    for v in d.values():
        if torch.is_tensor(v):
            total += _tensor_bytes(v)
        elif isinstance(v, dict):
            total += _dict_bytes(v)
    return total


from triton_decode_kernels import decode_rvq_triton_into, rvq_triton_supported

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def _load_model_config(compressed_dir: str):
    config = AutoConfig.from_pretrained(compressed_dir, trust_remote_code=True)
    if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
        text_cfg = config.text_config
        for key in list(vars(text_cfg).keys()):
            if not key.startswith("_") and not hasattr(config, key):
                config.__dict__[key] = getattr(text_cfg, key)
    raw_path = os.path.join(compressed_dir, "config.json")
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        src = raw.get("text_config", raw)
        for key, value in src.items():
            if not hasattr(config, key) and not isinstance(value, (dict, list)):
                config.__dict__[key] = value
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    return config


def _build_meta_model(compressed_dir: str):
    config = _load_model_config(compressed_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model


def _find_base_model(model):
    if hasattr(model, "get_model"):
        try:
            return model.get_model()
        except Exception:
            pass
    for attr in ("model", "language_model", "transformer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Could not locate the base decoder model.")


def _prefix_for_suffix(keys, suffix: str) -> str:
    for key in keys:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return ""


def _remap_entry_prefixes(
    entry_map: Dict[str, Dict[str, object]], model_state_keys: List[str]
) -> Dict[str, Dict[str, object]]:
    anchors = (
        "embed_tokens.weight",
        "lm_head.weight",
        "layers.0.input_layernorm.weight",
    )
    model_prefix = ""
    artifact_prefix = ""
    for anchor in anchors:
        model_prefix = _prefix_for_suffix(model_state_keys, anchor)
        artifact_prefix = _prefix_for_suffix(entry_map.keys(), anchor)
        if model_prefix or artifact_prefix:
            break
    if model_prefix == artifact_prefix:
        return entry_map
    remapped = {}
    for key, value in entry_map.items():
        if artifact_prefix and key.startswith(artifact_prefix):
            remapped[model_prefix + key[len(artifact_prefix) :]] = value
        else:
            remapped[key] = value
    return remapped


def _build_layer_plan(model, entry_map: Dict[str, Dict[str, object]]):
    model_state_keys = list(model.state_dict().keys())
    remapped = _remap_entry_prefixes(entry_map, model_state_keys)
    model_key_set = set(model_state_keys)
    usable = {key: value for key, value in remapped.items() if key in model_key_set}
    base_prefix = _prefix_for_suffix(model_state_keys, "embed_tokens.weight")
    embed_key = (
        f"{base_prefix}embed_tokens.weight"
        if f"{base_prefix}embed_tokens.weight" in usable
        else None
    )
    lm_head_prefix = _prefix_for_suffix(model_state_keys, "lm_head.weight")
    lm_head_key = (
        f"{lm_head_prefix}lm_head.weight"
        if f"{lm_head_prefix}lm_head.weight" in usable
        else None
    )

    layer_re = re.compile(rf"^{re.escape(base_prefix)}layers\.(\d+)\.")
    layer_groups: Dict[int, List[str]] = {}
    for key in usable:
        match = layer_re.match(key)
        if match:
            layer_idx = int(match.group(1))
            layer_groups.setdefault(layer_idx, []).append(key)

    norm_keys = [key for key in usable if key.startswith(f"{base_prefix}norm.")]
    layer_keys = [sorted(layer_groups[idx]) for idx in sorted(layer_groups)]
    base_model = _find_base_model(model)
    return {
        "base_prefix": base_prefix,
        "entries": usable,
        "base_model": base_model,
        "embed_key": embed_key,
        "lm_head_key": lm_head_key,
        "layer_keys": layer_keys,
        "norm_keys": sorted(norm_keys),
    }


def decode_into(
    entry: Dict[str, object], codebooks: Dict[str, torch.Tensor], out: torch.Tensor
):
    method = entry["method"]
    if method == "bf16_raw":
        flat_data = entry["data"].view(torch.int16)
        out.view(torch.int16).copy_(flat_data, non_blocking=True)
    elif method in ("rvq_groupwise", "rvq_mlp"):
        codebook = codebooks[entry["codebook_id"]]
        if not rvq_triton_supported(entry, out):
            raise ValueError(
                f"Triton kernel does not support entry shape {entry['shape']}"
            )
        decode_rvq_triton_into(entry, codebook, out)
    else:
        raise ValueError(f"Unsupported decode method: {method}")


def decode_to_tensor(
    entry: Dict[str, object],
    codebooks: Dict[str, torch.Tensor],
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> torch.Tensor:
    shape = tuple(entry["shape"])
    out = torch.empty(shape, dtype=runtime_dtype, device=device)
    decode_into(entry, codebooks, out)
    return out


def _make_dynamic_cache(base_model):
    cache = DynamicCache(config=base_model.config)
    num_layers = len(getattr(base_model, "layers", []))
    if not hasattr(cache, "conv_states"):
        cache.conv_states = [None] * num_layers
    if not hasattr(cache, "recurrent_states"):
        cache.recurrent_states = [None] * num_layers
    cache.has_previous_state = False
    return cache


def _layerwise_logits(
    model,
    plan,
    gpu_codebooks,
    embed_weight,
    lm_head_weight,
    input_ids,
    attention_mask,
    past_key_values,
    cache_position,
    device,
    runtime_dtype,
):
    base_model = plan["base_model"]
    hidden_states = F.embedding(input_ids, embed_weight)
    position_ids = cache_position.unsqueeze(0)

    position_embeddings = None
    if hasattr(base_model, "rotary_emb"):
        position_embeddings = base_model.rotary_emb(hidden_states, position_ids)

    # Use basic attention mask fallback if model doesn't implement create_causal_mask
    mask_kwargs = {
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }
    if (
        hasattr(base_model.forward, "__globals__")
        and "create_causal_mask" in base_model.forward.__globals__
    ):
        attn_mask = base_model.forward.__globals__["create_causal_mask"](
            input_embeds=hidden_states, **mask_kwargs
        )
    else:
        attn_mask = attention_mask

    # Setup slots and streams
    num_layers = len(base_model.layers)
    slots = [{}, {}]
    stream = torch.cuda.Stream(device=device)

    # Helper to decode a layer asynchronously
    def prefetch_layer(layer_idx, slot_idx):
        if layer_idx >= num_layers:
            return
        with torch.cuda.stream(stream):
            expected_keys = set()
            for key in plan["layer_keys"][layer_idx]:
                entry = plan["entries"][key]
                rel_key = key.split(f"layers.{layer_idx}.")[-1]
                expected_keys.add(rel_key)
                shape = tuple(entry["shape"])
                if (
                    rel_key not in slots[slot_idx]
                    or slots[slot_idx][rel_key].shape != shape
                ):
                    slots[slot_idx][rel_key] = torch.empty(
                        shape, dtype=runtime_dtype, device=device
                    )
                decode_into(entry, gpu_codebooks, slots[slot_idx][rel_key])
            for rel_key in list(slots[slot_idx].keys()):
                if rel_key not in expected_keys:
                    del slots[slot_idx][rel_key]

    # Initial prefetch for layer 0
    prefetch_layer(0, 0)

    for layer_idx, layer in enumerate(base_model.layers):
        torch.cuda.current_stream(device).wait_stream(
            stream
        )  # Wait for layer_idx decode to finish
        prefetch_layer(
            layer_idx + 1, (layer_idx + 1) % 2
        )  # Start decoding layer_idx + 1

        layer_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "cache_position": cache_position,
        }
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings

        hidden_states = functional_call(
            layer, slots[layer_idx % 2], (), layer_kwargs, strict=False
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

    hidden_states = base_model.norm(hidden_states)
    logits = F.linear(hidden_states[:, -1:, :], lm_head_weight)
    return logits[:, -1, :], past_key_values


def generate_greedy(
    model,
    tokenizer,
    plan,
    gpu_codebooks,
    embed_weight,
    lm_head_weight,
    prompt,
    max_new_tokens,
    device,
    runtime_dtype,
    report_time=False,
    report_memory=False,
):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    generated = input_ids
    generated_attention = attention_mask
    past_key_values = _make_dynamic_cache(plan["base_model"])

    t0 = time.perf_counter()
    with torch.inference_mode():
        # Prefill
        cache_position = torch.arange(generated.shape[1], device=device)
        logits, past_key_values = _layerwise_logits(
            model,
            plan,
            gpu_codebooks,
            embed_weight,
            lm_head_weight,
            generated,
            generated_attention,
            past_key_values,
            cache_position,
            device,
            runtime_dtype,
        )
        ttft = time.perf_counter() - t0
        past_key_values.has_previous_state = True

        if report_memory:
            _print_cuda_memory("after prefill")

        t1 = time.perf_counter()
        # Decode
        for step in range(max_new_tokens):
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if generated_attention is not None:
                next_mask = torch.ones(
                    (generated_attention.shape[0], 1),
                    dtype=generated_attention.dtype,
                    device=device,
                )
                generated_attention = torch.cat([generated_attention, next_mask], dim=1)
            if (
                tokenizer.eos_token_id is not None
                and (next_token == tokenizer.eos_token_id).all()
            ):
                break

            cache_position = torch.tensor([generated.shape[1] - 1], device=device)
            logits, past_key_values = _layerwise_logits(
                model,
                plan,
                gpu_codebooks,
                embed_weight,
                lm_head_weight,
                next_token,
                generated_attention,
                past_key_values,
                cache_position,
                device,
                runtime_dtype,
            )

            if report_memory and step == 0:
                _print_cuda_memory("after first decode step")

        t2 = time.perf_counter()

    if report_memory:
        _print_cuda_memory("after generation")

    tokens_generated = generated.shape[1] - input_ids.shape[1]
    tbt = (t2 - t1) / tokens_generated if tokens_generated > 0 else 0

    if report_time:
        print(
            f"[generation-time] time_to_first_token={ttft*1000:.1f} ms time_between_tokens={tbt*1000:.1f} ms/token generated_tokens={tokens_generated}"
        )
    else:
        print(
            f"\n[Performance] TTFT: {ttft*1000:.1f}ms | Decode: {1/tbt if tbt > 0 else 0:.1f} tokens/sec"
        )

    raw = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report-time", action="store_true")
    parser.add_argument("--report-memory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    runtime_dtype = torch.bfloat16

    print("Loading artifact into CPU...")
    artifact_path = os.path.join(args.model_dir, "compressed_model.pt")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)

    print("Moving codebooks and compressed tensors to GPU VRAM...")
    gpu_codebooks = {
        k: v.to(device, dtype=torch.float32)
        for k, v in artifact["shared_codebooks"].items()
    }
    gpu_entries = {}
    for k, v in artifact["entries"].items():
        gpu_entries[k] = {
            ek: ev.to(device) if torch.is_tensor(ev) else ev for ek, ev in v.items()
        }
    del artifact

    model = _build_meta_model(args.model_dir)
    plan = _build_layer_plan(model, gpu_entries)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    embed_weight = decode_to_tensor(
        plan["entries"][plan["embed_key"]], gpu_codebooks, device, runtime_dtype
    )
    lm_head_weight = (
        embed_weight
        if plan["lm_head_key"] is None
        else decode_to_tensor(
            plan["entries"][plan["lm_head_key"]], gpu_codebooks, device, runtime_dtype
        )
    )

    norm_bytes = 0
    for key in plan["norm_keys"]:
        val = decode_to_tensor(
            plan["entries"][key], gpu_codebooks, device, runtime_dtype
        )
        norm_bytes += _tensor_bytes(val)
        set_module_tensor_to_device(model, key, device, value=val, dtype=runtime_dtype)

    rotary_prefix = f"{plan['base_prefix']}rotary_emb."
    for name, _ in model.named_buffers():
        if name.startswith(rotary_prefix):
            set_module_tensor_to_device(model, name, device)

    if args.report_memory:
        torch.cuda.reset_peak_memory_stats()
        _print_cuda_memory("after load")

        compressed_bytes = _dict_bytes(gpu_entries) + _dict_bytes(gpu_codebooks)
        always_hot_bytes = (
            _tensor_bytes(embed_weight)
            + _tensor_bytes(lm_head_weight)
            + norm_bytes
        )
        print(
            f"[layerwise-breakdown] after load: compressed_artifact_tensor={compressed_bytes/(1024**2):.1f} MiB "
            f"always_hot={always_hot_bytes/(1024**2):.1f} MiB"
        )

    print("Starting inference...\n")
    response = generate_greedy(
        model,
        tokenizer,
        plan,
        gpu_codebooks,
        embed_weight,
        lm_head_weight,
        args.prompt,
        args.max_new_tokens,
        device,
        runtime_dtype,
        report_time=args.report_time,
        report_memory=args.report_memory,
    )
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
