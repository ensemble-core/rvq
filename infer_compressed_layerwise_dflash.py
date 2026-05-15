#!/usr/bin/env python3
"""
DFlash speculative decoding on top of the compressed layerwise target.

This is a correctness-first integration:
  - target weights are loaded through infer_compressed_layerwise_single_slot.py
  - target verification uses layerwise block forward with selected hidden states
  - DFlash drafts blocks from target hidden states
  - partial rejection crops the KV cache and recomputes Qwen3.5 linear-attention
    state for the accepted prefix
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel

import infer_compressed_layerwise_single_slot as target_base


DEFAULT_DRAFT_ID = "z-lab/Qwen3.5-4B-DFlash"


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits.reshape(-1, vocab_size) / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)


def cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except TypeError:
            return int(cache.get_seq_length(0))
    return 0


def crop_cache(cache, max_length: int):
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    raise TypeError(f"Cache type {type(cache).__name__} does not support crop().")


def mark_qwen35_previous_state(cache) -> None:
    marker = getattr(cache, "has_previous_state", None)
    if marker is not None and not callable(marker) and not marker:
        cache.has_previous_state = True


def find_qwen35_linear_modules(base_model):
    modules = []
    for name, module in base_model.named_modules():
        if (
            module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
            and hasattr(module, "layer_idx")
            and hasattr(module, "in_proj_qkv")
        ):
            layer_prefix = f"layers.{module.layer_idx}."
            slot_prefix = "linear_attn."
            if name.startswith(layer_prefix):
                slot_prefix = name[len(layer_prefix) :] + "."
            module._layerwise_slot_prefix = slot_prefix
            modules.append(module)
    return sorted(modules, key=lambda module: module.layer_idx)


def _clone_if_tensor(value):
    return value.clone() if torch.is_tensor(value) else value


def snapshot_qwen35_linear_cache(cache, linear_modules):
    if cache is None:
        return {}
    snapshots = {}
    layers = getattr(cache, "layers", None)
    for module in linear_modules:
        layer_idx = module.layer_idx
        if layers is not None and layer_idx < len(layers):
            layer = layers[layer_idx]
            snapshots[layer_idx] = SimpleNamespace(
                has_previous_state=bool(getattr(layer, "has_previous_state", False)),
                is_conv_states_initialized=bool(getattr(layer, "is_conv_states_initialized", False)),
                is_recurrent_states_initialized=bool(getattr(layer, "is_recurrent_states_initialized", False)),
                conv_states=_clone_if_tensor(getattr(layer, "conv_states", None)),
                recurrent_states=_clone_if_tensor(getattr(layer, "recurrent_states", None)),
            )
        elif hasattr(cache, "conv_states") and hasattr(cache, "recurrent_states"):
            snapshots[layer_idx] = SimpleNamespace(
                has_previous_state=bool(getattr(cache, "has_previous_state", False)),
                is_conv_states_initialized=True,
                is_recurrent_states_initialized=True,
                conv_states=_clone_if_tensor(cache.conv_states[layer_idx]),
                recurrent_states=_clone_if_tensor(cache.recurrent_states[layer_idx]),
            )
    return snapshots


class Qwen35LinearInputCapture:
    def __init__(self, linear_modules) -> None:
        self.linear_modules = linear_modules
        self.inputs: Dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        self.inputs = {}

        def capture(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            self.inputs[module.layer_idx] = hidden_states.detach()

        self.handles = [
            module.register_forward_pre_hook(capture, with_kwargs=True)
            for module in self.linear_modules
        ]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _slot_linear(slot_state, prefix: str, module_name: str, hidden_states):
    weight = slot_state[f"{prefix}{module_name}.weight"]
    bias = slot_state.get(f"{prefix}{module_name}.bias")
    return F.linear(hidden_states, weight, bias)


def collect_qwen35_linear_rollback_materials(module, slot_state, hidden_states):
    prefix = getattr(module, "_layerwise_slot_prefix", "linear_attn.")
    conv_bias = slot_state.get(f"{prefix}conv1d.bias")
    return SimpleNamespace(
        mixed_qkv=_slot_linear(slot_state, prefix, "in_proj_qkv", hidden_states).detach(),
        b=_slot_linear(slot_state, prefix, "in_proj_b", hidden_states).detach(),
        a=_slot_linear(slot_state, prefix, "in_proj_a", hidden_states).detach(),
        conv_weight=slot_state[f"{prefix}conv1d.weight"].detach().clone(),
        conv_bias=conv_bias.detach().clone() if torch.is_tensor(conv_bias) else None,
        a_log=slot_state[f"{prefix}A_log"].detach().clone(),
        dt_bias=slot_state[f"{prefix}dt_bias"].detach().clone(),
    )


def compute_qwen35_linear_prefix_state_from_materials(module, materials, previous_state, prefix_len: int):
    mixed_qkv = materials.mixed_qkv[:, :prefix_len, :].transpose(1, 2)
    b = materials.b[:, :prefix_len, :]
    a = materials.a[:, :prefix_len, :]
    batch_size, seq_len, _ = a.shape
    if seq_len <= 0:
        return previous_state.conv_states, previous_state.recurrent_states

    use_previous = (
        previous_state is not None
        and previous_state.has_previous_state
        and previous_state.conv_states is not None
        and previous_state.recurrent_states is not None
    )
    conv_state = previous_state.conv_states if use_previous else None
    recurrent_state = previous_state.recurrent_states if use_previous else None

    conv_weight = materials.conv_weight
    conv_bias = materials.conv_bias
    conv_weight_squeezed = conv_weight.squeeze(1)

    if use_previous and seq_len == 1 and getattr(module, "causal_conv1d_update", None) is not None:
        new_conv_state = conv_state.clone()
        mixed_qkv = module.causal_conv1d_update(
            mixed_qkv,
            new_conv_state,
            conv_weight_squeezed,
            conv_bias,
            module.activation,
        )
        if mixed_qkv.dim() == 2:
            mixed_qkv = mixed_qkv.unsqueeze(1)
        else:
            mixed_qkv = mixed_qkv.transpose(1, 2)
    else:
        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1) if use_previous else mixed_qkv
        new_conv_state = F.pad(conv_input, (module.conv_kernel_size - conv_input.shape[-1], 0))
        if getattr(module, "causal_conv1d_fn", None) is not None:
            mixed_qkv = module.causal_conv1d_fn(
                x=conv_input,
                weight=conv_weight_squeezed,
                bias=conv_bias,
                activation=module.activation,
                seq_idx=None,
            )
        else:
            padding = getattr(module.conv1d, "padding", (module.conv_kernel_size - 1,))
            if isinstance(padding, tuple):
                padding = padding[0]
            groups = getattr(module.conv1d, "groups", conv_weight.shape[0])
            mixed_qkv = F.silu(
                F.conv1d(
                    conv_input,
                    conv_weight,
                    conv_bias,
                    padding=padding,
                    groups=groups,
                )[:, :, : conv_input.shape[-1]]
            )
        if use_previous:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)
    beta = b.sigmoid()
    g = -materials.a_log.float().exp() * F.softplus(a.float() + materials.dt_bias.float())

    if module.num_v_heads // module.num_k_heads > 1:
        repeat = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    if use_previous and seq_len == 1 and getattr(module, "recurrent_gated_delta_rule", None) is not None:
        _, recurrent_state = module.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        _, recurrent_state = module.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state if use_previous else None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    return new_conv_state, recurrent_state


def overwrite_qwen35_linear_cache_with_materials(
    cache,
    linear_modules,
    previous_snapshots,
    rollback_materials,
    prefix_len: int,
) -> int:
    updated = 0
    for module in linear_modules:
        layer_idx = module.layer_idx
        if layer_idx not in rollback_materials or layer_idx not in previous_snapshots:
            continue
        conv_state, recurrent_state = compute_qwen35_linear_prefix_state_from_materials(
            module,
            rollback_materials[layer_idx],
            previous_snapshots[layer_idx],
            prefix_len,
        )

        layers = getattr(cache, "layers", None)
        if layers is not None and layer_idx < len(layers):
            layer = layers[layer_idx]
            if conv_state is not None:
                if getattr(layer, "conv_states", None) is None and hasattr(layer, "lazy_initialization"):
                    layer.lazy_initialization(conv_states=conv_state)
                layer.conv_states.copy_(conv_state)
                if hasattr(layer, "is_conv_states_initialized"):
                    layer.is_conv_states_initialized = True
            if recurrent_state is not None:
                if getattr(layer, "recurrent_states", None) is None and hasattr(layer, "lazy_initialization"):
                    layer.lazy_initialization(recurrent_states=recurrent_state)
                layer.recurrent_states.copy_(recurrent_state)
                if hasattr(layer, "is_recurrent_states_initialized"):
                    layer.is_recurrent_states_initialized = True
            if hasattr(layer, "has_previous_state"):
                layer.has_previous_state = True
            updated += 1
        elif hasattr(cache, "conv_states") and hasattr(cache, "recurrent_states"):
            cache.conv_states[layer_idx] = conv_state
            cache.recurrent_states[layer_idx] = recurrent_state
            updated += 1

    mark_qwen35_previous_state(cache)
    return updated


def build_attention_mask(prompt_attention_mask, total_len: int, device) -> Optional[torch.Tensor]:
    if prompt_attention_mask is None:
        return None
    if prompt_attention_mask.shape[1] >= total_len:
        return prompt_attention_mask[:, :total_len]
    extra = torch.ones(
        (prompt_attention_mask.shape[0], total_len - prompt_attention_mask.shape[1]),
        dtype=prompt_attention_mask.dtype,
        device=device,
    )
    return torch.cat([prompt_attention_mask, extra], dim=1)


def target_embed(input_ids, embed_weight):
    return F.embedding(input_ids, embed_weight)


def target_lm_head(hidden_states, lm_head_weight):
    return F.linear(hidden_states, lm_head_weight)


def extract_draft_hidden(draft_output):
    if torch.is_tensor(draft_output):
        return draft_output
    if hasattr(draft_output, "last_hidden_state"):
        return draft_output.last_hidden_state
    if hasattr(draft_output, "hidden_states") and draft_output.hidden_states is not None:
        return draft_output.hidden_states[-1]
    raise TypeError(f"Could not extract hidden states from draft output type {type(draft_output).__name__}.")


def layerwise_forward(
    model,
    plan,
    residency,
    embed_weight,
    lm_head_weight,
    input_ids,
    attention_mask,
    past_key_values,
    cache_position,
    selected_layer_ids: Optional[List[int]],
    logits_last_only: bool,
    profile_label: Optional[str],
    linear_capture: Optional[Qwen35LinearInputCapture] = None,
    linear_modules_by_layer: Optional[Dict[int, object]] = None,
    rollback_materials: Optional[Dict[int, object]] = None,
):
    base_model = plan["base_model"]
    device = input_ids.device
    hidden_states = F.embedding(input_ids, embed_weight)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = (
        base_model.rotary_emb(hidden_states, position_ids)
        if hasattr(base_model, "rotary_emb")
        else None
    )
    mask_mapping = target_base._build_attention_mask_mapping(
        base_model,
        hidden_states,
        attention_mask,
        cache_position,
        position_ids,
        past_key_values,
    )

    selected = []
    selected_set = set(selected_layer_ids or [])
    residency.set_active_profile_label(profile_label)
    try:
        residency.prepare_for_forward()
        for layer_idx, layer in enumerate(base_model.layers):
            residency.ensure_layer_ready(layer_idx)
            residency.start_prefetch_for_layer(layer_idx + 1)
            layer_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": target_base._select_layer_mask(layer, mask_mapping),
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "cache_position": cache_position,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            start_event = end_event = None
            if residency.collect_gpu_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(device))
            with target_base._nvtx_range(f"{profile_label or 'target'}_layer_{layer_idx:03d}_forward"):
                hidden_states = residency.run_layer(layer_idx, layer, layer_kwargs)
            if residency.collect_gpu_timing and start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(device))
            residency.record_gpu_forward_event(layer_idx, start_event, end_event)
            if layer_idx in selected_set:
                selected.append(hidden_states)
            if (
                rollback_materials is not None
                and linear_capture is not None
                and linear_modules_by_layer is not None
                and layer_idx in linear_modules_by_layer
                and layer_idx in linear_capture.inputs
            ):
                slot_state = residency.slot_states[residency.layer_to_slot[layer_idx]]
                rollback_materials[layer_idx] = collect_qwen35_linear_rollback_materials(
                    linear_modules_by_layer[layer_idx],
                    slot_state,
                    linear_capture.inputs[layer_idx],
                )
            residency.offload_layer(layer_idx)
    finally:
        residency.set_active_profile_label(None)

    hidden_states = base_model.norm(hidden_states)
    logits_input = hidden_states[:, -1:, :] if logits_last_only else hidden_states
    logits = F.linear(logits_input, lm_head_weight)
    selected_hidden = torch.cat(selected, dim=-1) if selected else None
    return SimpleNamespace(
        logits=logits,
        past_key_values=past_key_values,
        selected_hidden=selected_hidden,
    )


def run_draft_block(
    draft,
    draft_cache,
    target_hidden,
    block_output_ids,
    all_position_ids,
    start: int,
    block_size: int,
    embed_weight,
    lm_head_weight,
    temperature: float,
):
    if block_size <= 1:
        return block_output_ids
    noise_embedding = target_embed(block_output_ids, embed_weight)
    draft_start = cache_seq_len(draft_cache)
    draft_position_ids = all_position_ids[:, draft_start : start + block_size]
    draft_output = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=draft_position_ids,
        past_key_values=draft_cache,
        use_cache=True,
        is_causal=False,
    )
    draft_hidden = extract_draft_hidden(draft_output)
    draft_logits = target_lm_head(draft_hidden[:, 1 - block_size :, :], lm_head_weight)
    crop_cache(draft_cache, start)
    block_output_ids[:, 1:] = sample(draft_logits, temperature)
    return block_output_ids


@torch.inference_mode()
def dflash_layerwise_generate(
    *,
    model,
    tokenizer,
    residency,
    plan,
    embed_weight,
    lm_head_weight,
    draft,
    input_ids,
    attention_mask,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]],
    temperature: float,
    block_size: int,
    target_layer_ids: List[int],
    mask_token_id: int,
    report_memory: bool,
    report_time: bool,
    debug: bool,
):
    device = input_ids.device
    batch_size, num_input_tokens = input_ids.shape
    if batch_size != 1:
        raise ValueError("This prototype supports batch_size=1.")
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (batch_size, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    output_ids[:, :num_input_tokens] = input_ids
    all_position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    target_cache = target_base._make_dynamic_cache(plan["base_model"])
    draft_cache = target_base.DynamicCache()
    qwen35_linear_modules = find_qwen35_linear_modules(plan["base_model"])
    qwen35_linear_modules_by_layer = {module.layer_idx: module for module in qwen35_linear_modules}
    if debug:
        print("target_layer_ids", target_layer_ids)
        print("qwen35_linear_modules", len(qwen35_linear_modules))

    residency.reset_gpu_timing_events()
    residency.set_collect_gpu_timing(report_time)
    if report_time:
        torch.cuda.synchronize(device)
        prefill_start = time.perf_counter()

    prefill = layerwise_forward(
        model,
        plan,
        residency,
        embed_weight,
        lm_head_weight,
        input_ids,
        attention_mask,
        target_cache,
        torch.arange(num_input_tokens, device=device),
        target_layer_ids,
        logits_last_only=True,
        profile_label="prefill",
    )
    target_cache = prefill.past_key_values
    mark_qwen35_previous_state(target_cache)
    first_token = sample(prefill.logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
    target_hidden = prefill.selected_hidden

    if report_time:
        torch.cuda.synchronize(device)
        time_to_first_token = time.perf_counter() - prefill_start
    else:
        time_to_first_token = None
    target_base._maybe_report_cuda_memory(device, "after prefill", report_memory)

    accepted_lengths: List[int] = []
    target_verify_count = 0
    linear_rollback_count = 0
    start = num_input_tokens
    decode_step_times: List[float] = []

    try:
        while start < max_length:
            bs = min(block_size, max_length - start)
            if bs <= 0:
                break

            if report_time:
                torch.cuda.synchronize(device)
                block_start_time = time.perf_counter()
            else:
                block_start_time = None

            block_output_ids = output_ids[:, start : start + bs].clone()

            if bs > 1:
                block_output_ids = run_draft_block(
                    draft,
                    draft_cache,
                    target_hidden,
                    block_output_ids,
                    all_position_ids,
                    start,
                    bs,
                    embed_weight,
                    lm_head_weight,
                    temperature,
                )

            linear_snapshot = snapshot_qwen35_linear_cache(target_cache, qwen35_linear_modules)
            rollback_materials: Dict[int, object] = {}
            with Qwen35LinearInputCapture(qwen35_linear_modules) as linear_capture:
                verify = layerwise_forward(
                    model,
                    plan,
                    residency,
                    embed_weight,
                    lm_head_weight,
                    block_output_ids,
                    build_attention_mask(attention_mask, start + bs, device),
                    target_cache,
                    all_position_ids[0, start : start + bs],
                    target_layer_ids,
                    logits_last_only=False,
                    profile_label=f"verify_{len(accepted_lengths):03d}",
                    linear_capture=linear_capture,
                    linear_modules_by_layer=qwen35_linear_modules_by_layer,
                    rollback_materials=rollback_materials,
                )
            target_verify_count += 1
            target_cache = verify.past_key_values
            mark_qwen35_previous_state(target_cache)

            posterior = sample(verify.logits, temperature)
            if bs > 1:
                accepted = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            else:
                accepted = 0
            accepted_plus_bonus = min(accepted + 1, max_length - start)

            output_ids[:, start : start + accepted_plus_bonus] = block_output_ids[:, :accepted_plus_bonus]
            if start + accepted_plus_bonus < output_ids.shape[1] and accepted < posterior.shape[1]:
                output_ids[:, start + accepted_plus_bonus] = posterior[:, accepted]

            if accepted_plus_bonus < bs:
                if len(rollback_materials) != len(qwen35_linear_modules):
                    missing = sorted(
                        module.layer_idx
                        for module in qwen35_linear_modules
                        if module.layer_idx not in rollback_materials
                    )
                    raise RuntimeError(f"Missing Qwen3.5 linear-attention rollback materials for layers: {missing}")
                crop_cache(target_cache, start + accepted_plus_bonus)
                updated = overwrite_qwen35_linear_cache_with_materials(
                    target_cache,
                    qwen35_linear_modules,
                    linear_snapshot,
                    rollback_materials,
                    accepted_plus_bonus,
                )
                if updated != len(qwen35_linear_modules):
                    raise RuntimeError(
                        f"Updated {updated} Qwen3.5 linear-attention cache states, "
                        f"expected {len(qwen35_linear_modules)}."
                    )
                linear_rollback_count += 1
                target_hidden = verify.selected_hidden[:, :accepted_plus_bonus, :]
            else:
                target_hidden = verify.selected_hidden[:, :accepted_plus_bonus, :]

            start += accepted_plus_bonus
            accepted_lengths.append(int(accepted_plus_bonus))

            if report_time:
                torch.cuda.synchronize(device)
                decode_step_times.append(time.perf_counter() - block_start_time)
                if len(accepted_lengths) == 1:
                    target_base._maybe_report_cuda_memory(device, "after first verify block", report_memory)

            if stop_token_ids is not None:
                generated = output_ids[:, num_input_tokens:start]
                stop_tensor = torch.tensor(stop_token_ids, device=device)
                if torch.isin(generated, stop_tensor).any():
                    break
    finally:
        residency.set_collect_gpu_timing(False)

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=device)
        stop_positions = torch.isin(output_ids[0, num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
        if stop_positions.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + int(stop_positions[0]) + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    timing = {
        "time_to_first_token_s": time_to_first_token,
        "time_between_tokens_s": (
            sum(decode_step_times) / max(num_output_tokens, 1)
            if decode_step_times
            else None
        ),
        "generated_tokens": int(num_output_tokens),
    }
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        timing=timing,
        acceptance_lengths=accepted_lengths,
        target_verify_count=target_verify_count,
        linear_rollback_count=linear_rollback_count,
    )


def load_draft_model(model_id: str, device, attn_implementation: Optional[str]):
    kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        draft = AutoModel.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        draft = AutoModel.from_pretrained(model_id, **kwargs)
    return draft.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser(description="DFlash speculative decoding with compressed layerwise target.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_ID)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--slots", type=int, choices=(1, 2), default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-attn-implementation", default=None)
    parser.add_argument("--report-memory", action="store_true")
    parser.add_argument("--report-time", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    target_base._load_runtime()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    device = torch.device("cuda")
    if args.report_memory:
        torch.cuda.reset_peak_memory_stats(device)

    print("loading compressed target")
    model, tokenizer, residency, plan, embed_weight, lm_head_weight = target_base.load_slot_model(
        args.model_dir,
        device,
        args.slots,
    )
    target_base._maybe_report_cuda_memory(device, "after target load", args.report_memory)
    target_base._report_layerwise_breakdown(
        "after target load",
        residency,
        model,
        plan,
        embed_weight,
        lm_head_weight,
        args.report_memory,
    )

    print("loading dflash draft")
    draft = load_draft_model(args.draft_model, device, args.draft_attn_implementation)
    target_base._maybe_report_cuda_memory(device, "after draft load", args.report_memory)

    block_size = int(args.block_size or getattr(draft, "block_size", draft.config.block_size))
    mask_token_id = int(getattr(draft, "mask_token_id", draft.config.dflash_config["mask_token_id"]))
    target_layer_ids = list(getattr(draft, "target_layer_ids", draft.config.dflash_config["target_layer_ids"]))
    if args.debug:
        print("block_size", block_size)
        print("mask_token_id", mask_token_id)
        print("target_layer_ids", target_layer_ids)

    prompt_text = target_base._prepare_prompt(tokenizer, args.prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    stop_ids = []
    if tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
        stop_ids.extend(eos if isinstance(eos, list) else [eos])

    result = dflash_layerwise_generate(
        model=model,
        tokenizer=tokenizer,
        residency=residency,
        plan=plan,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        draft=draft,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids or None,
        temperature=args.temperature,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        mask_token_id=mask_token_id,
        report_memory=args.report_memory,
        report_time=args.report_time,
        debug=args.debug,
    )

    target_base._maybe_report_cuda_memory(device, "after generation", args.report_memory)
    target_base._report_layerwise_breakdown(
        "after generation",
        residency,
        model,
        plan,
        embed_weight,
        lm_head_weight,
        args.report_memory,
    )
    target_base._report_gpu_timing(residency, result.timing, args.report_time)

    mean_accept = sum(result.acceptance_lengths) / max(len(result.acceptance_lengths), 1)
    print(f"[dflash] block_size={block_size} mean_acceptance_length={mean_accept:.2f} "
          f"verify_blocks={result.target_verify_count} rollbacks={result.linear_rollback_count}")
    print(f"[dflash] acceptance_lengths={result.acceptance_lengths[:80]}")

    new_tokens = result.output_ids[0, result.num_input_tokens :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
