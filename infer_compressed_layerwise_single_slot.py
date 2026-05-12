#!/usr/bin/env python3
"""
Standalone slot-buffered GPU inference for compressed model folders.

This script keeps compressed layer tensors and codebooks on CUDA, decodes one
layer at a time into reusable CUDA slot buffers, and measures peak VRAM/timing.
It does not import `infer_compressed_layerwise.py` and intentionally has no
CPU-cache, hybrid, or `--decode-on-gpu` mode.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import inspect
import json
import math
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple


torch = None
F = None
init_empty_weights = None
set_module_tensor_to_device = None
DynamicCache = None
AutoConfig = None
AutoModelForCausalLM = None
AutoTokenizer = None
create_causal_mask = None
create_sliding_window_causal_mask = None
functional_call = None
decode_rvq_triton_into = None
rvq_triton_supported = None


def _load_runtime() -> None:
    global torch, F, init_empty_weights, set_module_tensor_to_device
    global DynamicCache, AutoConfig, AutoModelForCausalLM, AutoTokenizer
    global create_causal_mask, create_sliding_window_causal_mask, functional_call
    global decode_rvq_triton_into, rvq_triton_supported

    if torch is not None:
        return

    import torch as torch_mod
    import torch.nn.functional as F_mod
    from accelerate import init_empty_weights as init_empty_weights_mod
    from accelerate.utils.modeling import set_module_tensor_to_device as set_tensor_mod
    from transformers import AutoConfig as AutoConfig_mod
    from transformers import AutoModelForCausalLM as AutoModelForCausalLM_mod
    from transformers import AutoTokenizer as AutoTokenizer_mod
    from transformers.cache_utils import DynamicCache as DynamicCache_mod
    from transformers.masking_utils import create_causal_mask as create_causal_mask_mod

    try:
        from torch.func import functional_call as functional_call_mod
    except ImportError:  # pragma: no cover - older torch fallback.
        from torch.nn.utils.stateless import functional_call as functional_call_mod

    try:
        from transformers.masking_utils import (
            create_sliding_window_causal_mask as create_sliding_mod,
        )
    except ImportError:  # pragma: no cover - version-dependent helper.
        create_sliding_mod = None

    try:
        from triton_decode_kernels import (
            decode_rvq_triton_into as triton_decode_mod,
            rvq_triton_supported as triton_supported_mod,
        )
    except Exception:  # pragma: no cover - Triton backend is optional.
        triton_decode_mod = None

        def triton_supported_mod(entry, out):  # type: ignore[no-redef]
            return False

    torch = torch_mod
    F = F_mod
    init_empty_weights = init_empty_weights_mod
    set_module_tensor_to_device = set_tensor_mod
    DynamicCache = DynamicCache_mod
    AutoConfig = AutoConfig_mod
    AutoModelForCausalLM = AutoModelForCausalLM_mod
    AutoTokenizer = AutoTokenizer_mod
    create_causal_mask = create_causal_mask_mod
    create_sliding_window_causal_mask = create_sliding_mod
    functional_call = functional_call_mod
    decode_rvq_triton_into = triton_decode_mod
    rvq_triton_supported = triton_supported_mod


def _dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _decode_bf16_raw_into(entry: Dict[str, object], out):
    out.copy_(entry["data"].view(*entry["shape"]))
    return out.contiguous()


def _decode_int8_into(entry: Dict[str, object], out):
    q = entry["q"] if entry["q"].dtype == torch.float32 else entry["q"].to(torch.float32)
    scales = entry["scales"] if entry["scales"].dtype == torch.float32 else entry["scales"].to(torch.float32)
    tail = entry["tail"] if entry["tail"].dtype == torch.float32 else entry["tail"].to(torch.float32)
    trimmed = int(entry["trimmed_numel"])
    core = (q * scales.unsqueeze(1)).reshape(-1)[:trimmed]
    flat_out = out.view(-1)
    flat_out[:trimmed].copy_(core)
    if tail.numel():
        flat_out[trimmed : trimmed + tail.numel()].copy_(tail)
    return out.contiguous()


def _decode_rvq_into(entry: Dict[str, object], codebook, out, chunk: int = 262144):
    d = int(entry["d"])
    group_size = int(entry["group_size"])
    trimmed = int(entry["trimmed_numel"])
    scales = entry["scales"] if entry["scales"].dtype == torch.float32 else entry["scales"].to(torch.float32)
    indices = entry["indices"] if entry["indices"].dtype == torch.int64 else entry["indices"].to(torch.int64)
    tail = entry["tail"] if entry["tail"].dtype == torch.float32 else entry["tail"].to(torch.float32)

    codebook = codebook if codebook.dtype == torch.float32 else codebook.to(torch.float32)
    if not codebook.is_contiguous():
        codebook = codebook.contiguous()

    stages, nvec = indices.shape
    recon = torch.empty((nvec, d), dtype=torch.float32, device=codebook.device)
    stage_ids = torch.arange(stages, device=codebook.device)
    effective_chunk = max(1, chunk // max(stages, 1))
    for start in range(0, nvec, effective_chunk):
        end = min(start + effective_chunk, nvec)
        gathered = codebook[stage_ids[:, None], indices[:, start:end]]
        recon[start:end].copy_(gathered.sum(dim=0))

    core = (recon.reshape(-1, group_size) * scales.unsqueeze(1)).reshape(-1)[:trimmed]
    flat_out = out.view(-1)
    flat_out[:trimmed].copy_(core)
    if tail.numel():
        flat_out[trimmed : trimmed + tail.numel()].copy_(tail)
    return out.contiguous()


def _decode_entry_into(entry: Dict[str, object], shared_codebooks: Dict[str, object], out):
    method = entry["method"]
    if method == "bf16_raw":
        return _decode_bf16_raw_into(entry, out)
    if method == "int8_groupwise":
        return _decode_int8_into(entry, out)
    if method in ("rvq_groupwise", "rvq_mlp"):
        codebook = shared_codebooks[entry["codebook_id"]]
        if decode_rvq_triton_into is not None and rvq_triton_supported(entry, out):
            return decode_rvq_triton_into(entry, codebook, out)
        return _decode_rvq_into(entry, codebook, out)
    raise ValueError(f"Unknown method: {method}")


def _decode_entry(entry: Dict[str, object], shared_codebooks: Dict[str, object]):
    out = torch.empty(tuple(entry["shape"]), dtype=_dtype(entry["dtype"]), device=_entry_device(entry, shared_codebooks))
    return _decode_entry_into(entry, shared_codebooks, out)


def _entry_device(entry: Dict[str, object], shared_codebooks: Dict[str, object]):
    if entry["method"] in ("rvq_groupwise", "rvq_mlp"):
        return shared_codebooks[entry["codebook_id"]].device
    for value in entry.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _strip_control_tokens(text: str, eos_token: Optional[str] = None) -> str:
    tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
    if eos_token:
        tokens.append(eos_token)
    out = text.rstrip()
    changed = True
    while changed:
        changed = False
        for token in tokens:
            if token and out.endswith(token):
                out = out[: -len(token)].rstrip()
                changed = True
    return out


def _split_thinking(text: str) -> Tuple[Optional[str], str]:
    start = text.find("<think>")
    if start == -1:
        return None, text.strip()
    after_start = text[start + len("<think>") :]
    end = after_start.find("</think>")
    if end == -1:
        return after_start.strip() or None, text[:start].strip()
    think = after_start[:end].strip()
    answer = (text[:start] + after_start[end + len("</think>") :]).strip()
    return think or None, answer


def _format_generation(text: str, eos_token: Optional[str] = None) -> str:
    cleaned = _strip_control_tokens(text, eos_token=eos_token)
    _, answer = _split_thinking(cleaned)
    return answer or cleaned


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
        for key, value in raw.get("text_config", raw).items():
            if not hasattr(config, key) and not isinstance(value, (dict, list)):
                config.__dict__[key] = value

    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    return config


def _build_meta_model(compressed_dir: str):
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(_load_model_config(compressed_dir), trust_remote_code=True)


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


def _remap_entry_prefixes(entry_map: Dict[str, Dict[str, object]], model_state_keys: List[str]):
    anchors = ("embed_tokens.weight", "lm_head.weight", "layers.0.input_layernorm.weight")
    model_prefix = ""
    artifact_prefix = ""
    for anchor in anchors:
        model_prefix = _prefix_for_suffix(model_state_keys, anchor)
        artifact_prefix = _prefix_for_suffix(entry_map.keys(), anchor)
        if model_prefix or artifact_prefix:
            break
    if model_prefix == artifact_prefix:
        return entry_map

    print(f"Prefix remap: '{artifact_prefix}*' -> '{model_prefix}*'")
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
    usable = {key: value for key, value in remapped.items() if key in set(model_state_keys)}
    ignored = sorted(set(remapped) - set(usable))
    if ignored:
        print(f"Note: {len(ignored)} extra tensors ignored (e.g. {ignored[0]})")

    base_prefix = _prefix_for_suffix(model_state_keys, "embed_tokens.weight")
    embed_key = f"{base_prefix}embed_tokens.weight" if f"{base_prefix}embed_tokens.weight" in usable else None
    lm_head_prefix = _prefix_for_suffix(model_state_keys, "lm_head.weight")
    lm_head_key = f"{lm_head_prefix}lm_head.weight" if f"{lm_head_prefix}lm_head.weight" in usable else None

    layer_re = re.compile(rf"^{re.escape(base_prefix)}layers\.(\d+)\.")
    layer_groups: Dict[int, List[str]] = {}
    for key in usable:
        match = layer_re.match(key)
        if match:
            layer_groups.setdefault(int(match.group(1)), []).append(key)

    norm_prefix = f"{base_prefix}norm."
    norm_keys = sorted(key for key in usable if key.startswith(norm_prefix))
    layer_keys = [sorted(layer_groups[idx]) for idx in sorted(layer_groups)]
    base_model = _find_base_model(model)
    if len(layer_keys) != len(base_model.layers):
        raise RuntimeError(f"Layer/key mismatch: {len(layer_keys)} groups vs {len(base_model.layers)} layers.")

    return {
        "base_prefix": base_prefix,
        "entries": usable,
        "base_model": base_model,
        "embed_key": embed_key,
        "lm_head_key": lm_head_key,
        "layer_keys": layer_keys,
        "norm_keys": norm_keys,
    }


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _object_tensor_nbytes(obj) -> int:
    if torch.is_tensor(obj):
        return _tensor_nbytes(obj)
    if isinstance(obj, dict):
        return sum(_object_tensor_nbytes(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_object_tensor_nbytes(value) for value in obj)
    return 0


def _decoded_entry_nbytes(entry: Dict[str, object], runtime_dtype) -> int:
    return int(math.prod(int(dim) for dim in entry["shape"])) * torch.empty((), dtype=runtime_dtype).element_size()


def _tensor_identity(tensor):
    if tensor.device.type == "meta":
        return None
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


def _sum_unique_tensor_bytes(tensors: List[object]) -> int:
    total = 0
    seen = set()
    for tensor in tensors:
        identity = _tensor_identity(tensor)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        total += _tensor_nbytes(tensor)
    return total


def _resolve_named_tensor(model, tensor_name: str):
    module = model
    if "." in tensor_name:
        parts = tensor_name.split(".")
        for part in parts[:-1]:
            module = getattr(module, part)
        tensor_name = parts[-1]
    return module, tensor_name


def _materialize_keys(model, keys: List[str], entries, shared_codebooks, device, runtime_dtype) -> None:
    for key in keys:
        value = _decode_entry(entries[key], shared_codebooks)
        dtype = runtime_dtype if value.is_floating_point() else None
        set_module_tensor_to_device(model, key, device, value=value, dtype=dtype)


def _move_module_buffers_to_device(model, module_prefix: str, device) -> None:
    prefix = f"{module_prefix}."
    for name, _ in model.named_buffers():
        if name == module_prefix or name.startswith(prefix):
            set_module_tensor_to_device(model, name, device)


def _move_entry_tensors_to_gpu(entry: Dict[str, object], device) -> Dict[str, object]:
    moved = {}
    for key, value in entry.items():
        moved[key] = value.to(device=device, non_blocking=False) if torch.is_tensor(value) else value
    return moved


def _collect_always_hot_entry_keys(plan) -> List[str]:
    keys: Set[str] = set(plan["norm_keys"])
    if plan["embed_key"] is not None:
        keys.add(plan["embed_key"])
    if plan["lm_head_key"] is not None:
        keys.add(plan["lm_head_key"])

    rotary_prefix = f"{plan['base_prefix']}rotary_emb"
    rotary_prefix_dot = f"{rotary_prefix}."
    for key in plan["entries"]:
        if key == rotary_prefix or key.startswith(rotary_prefix_dot):
            keys.add(key)
    return sorted(keys)


def _drop_always_hot_compressed_entries(plan) -> int:
    removed = 0
    for key in _collect_always_hot_entry_keys(plan):
        entry = plan["entries"].pop(key, None)
        if entry is not None:
            removed += _object_tensor_nbytes(entry)
    return removed


class SlotResidency:
    def __init__(self, model, plan, shared_codebooks, device, runtime_dtype, slot_count: int) -> None:
        if slot_count not in (1, 2):
            raise ValueError("--slots must be 1 or 2.")
        self.model = model
        self.plan = plan
        self.device = device
        self.runtime_dtype = runtime_dtype
        self.slot_count = slot_count
        self.num_layers = len(plan["layer_keys"])
        self.decode_on_gpu = True
        self.gpu_entry_cache: Dict[str, Dict[str, object]] = {}
        self.layer_buffers_on_device: Set[int] = set()
        self.resident_layers: Set[int] = set()
        self.pending_layers: Set[int] = set()
        self.pending_ready_events: Dict[int, object] = {}
        self.prefetch_stream = torch.cuda.Stream(device=device) if slot_count == 2 else None
        self.slot_states: List[Dict[str, object]] = [dict() for _ in range(slot_count)]
        self.layer_to_slot: Dict[int, int] = {}
        self.slot_to_layer: Dict[int, int] = {}
        self.peak_working_window_bytes = 0
        self.collect_gpu_timing = False
        self.active_profile_label: Optional[str] = None
        self.gpu_layer_decode_events: List[Tuple[Optional[str], int, object, object]] = []
        self.gpu_layer_forward_events: List[Tuple[Optional[str], int, object, object]] = []
        self.compressed_artifact_file_bytes = 0
        self.layer_dense_bytes = {
            idx: sum(_decoded_entry_nbytes(plan["entries"][key], runtime_dtype) for key in layer_keys)
            for idx, layer_keys in enumerate(plan["layer_keys"])
        }
        self.layer_relative_keys = []
        for idx, layer_keys in enumerate(plan["layer_keys"]):
            prefix = f"{plan['base_prefix']}layers.{idx}."
            self.layer_relative_keys.append({key: key[len(prefix) :] if key.startswith(prefix) else key for key in layer_keys})
        self.shared_codebooks_device = {
            key: value.to(device=device, dtype=torch.float32).contiguous()
            for key, value in shared_codebooks.items()
        }

    def close(self) -> None:
        pass

    def set_collect_gpu_timing(self, enabled: bool) -> None:
        self.collect_gpu_timing = bool(enabled)

    def reset_gpu_timing_events(self) -> None:
        self.gpu_layer_decode_events.clear()
        self.gpu_layer_forward_events.clear()

    def set_active_profile_label(self, label: Optional[str]) -> None:
        self.active_profile_label = label

    def preload_compressed_entries_to_gpu(self) -> None:
        seen: Set[str] = set()
        for layer_keys in self.plan["layer_keys"]:
            for key in layer_keys:
                if key in seen:
                    continue
                self.gpu_entry_cache[key] = _move_entry_tensors_to_gpu(self.plan["entries"][key], self.device)
                seen.add(key)

    def warmup_triton_decode(self) -> None:
        if decode_rvq_triton_into is None:
            return
        warmed: Set[Tuple[object, ...]] = set()
        for layer_keys in self.plan["layer_keys"]:
            for key in layer_keys:
                entry = self.gpu_entry_cache[key]
                if entry.get("method") not in ("rvq_groupwise", "rvq_mlp"):
                    continue
                family = (
                    entry["method"],
                    tuple(entry["shape"]),
                    int(entry["d"]),
                    int(entry["group_size"]),
                    tuple(entry["indices"].shape),
                    str(self.runtime_dtype),
                )
                if family in warmed:
                    continue
                out = torch.empty(tuple(entry["shape"]), dtype=self.runtime_dtype, device=self.device)
                if rvq_triton_supported(entry, out):
                    decode_rvq_triton_into(entry, self.shared_codebooks_device[entry["codebook_id"]], out)
                    warmed.add(family)
                del out
        torch.cuda.synchronize(self.device)

    def decoded_cache_nbytes(self) -> int:
        return 0

    def compressed_store_nbytes(self) -> int:
        return _object_tensor_nbytes(self.gpu_entry_cache) + _object_tensor_nbytes(self.shared_codebooks_device)

    def _event_totals_by_label(self, events):
        if not events:
            return []
        torch.cuda.synchronize(self.device)
        totals: Dict[str, float] = {}
        for label, _, start_event, end_event in events:
            totals[label or "unlabeled"] = totals.get(label or "unlabeled", 0.0) + start_event.elapsed_time(end_event)
        return list(totals.items())

    def average_gpu_layer_decode_ms(self) -> Optional[float]:
        if not self.gpu_layer_decode_events:
            return None
        torch.cuda.synchronize(self.device)
        return sum(start.elapsed_time(end) for _, _, start, end in self.gpu_layer_decode_events) / len(
            self.gpu_layer_decode_events
        )

    def decode_ms_per_layer_index(self) -> List[Tuple[int, float]]:
        if not self.gpu_layer_decode_events:
            return []
        torch.cuda.synchronize(self.device)
        totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for _, layer_idx, start, end in self.gpu_layer_decode_events:
            totals[layer_idx] = totals.get(layer_idx, 0.0) + start.elapsed_time(end)
            counts[layer_idx] = counts.get(layer_idx, 0) + 1
        return [(idx, totals[idx] / counts[idx]) for idx in sorted(totals)]

    def decode_ms_per_token(self):
        return self._event_totals_by_label(self.gpu_layer_decode_events)

    def forward_ms_per_token(self):
        return self._event_totals_by_label(self.gpu_layer_forward_events)

    def prepare_for_forward(self) -> None:
        if self.num_layers:
            self.ensure_layer_ready(0)

    def ensure_layer_ready(self, layer_idx: int) -> None:
        ready = self.pending_ready_events.pop(layer_idx, None)
        if ready is not None:
            torch.cuda.current_stream(self.device).wait_event(ready)
            self.pending_layers.discard(layer_idx)
            self.resident_layers.add(layer_idx)
            self._update_peak_working_window_bytes()
            return
        if layer_idx in self.resident_layers:
            return
        self._decode_layer_sync(layer_idx)

    def start_prefetch_for_layer(self, layer_idx: int) -> None:
        if self.prefetch_stream is None or layer_idx >= self.num_layers:
            return
        if layer_idx in self.resident_layers or layer_idx in self.pending_layers:
            return
        self._decode_layer_async(layer_idx)

    def run_layer(self, layer_idx: int, layer, layer_kwargs):
        if layer_idx not in self.resident_layers:
            raise RuntimeError(f"Layer {layer_idx} is not decoded into the slot.")
        return functional_call(layer, self.slot_states[self.layer_to_slot[layer_idx]], (), layer_kwargs, strict=False)

    def offload_layer(self, layer_idx: int) -> None:
        if layer_idx in self.pending_layers:
            ready = self.pending_ready_events.pop(layer_idx, None)
            if ready is not None:
                torch.cuda.current_stream(self.device).wait_event(ready)
            self.pending_layers.discard(layer_idx)
            self._release_slot(layer_idx)
        if layer_idx in self.resident_layers:
            self.resident_layers.discard(layer_idx)
            self._release_slot(layer_idx)
        self._update_peak_working_window_bytes()

    def _move_layer_buffers_once(self, layer_idx: int) -> None:
        if layer_idx in self.layer_buffers_on_device:
            return
        _move_module_buffers_to_device(self.model, f"{self.plan['base_prefix']}layers.{layer_idx}", self.device)
        self.layer_buffers_on_device.add(layer_idx)

    def _acquire_slot(self, layer_idx: int) -> int:
        existing = self.layer_to_slot.get(layer_idx)
        if existing is not None:
            return existing
        for slot_idx in range(self.slot_count):
            if slot_idx not in self.slot_to_layer:
                self.layer_to_slot[layer_idx] = slot_idx
                self.slot_to_layer[slot_idx] = layer_idx
                return slot_idx
        raise RuntimeError("No free slot available.")

    def _release_slot(self, layer_idx: int) -> None:
        slot_idx = self.layer_to_slot.pop(layer_idx, None)
        if slot_idx is not None:
            self.slot_to_layer.pop(slot_idx, None)

    def _update_peak_working_window_bytes(self) -> None:
        active_layers = self.resident_layers | self.pending_layers
        current = sum(self.layer_dense_bytes[layer_idx] for layer_idx in active_layers)
        self.peak_working_window_bytes = max(self.peak_working_window_bytes, current)

    def _ensure_slot_tensor(self, layer_idx: int, slot_idx: int, key: str):
        rel_name = self.layer_relative_keys[layer_idx][key]
        entry = self.plan["entries"][key]
        shape = tuple(entry["shape"])
        slot_state = self.slot_states[slot_idx]
        cached = slot_state.get(rel_name)
        if cached is None or cached.shape != shape or cached.dtype != self.runtime_dtype or cached.device != self.device:
            cached = torch.empty(shape, dtype=self.runtime_dtype, device=self.device)
            slot_state[rel_name] = cached
        return cached

    def _prune_slot_state(self, layer_idx: int, slot_idx: int) -> None:
        expected = set(self.layer_relative_keys[layer_idx].values())
        slot_state = self.slot_states[slot_idx]
        for rel_name in list(slot_state.keys()):
            if rel_name not in expected:
                del slot_state[rel_name]

    def _decode_layer_sync(self, layer_idx: int) -> None:
        self._move_layer_buffers_once(layer_idx)
        slot_idx = self._acquire_slot(layer_idx)
        start_event = end_event = None
        if self.collect_gpu_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))

        with _nvtx_range(f"layer_{layer_idx:03d}_decode"):
            for key in self.plan["layer_keys"][layer_idx]:
                out = self._ensure_slot_tensor(layer_idx, slot_idx, key)
                _decode_entry_into(self.gpu_entry_cache[key], self.shared_codebooks_device, out)

        self._prune_slot_state(layer_idx, slot_idx)

        if self.collect_gpu_timing and start_event is not None and end_event is not None:
            end_event.record(torch.cuda.current_stream(self.device))
            self.gpu_layer_decode_events.append((self.active_profile_label, layer_idx, start_event, end_event))

        self.resident_layers.add(layer_idx)
        self._update_peak_working_window_bytes()

    def _decode_layer_async(self, layer_idx: int) -> None:
        self._move_layer_buffers_once(layer_idx)
        slot_idx = self._acquire_slot(layer_idx)
        start_event = end_event = None
        with torch.cuda.stream(self.prefetch_stream):
            if self.collect_gpu_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(self.prefetch_stream)
            with _nvtx_range(f"layer_{layer_idx:03d}_decode_prefetch"):
                for key in self.plan["layer_keys"][layer_idx]:
                    out = self._ensure_slot_tensor(layer_idx, slot_idx, key)
                    _decode_entry_into(self.gpu_entry_cache[key], self.shared_codebooks_device, out)
            if end_event is not None:
                end_event.record(self.prefetch_stream)
            ready = torch.cuda.Event()
            ready.record(self.prefetch_stream)

        self._prune_slot_state(layer_idx, slot_idx)
        if self.collect_gpu_timing and start_event is not None and end_event is not None:
            self.gpu_layer_decode_events.append((self.active_profile_label, layer_idx, start_event, end_event))
        self.pending_layers.add(layer_idx)
        self.pending_ready_events[layer_idx] = ready
        self._update_peak_working_window_bytes()

    def record_gpu_forward_event(self, layer_idx: int, start_event, end_event) -> None:
        if self.collect_gpu_timing and start_event is not None and end_event is not None:
            self.gpu_layer_forward_events.append((self.active_profile_label, layer_idx, start_event, end_event))


def _get_forward_global(module, name: str):
    return getattr(module.forward, "__globals__", {}).get(name)


def _build_attention_mask_mapping(base_model, hidden_states, attention_mask, cache_position, position_ids, past_key_values):
    if isinstance(attention_mask, dict):
        return attention_mask

    mask_kwargs = {
        "config": base_model.config,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    create_causal = _get_forward_global(base_model, "create_causal_mask") or create_causal_mask
    create_sliding = _get_forward_global(base_model, "create_sliding_window_causal_mask") or create_sliding_window_causal_mask

    try:
        full_attention = create_causal(inputs_embeds=hidden_states, **mask_kwargs)
    except TypeError:
        full_attention = create_causal(input_embeds=hidden_states, **mask_kwargs)

    mapping = {"full_attention": full_attention}
    if hasattr(base_model, "_update_linear_attn_mask"):
        update_linear_mask = base_model._update_linear_attn_mask
        try:
            params = inspect.signature(update_linear_mask).parameters
        except (TypeError, ValueError):
            params = {}
        if "past_key_values" in params:
            mapping["linear_attention"] = update_linear_mask(attention_mask, past_key_values)
        elif "cache_position" in params:
            mapping["linear_attention"] = update_linear_mask(attention_mask, cache_position)
        else:
            try:
                mapping["linear_attention"] = update_linear_mask(attention_mask, past_key_values)
            except (TypeError, AttributeError):
                mapping["linear_attention"] = update_linear_mask(attention_mask, cache_position)

    if getattr(base_model, "has_sliding_layers", False) and create_sliding is not None:
        try:
            mapping["sliding_attention"] = create_sliding(inputs_embeds=hidden_states, **mask_kwargs)
        except TypeError:
            mapping["sliding_attention"] = create_sliding(input_embeds=hidden_states, **mask_kwargs)
    return mapping


def _select_layer_mask(layer, mask_mapping):
    if hasattr(layer, "layer_type"):
        return mask_mapping.get(layer.layer_type, mask_mapping["full_attention"])
    if hasattr(layer, "attention_type"):
        return mask_mapping.get(layer.attention_type, mask_mapping["full_attention"])
    return mask_mapping["full_attention"]


def _make_dynamic_cache(base_model):
    cache = None
    for name in ("Qwen3_5DynamicCache", "Qwen3NextDynamicCache", "DynamicCache"):
        cache_cls = _get_forward_global(base_model, name)
        if cache_cls is None:
            continue
        try:
            cache = cache_cls(config=base_model.config)
        except TypeError:
            cache = cache_cls()
        break
    if cache is None:
        cache = DynamicCache(config=base_model.config)

    num_layers = len(getattr(base_model, "layers", []))
    if not hasattr(cache, "conv_states"):
        cache.conv_states = [None] * num_layers
    if not hasattr(cache, "recurrent_states"):
        cache.recurrent_states = [None] * num_layers
    if not hasattr(cache, "has_previous_state"):
        cache.has_previous_state = False
    return cache


@contextmanager
def _nvtx_range(label: str):
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "nvtx"):
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _synchronize_device(device) -> None:
    torch.cuda.synchronize(device)


def _maybe_report_cuda_memory(device, label: str, enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"[cuda-memory] {label}: allocated={allocated:.1f} MiB reserved={reserved:.1f} MiB peak={peak:.1f} MiB")


def _bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024**2)


def _always_hot_tensor_bytes(model, plan, embed_weight, lm_head_weight) -> int:
    tensors = [embed_weight]
    if lm_head_weight is not embed_weight:
        tensors.append(lm_head_weight)

    for key in plan["norm_keys"]:
        module, leaf_name = _resolve_named_tensor(model, key)
        tensor = getattr(module, leaf_name)
        if torch.is_tensor(tensor):
            tensors.append(tensor)

    rotary_prefix = f"{plan['base_prefix']}rotary_emb"
    rotary_dot = f"{rotary_prefix}."
    for name, buffer in model.named_buffers():
        if name == rotary_prefix or name.startswith(rotary_dot):
            tensors.append(buffer)
    return _sum_unique_tensor_bytes(tensors)


def _report_layerwise_breakdown(label, residency, model, plan, embed_weight, lm_head_weight, enabled: bool) -> None:
    if not enabled:
        return
    print(
        f"[layerwise-breakdown] {label}: "
        f"compressed_store_tensor={_bytes_to_mib(residency.compressed_store_nbytes()):.1f} MiB "
        f"compressed_artifact_file={_bytes_to_mib(residency.compressed_artifact_file_bytes):.1f} MiB "
        f"decoded_cache={_bytes_to_mib(residency.decoded_cache_nbytes()):.1f} MiB "
        f"always_hot={_bytes_to_mib(_always_hot_tensor_bytes(model, plan, embed_weight, lm_head_weight)):.1f} MiB "
        f"peak_working_window={_bytes_to_mib(residency.peak_working_window_bytes):.1f} MiB"
    )


def _print_named_ms_chunks(tag: str, items: List[Tuple[str, float]], chunk_size: int = 8) -> None:
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        print(f"{tag} " + " ".join(f"{name}={value:.1f}ms" for name, value in chunk))


def _report_gpu_timing(residency, timing: Dict[str, object], enabled: bool) -> None:
    if not enabled:
        return
    ttft = "n/a" if timing["time_to_first_token_s"] is None else f"{timing['time_to_first_token_s'] * 1000:.1f} ms"
    tbt = "n/a" if timing["time_between_tokens_s"] is None else f"{timing['time_between_tokens_s'] * 1000:.1f} ms/token"
    print(f"[generation-time] time_to_first_token={ttft} time_between_tokens={tbt} generated_tokens={timing['generated_tokens']}")

    avg_ms = residency.average_gpu_layer_decode_ms()
    if avg_ms is None:
        return
    print(f"[gpu-layer-decode] average={avg_ms:.1f} ms/layer events={len(residency.gpu_layer_decode_events)}")
    _print_named_ms_chunks(
        "[gpu-layer-decode-per-layer]",
        [(f"layer_{idx:03d}", ms) for idx, ms in residency.decode_ms_per_layer_index()],
    )
    _print_named_ms_chunks("[gpu-decode-per-token]", residency.decode_ms_per_token())
    _print_named_ms_chunks("[gpu-forward-per-token]", residency.forward_ms_per_token())


def _layerwise_logits(
    model,
    plan,
    residency: SlotResidency,
    embed_weight,
    lm_head_weight,
    input_ids,
    attention_mask,
    past_key_values=None,
    use_cache: bool = False,
    cache_position=None,
    profile_pass_label: Optional[str] = None,
):
    base_model = plan["base_model"]
    device = input_ids.device
    hidden_states = F.embedding(input_ids, embed_weight)
    if cache_position is None:
        cache_position = torch.arange(hidden_states.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids) if hasattr(base_model, "rotary_emb") else None

    mask_mapping = _build_attention_mask_mapping(
        base_model,
        hidden_states,
        attention_mask,
        cache_position,
        position_ids,
        past_key_values,
    )

    residency.set_active_profile_label(profile_pass_label)
    try:
        residency.prepare_for_forward()
        for layer_idx, layer in enumerate(base_model.layers):
            residency.ensure_layer_ready(layer_idx)
            residency.start_prefetch_for_layer(layer_idx + 1)
            layer_kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": _select_layer_mask(layer, mask_mapping),
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            start_event = end_event = None
            if residency.collect_gpu_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(device))
            with _nvtx_range(f"layer_{layer_idx:03d}_forward"):
                hidden_states = residency.run_layer(layer_idx, layer, layer_kwargs)
            if residency.collect_gpu_timing and start_event is not None and end_event is not None:
                end_event.record(torch.cuda.current_stream(device))
            residency.record_gpu_forward_event(layer_idx, start_event, end_event)
            residency.offload_layer(layer_idx)
    finally:
        residency.set_active_profile_label(None)

    hidden_states = base_model.norm(hidden_states)
    logits = F.linear(hidden_states[:, -1:, :], lm_head_weight)
    return logits[:, -1, :], past_key_values


def _keep_tied_head(plan, model) -> bool:
    return bool(getattr(model.config, "tie_word_embeddings", False)) or plan["lm_head_key"] is None


def load_slot_model(compressed_dir: str, device, slot_count: int):
    artifact_path = os.path.join(compressed_dir, "compressed_model.pt")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Missing {artifact_path}")

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    entry_map = artifact["entries"]
    shared_codebooks = artifact["shared_codebooks"]

    model = _build_meta_model(compressed_dir)
    plan = _build_layer_plan(model, entry_map)
    tokenizer = AutoTokenizer.from_pretrained(compressed_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    runtime_dtype = torch.bfloat16
    if plan["embed_key"] is None:
        raise RuntimeError("Could not find embed_tokens.weight in the compressed artifact.")
    embed_weight = _decode_entry(plan["entries"][plan["embed_key"]], shared_codebooks).to(device=device, dtype=runtime_dtype)
    lm_head_weight = (
        embed_weight
        if _keep_tied_head(plan, model)
        else _decode_entry(plan["entries"][plan["lm_head_key"]], shared_codebooks).to(device=device, dtype=runtime_dtype)
    )

    if plan["norm_keys"]:
        _materialize_keys(model, plan["norm_keys"], plan["entries"], shared_codebooks, device, runtime_dtype)
    _move_module_buffers_to_device(model, f"{plan['base_prefix']}rotary_emb", device)

    residency = SlotResidency(model, plan, shared_codebooks, device, runtime_dtype, slot_count)
    residency.compressed_artifact_file_bytes = os.path.getsize(artifact_path)
    residency.preload_compressed_entries_to_gpu()
    residency.warmup_triton_decode()
    residency.dropped_always_hot_tensor_bytes = _drop_always_hot_compressed_entries(plan)
    gc.collect()

    model.eval()
    return model, tokenizer, residency, plan, embed_weight, lm_head_weight


def _prepare_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_greedy(
    model,
    tokenizer,
    residency: SlotResidency,
    plan,
    embed_weight,
    lm_head_weight,
    prompt: str,
    max_new_tokens: int,
    device,
    report_memory: bool,
    report_time: bool,
):
    inputs = tokenizer(_prepare_prompt(tokenizer, prompt), return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated = input_ids
    generated_attention = attention_mask
    time_to_first_token_s = None
    decode_step_times: List[float] = []
    residency.reset_gpu_timing_events()
    residency.set_collect_gpu_timing(report_time)

    try:
        with torch.inference_mode():
            past_key_values = _make_dynamic_cache(plan["base_model"])
            if report_time:
                _synchronize_device(device)
                generation_start = time.perf_counter()
            logits, past_key_values = _layerwise_logits(
                model,
                plan,
                residency,
                embed_weight,
                lm_head_weight,
                generated,
                generated_attention,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=torch.arange(generated.shape[1], device=device),
                profile_pass_label="prefill",
            )
            if report_time:
                _synchronize_device(device)
                time_to_first_token_s = time.perf_counter() - generation_start
            if hasattr(past_key_values, "has_previous_state") and not past_key_values.has_previous_state:
                past_key_values.has_previous_state = True
            _maybe_report_cuda_memory(device, "after prefill", report_memory)

            for step in range(max_new_tokens):
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if generated_attention is not None:
                    next_mask = torch.ones((generated_attention.shape[0], 1), dtype=generated_attention.dtype, device=device)
                    generated_attention = torch.cat([generated_attention, next_mask], dim=1)
                if tokenizer.eos_token_id is not None and bool((next_token == tokenizer.eos_token_id).all()):
                    break

                if report_time:
                    _synchronize_device(device)
                    step_start = time.perf_counter()
                logits, past_key_values = _layerwise_logits(
                    model,
                    plan,
                    residency,
                    embed_weight,
                    lm_head_weight,
                    next_token,
                    generated_attention,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.tensor([generated.shape[1] - 1], device=device),
                    profile_pass_label=f"decode_step_{step:03d}",
                )
                if report_time:
                    _synchronize_device(device)
                    decode_step_times.append(time.perf_counter() - step_start)
                if step == 0:
                    _maybe_report_cuda_memory(device, "after first decode step", report_memory)
    finally:
        residency.set_collect_gpu_timing(False)

    new_tokens = generated[0, input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    timing = {
        "time_to_first_token_s": time_to_first_token_s,
        "time_between_tokens_s": (sum(decode_step_times) / len(decode_step_times)) if decode_step_times else None,
        "generated_tokens": int(new_tokens.shape[0]),
    }
    return _format_generation(raw, eos_token=tokenizer.eos_token), timing


def parse_args():
    parser = argparse.ArgumentParser(description="Slot-buffered GPU compressed-model inference benchmark.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--slots",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of decoded layer slots: 1 for single-slot, 2 for ping-pong prefetch.",
    )
    parser.add_argument("--report-memory", action="store_true")
    parser.add_argument("--report-time", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _load_runtime()
    if not torch.cuda.is_available():
        raise ValueError("This script requires CUDA.")
    device = torch.device("cuda")
    if args.report_memory:
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer, residency, plan, embed_weight, lm_head_weight = load_slot_model(args.model_dir, device, args.slots)
    _maybe_report_cuda_memory(device, "after load", args.report_memory)
    _report_layerwise_breakdown("after load", residency, model, plan, embed_weight, lm_head_weight, args.report_memory)

    try:
        text, timing = generate_greedy(
            model,
            tokenizer,
            residency,
            plan,
            embed_weight,
            lm_head_weight,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            report_memory=args.report_memory,
            report_time=args.report_time,
        )
    finally:
        residency.close()

    _maybe_report_cuda_memory(device, "after generation", args.report_memory)
    _report_layerwise_breakdown("after generation", residency, model, plan, embed_weight, lm_head_weight, args.report_memory)
    _report_gpu_timing(residency, timing, args.report_time)
    print(text)


if __name__ == "__main__":
    main()
