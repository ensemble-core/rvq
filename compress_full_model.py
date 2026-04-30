#!/usr/bin/env python3
"""
Residual Vector Quantization - Standalone compressed model packer.

Creates a compressed folder that does NOT require the original base model folder
at inference time.

Compression policy:
  - ALL 2D+ tensors: RVQ + per-group scaling
  - Small 1D tensors: keep bf16 raw
  - Configurable codebook sharing:
      * global: one shared codebook for all 2D tensors
      * family: one shared codebook per tensor family (recommended default)
      * tensor: one codebook per tensor (slow, quality-first)
"""

import argparse
import json
import math
import os
import random
import re
import shutil
from glob import glob
from typing import Dict, List

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm


def ensure_model_files(model_dir: str, repo_id: str, auto_download: bool) -> None:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    shards = glob(os.path.join(model_dir, "model-*.safetensors"))
    if os.path.exists(idx_path) or os.path.exists(single) or len(shards) > 0:
        return
    if not auto_download:
        raise FileNotFoundError(
            f"No model files found under {model_dir}. Use --auto-download to fetch."
        )
    print(f"Local model files not found under {model_dir}. Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id, local_dir=model_dir)
    print("Download complete.")


def load_weight_map(model_dir: str) -> Dict[str, str]:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        out = {}
        with safe_open(single, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = "model.safetensors"
        return out

    out = {}
    for sp in sorted(glob(os.path.join(model_dir, "model-*.safetensors"))):
        shard = os.path.basename(sp)
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = shard
    if not out:
        raise FileNotFoundError(f"No readable safetensors under {model_dir}")
    return out


def load_tensor(model_dir: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map[key]
    sp = os.path.join(model_dir, shard)
    with safe_open(sp, framework="pt", device="cpu") as f:
        return f.get_tensor(key).contiguous()


def load_tensor_shapes(model_dir: str, weight_map: Dict[str, str]) -> Dict[str, List[int]]:
    """Read tensor shapes from safetensors headers (fast, no data loaded)."""
    shapes: Dict[str, List[int]] = {}
    seen_shards: set = set()
    for key, shard in weight_map.items():
        if shard in seen_shards:
            continue
        seen_shards.add(shard)
        sp = os.path.join(model_dir, shard)
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                shapes[k] = list(f.get_slice(k).get_shape())
    return shapes


def build_logical_state_keys(weight_map: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Build logical state keys from raw checkpoint keys.

    Supports Hard-EM wrapped checkpoints where MLP weights are stored as:
      *.weight_fp + *.codebooks + *.indices + *.scales + *.tail
    and maps them back to logical *.weight.
    """
    logical: Dict[str, Dict[str, str]] = {}
    ignored_suffixes = (".codebooks", ".indices", ".scales", ".tail")

    for k in weight_map.keys():
        if k.endswith(ignored_suffixes):
            continue
        if k.endswith(".weight_fp"):
            continue
        logical[k] = {"kind": "direct", "key": k}

    for k in weight_map.keys():
        if not k.endswith(".weight_fp"):
            continue
        prefix = k[: -len("weight_fp")]  # keeps trailing dot
        logical_key = f"{prefix}weight"
        codebooks = f"{prefix}codebooks"
        indices = f"{prefix}indices"
        scales = f"{prefix}scales"
        tail = f"{prefix}tail"
        if (
            codebooks in weight_map
            and indices in weight_map
            and scales in weight_map
            and tail in weight_map
        ):
            logical[logical_key] = {
                "kind": "hardem",
                "weight_fp": k,
                "codebooks": codebooks,
                "indices": indices,
                "scales": scales,
                "tail": tail,
            }
    return logical


def reconstruct_hardem_weight(model_dir: str, weight_map: Dict[str, str], spec: Dict[str, str]) -> torch.Tensor:
    weight_fp = load_tensor(model_dir, weight_map, spec["weight_fp"]).to(torch.float32)
    codebooks = load_tensor(model_dir, weight_map, spec["codebooks"]).to(torch.float32)
    indices = load_tensor(model_dir, weight_map, spec["indices"]).to(torch.long)
    scales = load_tensor(model_dir, weight_map, spec["scales"]).to(torch.float32)
    tail = load_tensor(model_dir, weight_map, spec["tail"]).to(torch.float32)

    stages, nvec = indices.shape
    d = int(codebooks.shape[2])
    if int(codebooks.shape[0]) != stages:
        raise ValueError("HardEM reconstruct: stage mismatch")

    recon = torch.zeros((nvec, d), dtype=torch.float32)
    for r in range(stages):
        recon += codebooks[r][indices[r]]

    trimmed_numel = nvec * d
    if scales.numel() == 0 or trimmed_numel % scales.numel() != 0:
        raise ValueError("HardEM reconstruct: invalid scales")
    group_size = trimmed_numel // scales.numel()
    core = recon.reshape(-1, group_size) * scales.unsqueeze(1)
    core_flat = core.reshape(-1)
    full = torch.cat([core_flat, tail], dim=0)
    if full.numel() != weight_fp.numel():
        if full.numel() > weight_fp.numel():
            full = full[: weight_fp.numel()]
        else:
            full = torch.cat([full, torch.zeros(weight_fp.numel() - full.numel())], dim=0)
    return full.view_as(weight_fp).to(weight_fp.dtype).contiguous()


def load_effective_tensor(model_dir: str, weight_map: Dict[str, str], spec: Dict[str, str]) -> torch.Tensor:
    if spec["kind"] == "direct":
        return load_tensor(model_dir, weight_map, spec["key"])
    if spec["kind"] == "hardem":
        return reconstruct_hardem_weight(model_dir, weight_map, spec)
    raise ValueError(f"Unknown tensor spec kind: {spec['kind']}")


def copy_support_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    keep = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
    for name in keep:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def pairwise_sqdist(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    c2 = (c * c).sum(dim=1).unsqueeze(0)
    return x2 + c2 - 2.0 * (x @ c.t())


def assign_codes(x: torch.Tensor, c: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = x.shape[0]
    idx = torch.empty((n,), dtype=torch.int64)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        idx[s:e] = torch.argmin(pairwise_sqdist(x[s:e], c), dim=1)
    return idx


def kmeans_train(x: torch.Tensor, k: int, iters: int, seed: int, chunk_size: int) -> torch.Tensor:
    n, d = x.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centroids = x[init_idx].clone()
    for _ in range(iters):
        sums = torch.zeros((k, d), dtype=torch.float32)
        counts = torch.zeros((k,), dtype=torch.int64)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            xb = x[s:e]
            a = torch.argmin(pairwise_sqdist(xb, centroids), dim=1)
            counts += torch.bincount(a, minlength=k)
            sums.index_add_(0, a, xb)
        nz = counts > 0
        centroids[nz] = sums[nz] / counts[nz].unsqueeze(1)
        dead = (~nz).nonzero(as_tuple=False).view(-1)
        if dead.numel() > 0:
            refill = torch.randperm(n, generator=g)[: dead.numel()]
            centroids[dead] = x[refill]
    return centroids


def normalize_groups(flat: torch.Tensor, group_size: int, eps: float = 1e-8):
    trimmed = (flat.numel() // group_size) * group_size
    core = flat[:trimmed]
    tail = flat[trimmed:]
    if trimmed == 0:
        return None, None, tail
    groups = core.view(-1, group_size)
    scales = groups.abs().amax(dim=1).clamp_min(eps)
    norm = groups / scales.unsqueeze(1)
    return norm.contiguous(), scales.contiguous(), tail.contiguous()


def to_vectors(norm_groups: torch.Tensor, d: int) -> torch.Tensor:
    flat = norm_groups.view(-1)
    usable = (flat.numel() // d) * d
    flat = flat[:usable]
    return flat.view(-1, d).contiguous()


def tensor_family(name: str) -> str:
    if name in ("model.embed_tokens.weight", "embed_tokens.weight"):
        return "embed_tokens"
    if name == "lm_head.weight":
        return "lm_head"
    m = re.search(r'\.mlp\b.*\.(gate_proj|up_proj|down_proj)\.weight$', name)
    if m:
        return f"mlp_{m.group(1)}"
    m = re.search(r'\.self_attn\b.*\.(q_proj|k_proj|v_proj|o_proj)\.weight$', name)
    if m:
        return f"attn_{m.group(1)}"
    return "other_2d"


def family_root(name: str) -> str:
    fam = tensor_family(name)
    if fam.startswith("mlp_"):
        return "mlp"
    if fam.startswith("attn_"):
        return "attn"
    if fam == "embed_tokens":
        return "embed_tokens"
    if fam == "lm_head":
        return "lm_head"
    return "other_2d"


def is_known_2d_weight(name: str) -> bool:
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return True
    if re.match(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$", name):
        return True
    if re.match(r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$", name):
        return True
    return False


def train_shared_rvq_codebooks(
    model_dir: str,
    weight_map: Dict[str, str],
    logical_map: Dict[str, Dict[str, str]],
    names: List[str],
    d: int,
    group_size: int,
    stages: int,
    k: int,
    max_vectors: int,
    kmeans_iters: int,
    chunk_size: int,
    seed: int,
) -> torch.Tensor:
    g = random.Random(seed)
    per_tensor = max(1, max_vectors // max(1, len(names)))
    allv = []
    for name in tqdm(names, desc="Collect train vectors", leave=False):
        t = load_effective_tensor(model_dir, weight_map, logical_map[name]).to(torch.float32).view(-1)
        norm_groups, _, _ = normalize_groups(t, group_size=group_size)
        if norm_groups is None:
            continue
        x = to_vectors(norm_groups, d=d)
        if x.shape[0] > per_tensor:
            idx = torch.tensor(g.sample(range(x.shape[0]), per_tensor), dtype=torch.int64)
            x = x[idx]
        allv.append(x)
    train_x = torch.cat(allv, dim=0)
    if train_x.shape[0] > max_vectors:
        idx = torch.tensor(g.sample(range(train_x.shape[0]), max_vectors), dtype=torch.int64)
        train_x = train_x[idx]

    residual = train_x.clone()
    codebooks = []
    for r in range(stages):
        cb = kmeans_train(
            residual, k=k, iters=kmeans_iters, seed=seed + 31 * r, chunk_size=chunk_size
        )
        idx = assign_codes(residual, cb, chunk_size=chunk_size)
        residual = residual - cb[idx]
        codebooks.append(cb)
    return torch.stack(codebooks, dim=0).to(torch.float16).contiguous()


def encode_rvq_tensor(
    t: torch.Tensor,
    codebooks: torch.Tensor,
    d: int,
    group_size: int,
    chunk_size: int,
    codebook_id: str,
) -> Dict[str, object]:
    flat = t.to(torch.float32).view(-1)
    norm_groups, scales, tail = normalize_groups(flat, group_size=group_size)
    if norm_groups is None:
        raise RuntimeError("Tensor too small for RVQ group size.")

    x = to_vectors(norm_groups, d=d)
    cb = codebooks.to(torch.float32)
    stages = cb.shape[0]
    residual = x.clone()
    codes = []
    recon = torch.zeros_like(x)
    for r in range(stages):
        idx = assign_codes(residual, cb[r], chunk_size=chunk_size)
        part = cb[r][idx]
        recon += part
        residual -= part
        codes.append(idx)
    codes = torch.stack(codes, dim=0)

    recon_flat = recon.view(-1)
    recon_groups = recon_flat.view(-1, group_size)
    recon_core = (recon_groups * scales.unsqueeze(1)).reshape(-1)
    ref_core = flat[: recon_core.numel()]
    err = recon_core - ref_core
    mse = float((err * err).mean().item())
    denom = float(torch.sqrt((ref_core * ref_core).mean()).item()) + 1e-12
    rel_rmse = math.sqrt(mse) / denom

    k = cb.shape[1]
    code_dtype = torch.uint8 if k <= 256 else torch.int16
    return {
        "method": "rvq_groupwise",
        "codebook_id": codebook_id,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "group_size": int(group_size),
        "d": int(d),
        "trimmed_numel": int(recon_core.numel()),
        "scales": scales.to(torch.float16),
        "indices": codes.to(code_dtype).contiguous(),
        "tail": tail.to(torch.bfloat16),
        "mse": mse,
        "rel_rmse": rel_rmse,
    }


def encode_int8_groupwise(t: torch.Tensor, group_size: int) -> Dict[str, object]:
    flat = t.to(torch.float32).view(-1)
    trimmed = (flat.numel() // group_size) * group_size
    core = flat[:trimmed]
    tail = flat[trimmed:]

    if trimmed == 0:
        return {
            "method": "bf16_raw",
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "data": t.to(torch.bfloat16).contiguous(),
        }

    groups = core.view(-1, group_size)
    scales = groups.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(groups / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    recon = (q.to(torch.float32) * scales.unsqueeze(1)).reshape(-1)
    err = recon - core
    mse = float((err * err).mean().item())
    denom = float(torch.sqrt((core * core).mean()).item()) + 1e-12
    rel_rmse = math.sqrt(mse) / denom

    return {
        "method": "int8_groupwise",
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "group_size": int(group_size),
        "trimmed_numel": int(trimmed),
        "q": q.contiguous(),
        "scales": scales.to(torch.float16).contiguous(),
        "tail": tail.to(torch.bfloat16).contiguous(),
        "mse": mse,
        "rel_rmse": rel_rmse,
    }


def estimate_entry_bytes(entry: Dict[str, object]) -> int:
    m = entry["method"]
    if m == "bf16_raw":
        return entry["data"].numel() * 2
    if m == "int8_groupwise":
        return (
            entry["q"].numel() * entry["q"].element_size()
            + entry["scales"].numel() * 2
            + entry["tail"].numel() * 2
        )
    if m == "rvq_groupwise":
        return (
            entry["indices"].numel() * entry["indices"].element_size()
            + entry["scales"].numel() * 2
            + entry["tail"].numel() * 2
        )
    raise ValueError(f"Unknown method: {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--repo-id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--auto-download", action="store_true")
    ap.add_argument("--overwrite", action="store_true")

    # RVQ config for all 2D tensors
    ap.add_argument("--rvq-d", type=int, default=8)
    ap.add_argument("--rvq-stages", type=int, default=4)
    ap.add_argument("--rvq-k", type=int, default=256)
    ap.add_argument("--rvq-group-size", type=int, default=128)
    ap.add_argument("--rvq-share-mode", type=str, default="family", choices=["global", "family", "tensor"])
    ap.add_argument("--rvq-max-train-vectors-per-codebook", type=int, default=400000)
    ap.add_argument("--rvq-kmeans-iters", type=int, default=12)
    ap.add_argument("--rvq-chunk-size", type=int, default=65536)
    ap.add_argument("--int8-group-size", type=int, default=128)

    # Per-family policy for 2D tensors.
    ap.add_argument("--policy-mlp", type=str, default="rvq", choices=["rvq", "int8", "bf16"])
    ap.add_argument("--policy-attn", type=str, default="rvq", choices=["rvq", "int8", "bf16"])
    ap.add_argument("--policy-embed", type=str, default="rvq", choices=["rvq", "int8", "bf16"])
    ap.add_argument("--policy-lm-head", type=str, default="rvq", choices=["rvq", "int8", "bf16"])
    ap.add_argument("--policy-other-2d", type=str, default="rvq", choices=["rvq", "int8", "bf16"])
    ap.add_argument("--keep-1d-bf16", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.rvq_group_size % args.rvq_d != 0:
        raise ValueError("--rvq-group-size must be divisible by --rvq-d")

    ensure_model_files(args.model_dir, args.repo_id, args.auto_download)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.listdir(args.output_dir) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty. Use --overwrite.")

    copy_support_files(args.model_dir, args.output_dir)
    weight_map = load_weight_map(args.model_dir)
    logical_map = build_logical_state_keys(weight_map)
    raw_shapes = load_tensor_shapes(args.model_dir, weight_map)
    all_keys = sorted(logical_map.keys())

    def _is_2d_weight(key: str) -> bool:
        spec = logical_map[key]
        if spec["kind"] == "hardem":
            return True
        raw_key = spec.get("key", key)
        if raw_key in raw_shapes:
            return len(raw_shapes[raw_key]) == 2
        return False

    keys_2d = [k for k in all_keys if _is_2d_weight(k)]

    policy_map = {
        "mlp": args.policy_mlp,
        "attn": args.policy_attn,
        "embed_tokens": args.policy_embed,
        "lm_head": args.policy_lm_head,
        "other_2d": args.policy_other_2d,
    }
    tensor_method: Dict[str, str] = {}
    for k in keys_2d:
        tensor_method[k] = policy_map[family_root(k)]

    # Build codebook groups according to share mode.
    codebook_groups: Dict[str, List[str]] = {}
    rvq_keys_2d = [k for k in keys_2d if tensor_method[k] == "rvq"]
    if rvq_keys_2d:
        if args.rvq_share_mode == "global":
            codebook_groups["global_2d"] = rvq_keys_2d
        elif args.rvq_share_mode == "family":
            for k in rvq_keys_2d:
                fid = tensor_family(k)
                codebook_groups.setdefault(fid, []).append(k)
        else:  # tensor
            for k in rvq_keys_2d:
                codebook_groups[f"tensor::{k}"] = [k]

    print(f"2D tensors: {len(keys_2d)}")
    print(
        "Policy counts: "
        f"rvq={sum(1 for k in keys_2d if tensor_method[k] == 'rvq')}, "
        f"int8={sum(1 for k in keys_2d if tensor_method[k] == 'int8')}, "
        f"bf16={sum(1 for k in keys_2d if tensor_method[k] == 'bf16')}"
    )
    print(f"Codebook groups ({args.rvq_share_mode}): {len(codebook_groups)}")

    shared_codebooks = {}
    for i, (cbid, names) in enumerate(codebook_groups.items()):
        print(f"\nTraining codebook [{i+1}/{len(codebook_groups)}] {cbid} on {len(names)} tensors ...")
        shared_codebooks[cbid] = train_shared_rvq_codebooks(
            model_dir=args.model_dir,
            weight_map=weight_map,
            logical_map=logical_map,
            names=names,
            d=args.rvq_d,
            group_size=args.rvq_group_size,
            stages=args.rvq_stages,
            k=args.rvq_k,
            max_vectors=args.rvq_max_train_vectors_per_codebook,
            kmeans_iters=args.rvq_kmeans_iters,
            chunk_size=args.rvq_chunk_size,
            seed=args.seed + i * 1000,
        )

    tensor_to_codebook = {}
    for cbid, names in codebook_groups.items():
        for n in names:
            tensor_to_codebook[n] = cbid

    entries = {}
    summary = {"per_tensor": {}, "totals": {}, "config": vars(args)}
    total_ref = 0
    total_cmp = 0
    rel_errs = []

    print("\nEncoding all tensors ...")
    for key in tqdm(all_keys):
        t = load_effective_tensor(args.model_dir, weight_map, logical_map[key])
        ref_bytes = t.numel() * 2  # bf16 reference
        if key in tensor_method:
            method = tensor_method[key]
            if method == "rvq":
                cbid = tensor_to_codebook[key]
                entry = encode_rvq_tensor(
                    t=t,
                    codebooks=shared_codebooks[cbid],
                    d=args.rvq_d,
                    group_size=args.rvq_group_size,
                    chunk_size=args.rvq_chunk_size,
                    codebook_id=cbid,
                )
                rel = float(entry["rel_rmse"])
            elif method == "int8":
                entry = encode_int8_groupwise(t, group_size=args.int8_group_size)
                rel = float(entry["rel_rmse"])
            else:
                entry = {
                    "method": "bf16_raw",
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "data": t.to(torch.bfloat16).contiguous(),
                }
                rel = 0.0
        else:
            if args.keep_1d_bf16 and t.ndim <= 1:
                entry = {
                    "method": "bf16_raw",
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "data": t.to(torch.bfloat16).contiguous(),
                }
                rel = 0.0
            else:
                entry = {
                    "method": "bf16_raw",
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "data": t.to(torch.bfloat16).contiguous(),
                }
                rel = 0.0

        cmp_bytes = estimate_entry_bytes(entry)
        total_ref += ref_bytes
        total_cmp += cmp_bytes
        rel_errs.append(rel)
        entries[key] = entry
        summary["per_tensor"][key] = {
            "method": entry["method"],
            "shape": list(t.shape),
            "bf16_bytes": ref_bytes,
            "compressed_bytes": cmp_bytes,
            "rel_rmse": rel,
        }

    artifact = {
        "format_version": 1,
        "model_type": "qwen3",
        "shared_codebooks": shared_codebooks,
        "entries": entries,
    }
    artifact_path = os.path.join(args.output_dir, "compressed_model.pt")
    torch.save(artifact, artifact_path)

    summary["totals"] = {
        "num_tensors": len(entries),
        "bf16_total_bytes": total_ref,
        "compressed_total_bytes": total_cmp,
        "compression_vs_bf16": total_ref / max(total_cmp, 1),
        "avg_rel_rmse": float(sum(rel_errs) / max(len(rel_errs), 1)),
    }
    summary_path = os.path.join(args.output_dir, "compression_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Compressed folder: {args.output_dir}")
    print(f"Artifact         : {artifact_path}")
    print(f"Summary          : {summary_path}")
    print(f"Compression      : {summary['totals']['compression_vs_bf16']:.2f}x vs bf16")


if __name__ == "__main__":
    main()

