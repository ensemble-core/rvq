#!/usr/bin/env python3
"""
Triton decode kernels for compressed layerwise inference.

This module currently implements a fused RVQ decode kernel for the common
artifact layout produced by `compress_full_model.py`:
  - RVQ stages selected by `indices`
  - shared `codebook` tensor
  - one scale per group of vectors
  - optional tail copied after the core decode

The first implementation is intentionally specialized to the current default
artifact shape (`d == 8`). That keeps the kernel simple and targets the hot
path seen in profiling before generalizing further.
"""

from __future__ import annotations

from typing import Dict

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def rvq_triton_supported(entry: Dict[str, object], out: torch.Tensor) -> bool:
    if not triton_available():
        return False
    if not out.is_cuda:
        return False
    if int(entry["d"]) != 8:
        return False
    if entry["method"] not in ("rvq_groupwise", "rvq_mlp"):
        return False
    return True


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["N", "STAGES", "GROUP_VECS", "OUT_KIND"],
)
@triton.jit
def _rvq_decode_into_kernel(
    indices_ptr,
    codebook_ptr,
    scales_ptr,
    out_ptr,
    N,
    D,
    TRIMMED,
    stride_is,
    stride_in,
    stride_cs,
    stride_ck,
    stride_cd,
    GROUP_VECS: tl.constexpr,
    STAGES: tl.constexpr,
    OUT_KIND: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 8)

    acc = tl.zeros((BLOCK_N, 8), dtype=tl.float32)

    for stage_idx in tl.static_range(STAGES):
        code_idx = tl.load(
            indices_ptr + stage_idx * stride_is + offs_n * stride_in,
            mask=offs_n < N,
            other=0,
        ).to(tl.int32)

        codebook_ptrs = (
            codebook_ptr
            + stage_idx * stride_cs
            + code_idx[:, None] * stride_ck
            + offs_d[None, :] * stride_cd
        )
        acc += tl.load(
            codebook_ptrs,
            mask=offs_n[:, None] < N,
            other=0.0,
        )

    scale_idx = offs_n // GROUP_VECS
    scales = tl.load(scales_ptr + scale_idx, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc * scales[:, None]

    flat_offs = offs_n[:, None] * D + offs_d[None, :]
    mask = (offs_n[:, None] < N) & (flat_offs < TRIMMED)

    if OUT_KIND == 0:
        tl.store(out_ptr + flat_offs, acc.to(tl.float16), mask=mask)
    elif OUT_KIND == 1:
        tl.store(out_ptr + flat_offs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + flat_offs, acc, mask=mask)


def decode_rvq_triton_into(
    entry: Dict[str, object],
    codebook: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton is not available.")

    d = int(entry["d"])
    group_size = int(entry["group_size"])
    trimmed = int(entry["trimmed_numel"])

    if d != 8:
        raise NotImplementedError(f"Triton RVQ decode currently expects d=8, got d={d}.")
    if group_size % d != 0:
        raise ValueError(f"group_size={group_size} must be divisible by d={d}.")
    if not out.is_cuda:
        raise ValueError("Triton RVQ decode requires a CUDA output tensor.")
    if not out.is_contiguous():
        raise ValueError("Triton RVQ decode requires a contiguous output tensor.")

    indices = entry["indices"]
    scales = entry["scales"]
    tail = entry["tail"]

    if not indices.is_cuda:
        raise ValueError("Triton RVQ decode requires CUDA indices.")
    if not scales.is_cuda:
        raise ValueError("Triton RVQ decode requires CUDA scales.")
    if not codebook.is_cuda:
        raise ValueError("Triton RVQ decode requires a CUDA codebook.")

    if not indices.is_contiguous():
        indices = indices.contiguous()
    if not scales.is_contiguous():
        scales = scales.contiguous()
    if codebook.dtype != torch.float32 or not codebook.is_contiguous():
        codebook = codebook.to(dtype=torch.float32).contiguous()

    stages, nvec = indices.shape
    group_vecs = group_size // d
    flat_out = out.view(-1)
    core_out = flat_out[:trimmed]

    if out.dtype == torch.float16:
        out_kind = 0
    elif out.dtype == torch.bfloat16:
        out_kind = 1
    elif out.dtype == torch.float32:
        out_kind = 2
    else:
        raise NotImplementedError(f"Unsupported Triton RVQ output dtype: {out.dtype}")

    grid = lambda meta: (triton.cdiv(nvec, meta["BLOCK_N"]),)

    _rvq_decode_into_kernel[grid](
        indices,
        codebook,
        scales,
        core_out,
        nvec,
        d,
        trimmed,
        indices.stride(0),
        indices.stride(1),
        codebook.stride(0),
        codebook.stride(1),
        codebook.stride(2),
        GROUP_VECS=group_vecs,
        STAGES=stages,
        OUT_KIND=out_kind,
    )

    if tail.numel():
        flat_out[trimmed : trimmed + tail.numel()].copy_(tail)

    return out


__all__ = [
    "decode_rvq_triton_into",
    "rvq_triton_supported",
    "triton_available",
]
