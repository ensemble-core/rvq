#!/usr/bin/env python3
"""
Single-slot/no-prefetch benchmark for GPU-resident compressed inference.

This script reuses `infer_compressed_layerwise.py` and overrides only the
LayerResidencyManager behavior needed for the benchmark:
  - one reusable decoded layer slot
  - no next-layer prefetch in GPU decode mode
  - no CPU decoded-cache or hybrid modes exposed through the CLI
"""

import argparse


def _make_single_slot_manager(base):
    class SingleSlotLayerResidencyManager(base.LayerResidencyManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.decode_on_gpu:
                self.functional_slots = True
                self.slot_states = [dict()]
                self.layer_to_slot.clear()
                self.slot_to_layer.clear()

        def start_prefetch_for_layer(self, layer_idx: int) -> None:
            if self.decode_on_gpu:
                return
            return super().start_prefetch_for_layer(layer_idx)

    return SingleSlotLayerResidencyManager


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-slot/no-prefetch benchmark for GPU-resident compressed "
            "layerwise inference."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--decode-on-gpu",
        action="store_true",
        required=True,
        help="Required. Keep compressed weights on GPU and decode one layer slot at a time.",
    )
    parser.add_argument("--report-memory", action="store_true")
    parser.add_argument("--report-time", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import infer_compressed_layerwise as base

    device = base._choose_device("auto")
    if device.type != "cuda":
        raise ValueError("This benchmark supports only --decode-on-gpu on CUDA.")

    if args.report_memory:
        torch.cuda.reset_peak_memory_stats(device)

    original_manager = base.LayerResidencyManager
    base.LayerResidencyManager = _make_single_slot_manager(base)
    try:
        model, tokenizer, residency, plan, embed_weight, lm_head_weight = base.load_layerwise_model(
            args.model_dir,
            device=device,
            runtime_dtype=torch.bfloat16,
            layer_window=1,
            decode_on_gpu=True,
            gpu_decode_to_cpu_cache=False,
            functional_slots=True,
        )
    finally:
        base.LayerResidencyManager = original_manager

    base._maybe_report_cuda_memory(device, "after load", args.report_memory)
    base._maybe_report_layerwise_breakdown(
        label="after load",
        residency=residency,
        model=model,
        plan=plan,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        enabled=args.report_memory,
    )

    try:
        text, timing = base.generate_greedy(
            model,
            tokenizer,
            residency,
            plan,
            embed_weight,
            lm_head_weight,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            enable_thinking=False,
            hybrid_decode=False,
            report_memory=args.report_memory,
            report_time=args.report_time,
        )
    finally:
        residency.close()

    base._maybe_report_cuda_memory(device, "after generation", args.report_memory)
    base._maybe_report_layerwise_breakdown(
        label="after generation",
        residency=residency,
        model=model,
        plan=plan,
        embed_weight=embed_weight,
        lm_head_weight=lm_head_weight,
        enabled=args.report_memory,
    )
    base._maybe_report_generation_timing(enabled=args.report_time, **timing)
    base._maybe_report_gpu_layer_decode_timing(residency=residency, enabled=args.report_time)
    print(text)


if __name__ == "__main__":
    main()
