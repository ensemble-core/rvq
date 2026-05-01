# RVQ-Triton: Low-VRAM Layerwise Inference for Compressed LLMs

This repository provides a highly optimized, 100% GPU-streaming pipeline for running extremely large language models on consumer hardware. By combining **Residual Vector Quantization (RVQ)** with a **2-slot ping-pong layerwise decoding engine**, this project massively reduces Peak VRAM requirements while bypassing traditional PCIe host-to-device bottlenecks.

## 🚀 Key Features

*   **Massive VRAM Savings:** Cuts Peak VRAM consumption by nearly 50%. A Qwen3.5-4B parameter model runs comfortably in ~4.3 GB of VRAM instead of 8.1 GB.
*   **Zero-CPU Architecture:** All codebooks, indices, and scales are loaded directly into VRAM at startup. No host-memory cache or background multi-threading is used during generation.
*   **Triton-Accelerated Decoding:** Layer weights are decompressed entirely on the GPU in real-time using highly optimized custom Triton kernels.
*   **Ping-Pong Streaming:** Utilizes a strict 2-slot memory buffer. While PyTorch executes Layer $i$ on Stream A, Triton asynchronously decompresses Layer $i+1$ into the alternating slot on Stream B.

## 📦 File Overview

*   `compress_full_model.py`: Script to compress a standard HuggingFace model using Residual Vector Quantization.
*   `infer_compressed_gpu_only.py`: The minimalist (~400 lines) inference engine that performs layerwise decoding on the GPU.
*   `triton_decode_kernels.py`: The custom Triton kernels used for lightning-fast on-device decompression.
*   `benchmark_baseline.py`: A script for benchmarking the uncompressed baseline model for performance comparisons.

## ⚙️ Requirements

*   Python 3.10+
*   PyTorch (CUDA enabled)
*   Transformers & Accelerate
*   [Triton](https://github.com/openai/triton)

## 🛠️ Usage

### 1. Compress the Model
First, compress your standard HuggingFace model into an RVQ artifact.

```bash
python compress_full_model.py \
    --model-dir Qwen3.5-4B \
    --output-dir Qwen3.5-4B-rvq
```

### 2. Run Layerwise Inference
Run the GPU-only streaming decoder. You can monitor performance metrics by appending the reporting flags.

```bash
python infer_compressed_gpu_only.py \
    --model-dir Qwen3.5-4B-rvq \
    --prompt "What is the capital of France?" \
    --max-new-tokens 128 \
    --report-time \
    --report-memory
```

## 📊 Benchmarks

Performance comparison on **Qwen3.5-4B** (Uncompressed Baseline vs. Layerwise RVQ):

| Metric | Base Model (Qwen3.5-4B) | Layerwise RVQ (Qwen3.5-4B-rvq) | Difference |
| :--- | :--- | :--- | :--- |
| **Peak VRAM Allocated** | 8,187 MiB | **4,311 MiB** | **47% Reduction** (Saving ~3.8 GB) |
| **Time to First Token** | **340 ms** | 464 ms | 1.3x slower |
| **Decode Speed** | **43 ms/token** | 67 ms/token | 1.5x slower |

