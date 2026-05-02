# llama.cpp — Kepler GPU Support (Tesla K80/K40/K20, sm_35/sm_37)

This fork adds support for **NVIDIA Kepler GPUs** (compute capability 3.5 and 3.7 — Tesla K80, K40, K20) in llama.cpp. Kepler support was dropped from CUDA 12.x onwards, making it impossible to build llama.cpp with CUDA acceleration on systems running CUDA 11.x targeting these GPUs — until this patch.

---

## Why Kepler GPUs Don't Work with Official llama.cpp

NVIDIA Kepler (sm_37) was the last architecture supported by **CUDA 11.x**. Starting with CUDA 12.0, sm_37 support was removed entirely.

Official llama.cpp's CMake build logic (since ~b8000) does not include `sm_37` in the auto-detected CUDA architecture list, even when building with CUDA 11.x. Additionally, llama.cpp added BF16 matrix multiplication via cuBLAS `GemmEx`, but **BF16 in cuBLAS requires Ampere (sm_80) or newer** — calling it on Kepler causes a runtime crash (`CUBLAS_STATUS_NOT_SUPPORTED`).

Two bugs combined to make Kepler completely non-functional:
1. `sm_37` was never compiled into the binary → GPU not recognized at runtime
2. BF16 cuBLAS path was invoked unconditionally for all NVIDIA GPUs → crash on any pre-Ampere card

---

## Based on

| Field        | Value |
|---|---|
| Upstream     | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Base tag     | **b8990** |
| Base commit  | `a95a11e5b` (3 commits ahead of b8990) |
| Build number | 8993 |
| CUDA version tested | CUDA 11.4 |
| Driver tested | 470.256.02 |

---

## Changes (2 files modified)

### 1. `ggml/src/ggml-cuda/CMakeLists.txt`

Adds `37-virtual` to the CUDA architecture list when building with CUDA < 12, so that Kepler PTX code is compiled and the GPU is recognised at runtime.

```diff
 if (CUDAToolkit_VERSION VERSION_LESS "13")
+    # 35 == Tesla K40/K20 (Kepler GK110), 37 == Tesla K80 (Kepler GK210)
+    # Both are last supported in CUDA 11.x
+    if (CUDAToolkit_VERSION VERSION_LESS "12")
+        list(APPEND CMAKE_CUDA_ARCHITECTURES 35-virtual 37-virtual)
+    endif()
     list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
 endif ()
```

### 2. `ggml/src/ggml-cuda/ggml-cuda.cu`

Restricts BF16 cuBLAS `GemmEx` to Ampere and newer. Without this fix, any GGUF model with BF16 weights causes a `CUBLAS_STATUS_NOT_SUPPORTED` crash on Kepler.

```diff
-const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
+// BF16 cuBLAS GemmEx requires Ampere (cc >= 800); Kepler/Maxwell/Pascal/Volta do not support it.
+const bool supports_bf16 = (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) || GGML_CUDA_CC_IS_AMD(cc) ||
     (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
```

---

## Requirements

- NVIDIA Kepler GPU **sm_35 or sm_37**:
  | GPU | Chip | Compute | Status |
  |---|---|---|---|
  | Tesla K80 | GK210 | sm_37 | ✅ tested |
  | Tesla K40 | GK110B | sm_35 | ⚠️ compiled, not tested |
  | Tesla K20 | GK110 | sm_35 | ⚠️ compiled, not tested |
  | GTX 780 Ti | GK110 | sm_35 | ⚠️ compiled, not tested |
- **CUDA Toolkit 11.x** (11.0–11.8) — Kepler is **not** supported in CUDA 12+
- Driver ≥ 470.x (matching CUDA 11.4)
- GCC ≤ 10 (GCC 9/10 recommended — CUDA 11.x does not support GCC 11+)
- CMake ≥ 3.18
- Linux x86_64

---

## Build Instructions

**For Tesla K80 (sm_37):**
```bash
git clone https://github.com/babal35/llamacpp-kepler
cd llamacpp-kepler
mkdir build && cd build

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF

make -j$(nproc)
```

**For Tesla K40 / K20 / GTX 780 Ti (sm_35, untested):**
```bash
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=35 \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF
```

> **`-DGGML_CUDA_GRAPHS=OFF`** is mandatory — CUDA Graphs require sm_60+ and will crash on Kepler.
> **`-DGGML_NATIVE=OFF`** prevents CMake from auto-detecting "native" architecture (which would fail on an x86 host building for Kepler).

---

## Inference Commands

### Single K80 (1 GPU)

```bash
./bin/llama-cli \
  --model /path/to/model.gguf \
  -ngl 999 \
  -c 4096 \
  -n 512 \
  -p "Your prompt here"
```

### Dual K80 (2 GPUs, e.g. one Tesla K80 card = 2x GK210)

```bash
./bin/llama-cli \
  --model /path/to/model.gguf \
  -ngl 999 \
  --tensor-split 1,1 \
  -c 4096 \
  -n 512 \
  -p "Your prompt here"
```

> A Tesla K80 card exposes **2 independent GK210 GPUs** (each ~12 GB). Use `--tensor-split 1,1` to distribute the model evenly across both.

### Interactive / Chat mode

```bash
./bin/llama-cli \
  --model /path/to/model.gguf \
  -ngl 999 \
  --tensor-split 1,1 \
  -c 4096 \
  --interactive-first \
  --chat-template llama3
```

---

## Tested Models & Performance

All tests run on a single Tesla K80 card (**2×GK210**, 2×11441 MiB = 22882 MiB total), with `--tensor-split 1,1`.

| Model | Quant | VRAM used | Prompt t/s | Gen t/s |
|---|---|---|---|---|
| [GPT-OSS-20B MoE](https://huggingface.co/collections/microsoft/phi-4-677b5ade6bc4cbcf92e24d79) (mxfp4 GGUF) | mxfp4 | ~22 GB (2×K80) | ~28 | ~25 |
| [Gemma4 26B A4B](https://huggingface.co/google/gemma-3-27b) (MoE) | Q4_K_M | ~22 GB (2×K80) | ~22 | ~20 |

> Both models fit entirely on the two K80 GPUs with no CPU offload. Tokens/s measured with a warm context, 512 token generation.

---

## What Works vs. Official llama.cpp

| Feature | This fork | Official llama.cpp (b8990+) |
|---|---|---|
| Kepler GPU (sm_35/sm_37) | ✅ | ❌ (not compiled) |
| FP16 inference | ✅ | ✅ (Pascal+) |
| Q4/Q8 quantized models | ✅ | ✅ |
| BF16 models (e.g. Gemma, GPT-OSS) | ✅ (falls back to FP16 cuBLAS) | ❌ crash on Kepler |
| MoE models (multi-expert) | ✅ | ✅ |
| CUDA Graphs | ❌ disabled (sm_60+ required) | ✅ (enabled by default) |
| Flash Attention | ❌ (sm_80+ required) | ✅ (Ampere+) |
| BF16 fast path (Ampere+) | N/A | ✅ |
| Multi-GPU tensor split | ✅ | ✅ |
| CPU-only inference | ✅ | ✅ |
| All other llama.cpp features | ✅ | ✅ |

---

## Why Not Submit Upstream?

NVIDIA officially dropped Kepler in CUDA 12.0. The CUDA 11.x branch is EOL (End of Life) and the official llama.cpp project targets actively supported toolchains. A patch for an EOL architecture is unlikely to be accepted. This fork preserves Kepler support for those who still have K80/K40/K20 hardware.

---

## Pre-compiled Binaries

Pre-compiled binaries are **not provided** because:
- The compiled `libggml-cuda.so` is ~91 MB and tied to the exact CUDA 11.x version used
- Shared libraries require matching runtime linker paths
- It is straightforward to compile from source with the instructions above

---

## License

Same as upstream llama.cpp — [MIT License](LICENSE)
