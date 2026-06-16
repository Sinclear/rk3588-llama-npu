# RK3588 NPU LLM Inference — llama.cpp Optimizations

Optimized llama.cpp backend for RK3588 NPU (Orange Pi 5 Max). Includes tiling cache, madvise page release, and C-bound skip optimizations.

## Key Results

**With `-t 4 taskset 0xf0` (big cores only, recommended):**

| Model | Params | Quant | RSS | Decode | Quality |
|-------|--------|-------|-----|--------|---------|
| **Gemma-4 E2B** | **4.6B dense** | **Q8_0** | **4.7 GB** | **9.1 tok/s** | **5/5** |
| Qwen2.5-Coder-3B | 3B dense | Q8_0 | 3.0 GB | 8.1 tok/s | 4/5 |
| Qwen3.5-4B | 4B dense | Q8_0 | 4.3 GB | 5.9 tok/s | 5/5 |
| Gemma-4 E4B | 7.5B dense | Q8_0 | 7.6 GB | 5.0 tok/s | 5/5 |
| T-lite 8B | 8B dense | Q8_0 | 7.5 GB | 4.1 tok/s | 5/5 |
| LFM2.5-8B-A1B | 8B MoE | Q8_0 | 13.3 GB | 10 tok/s (CPU) | OK |

**Without taskset (all 8 cores, big+LITTLE) — slower:**

| Model | Decode (8c) | Decode (4 big) | Speedup |
|-------|-------------|----------------|---------|
| Gemma-4 E2B Q8_0 | 6.9 tok/s | 9.1 tok/s | +32% |
| Qwen2.5-Coder-3B Q8_0 | 6.1 tok/s | 8.1 tok/s | +33% |
| Qwen3.5-4B Q8_0 | 4.5 tok/s | 5.9 tok/s | +31% |
| Gemma-4 E4B Q8_0 | 4.1 tok/s | 5.0 tok/s | +22% |
| T-lite 8B Q8_0 | 3.5 tok/s | 4.1 tok/s | +17% |

RK3588 is big.LITTLE: 4×A76 @ 2.3 GHz (big) + 4×A55 @ 1.8 GHz (LITTLE).
LITTLE cores slow down NPU-bound inference. Use `-t 4` + `taskset 0xf0` for best performance.

## Optimizations

1. **Tiling Cache v5** — precomputed segments/offsets, eliminates repeated computation (168 matmul/token)
2. **madvise Page Release** — frees duplicated GGUF mmap pages after NPU DMA, RSS 13.5→7.5 GB
3. **C-bound Skip** — skip `rknn_matmul_set_io_mem` for C buffer after first call
4. **B-bound Skip** — skip `rknn_matmul_set_io_mem` for B buffer after first call (original)

## Files

- `ggml-rknpu2.cpp` — main backend with all optimizations (madvise + tiling cache v5 + c_bound)
- `tiling_cache.h` / `tiling_cache.cpp` — tiling cache implementation
- `rknpu2-configuration.cpp` / `.h` — RKNN configuration (includes Gemma-4 Q8_0 patterns)
- `CMakeLists.txt` — build config (includes tiling_cache)
- `ggml-rknpu2.patch` — diff against upstream rk-llama.cpp
- `REPORT.md` — full research report (Russian)

## Build

```bash
cd /path/to/llama.cpp
# Copy files to ggml/src/ggml-rknpu2/
cmake -B build -DLLAMA_RKNPU2=ON
cmake --build build --target llama-server -j4
```

## Run

```bash
ulimit -n 65536
taskset 0xf0 ./build/bin/llama-server -m model-Q8_0.gguf --host 0.0.0.0 --port 8083 -ngl 99 -c 512 -t 4
```

**Important:** Use `taskset 0xf0` and `-t 4` to pin to big cores (A76). Without this,
LITTLE cores (A55) slow down NPU-bound inference by 17-33%.

## NPU Decode Profile (T-lite 8B Q8_0, 168 matmul/token)

| Phase | Time | Share |
|-------|------|-------|
| NPU_run | 153-166 ms | 80% |
| C_sync | 23-38 ms | 17% |
| A_prep | 5-7 ms | 3% |
| B_ctx | 0.5 ms | 0.3% |

NPU compute is the bottleneck — further code optimization is not possible.

## Based On

- [rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp) — upstream RK3588 NPU backend
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGML/llama.cpp framework