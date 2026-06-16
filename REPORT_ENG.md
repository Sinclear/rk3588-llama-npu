# Report: NPU LLM Inference Optimization on RK3588

Date: 2026-06-16
Platform: Orange Pi 5 Max (RK3588, 16 GB RAM, 3 NPU cores @ 1 GHz)

---

## Summary

Full optimization pipeline for llama.cpp with ggml-rknpu2 backend on RK3588 was investigated.
Result: RSS reduced from 13.5 GB to 3.0 GB, decode accelerated from 3.5 to 9.1 tok/s
(via model selection + big-core pinning). NPU compute = 80% bottleneck for dense 8B,
further code optimization is impossible — hardware limitation.

**Best results (big cores only, taskset 0xf0 -t 4):**
- Gemma-4 E2B Q8_0: 9.1 tok/s, quality 5/5, RSS 4.7 GB ← BEST DENSE
- Qwen2.5-Coder-3B Q8_0: 8.1 tok/s, quality 4/5, RSS 3.0 GB ← BEST RSS
- LFM2.5-8B-A1B Q8_0: 10 tok/s, MoE, quality OK, RSS 13.3 GB ← BEST SPEED
- T-lite 8B Q8_0: 4.1 tok/s, quality 5/5, RSS 7.5 GB ← BASELINE

---

## 1. Platform and Baseline

### 1.1 Hardware

- Orange Pi 5 Max (RK3588)
- 16 GB LPDDR4x RAM
- 4×A76 @ 2.3 GHz (big) + 4×A55 @ 1.8 GHz (LITTLE) — big.LITTLE
- 3 NPU cores @ 1 GHz (topology0/1/2)
- NPU: matmul up to 256×256 (K, N), M=1 for decode
- CMA memory: 3 GB reserved for NPU

### 1.2 Baseline Models

- T-lite-it-2.1 Q4_K_M (4.7 GB) — does NOT use NPU (type not supported)
- T-lite-it-1.0 Q8_0 (8.1 GB) — uses NPU (W8A8 pipeline)

### 1.3 Baseline Performance

- Decode Q8_0: ~3.5 tok/s (NPU W8A8)
- Decode Q4_K_M: ~3.5-3.8 tok/s (CPU fallback, faster due to less data)
- RSS: 13.5 GB (GGUF mmap + RKNPU buffer duplication)
- rkllm W4A4 for comparison: ~13 tok/s, but 53% quality

---

## 2. Code Optimizations

### 2.1 Step 1: Tiling Cache (v1→v5)

Goal: eliminate repeated computation of segments/offsets (168 matmul/token)

**v1 (FULL CACHE) — BROKEN:**
- Cached segments, pipeline, B-matrix offsets
- Bug: segment_size_bytes != io_attr.B.size (RKNN padding)
- Bug: offset accumulation via all_n_segments loses inactive positions

**v2 (SEGMENTS ONLY) — VALIDATED:**
- Caches: segments (K/N), pipeline config, matmul_type, use_hadamard
- Does NOT cache: B-matrix offsets
- Decode: ~3.8 tok/s, correct output

**v3 (B OFFSETS) — VALIDATED:**
- b_offsets[k_idx][n_idx_all] — precomputed B-matrix offsets
- Reproduces original offset accumulation logic
- Decode: ~3.77 tok/s

**v4 (BMatrixInfo + ACTIVE_IDX) — VALIDATED:**
- BMatrixInfo { offset_in_dma, active_idx }
- O(active) direct access instead of O(active x all) matching loop
- Decode: ~3.79 tok/s

**v5 (C + c_bound) — VALIDATED (FINAL):**
- c_bound: skip set_io_mem C after first call (analogous to b_bound)
- a_bound: CRASHES — RKNN requires set_io_mem A before every mem_sync
- Decode: ~3.78 tok/s, quality 5/5

Conclusion: tiling cache does not increase speed (bottleneck = NPU compute 80%),
but eliminates nested matching loop and improves stability.

### 2.2 Step 2: GGUF Page Cache Release (madvise)

Goal: eliminate duplication of GGUF mmap (7.2 GB) + RKNPU buffer (6.2 GB) = 13.5 GB

Key discovery:
- data in set_tensor = GGUF mmap pointer (MAP_SHARED, PROT_READ)
- tensor->data = RKNPU virtual buffer (MAP_PRIVATE | MAP_ANONYMOUS)
- Previous attempt to madvise(tensor->data) was WRONG — freed needed NPU pages

Implementation (~10 lines):
After rknn_mem_sync(TO_DEVICE) for NPU tensors:
  madvise(data, aligned_size, MADV_DONTNEED) — frees GGUF pages
GGUF opened PROT_READ — pages are clean, kernel frees without data loss

Results:

| Metric       | Before     | After      | Delta         |
|--------------|------------|------------|---------------|
| RSS          | 13.5 GB    | 7.5 GB     | -6.0 GB (44%) |
| Decode       | 3.5 tok/s  | 3.5 tok/s  | unchanged     |
| Quality      | 5/5        | 5/5        | OK            |
| GGUF released| 0          | 6.46 GB    | +196 tensors  |

### 2.3 Step 2b: NPU Weight Cache (ALREADY IMPLEMENTED in upstream)

- b_bound: skip rknn_create_mem_from_fd and set_io_mem B after first call
- mem_B shared_ptr cached in rknpu_matmul_context
- B_ctx = 0.3% of decode time — NOT a bottleneck

### 2.4 Step 2c: C_sync Optimization — NO SPEEDUP

Test 1: skip mem_sync(FROM_DEVICE):
- Quality OK (ARM cache-coherent)
- Speed DROPPED: 3.07 vs 3.5 tok/s
- Without cache invalidate, CPU reads stale from L2, which is slower

Test 2: parallel C_sync (#pragma omp):
- No significant improvement (~3.5 tok/s)
- 3 segments, thread creation overhead > gain

Conclusion: C_sync = necessary cache invalidation, cannot be optimized away

### 2.5 Decode Profiling (168 matmul/token, T-lite 8B Q8_0)

| Phase   | Time       | Share |
|---------|------------|-------|
| B_ctx   | 0.5 ms     | 0.3%  |
| A_prep  | 5-7 ms     | 3%    |
| NPU_run | 153-166 ms | 80%   |
| C_sync  | 23-38 ms   | 17%   |

NPU compute = 80% — hardware limitation, not fixable in software.

---

## 3. Architectural Limitations

### 3.1 Batch Matmul — IMPOSSIBLE for decode

- RKNN matmul: C(M,N) = A(M,K) × B(K,N), M = batch dimension
- Decode: M=1 (one token at a time), cannot increase
- rknn_matmul_run — blocking call, no async API
- Software pipelining (overlap CPU/NPU) impossible
- Prefill already works with M>1 via next_power_of_two (49 tok/s)

### 3.2 W4A4 Matmul — REJECTED

- W4A4 = model compression, not hardware acceleration
- rkllm W4A4: 13 tok/s, but 53% quality — unusable
- Quality W4A4 requires separate training (months of research)
- Alternative: 4B Q8_0 model gives same size with full quality

### 3.3 Q4_K_M → W8A8 — REJECTED

- Won't speed up: W8A8 at M=1 = same bottleneck
- NPU offload for Q4_K_M is pointless — CPU decode is faster

---

## 4. Model Comparison on RK3588

### 4.1 Methodology

- Device: Orange Pi 5 Max (RK3588, 16 GB RAM)
- Backend: llama.cpp + ggml-rknpu2 (with madvise patch)
- Context: 512 tokens
- Quantization: Q8_0 (except where noted)
- Quality: 5 questions (capital, speed of light, author, element, mountain)
- CPU: taskset 0xf0 -t 4 (big cores only, A76 @ 2.3 GHz)

**IMPORTANT:** RK3588 is big.LITTLE (4×A76 + 4×A55). LITTLE cores slow down
NPU inference by 17-33%. All results below are on big cores.

### 4.2 Dense Models on NPU (big cores only)

| Model             | Params     | Quant | RSS   | NPU buf | Decode   | Quality |
|-------------------|------------|-------|-------|---------|----------|---------|
| Gemma-4 E2B       | 4.6B dense | Q8_0  | 4.7 GB| 2.2 GB  | 9.1 tok/s| 5/5     |
| Qwen2.5-Coder-3B | 3B dense   | Q8_0  | 3.0 GB| 2.9 GB  | 8.1 tok/s| 4/5     |
| Qwen3.5-4B        | 4B dense   | Q8_0  | 4.3 GB| 4.0 GB  | 5.9 tok/s| 5/5     |
| Gemma-4 E4B       | 7.5B dense | Q8_0  | 7.6 GB| 4.4 GB  | 5.0 tok/s| 5/5     |
| T-lite 8B         | 8B dense   | Q8_0  | 7.5 GB| 6.2 GB  | 4.1 tok/s| 5/5     |
| Gemma-4 E4B       | 7.5B dense | Q4_0  | 5.8 GB| 2.8 GB  | 4.3 tok/s| 2/5!!!  |

### 4.3 big.LITTLE Impact on Speed

| Model                  | 8 cores (big+LITTLE) | 4 big      | Speedup |
|------------------------|----------------------|------------|---------|
| Gemma-4 E2B Q8_0      | 6.9 tok/s            | 9.1 tok/s  | +32%    |
| Qwen2.5-Coder-3B Q8_0 | 6.1 tok/s            | 8.1 tok/s  | +33%    |
| Qwen3.5-4B Q8_0       | 4.5 tok/s            | 5.9 tok/s  | +31%    |
| Gemma-4 E4B Q8_0      | 4.1 tok/s            | 5.0 tok/s  | +22%    |
| T-lite 8B Q8_0         | 3.5 tok/s            | 4.1 tok/s  | +17%    |

Cause: LITTLE cores (A55 @ 1.8 GHz) are slower at CPU-side work
(dequantize, quantize, pack, layernorm). With 8 threads, the scheduler
assigns work to LITTLE cores, creating a bottleneck.
Fix: taskset 0xf0 + -t 4 to pin to big cores only (A76).

### 4.4 MoE Model (CPU-only)

| Model          | Params          | Quant | RSS    | NPU buf | Decode   | Quality |
|----------------|-----------------|-------|--------|---------|----------|---------|
| LFM2.5-8B-A1B  | 8B MoE (1B act) | Q8_0  | 13.3 GB| 0.7 GB  | 10 tok/s | OK+CoT  |

### 4.5 Key Observations

1. Model size vs speed correlation:
   - Fewer params → faster decode (inverse correlation)
   - Gemma-4 E2B (4.6B) faster than Qwen-Coder-3B (3B) — architectural difference

2. NPU utilization:
   - Dense Q8_0: 100% weights on NPU (W8A8 pipeline)
   - MoE: only 8% weights on NPU (shared layers), experts on CPU
   - Q4_0: supported via W4A4_HADAMARD, but quality is POOR

3. Q4_0 quality degradation:
   - Gemma-4 E4B Q4_0: Element 79 = "Mougn" (???), speed of light = 770 m/s (???)
   - Q8_0 same model: Element 79 = Gold (Au), speed of light = 299,792,458 m/s
   - Conclusion: Q4_0 on RK3588 is unsuitable for accuracy-sensitive tasks

4. Gemma-4 E2B/E4B are DENSE models (NOT MoE):
   - "E2B" = actually 4.6B parameters (size_label=4.6B, n_expert=0)
   - "E4B" = actually 7.5B parameters (size_label=7.5B, n_expert=0)
   - E2B/E4B naming is Google marketing ("E" = "efficient")

5. Tool calling via system prompt:
   - Qwen3.5-4B and Qwen2.5-Coder-3B: correct JSON with explicit system prompt
   - OpenAI tools API doesn't work — chat template doesn't process tools parameter
   - Parallel function calling works (JSON array)
   - Multi-turn (call → response → answer) works

6. Chat with reasoning models:
   - Qwen3.5-4B: requires enable_thinking=false, otherwise reasoning_content eats output
   - LFM2.5-8B-A1B: CoT in reasoning_content, content empty without configuration
   - Gemma-4: requires enable_thinking=false

---

## 5. Technical Discoveries

### 5.1 RKNN API Semantics

- rknn_matmul_set_io_mem A: REQUIRED before every mem_sync TO_DEVICE
  (a_bound crashes server — RKNN uses set_io_mem as DMA trigger)
- rknn_matmul_set_io_mem B: can skip after first call (b_bound)
- rknn_matmul_set_io_mem C: can skip after first call (c_bound)
- rknn_mem_sync(TO_DEVICE): DMA transfer of data to NPU
- rknn_mem_sync(FROM_DEVICE): cache invalidation for CPU result reading
- rknn_matmul_run: BLOCKING call, no async API

### 5.2 RKNN Matmul Tiling

- RK3588 NPU: max K=256, max N=256 per core, M=1 for decode
- Large matrices split into tiles (segments)
- segment_size from io_attr.B.size (RKNN ground truth), not mathematical calculation
- Active N-segments: size_n > 0, inactive: size_n = 0 (padding)

### 5.3 NPU Memory

- RKNPU buffer: MAP_PRIVATE | MAP_ANONYMOUS (virtual, 6.2 GB for 8B)
- RKNPU DMA: via IOMMU domains (physical CMA memory, 3 GB)
- rknn_create_mem_from_fd: DMA buffer for B-matrix weights (cached via b_bound)
- C-matrix: c_buffer_cache (shared_ptr, reused between matmul)
- A-matrix: a_buffer_cache (shared_ptr, reused between matmul)

### 5.4 GGUF Page Release (madvise)

- data in set_tensor = GGUF mmap pointer (MAP_SHARED, PROT_READ)
- tensor->data = RKNPU virtual buffer (MAP_PRIVATE | MAP_ANONYMOUS)
- madvise(data, MADV_DONTNEED) after mem_sync — frees GGUF pages
- madvise(tensor->data) — WRONG, frees needed NPU pages
- Result: RSS 13.5 → 7.5 GB (6 GB savings)

### 5.5 Q4_K_M vs Q8_0 on RK3588

| Parameter     | Q4_K_M       | Q8_0          |
|---------------|--------------|---------------|
| NPU support   | NO           | YES (W8A8)    |
| NPU load      | 0%           | 100%          |
| Decode        | 3.5-3.8 tok/s| 3.3-3.6 tok/s |
| RSS (madvise) | ~5 GB        | ~7.5 GB       |
| GGUF size     | 4.7 GB       | 8.1 GB        |

Q4_K_M is faster because CPU fallback is faster than CPU Q8_0 decode (less data)

### 5.6 Comparison with Upstream rk-llama.cpp

Benchmark: upstream (github.com/invisiofficial/rk-llama.cpp) vs optimized,
both built on big cores (taskset 0xf0 -t 4).

| Model                  | Upstream tok/s | Optimized tok/s | Upstream RSS | Optimized RSS | RSS Savings  |
|------------------------|----------------|-----------------|--------------|---------------|--------------|
| Gemma-4 E2B Q8_0      | 9.3            | 9.3             | 7.4 GB       | 5.2 GB        | -2.2 GB (30%)|
| T-lite 8B Q8_0        | 4.1            | 4.0             | 13.7 GB      | 7.3 GB        | -6.4 GB (47%)|
| Qwen2.5-Coder-3B Q8_0 | 8.0            | 8.5             | 6.1 GB       | 3.1 GB        | -3.0 GB (49%)|

Conclusion: decode speed unchanged (NPU compute = 80% bottleneck, not fixable
in software). Main optimization effect — RSS reduction by 2-6 GB (madvise
frees GGUF mmap pages after NPU DMA). For models with large NPU buffer
the effect is greater (47-49% RSS savings).

---

## 6. Files

Source code: [github.com/Sinclear/rk3588-llama-npu](https://github.com/Sinclear/rk3588-llama-npu)

- `ggml-rknpu2.cpp` — main backend (madvise + tiling cache v5 + c_bound)
- `tiling_cache.h/cpp` — tiling cache implementation (v5, C + c_bound)
- `rknpu2-configuration.h/cpp` — RKNN configuration (including Gemma-4 Q8_0)
- `ggml-rknpu2.patch` — diff against upstream rk-llama.cpp

Tested models (HuggingFace, unsloth):

- T-lite-it-1.0 Q8_0 (baseline, 8.1 GB)
- Qwen3.5-4B Q8_0 (4.2 GB)
- Qwen2.5-Coder-3B-Instruct-128K Q8_0 (3.1 GB)
- Gemma-4 E2B Q8_0 (4.8 GB)
- Gemma-4 E4B Q8_0 (7.7 GB)
- Gemma-4 E4B Q4_0 (4.6 GB)
- LFM2.5-8B-A1B Q8_0 (8.4 GB)

---

## 7. Conclusions and Recommendations

### 7.1 Code Optimization — Exhausted

All software optimizations for dense 8B Q8_0 on RK3588 are exhausted:
- NPU compute = 80% of time (hardware limitation)
- C_sync = 17% (cache invalidation, cannot remove)
- A_prep = 3% (minimal)
- B_ctx = 0.3% (already cached)
- Batch matmul impossible (M=1 for decode, rknn_matmul_run blocking)
- Software pipelining impossible (no async API)

### 7.2 Best Model Choice for RK3588

By speed (decode, big cores only):
1. LFM2.5-8B-A1B Q8_0: 10 tok/s (MoE, CPU-only, RSS 13.3 GB)
2. Gemma-4 E2B Q8_0: 9.1 tok/s (dense, NPU, RSS 4.7 GB, quality 5/5)
3. Qwen2.5-Coder-3B Q8_0: 8.1 tok/s (dense, NPU, RSS 3.0 GB, quality 4/5)

By quality:
1. Gemma-4 E2B Q8_0: 5/5, 9.1 tok/s
2. Qwen3.5-4B Q8_0: 5/5, 5.9 tok/s
3. T-lite 8B Q8_0: 5/5, 4.1 tok/s

By RSS:
1. Qwen2.5-Coder-3B Q8_0: 3.0 GB, 8.1 tok/s
2. Qwen3.5-4B Q8_0: 4.3 GB, 5.9 tok/s
3. Gemma-4 E2B Q8_0: 4.7 GB, 9.1 tok/s

**Recommendation: Gemma-4 E2B Q8_0 + taskset 0xf0 -t 4 — best balance**

### 7.3 Future Directions

- Smaller model (1.5B-2B): even faster, but worse quality
- Custom fused ops on NPU: LayerNorm+matmul (requires RKNN SDK access)
- Speculative decoding: small model + verification by large model
- MoE architectures: LFM2.5-8B-A1B gives 10 tok/s on CPU
- Q4_0: supported by NPU, but severe quality degradation

### 7.4 Not Recommended

- Q4_K_M: not supported by NPU (0% offload)
- Q4_0: severe quality degradation (hallucinations)
- W4A4 matmul: 53% quality, unusable
- Batch matmul: impossible for decode (M=1, blocking API)

### 7.5 Launch Command

Required parameters for maximum performance:

```bash
ulimit -n 65536
taskset 0xf0 ./build/bin/llama-server \
  -m model-Q8_0.gguf \
  --host 0.0.0.0 --port 8083 \
  -ngl 99 -c 512 -t 4
```

- `taskset 0xf0` + `-t 4`: pin to big cores (A76), +17-33% speedup
- `ulimit -n 65536`: sufficient file descriptors for NPU contexts
- `-ngl 99`: all layers on NPU (W8A8 pipeline)