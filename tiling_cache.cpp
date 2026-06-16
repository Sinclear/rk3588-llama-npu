/*
 * Tiling cache v4 implementation (Variant C: BMatrixInfo with active_idx)
 */

#include "tiling_cache.h"

namespace rknpu2_tiling_cache {

TilingCache& get_tiling_cache() {
    static TilingCache instance;
    return instance;
}

const TilingPlan* TilingCache::get_or_compute(
    int M, int K, int N, int type_hint,
    const rknpu2_configuration::Rknpu2HardwarePipeline* pipeline,
    const rknpu2_configuration::Rknpu2DeviceConfig& config
) {
    CacheKey key = std::make_tuple(M, K, N, type_hint);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            stats_hits_++;
            return &it->second;
        }
    }

    stats_misses_++;

    TilingPlan plan;

    plan.M_op = M;
    if (M > 1) {
        plan.M_op = rknpu2_calibration::next_power_of_two(M);
    }

    plan.use_hadamard = pipeline->use_hadamard;
    plan.K_op = plan.use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

    int k_limit = config.max_k_limit;
    if (pipeline->effective_k > 0) {
        k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
    }
    plan.k_segments = compute_k_segments(plan.K_op, k_limit, pipeline->k_align);

    int alignment = pipeline->n_align;
    auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);
    plan.all_n_segments = all_n_segments;

    // Build active segments and mapping: all_n_idx -> active_idx
    plan.n_segments.clear();
    std::vector<int> all_to_active(all_n_segments.size(), -1);
    for (size_t n_idx = 0; n_idx < all_n_segments.size(); ++n_idx) {
        if (all_n_segments[n_idx].size_n > 0) {
            all_to_active[n_idx] = (int)plan.n_segments.size();
            plan.n_segments.push_back(all_n_segments[n_idx]);
        }
    }
    plan.num_active_segments = (int)plan.n_segments.size();

    plan.matmul_type = pipeline->mm_type;
    plan.k_align = pipeline->k_align;
    plan.n_align = pipeline->n_align;
    plan.npu_type_a = (int)pipeline->npu_type_a;
    plan.npu_type_b = (int)pipeline->npu_type_b;
    plan.npu_type_c = (int)pipeline->npu_type_c;
    plan.effective_k = pipeline->effective_k;

    plan.type_size_packed = 0;
    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) plan.type_size_packed = 2;
    else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) plan.type_size_packed = 1;

    // Variant C: precompute b_info with offset_in_dma + active_idx
    // Exact same accumulation logic as original code
    plan.b_info.resize(plan.k_segments.size());
    size_t current_offset_in_tensor = 0;

    for (size_t k_idx = 0; k_idx < plan.k_segments.size(); ++k_idx) {
        const int K_seg_op = plan.k_segments[k_idx].size_k;
        plan.b_info[k_idx].resize(all_n_segments.size());

        for (size_t n_idx = 0; n_idx < all_n_segments.size(); ++n_idx) {
            const auto& n_seg = all_n_segments[n_idx];

            plan.b_info[k_idx][n_idx].offset_in_dma = current_offset_in_tensor;
            plan.b_info[k_idx][n_idx].active_idx = all_to_active[n_idx];

            if (n_seg.size_n > 0) {
                if (plan.type_size_packed > 0) {
                    current_offset_in_tensor += (size_t)n_seg.size_n * K_seg_op * plan.type_size_packed;
                } else {
                    current_offset_in_tensor += (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = cache_.emplace(key, std::move(plan));
        return &result.first->second;
    }
}

} // namespace rknpu2_tiling_cache