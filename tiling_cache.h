/*
 * Tiling cache v4 for ggml-rknpu2 backend
 *
 * Variant C: pre-compute BMatrixInfo with active_idx mapping.
 * Eliminates the O(active × all) matching loop on every matmul call.
 */

#ifndef GGML_RKNPU2_TILING_CACHE_H
#define GGML_RKNPU2_TILING_CACHE_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include "rknpu2-configuration.h"
#include "rknpu2-calibration.h"

namespace rknpu2_tiling_cache {

struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();
    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);
    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];
        if (i < remaining / alignment) seg.size_n += alignment;
        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;
    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }
    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

struct TilingPlan {
    int M_op;
    int K_op;
    std::vector<MatrixSegmentK> k_segments;
    std::vector<MatrixSegmentN> all_n_segments;
    std::vector<MatrixSegmentN> n_segments;       // active only
    int num_active_segments;

    rknn_matmul_type matmul_type;
    int k_align;
    int n_align;
    int npu_type_a;
    int npu_type_b;
    int npu_type_c;
    bool use_hadamard;
    int effective_k;
    int type_size_packed;

    // Variant C: B-matrix info with active_idx mapping
    struct BMatrixInfo {
        size_t offset_in_dma;  // precomputed offset
        int active_idx;        // index in active_n_segments, -1 if inactive
    };
    // b_info[k_idx][n_idx_in_all]
    std::vector<std::vector<BMatrixInfo>> b_info;
};

using CacheKey = std::tuple<int, int, int, int>;

struct KeyHash {
    size_t operator()(const CacheKey& k) const {
        size_t h = std::hash<int>{}(std::get<0>(k));
        h ^= std::hash<int>{}(std::get<1>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<2>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(std::get<3>(k)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class TilingCache {
public:
    const TilingPlan* get_or_compute(
        int M, int K, int N, int type_hint,
        const rknpu2_configuration::Rknpu2HardwarePipeline* pipeline,
        const rknpu2_configuration::Rknpu2DeviceConfig& config
    );

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        stats_hits_ = 0;
        stats_misses_ = 0;
    }

    int hits() const { return stats_hits_; }
    int misses() const { return stats_misses_; }
    size_t size() const { return cache_.size(); }

private:
    std::unordered_map<CacheKey, TilingPlan, KeyHash> cache_;
    mutable std::mutex mutex_;
    int stats_hits_ = 0;
    int stats_misses_ = 0;
};

TilingCache& get_tiling_cache();

} // namespace rknpu2_tiling_cache

#endif