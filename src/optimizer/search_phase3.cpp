// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/search_phase3.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace symplex::optimizer {

namespace {

/// Compute a "naive" baseline latency for comparison.
/// The naive config is the smallest tile that fits the Tensor Core
/// native dimensions (m, n, k) with no further optimisation.
double naive_baseline_latency(
    const hardware::HardwareTarget& target,
    size_t ndim
) {
    const auto& tc = target.gpu.tensor_core;

    ExtendedTileConfig naive;
    if (ndim >= 3) {
        naive.inner_tiles = {tc.m, tc.n, tc.k};
        naive.outer_tiles = {tc.m, tc.n, tc.k};
        naive.compute_ops  = compute_matmul_ops(tc.m, tc.n, tc.k);
        naive.bytes_moved  = compute_matmul_bytes(
            tc.m, tc.n, tc.k, target.bytes_per_element);
    } else {
        naive.inner_tiles = {tc.m, tc.n};
        naive.outer_tiles = {tc.m, tc.n};
        int64_t vol = tc.m * tc.n;
        naive.compute_ops  = 2 * vol;
        naive.bytes_moved  = 3 * vol * target.bytes_per_element;
    }
    naive.operational_intensity = naive.calc_operational_intensity();

    return estimate_latency_ns(naive, target);
}

/// Re-estimate latency with an occupancy-aware model.
/// The roofline model gives a lower bound; low occupancy degrades the
/// achievable fraction of peak.  We apply an occupancy derating factor:
///   effective_flops = peak_flops * min(1.0, occupancy / max_warps)
double occupancy_derated_latency(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    double peak_flops = target.peak_flops_fp16();
    double peak_bw    = target.gpu.memory.global_bw_gbps * 1e9;

    // Occupancy derating: if we don't saturate the SM, we achieve
    // a fraction of peak proportional to active-warps / max-warps.
    double occ_ratio = 1.0;
    if (target.gpu.sm.max_warps > 0 && cfg.occupancy > 0) {
        occ_ratio = std::min(
            static_cast<double>(cfg.occupancy) /
            static_cast<double>(target.gpu.sm.max_warps),
            1.0);
    } else if (cfg.occupancy == 0) {
        occ_ratio = 0.1;  // Extremely low – penalise heavily
    }

    // Pipeline overlap: with software pipelining, memory and compute
    // can partially overlap.  The effective latency approaches
    // max(compute, memory) * (1 - overlap_factor) + min(compute, memory) * overlap_factor
    // For a 2-stage pipeline with perfect overlap on the steady state:
    double pipeline_overlap = (target.pipeline_stages > 1) ? 0.15 : 0.0;

    double compute_time_s = (peak_flops > 0.0 && occ_ratio > 0.0)
        ? static_cast<double>(cfg.compute_ops) / (peak_flops * occ_ratio)
        : std::numeric_limits<double>::max();
    double memory_time_s  = (peak_bw > 0.0)
        ? static_cast<double>(cfg.bytes_moved) / peak_bw
        : std::numeric_limits<double>::max();

    // Apply pipeline overlap reduction
    double roofline_time = std::max(compute_time_s, memory_time_s);
    double min_time      = std::min(compute_time_s, memory_time_s);
    double effective_time = roofline_time * (1.0 - pipeline_overlap)
                          + min_time * pipeline_overlap;

    return effective_time * 1e9;  // ns
}

} // anonymous namespace

// ── Empirical profiling placeholder ──────────────────────────────────────

double empirical_profile_latency(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& /*target*/
) {
    // Placeholder: in production this would:
    //   1. Compile a parameterised matmul kernel with the given tile sizes.
    //   2. Launch it on the GPU with CUDA events for timing.
    //   3. Return the measured median latency over multiple iterations.
    //
    // For now we simply return the analytical estimate.
    return cfg.estimated_latency_ns;
}

// ── Phase 3 implementation ───────────────────────────────────────────────

SearchPhase3Result phase3_occupancy_sieve(
    std::vector<ExtendedTileConfig> candidates,
    const hardware::HardwareTarget& target,
    size_t top_n
) {
    SearchPhase3Result result;

    if (candidates.empty()) {
        // No candidates survived earlier phases – return empty result.
        return result;
    }

    // Step 1: take only the top_n candidates (they are already sorted
    //         by phase 2 score).
    if (candidates.size() > top_n) {
        candidates.resize(top_n);
    }

    // Step 2: compute occupancy-aware latency for each candidate.
    size_t ndim = candidates.front().inner_tiles.size();

    for (auto& cfg : candidates) {
        // Re-estimate occupancy if not already populated
        if (cfg.occupancy <= 0) {
            cfg.occupancy = estimate_occupancy(cfg, target);
        }

        cfg.estimated_latency_ns = occupancy_derated_latency(cfg, target);

        // Optionally override with empirical profiling (currently a no-op
        // that returns the analytical value).
        double empirical_ns = empirical_profile_latency(cfg, target);
        // Blend: 80% analytical + 20% empirical when empirical is available
        // (currently both are the same, so no change).
        cfg.estimated_latency_ns = 0.8 * cfg.estimated_latency_ns
                                 + 0.2 * empirical_ns;
    }

    // Step 3: rank by estimated latency (ascending = fastest first).
    std::sort(candidates.begin(), candidates.end(),
        [](const ExtendedTileConfig& a, const ExtendedTileConfig& b) {
            return a.estimated_latency_ns < b.estimated_latency_ns;
        });

    result.ranked_candidates = std::move(candidates);
    result.best_config = result.ranked_candidates.front();

    // Step 4: compute speedup vs naive baseline.
    double baseline_ns = naive_baseline_latency(target, ndim);
    if (baseline_ns > 0.0 && result.best_config.estimated_latency_ns > 0.0) {
        result.estimated_speedup_vs_baseline =
            baseline_ns / result.best_config.estimated_latency_ns;
    }

    return result;
}

} // namespace symplex::optimizer
