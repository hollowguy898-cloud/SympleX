// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/superoptimizer.h"
#include <sstream>
#include <algorithm>

namespace symplex::optimizer {

// ── Constructor ──────────────────────────────────────────────────────────

Superoptimizer::Superoptimizer(hardware::HardwareTarget target)
    : target_(std::move(target))
{}

// ── Accessor ─────────────────────────────────────────────────────────────

const hardware::HardwareTarget& Superoptimizer::target() const {
    return target_;
}

// ── Main optimization pipeline ───────────────────────────────────────────

SuperoptimizerResult Superoptimizer::optimize(
    const polyhedral::IterationSpace& ispace,
    size_t max_tensor_dim
) {
    SuperoptimizerResult result;

    // Determine the number of tiled dimensions from the iteration space.
    // For matmul this is 3 (M, N, K); for conv2d this can be higher but
    // we collapse to 3 for the Tensor Core search.
    const auto& stmts = ispace.statements();
    size_t ndim = 0;
    if (!stmts.empty()) {
        ndim = stmts[0].domain.ndim();
    }

    // The superoptimizer search is currently designed for 2-D and 3-D
    // tiled contractions (the dominant pattern in AI workloads).  For
    // higher-dimensional iteration spaces we map to 3-D by collapsing
    // batch and channel dimensions into the outer tile.
    size_t search_ndim = std::min(ndim, size_t(3));
    if (search_ndim < 2) {
        // Degenerate case: nothing meaningful to tile.
        result.summary = "Iteration space has < 2 dimensions; "
                         "no tiling search performed.";
        return result;
    }

    // ── Phase 1: Roofline Filter ─────────────────────────────────────
    auto phase1_results = phase1_roofline_pruning(
        search_ndim, target_, static_cast<int64_t>(max_tensor_dim));

    if (phase1_results.empty()) {
        result.summary = "Phase 1 (roofline pruning) eliminated all "
                         "candidates.  Problem may be too small for "
                         "the target hardware.";
        return result;
    }

    size_t phase1_count = phase1_results.size();

    // ── Phase 2: Symmetry Alignment ──────────────────────────────────
    auto phase2_results = phase2_symmetry_alignment(
        std::move(phase1_results), target_);

    if (phase2_results.empty()) {
        result.summary = "Phase 2 (symmetry alignment) eliminated all "
                         "candidates after phase 1.";
        return result;
    }

    size_t phase2_count = phase2_results.size();

    // ── Phase 3: Occupancy Sieve ─────────────────────────────────────
    auto phase3_result = phase3_occupancy_sieve(
        std::move(phase2_results), target_, 20);

    if (phase3_result.ranked_candidates.empty()) {
        result.summary = "Phase 3 (occupancy sieve) found no valid "
                         "candidates.";
        return result;
    }

    // ── Build result ─────────────────────────────────────────────────
    const auto& best = phase3_result.best_config;

    // Map the search result back to the original ndim if we collapsed
    // dimensions.  For ndim > 3, we keep the 3-D inner tile from the
    // search and set outer tiles to 1 for the extra dimensions.
    if (ndim <= 3) {
        result.best_tile = schedule::TileConfig(
            best.outer_tiles, best.inner_tiles);
    } else {
        // Pad with 1s for the extra outer dimensions
        std::vector<int64_t> outer = best.outer_tiles;
        std::vector<int64_t> inner = best.inner_tiles;
        for (size_t i = 3; i < ndim; ++i) {
            outer.push_back(1);
            inner.push_back(1);
        }
        result.best_tile = schedule::TileConfig(
            std::move(outer), std::move(inner));
    }

    result.estimated_latency_ns = best.estimated_latency_ns;
    result.speedup_vs_naive     = phase3_result.estimated_speedup_vs_baseline;

    // ── Build summary string ─────────────────────────────────────────
    std::ostringstream oss;
    oss << "Superoptimizer result for '" << ispace.name() << "' on "
        << target_.to_string() << ":\n"
        << "  Best tile: " << result.best_tile.to_string() << "\n"
        << "  Estimated latency: " << result.estimated_latency_ns << " ns\n"
        << "  Speedup vs naive: " << result.speedup_vs_naive << "x\n"
        << "  Phase 1 candidates: " << phase1_count
        << " (roofline pruning)\n"
        << "  Phase 2 candidates: " << phase2_count
        << " (symmetry alignment)\n"
        << "  Phase 3 candidates: " << phase3_result.ranked_candidates.size()
        << " (occupancy sieve, top 20 evaluated)\n"
        << "  Best config OI: " << best.operational_intensity << " FLOPS/byte\n"
        << "  Compute-bound: " << (best.compute_bound ? "yes" : "no") << "\n"
        << "  Occupancy: " << best.occupancy << " warps/SM\n"
        << "  Score (phase 2): " << best.score;

    result.summary = oss.str();

    return result;
}

} // namespace symplex::optimizer
