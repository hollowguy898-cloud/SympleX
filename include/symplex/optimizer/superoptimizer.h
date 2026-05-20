// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/optimizer/tile_config.h"
#include "symplex/optimizer/search_phase1.h"
#include "symplex/optimizer/search_phase2.h"
#include "symplex/optimizer/search_phase3.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/schedule/tiling.h"
#include <string>
#include <sstream>

namespace symplex::optimizer {

/// Result of the full 3-phase superoptimization search.
struct SuperoptimizerResult {
    schedule::TileConfig best_tile;             ///< Best tile configuration found
    double estimated_latency_ns  = 0.0;         ///< Estimated kernel latency
    double speedup_vs_naive      = 0.0;         ///< Speedup vs naive single-MMA tile
    std::string summary;                        ///< Human-readable summary

    /// Was a valid configuration found?
    [[nodiscard]] bool valid() const {
        return !best_tile.inner_tiles.empty();
    }
};

/// Superoptimizer: orchestrates the 3-phase search for the optimal
/// tile configuration for a given iteration space and hardware target.
///
///   Phase 1 – Roofline Filter:       prune memory-bound configs
///   Phase 2 – Symmetry Alignment:    enforce Tensor Core alignment & score
///   Phase 3 – Occupancy Sieve:       analytical latency ranking
///
class Superoptimizer {
public:
    /// Construct a superoptimizer for a specific hardware target.
    explicit Superoptimizer(hardware::HardwareTarget target);

    /// Run the full 3-phase optimization search for the given
    /// iteration space.
    ///
    /// \param ispace          The polyhedral iteration space to optimize.
    /// \param max_tensor_dim  Upper bound on any single tile dimension.
    /// \return                SuperoptimizerResult with the best tile.
    SuperoptimizerResult optimize(
        const polyhedral::IterationSpace& ispace,
        size_t max_tensor_dim = 1024
    );

    /// Access the hardware target.
    [[nodiscard]] const hardware::HardwareTarget& target() const;

private:
    hardware::HardwareTarget target_;
};

} // namespace symplex::optimizer
