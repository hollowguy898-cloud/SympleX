// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/superoptimizer.h"
#include "symplex/optimizer/search_phase1.h"
#include "symplex/optimizer/search_phase2.h"
#include "symplex/optimizer/search_phase3.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include <iostream>
#include <cassert>

using namespace symplex::optimizer;
using namespace symplex::polyhedral;
using namespace symplex::hardware;

int main() {
    // Test 1: Phase 1 roofline pruning
    {
        HardwareTarget target = HardwareTarget::H100();
        auto candidates = phase1_roofline_pruning(3, target, 256);
        assert(!candidates.empty());
        std::cout << "[PASS] Phase 1 roofline pruning (" << candidates.size() << " candidates)\n";
    }

    // Test 2: Phase 2 symmetry alignment
    {
        HardwareTarget target = HardwareTarget::H100();
        auto phase1 = phase1_roofline_pruning(3, target, 256);
        auto phase2 = phase2_symmetry_alignment(std::move(phase1), target);
        assert(!phase2.empty());
        std::cout << "[PASS] Phase 2 symmetry alignment (" << phase2.size() << " candidates)\n";
    }

    // Test 3: Phase 3 occupancy sieve
    {
        HardwareTarget target = HardwareTarget::H100();
        auto phase1 = phase1_roofline_pruning(3, target, 256);
        auto phase2 = phase2_symmetry_alignment(std::move(phase1), target);
        auto result = phase3_occupancy_sieve(std::move(phase2), target, 10);
        assert(result.best_config.inner_volume() > 0);
        std::cout << "[PASS] Phase 3 occupancy sieve (best OI="
                  << result.best_config.operational_intensity << ")\n";
    }

    // Test 4: Full superoptimizer
    {
        HardwareTarget target = HardwareTarget::H100();
        Superoptimizer opt(target);
        auto ispace = make_matmul_iteration_space(1024, 1024, 512);
        auto result = opt.optimize(ispace, 256);
        assert(result.best_tile.inner_tiles.size() > 0);
        assert(result.estimated_latency_ns > 0);
        std::cout << "[PASS] Superoptimizer (latency=" << result.estimated_latency_ns
                  << "ns, speedup=" << result.speedup_vs_naive << "x)\n";
    }

    std::cout << "All optimizer tests passed!\n";
    return 0;
}
