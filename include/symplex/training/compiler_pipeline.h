// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/optimizer/superoptimizer.h"
#include "symplex/codegen/code_generator.h"
#include "symplex/schedule/schedule_map.h"
#include <string>
#include <vector>
#include <memory>

namespace symplex::training {

struct PipelineResult {
    std::string ptx_source;
    std::string kernel_name;
    schedule::TileConfig optimal_tile;
    double estimated_latency_ns;
    double speedup_vs_naive;
    std::vector<int64_t> grid_dims;
    std::vector<int64_t> block_dims;
    bool valid;
    std::string error;
};

class CompilerPipeline {
public:
    explicit CompilerPipeline(hardware::HardwareTarget target);

    // Run the full pipeline: iteration space -> optimized PTX kernel
    PipelineResult compile(const polyhedral::IterationSpace& ispace);

    // Compile with a specific tile configuration (skip optimization)
    PipelineResult compile_with_tile(
        const polyhedral::IterationSpace& ispace,
        const schedule::TileConfig& tile
    );

    // Compile for matmul specifically
    PipelineResult compile_matmul(int64_t M, int64_t N, int64_t K);

    // Compile for conv2d
    PipelineResult compile_conv2d(
        int64_t batch, int64_t oc, int64_t ic,
        int64_t oh, int64_t ow, int64_t kh, int64_t kw,
        int64_t stride = 1, int64_t pad = 0
    );

    const hardware::HardwareTarget& target() const;

private:
    hardware::HardwareTarget target_;
    optimizer::Superoptimizer superopt_;
    codegen::CodeGenerator codegen_;
};

} // namespace symplex::training
