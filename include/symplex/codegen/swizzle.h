// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include <cstdint>
#include <string>

namespace symplex::codegen {

/// SwizzleGenerator: generates shared-memory swizzle transforms to
/// eliminate bank conflicts in GPU kernels.
///
/// NVIDIA shared memory has 32 banks, each 4 bytes wide. Consecutive
/// 4-byte words map to consecutive banks (bank = address/4 % 32).
/// When multiple threads in a warp access the same bank, a bank
/// conflict occurs and accesses are serialized.
///
/// Swizzling remaps column indices via XOR patterns so that
/// concurrent warp accesses are spread across different banks.
class SwizzleGenerator {
public:
    explicit SwizzleGenerator(const hardware::HardwareTarget& target);

    /// Generate PTX inline-assembly comment block and XOR swizzle
    /// transform for writing into shared memory.
    /// @param tile_m  Row dimension of the tile in shared memory
    /// @param tile_n  Column dimension of the tile in shared memory
    /// @param stride  Leading dimension (row stride) in elements
    /// @return PTX comment + inline asm performing the swizzle on store
    std::string emit_smem_swizzle(
        int64_t tile_m, int64_t tile_n, int64_t stride
    ) const;

    /// Generate the inverse swizzle for reading from shared memory.
    /// For XOR-based swizzle the inverse is the same operation.
    std::string emit_smem_unswizzle(
        int64_t tile_m, int64_t tile_n, int64_t stride
    ) const;

    /// Compute the swizzled shared-memory byte offset for element (row, col).
    /// The swizzle XORs the upper bits of the column index with the row index
    /// to spread accesses across banks.
    int64_t swizzle_address(int64_t row, int64_t col, int64_t stride) const;

    /// Check whether a given tile layout has shared-memory bank conflicts
    /// when accessed by a full warp of 32 threads.
    bool has_bank_conflicts(
        int64_t tile_m, int64_t tile_n, int64_t stride
    ) const;

    /// Find the optimal number of XOR swizzle bits (0-4) that eliminates
    /// all bank conflicts, or the value that minimises conflicts.
    int find_optimal_swizzle_bits(
        int64_t tile_m, int64_t tile_n, int64_t stride
    ) const;

private:
    hardware::HardwareTarget target_;

    static constexpr int NUM_SMEM_BANKS   = 32;   // Standard for modern NVIDIA GPUs
    static constexpr int SMEM_BANK_BYTES  = 4;    // 4 bytes per bank

    /// Internal: simulate warp accesses and return the number of bank conflicts
    /// for a given swizzle_bits value.
    int count_bank_conflicts(
        int64_t tile_m, int64_t tile_n, int64_t stride, int swizzle_bits
    ) const;

    /// Compute the bank index for a given byte address.
    static int bank_of(int64_t byte_addr) {
        return static_cast<int>((byte_addr / SMEM_BANK_BYTES) % NUM_SMEM_BANKS);
    }
};

} // namespace symplex::codegen
