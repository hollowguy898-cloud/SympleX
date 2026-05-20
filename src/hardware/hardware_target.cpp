// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
#include "symplex/hardware/hardware_target.h"
namespace symplex::hardware {
// HardwareTarget is header-inline. This TU verifies compilation and provides factory validation.
HardwareTarget detect_hardware_target() {
    // Try CUDA detection; fall back to Generic
#ifdef SYMPLEX_ENABLE_CUDA
    // Would query cudaGetDeviceProperties here
    // For now return H100 as the default high-end target
    return HardwareTarget::H100();
#else
    return HardwareTarget::Generic();
#endif
}
} // namespace symplex::hardware
