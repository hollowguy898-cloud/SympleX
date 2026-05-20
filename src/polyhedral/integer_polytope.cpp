// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/integer_polytope.h"

// All functionality is header-inline for this module.
// This translation unit exists for potential future explicit template
// instantiations and to ensure the header compiles standalone.

namespace symplex::polyhedral {

// Validate that the header-only implementation compiles correctly.
static_assert(std::is_default_constructible_v<Inequality>);
static_assert(std::is_default_constructible_v<Equality>);

} // namespace symplex::polyhedral
