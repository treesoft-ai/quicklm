#pragma once

#include "architecture.hpp"
#include <memory>
#include <string>

// Compile-in registry of architecture descriptors. All known architectures are
// registered at first use; the runtime looks one up by model_type.
namespace arch {

// Return the descriptor whose matches(model_type) is true, or nullptr if none.
const IArchitecture* find_by_model_type(const std::string& model_type);

} // namespace arch
