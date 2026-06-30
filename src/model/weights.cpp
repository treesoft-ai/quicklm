#include "weights.hpp"
#include <filesystem>

namespace fs = std::filesystem;

bool ModelWeights::load_from_dir(const std::string& model_dir) {
    if (!fs::exists(model_dir)) {
        return false;
    }

    // Look for all .safetensors files in the directory
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            auto loader = std::make_unique<SafetensorsLoader>();
            if (loader->open(entry.path().string())) {
                loaders.push_back(std::move(loader));
            }
        }
    }
    return !loaders.empty();
}

SafetensorsLoader* ModelWeights::find_loader(const std::string& name, std::string& resolved) const {
    for (const auto& loader : loaders) {
        if (loader->has_tensor(name)) {
            resolved = name;
            return loader.get();
        }
    }
    // Fall back to the architecture's name remap (e.g. Qwen's model.* -> model.language_model.*)
    if (remap) {
        std::string alt = remap(name);
        if (!alt.empty() && alt != name) {
            for (const auto& loader : loaders) {
                if (loader->has_tensor(alt)) {
                    resolved = alt;
                    return loader.get();
                }
            }
        }
    }
    return nullptr;
}

Tensor ModelWeights::get_tensor(const std::string& name) const {
    std::string resolved;
    SafetensorsLoader* loader = find_loader(name, resolved);
    if (!loader) {
        throw std::runtime_error("Tensor not found in any safetensors files: " + name);
    }
    return loader->get_tensor(resolved);
}

Tensor ModelWeights::get_weight(const std::string& name) const {
    std::string resolved;
    SafetensorsLoader* loader = find_loader(name, resolved);
    if (!loader) {
        throw std::runtime_error("Tensor not found in any safetensors files: " + name);
    }
    return loader->get_tensor_keep_bf16(resolved);
}

bool ModelWeights::has_tensor(const std::string& name) const {
    std::string resolved;
    return find_loader(name, resolved) != nullptr;
}
