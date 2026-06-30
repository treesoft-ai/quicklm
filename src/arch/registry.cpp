#include "registry.hpp"
#include "qwen3_5.hpp"
#include <memory>
#include <vector>

namespace arch {

// Lazily-built list of all compiled-in architectures. To add an architecture,
// implement IArchitecture in src/arch/<name>.{hpp,cpp} and push it here.
static const std::vector<std::unique_ptr<IArchitecture>>& all() {
    static const std::vector<std::unique_ptr<IArchitecture>> registry = [] {
        std::vector<std::unique_ptr<IArchitecture>> v;
        v.push_back(std::make_unique<Qwen3_5Architecture>());
        return v;
    }();
    return registry;
}

const IArchitecture* find_by_model_type(const std::string& model_type) {
    for (const auto& a : all()) {
        if (a->matches(model_type)) {
            return a.get();
        }
    }
    return nullptr;
}

} // namespace arch
