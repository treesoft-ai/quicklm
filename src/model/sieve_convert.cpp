#include "sieve_convert.hpp"

#include "scout.hpp"
#include "safetensors.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace {

// Deterministic per-tensor seed so re-running the converter on an unchanged
// checkpoint reproduces byte-identical scout files (useful for diffing / CI).
uint64_t seed_for_tensor(const std::string& name) {
    return ScoutTable::hash_name(name) ^ 0xD1B54A32D192ED03ULL;
}

}  // namespace

bool run_sieve_convert(const std::string& model_dir, uint32_t rank, uint32_t sketch_bits,
                        const std::string& tensor_filter) {
    if (!fs::exists(model_dir)) {
        std::cerr << "Error: model directory does not exist: " << model_dir << std::endl;
        return false;
    }

    std::vector<std::unique_ptr<SafetensorsLoader>> loaders;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            auto loader = std::make_unique<SafetensorsLoader>();
            if (loader->open(entry.path().string())) {
                loaders.push_back(std::move(loader));
            }
        }
    }
    if (loaders.empty()) {
        std::cerr << "Error: no .safetensors files found in: " << model_dir << std::endl;
        return false;
    }

    fs::path out_dir = fs::path(model_dir) / "sieve_scouts";
    fs::create_directories(out_dir);

    std::ofstream manifest((out_dir / "manifest.txt").string(), std::ios::trunc);

    int built = 0, skipped = 0;
    for (const auto& loader : loaders) {
        for (const std::string& name : loader->get_tensor_names()) {
            if (!tensor_filter.empty() && name.find(tensor_filter) == std::string::npos) continue;

            // Scout tables are for dense-matmul weight matrices only: 2D,
            // [d_out, h]. Skip biases/norms (1D) and anything else-shaped
            // (e.g. conv kernels, per-head state buffers) — those aren't
            // matmul B operands this scheme ranks rows of.
            Tensor probe = loader->get_tensor_keep_bf16(name);
            if (probe.shape.size() != 2 || !probe.is_bf16()) {
                skipped++;
                continue;
            }
            uint32_t d_out = uint32_t(probe.shape[0]);
            uint32_t h = uint32_t(probe.shape[1]);
            if (d_out == 0 || h == 0) {
                skipped++;
                continue;
            }
            // Rank/sketch must not exceed what the matrix can support.
            uint32_t eff_rank = std::min(rank, std::min(d_out, h));

            std::cout << "Building scout for '" << name << "' [" << d_out << ", " << h
                      << "] rank=" << eff_rank << " sketch_bits=" << sketch_bits << " ... " << std::flush;

            ScoutTable table = ScoutTable::build_from_bf16(
                probe.bf16_data, d_out, h, eff_rank, sketch_bits, seed_for_tensor(name), name);

            // Filesystem-safe file name: tensor names contain '.' which is
            // fine on both Win32/POSIX, but keep it simple/explicit anyway.
            std::string filename = name;
            for (char& c : filename) {
                if (c == '/' || c == '\\' || c == ':') c = '_';
            }
            filename += ".qlsc";
            fs::path out_path = out_dir / filename;

            if (!table.save(out_path.string())) {
                std::cerr << "FAILED to write " << out_path.string() << std::endl;
                continue;
            }
            std::cout << "ok -> " << out_path.filename().string() << std::endl;
            manifest << name << "\t" << filename << "\t" << d_out << "\t" << h << "\t" << eff_rank << "\n";
            built++;
        }
    }

    std::cout << "Sieve scout conversion done: " << built << " built, " << skipped
              << " skipped (non-matrix tensors), output: " << out_dir.string() << std::endl;
    return built > 0;
}
