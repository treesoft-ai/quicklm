cmake_minimum_required(VERSION 3.12)
project(QuickLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for standard C++ compilers on Windows
if(MSVC)
    add_compile_options(/O2 /Oi /Ot /GL /EHsc)
    add_compile_options(/arch:AVX2)
    # Remove std::min and std::max macro pollution from windows.h
    add_compile_definitions(NOMINMAX)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# Source files
set(SOURCES
    src/ops/math_ops.cpp
    src/io/safetensors.cpp
    src/io/tokenizer.cpp
    src/io/chat_template.cpp
    src/model/weights.cpp
    src/model/model.cpp
    src/arch/registry.cpp
    src/arch/qwen3_5.cpp
    src/main.cpp
    src/model/attention.cpp
)

# Header files
set(HEADERS
    src/tensor/tensor.hpp
    src/ops/math_ops.hpp
    src/io/json_parser.hpp
    src/io/safetensors.hpp
    src/io/tokenizer.hpp
    src/io/chat_template.hpp
    src/model/config.hpp
    src/model/weights.hpp
    src/model/modules.hpp
    src/model/model.hpp
    src/model/attention.hpp
    src/arch/architecture.hpp
    src/arch/registry.hpp
    src/arch/qwen3_5.hpp
)

add_executable(quicklm ${SOURCES} ${HEADERS})

# Source lives in domain subfolders; includes are bare filenames, so expose each
# subfolder as an include dir and the bare #includes resolve regardless of folder.
target_include_directories(quicklm PRIVATE
    src/tensor
    src/ops
    src/model
    src/io
    src/arch
)

# Link standard libraries (Threads)
find_package(Threads REQUIRED)
target_link_libraries(quicklm PRIVATE Threads::Threads)
