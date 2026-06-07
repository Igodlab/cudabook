cmake_minimum_required(VERSION 3.28)
project(pmpp LANGUAGES CXX CUDA)

# Global standards & arch — inherited by all subprojects
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Point cmake to pixi-managed packages
list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

# Single compile_commands.json at root — clangd picks this up automatically
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Chapters ────────────────────────────────────────────────────────────────
# Add a line here each time you start a new chapter
add_subdirectory(src/ch03)
add_subdirectory(src/ch04)
add_subdirectory(src/ch05)
