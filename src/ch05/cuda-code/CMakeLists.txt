# ── Chapter 5 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by many chapters' utils
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# utils — scoped within each chapter (not shared with other chapters)
# -----------------------------------------------------------------------
add_library(ch05_utils STATIC utils.cpp)

target_include_directories(ch05_utils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} # utils.hpp visible to all ch05 targets
)

target_link_libraries(ch05_utils PUBLIC
  CUDA::cudart
)

# -----------------------------------------------------------------------
# Helper scoped to this chapter only
# -----------------------------------------------------------------------
function(add_ch05_exercise target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ch05_utils CUDA::cudart)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

# -----------------------------------------------------------------------
# cuda-code targets — one line each
# -----------------------------------------------------------------------
add_ch05_exercise(ch05_tile_matmul         tile_matmul.cu)
add_ch05_exercise(ch05_dynamic_tile_matmul dynamic_tile_matmul.cu)
# add_ch05_exercise(ch05_optimal_tile_matmul optimal_tile_matmul.cu)
