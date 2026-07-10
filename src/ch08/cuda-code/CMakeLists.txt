# ── Chapter 6 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by many chapters' utils
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# utils — scoped within each chapter (not shared with other chapters)
# -----------------------------------------------------------------------
add_library(ch06_utils STATIC utils.cpp)

target_include_directories(ch06_utils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} # utils.hpp visible to all ch06 targets
)

target_link_libraries(ch06_utils PUBLIC
  CUDA::cudart
)

# -----------------------------------------------------------------------
# Helper scoped to this chapter only
# -----------------------------------------------------------------------
function(add_ch06_exercise target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ch06_utils CUDA::cudart)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

# -----------------------------------------------------------------------
# cuda-code targets — one line each
# -----------------------------------------------------------------------
add_ch06_exercise(ch06_parallel_stencil          parallel_stencil.cu)
# add_ch06_exercise(ch06_tile_stencil              tile_stencil.cu)
# add_ch06_exercise(ch06_thread_coarsening_stencil thread_coarsening_stencil.cu)
# add_ch06_exercise(ch06_register_tiling_stencil   register_tiling_stencil.cu)

