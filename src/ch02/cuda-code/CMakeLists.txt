# ── Chapter 2 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by many chapters' utils
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# utils — scoped within each chapter (not shared with other chapters)
# -----------------------------------------------------------------------
# None for ch02


# -----------------------------------------------------------------------
# Helper scoped to this chapter only
# -----------------------------------------------------------------------
function(add_ch02_exercise target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

# -----------------------------------------------------------------------
# cuda-code targets — one line each
# -----------------------------------------------------------------------
add_ch02_exercise(ch02_vector_addition         vecAdd.cu)
