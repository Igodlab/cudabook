# ── Chapter 4 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by many chapters' utils
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# utils — scoped within each chapter (not shared with other chapters)
# -----------------------------------------------------------------------
# None for ch04


# -----------------------------------------------------------------------
# Helper scoped to this chapter only
# -----------------------------------------------------------------------
function(add_ch04_exercise target source)
    add_executable(${target} ${source})
    # target_link_libraries(${target} PRIVATE ch04_utils CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

# -----------------------------------------------------------------------
# cuda-code targets — one line each
# -----------------------------------------------------------------------
add_ch04_exercise(ch04_device_properties    cuda-code/device_properties.cu)
add_ch04_exercise(ch04_opti_ch03_ex01       cuda-code/ch04_opti_ch03_ex01.cu)
