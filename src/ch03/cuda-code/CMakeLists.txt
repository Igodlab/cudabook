# ── Chapter 3 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by many chapters' utils
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# utils — scoped within each chapter (not shared with other chapters)
# -----------------------------------------------------------------------
add_library(ch03_utils STATIC utils.cpp)

target_include_directories(ch03_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # utils.hpp visible to all ch03 targets
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ch03_utils PUBLIC
    ${OpenCV_LIBS}
)

# -----------------------------------------------------------------------
# Helper scoped to this chapter only
# -----------------------------------------------------------------------
function(add_ch03_exercise target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ch03_utils CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

# -----------------------------------------------------------------------
# cuda-code targets — one line each
# -----------------------------------------------------------------------
add_ch03_exercise(ch03_vector_addition       vector_addition.cu)
add_ch03_exercise(ch03_color_to_grey         color_to_grey.cu)
add_ch03_exercise(ch03_image_blur            image_blur.cu)
add_ch03_exercise(ch03_matrix_multiplication matrix_multiplication.cu)
add_ch03_exercise(ch03_ex01                  ch03_ex01.cu)
add_ch03_exercise(ch03_ex02                  ch03_ex02.cu)
