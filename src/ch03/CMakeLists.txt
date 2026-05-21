# ── Chapter 3 ───────────────────────────────────────────────────────────────
# find_package calls are inherited from root but OpenCV needs to be found here
# since it's used by ch03's utils
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------
# ch03 utils — scoped to this chapter (not shared with other chapters)
# -----------------------------------------------------------------------
add_library(ch03_utils STATIC cuda-code/utils.cpp)

target_include_directories(ch03_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda-code   # utils.hpp visible to all ch03 targets
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ch03_utils PUBLIC
    ${OpenCV_LIBS}
)

# -----------------------------------------------------------------------
# Helper — same pattern as before, scoped to ch03
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
add_ch03_exercise(ch03_vector_addition       cuda-code/vector_addition.cu)
add_ch03_exercise(ch03_color_to_grey         cuda-code/color_to_grey.cu)
add_ch03_exercise(ch03_image_blur            cuda-code/image_blur.cu)
add_ch03_exercise(ch03_matrix_multiplication cuda-code/matrix_multiplication.cu)
add_ch03_exercise(ch03_ex01 cuda-code/ch03_ex01.cu)
add_ch03_exercise(ch03_ex02 cuda-code/ch03_ex02.cu)
