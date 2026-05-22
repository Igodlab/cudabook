# Chapter 3 — Multidimensional Grids and Data
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [color_to_grey.cu](src/ch03/cuda-code/color_to_grey.cu) | Grid/block model, boundary conditions, per-thread pixel mapping |
| Example 2 | [image_blur.cu](src/ch03/cuda-code/image_blur.cu) | 2D grid indexing, neighborhood access, RGB stride convention |
| Example 3 | [matrix_multiplication.cu](src/ch03/cuda-code/matrix_multiplication.cu) | One output element per thread, row-major indexing, BLAS fundamentals |
| Exercise 1 | [ch03_ex01.cu](src/ch03/cuda-code/ch03_ex01.cu) | Row/Column Matmul Variants. One output row or column per thread, coalescing tradeoffs |
| Exercise 2 | [ch03_ex02.cu](src/ch03/cuda-code/ch03_ex02.cu) | Matrix-Vector Multiplication. Dot product per thread, 1D grid design |
| Exercise 3 | Grid and Block Dimensions | Interpreting launch configs, counting total threads |
| Exercise 4 | 2D Flat Indexing | Row-major vs column-major element addressing |
| Exercise 5 | 3D Tensor Flat Indexing | Row-major addressing for rank-3 tensors |

---

## Book Examples

### Color to Greyscale

Kernel that shows how to get started with the grids and blocks GPU model using an example of
converting a color image to greyscale. Introduces boundary conditions to account for excess
threads larger than image pixels.

```cuda
__global__ void colorToGreyscaleKernel(
    unsigned char* Pout,
    unsigned char* Pin,
    int width, int height) { ... }
```

### Image Blur

A more complex kernel that operates on an RGB color image using a 3-strided flat vector
(row-major convention). Each thread computes one output pixel by averaging its neighborhood.

```cuda
__global__ void imageBlurKernel(
    unsigned char* Pout,
    unsigned char* Pin,
    int width, int height) { ... }
```

### Matrix Multiplication

Fundamental BLAS example. One output matrix element per thread. Each thread computes the
dot product of one row of M and one column of N.

```cuda
__global__ void matmulKernel(
    float* M, float* N, float* A,
    int m, int k, int n) { ... }
```

---

## Exercises

### Exercise 1 — Row and Column Matmul Variants

Two kernel variants for matrix multiplication where M ∈ R^{m×k} and N ∈ R^{k×n}:

- **1.a** — one thread computes an entire output row-vector
- **1.b** — one thread computes an entire output column-vector

**Pros/cons analysis:**

| | Row kernel | Column kernel |
|---|---|---|
| M access | coalesced | strided |
| N access | strided | coalesced |
| Grid | 1D over m | 1D over n |
| Work per thread | n*k ops | m*k ops |

Neither is optimal. Both carry one uncoalesced access pattern. The tiled shared memory
approach (ch. 5) resolves this.

```cuda
__global__ void matmulRowKernel(float* M, float* N, float* A, int m, int k, int n) { ... }
__global__ void matmulColKernel(float* M, float* N, float* B, int m, int k, int n) { ... }
```

### Exercise 2 — Matrix-Vector Multiplication

Kernel where each thread computes one full dot product between a matrix row and the input
vector. Grid is 1D over the number of matrix rows.

```cuda
__global__ void matvecKernel(float* M, float* v, float* out, int rows, int cols) { ... }
```

### Exercise 3 — Grid and Block Dimension Analysis

Given a kernel launch configuration, determine:

- `gridDim`, `blockDim` values
- total number of threads launched
- which threads are active vs idle (boundary padding)

### Exercise 4 — 2D Flat Indexing

Given a 2D matrix stored as a flat vector, express element `(i, j)` in:

- **Row-major:** `A[i * n + j]`
- **Column-major:** `A[j * m + i]`

### Exercise 5 — 3D Tensor Flat Indexing

Given a 3D tensor of shape `(d, m, n)` stored flat in row-major order, element `(i, j, k)` is:

```
T[i * (m * n) + j * n + k]
```

```cuda
__global__ void foo_kernel(float* a , float* b , unsigned int M,
    unsigned int N) {
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row < M && col < N) {
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}
void foo(float* a_d , float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16 , 32);
    dim3 gd((N - 1)/16 + 1 , (M - 1)/32 + 1);
    foo_kernel <<<gd, bd>>>(a_d , b_d , M , N);
}
```
