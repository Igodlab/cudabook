# Chapter 3 — Multidimensional Grids and Data
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [Color to greysclae](cuda-code#color-to-greyscale) | Grid/block model, boundary conditions, per-thread pixel mapping |
| Example 2 | [Image blur](image_blur.cu) | 2D grid indexing, neighborhood access, RGB stride convention |
| Example 3 | [Matrix Multiplication](matrix_multiplication.cu) | One output element per thread, row-major indexing, BLAS fundamentals |
| Exercise 1 | [Row/Column matmul variants](ch03_ex01.cu) | One output row or column per thread, coalescing tradeoffs |
| Exercise 2 | [Matrix-vector multiplication](ch03_ex02.cu) | Dot product per thread, 1D grid design |
| Exercise 3 | Grid and Block Dimensions | Interpreting launch configs, counting total threads |
| Exercise 4 | 2D Flat Indexing | Row-major vs column-major element addressing |
| Exercise 5 | 3D Tensor Flat Indexing | Row-major addressing for rank-3 tensors |

---

## Book Examples

### Color to greyscale ([color_to_grey.cu](color_to_grey.cu))

Kernel that shows how to get started with the grids and blocks GPU model using an example of converting a color image to greyscale. Introduces boundary conditions to account for excess threads larger than image pixels.

```cuda
__global__
void coloToGrayscaleConvertion(
    unsigned char *Pout,
    unsigned char *Pin,
    int width,
    int height) 
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // Get 1D offset for the greyscale output
    int grayOffset = row * width + col;

    // Input has 3x more dimensions due to rgb color channels
    int rgbOffset = grayOffset * CHANNELS;

    // Each pixel requires three bytes (one for each color channel)
    // OpenCV reads images as bgr so we need to account for that
    unsigned char b = Pin[rgbOffset];     // blue
    unsigned char g = Pin[rgbOffset + 1]; // green
    unsigned char r = Pin[rgbOffset + 2]; // red

    // Compute greys
    Pout[grayOffset] = 0.299*r + 0.587*g + 0.114*b;
  }
}
```

### Image Blur

A more complex kernel that operates on an RGB color image using a 3-strided flat vector (row-major convention). Each thread computes one output pixel by averaging its neighborhood.

```cuda
__global__ void imageBlurKernel(
    unsigned char* Pout,
    unsigned char* Pin,
    int width, int height) { ... }
```

### Matrix Multiplication

Fundamental BLAS example. One output matrix element per thread. Each thread computes the dot product of one row of M and one column of N.

```cuda
__global__ void matmulKernel(
    float* M, float* N, float* A,
    int m, int k, int n) { ... }
```

---

## Exercises

### Exercise 1 — Row and Column Matmul Variants

Two kernel variants for matrix multiplication where $M\in\mathbb{R}^{m\times k}$ and $N \in \mathbb{R}^{k\times n}$:

- **1.a** — one thread computes an entire output row-vector
- **1.b** — one thread computes an entire output column-vector

Neither is optimal. Both carry one uncoalesced access pattern. The tiled shared memory approach (ch. 5) resolves this.

```cuda
__global__ void matmulRowKernel(float* M, float* N, float* A, int m, int k, int n) { ... }
__global__ void matmulColKernel(float* M, float* N, float* B, int m, int k, int n) { ... }
```

### Exercise 2 — Matrix-Vector Multiplication

Kernel where each thread computes one full dot product between a matrix row and the input vector. Grid is 1D over the number of matrix rows.

```cuda
__global__ void matvecKernel(float* M, float* v, float* out, int rows, int cols) { ... }
```

### Exercise 3 — Grid and Block Dimension Analysis

Given the following kernel and configuration execution parameters `bd(16,32)` & `gd(19,5)` (note that C/C++ drops the decimal part in integer division if no float is specified in `gd`), we have:
- *a)* 512 threads per block
- *b)* $\text{threads\/block} \times (\text{gridDim.x} \times \text{gridDim.y})=48640$ threads in the grid
- *c)* 95 grids
- *d)* The threads that execute the code in line `05` are $M\times N=45000$ (which means that there is 3640 inactive threads)

```cuda
__global__ void foo_kernel(float* a , float* b , unsigned int M, unsigned int N) {
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

### Exercise 4 — 2D Flat Indexing

Given a 2D matrix $M\in\mathbb{R}^{m\times n}$ stored as a flat vector, express element `M[i, j]` in:

- **Row-major:** `M[i * n + j]`
- **Column-major:** `M[j * m + i]`

### Exercise 5 — 3D Tensor Flat Indexing

Given a 3D tensor $T\in\mathbb{R}^{m\times n\times r}$ flattened in row-major order, element `T[i, j, k]` is `T[i * n * r + j * r + k]`
