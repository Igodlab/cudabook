# Chapter 3 — Multidimensional Grids and Data

<!--toc:start-->
- [Chapter 3 — Multidimensional Grids and Data](#chapter-3-multidimensional-grids-and-data)
  - [Summary](#summary)
  - [Book Examples](#book-examples)
    - [Color to greyscale](#color-to-greyscale)
    - [Image blur](#image-blur)
    - [Matrix multiplication](#matrix-multiplication)
  - [Exercises](#exercises)
    - [Exercise 1](#exercise-1)
    - [Exercise 2](#exercise-2)
    - [Exercise 3](#exercise-3)
    - [Exercise 4](#exercise-4)
    - [Exercise 5](#exercise-5)
<!--toc:end-->

**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [Color to greysclae](#color-to-greyscale) | Grid/block model, boundary conditions, per-thread pixel mapping |
| Example 2 | [Image blur](#image-blur) | 2D grid indexing, neighborhood access, RGB stride convention |
| Example 3 | [Matrix multiplication](#matrix-multiplication) | One output element per thread, row-major indexing, BLAS fundamentals |
| Exercise 1 | [Row/Column matmul variants](#exercise-1) | One output row or column per thread, coalescing tradeoffs |
| Exercise 2 | [Matrix-vector multiplication](#exercise-2) | Dot product per thread, 1D grid design |
| Exercise 3 | [Grid and block dimensions](#exercise-3) | Interpreting launch configs, counting total threads |
| Exercise 4 | [2D flat indexing](#exercise-4) | Row-major vs column-major element addressing |
| Exercise 5 | [3D flat indexing](#exercise-5) | Row-major addressing for rank-3 tensors |

---
> [!IMPORTANT] 
> **Notation for R-rank tensors in the book**
> 
> We will follow the subscript notation for an R-rank covariant tensor that expresses indexes from ==*right-to-left* (from *fast-to-slow varying index*)==. Moreover, we'll be consistent in **both** CUDA code notation and mathematical expressions!
> 
> The generalized notation for addressing a $R$-rank tensor element with dimensions $T\in\mathbb{R}^{d_{R-1} \times \cdots \times d_1 \times d_0}$ (*slow←fast*) is via its indexes $T_{i_{R-1},\ldots,i_1,i_0}$ (*slow←fast*), respectively.
> 
> The generalized stride $s_k=\prod_{j=0}^{k-1}d_j$ is needed to compute the index in a row-major flattened tensor: $\text{flat(index)}=\sum_{r=0}^{R-1}i_rs_r$. For example:
> - 3D tensor $T\in\mathbb{R}^{d_2\times d_1\times d_0}$ element $T_{i_2,i_1,i_0}$ as row-major $T_{i_0 + i_1\times d_0 + i_2\times(d_0\times d_1)}$
> - 4D tensor $T\in\mathbb{R}^{d_3\times d_2\times d_1\times d_0}$ element $T_{i_3,i_2,i_1,i_0}$ as row-major $T_{i_0 + i_1\times d_0 + i_2\times(d_0\times d_1) + i_3\times(d_0\times d_1\times d_2)}$
> 
> Throughout the book we'll use variations of letters so here is a useful table (up to 4-rank tensors):
>
> | Description | Genearlized | Unespecific | Deep Learning |
> |---|---|---|---|
> | *fastest index*, dim-0, cols, width | $i_0\in[0, d_0]$ | $i\in[0,m]$ | $w\in[0,W]$ |
> | dim-1, rows, height                 | $i_1\in[0, d_1]$ | $j\in[0,n]$ | $h\in[0,H]$ |
> | dim-2, depth, channels              | $i_2\in[0, d_2]$ | $k\in[0,p]$ | $c\in[0,C]$ |
> | dim-3, sample, batch                | $i_3\in[0, d_3]$ | $l\in[0,q]$ | $n\in[0,N]$ |



## Book Examples

### Color to greyscale 

[color_to_grey.cu](color_to_grey.cu) shows how to get started with the grids and blocks GPU model using an example of converting a color image to greyscale. Introduces boundary conditions to account for excess threads larger than image pixels.

```cuda
__global__ void coloToGrayscaleConvertion(
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

### Image blur

[image_blur.cu](image_blur.cu) shows a complex kernel that operates on an RGB color image using a 3-strided flat vector (row-major convention). Each thread computes one output pixel by averaging its neighborhood.

```cuda
__global__ void imageBlur(
    unsigned char* Pin,
    unsigned char* Pout,
    int width,
    int height,
    int blur_radii) 
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int channel = threadIdx.z; // (b,g,r)=(0,1,2)

  if (col < width && row < height) {
    int acc = 0;
    int pixels = 0;

    for (int blurRow = -blur_radii; blurRow < blur_radii + 1; ++blurRow) {
      for (int blurCol = -blur_radii; blurCol < blur_radii + 1; ++blurCol) {
        int currCol = col + blurCol;
        int currRow = row + blurRow;

        if (currCol >= 0 && currCol < width && currRow >= 0 && currRow < height) {
          acc += Pin[(currRow * width + currCol) * CHANNELS + channel];
          ++pixels;
        }
      }
    }
    Pout[(row * width + col) * CHANNELS + channel] = (unsigned char)(acc / pixels);
  }
}
```

### Matrix multiplication 

[matrix_multiplication.cu](matrix_multiplication.cu) is the fundamental BLAS example, one output matrix element per thread. Each thread computes the dot product of one row of M and one column of N.

```cuda
__global__ void MatrixMulKernel(
    float* M,
    float* N,
    float* P,
    int height,
    int width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float acc = 0;
    // Compute P[j,i] = \sum_k^p M[j,k] * N[k,i]
    for (int k = 0; k < width; ++k) {
      acc += M[row*width+k] * N[k*width+col];
    }
    P[row*width+col] = acc;
  }
}
```

---

## Exercises

### Exercise 1

[ch03_ex01.cu](ch03_ex01.cu) shows two kernel variants for matrix multiplication where $M\in\mathbb{R}^{n\times p}$ and $N \in \mathbb{R}^{p\times m}$:

- **1.a** — one thread computes an entire output row-vector `A[row][:]` $\leftarrow \left[\sum_k^{p-1} M_{\text{row},k}N_{k,0}, \ldots, \sum_k^{p-1} M_{\text{row},k}N_{k,m-1}\right]$
- **1.b** — one thread computes an entire output column-vector `B[:][col]` $\leftarrow \left[\sum_k^{p-1} M_{0,k}N_{k,\text{col}}, \ldots, \sum_k^{p-1} M_{n-1,k}N_{k,\text{col}}\right]^T$
- **1.c** - Neither is optimal. Both carry one uncoalesced access pattern. The tiled shared memory approach (Ch. 5) resolves this

<img src="../../../images/ch03/ch03_ex01-sol.png" width="100%">

```cuda
// A - write a kernel that has each thread produce one output matrix row
__global__ void matmulRowKernel(
    float* M, // \in R^{m x l}
    float* N, // \in R^{l x n}
    float* A, // \in R^{m x n}
    int m,
    int l,
    int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    // Compute output row-vector A[row,:]
    // populate all j-th (\in n) elements of row-vector A[row,j] = \sum_k^{l-1} M[row,k] * N[k,j]
    for (int Nj = 0; Nj < n; ++Nj) {
      float acc = 0.0f;
      for (int k = 0; k < l; ++k) {
        acc += M[row * l + k] * N[k * n + Nj];
      }
      A[row * n + Nj] = acc;
    }
  }
}

// B - write a kernel that has each thread produce one output matrix column
__global__ void matmulColKernel(
    float* M,
    float* N,
    float* B,
    int m,
    int l,
    int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < n) {
    // Compute output col-vector B[:,col]
    // populate all i-th (\in m) elements of col-vector B[i,col] = \sum_k^{l-1} M[i,k] * N[k,col]
    for (int Mi = 0; Mi < m; ++Mi) {
      float acc = 0.0f;
      for (int k = 0; k < l; ++k){
        acc += M[Mi * l + k] * N[k * n + col];
      }
      // Populate Mi-th element of column-vector B[:,col]
      B[Mi * n + col] = acc;
    }
  }
}
```

### Exercise 2
[ch03_ex02.cu](ch03_ex02.cu) kernel where each thread computes one full dot product between a square matrix row `B[row,:]` and the input vector `c`. Grid is 1D over the number of matrix rows.

```cuda
__global__ void matVecKernel(
    float* B,
    float* c,
    float* a,
    int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += B[row * n + j] * c[j];
    }
    a[row] = acc;
  }
}
```

### Exercise 3

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

### Exercise 4

Given a 2D matrix $M\in\mathbb{R}^{m\times n}$ stored as a flat vector, express element `M[j][i]` in:

- **Row-major:** `M[i * n + j]`
- **Column-major:** `M[j * m + i]`

### Exercise 5

Given a 3D tensor $M\in\mathbb{R}^{m\times n\times l}$ in row-major order the leftmost index varies fastest, so the strides are:
- $i$ (rows) has stride $n \times r$
- $j$ (cols) has stride $r$
- $k$ (depth) has stride $1$

So the element `T[z=5][y=20][x=10]` of a $(\text{row, col, depth})=(m,n,l)=(500, 400, 300)$ tensor is accessed (in row-major) as `T[5*500*400 + 20*400 + 10] = T[1008010]`
