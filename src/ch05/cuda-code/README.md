# Chapter 5 — Memory architecture and data locality
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [Tiled matmul](#example-1) | Grid/block model, boundary conditions, per-thread pixel mapping |
| Exercise 1 | [Row/Column matmul variants](#exercise-1) | One output row or column per thread, coalescing tradeoffs |
| Exercise 2 | [Matrix-vector multiplication](#exercise-2) | Dot product per thread, 1D grid design |
| Exercise 3 | [Grid and block dimensions](#exercise-3) | Interpreting launch configs, counting total threads |
| Exercise 4 | [2D flat indexing](#exercise-4) | Row-major vs column-major element addressing |
| Exercise 5 | [3D flat indexing](#exercise-5) | Row-major addressing for rank-3 tensors |
| Exercise 6 | [3D flat indexing](#exercise-6) | Row-major addressing for rank-3 tensors |
| Exercise 7 | [3D flat indexing](#exercise-7) | Row-major addressing for rank-3 tensors |
| Exercise 8 | [3D flat indexing](#exercise-8) | Row-major addressing for rank-3 tensors |
| Exercise 9 | [3D flat indexing](#exercise-9) | Row-major addressing for rank-3 tensors |
| Exercise 10 | [3D flat indexing](#exercise-10) | Row-major addressing for rank-3 tensors |
| Exercise 11 | [3D flat indexing](#exercise-11) | Row-major addressing for rank-3 tensors |
| Exercise 12 | [3D flat indexing](#exercise-12) | Row-major addressing for rank-3 tensors |

---

## Book Examples

### Example 1


[tiled_matmul.cu](tiled_matmul.cu) improves on Chapter 3's matmul introducing *tiles* to reduce the number of reads from global memory. Tiles basically improve bandwidth by `TILE` dimension times by cooperatively loading a subset of input data to `__shared__` memory (scope is a thread block). The working logic is exemplified in the figure below

<img src="../../../images/ch05/tiled-matmul.png" width="100%">

example of kernel:

```cuda
#define TILE 16

__global__ void SquareMatmulTiled(
    float* M,
    float* N,
    float* P,
    int n)
{
  __shared__ float Mds[TILE][TILE];
  __shared__ float Nds[TILE][TILE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int col = bx * TILE + tx;
  int row = by * TILE + ty;

  /* Loop over */
  float acc = 0.0f;
  for (int ph = 0; ph < ceil(n/(float)TILE); ++ph) {
    /* Collaborative loading of M, N 
     * boundary conditions for tiles
     */
    if (row < n && (ph*TILE + tx) < n) {
      Mds[ty][tx] = M[row*n + tx + ph*TILE];
    } else Mds[ty][tx] = 0.0f;

    if (col < n && (ph*TILE + ty) < n) {
      Nds[ty][tx] = N[(ty + ph*TILE)*n + col];
    } else Nds[ty][tx] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  /* Boundary conditions for output matrix */
  if (row < n && col < n) {
    P[row*n + col] = acc;
  }
}
```

---

## Exercises

### Exercise 1

Matrix addition is performed elelemt by element (in-place) therefore we can use shared memory to cooperatively load from tiles to reduce global memory bandwidth consumption if we change our thread configuration ie. one thread computes an entire column/row output vector. Otherwise, if we instruct a thread to produce one output element we cannot leverage shared memory usage.

### Exercise 2

<img src="../../../images/ch05/ch05_ex02-sol.png" width="100%">

### Exercise 3

If no `__syncthreads()` are placed after *(i)* cooperatively tile load step and *(ii)* output value accumulator calculation step for every phase; our code will fail *read-after-write & write-after-read dependences*. Meaning that:
- we could have incomplete tile loads like uninitialized values or old values in dynamically and statically-sized tiles, respectively. 
- we could be writing corrupted values to any output element `accumulator`

### Exercise 4

Ignoring register and shared memory capacity limitations - shared memory is still beneficial because is shared accross all threads in the same block as opposed to registers which are evenly allocated accross all threads in SM. So even in the hypothetical case of registers having the capacity to store arbitrary-large variables these would have to be read from global memory for every thread which is expensive.

### Exercise 5

If we use a 32 x 32 tile in squared matmul the reduction of memory bandwidth is 32 for each M and N matrx dimensions.

### Exercise 6

Variables defined in local memory have thread-level scope so the 1000-block kernel will create such local variable 512000 times (512 times per thread per block).

### Exercise 7

Defining Exercise 6's varible from local to shared memory will make the program create such variable 1000 times (once every block).

### Exercise 8

Performing squared matmul of dimensions $N \times N$ every element of the input matrices is requested from gobal memory:
- **8.a.** $2N^2$ times without tiling.
- **8.b.** $2N^2/T$ with $T\times T$ tiling.

### Exercise 9

A kernel performs 36 FLOP and 7 4-Bytes global memory reads per thread which gives a *computational-intensity* of

$$
\frac{36 \text{ FLOP}}{7\times 4 \text{ B}} = 1.29 \text{ FLOP/B}
$$

Given the following device properties at peak capacity:

<img src="../../../images/ch05/roofline-model.png" width="50%">

- **9.a.** 200 GFLOPS & 100 GB/s - yields a *compute-intensity* of $2 \text{ FLOP/B}(> 1.29 \text{ FLOP/B})$ which indicates that the kernel is *memory-bound*.
- **9.b.** 300 GFLOPS & 250 GB/s - gives $1.2 \text{FLOP/B}(< 1.29 \text{ FLOP/B})$ *compute-intensity* so the kernel is *compute-bound*.

### Exercise 10

```cuda
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height)

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
  __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

  int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

  blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
  
  A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

As we've seen in [Exercise 3](#exercise-3) **10.b.** if we ommit a `__syncthreads()` after cooperative tile loading to `__shared__` memory, then we're making our code vulnerable to *read-after-write* failures. **10.a.** Thus, the code will only run with guarantees for `BLOCK_WIDTH = 1` 

### Exercise 11

```cuda
__global__ void foo_kernel(float* a, float* b) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  float x[4];
  __shared__ float y_s;
  __shared__ float b_s[128];
  for (unsigned int j = 0; j < 4; ++j) {
    x[j] = a[j*blocks.x*gridDim.x + i];
  }
  if (threadIdx.x == 0) {
    y_s = 7.4f;
  }
  b_s[threadIdx.x] = b[i];
  __syncthreads();
  b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*[3]
         + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}
void foo(int* a_d, int* b_d) {
  unsigned int N = 1024;
  foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```

- **11.a.** - there are `gridDim.x * blockDim.x = 8 * 128 = 1024` threads and versions of `i` in total.
- **11.b.** - one `x[]` per thread so 1024.
- **11.c.** - there are as many versions of `y_s` as thread blocks → 128.
- **11.d.** - one shared memory `y_s` per thread block so 128.
- **11.e.** - the amount of used shared memory per thread block is `sizeof(y_s) + sizeof(b_s) = 4 B + (128 * 4) B = 516 B`.
- **11.f.** - each thread makes 4 + 

### Exercise 12


