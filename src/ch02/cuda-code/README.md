# Chapter 2 — Heterogenous data-parallel computing
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [Vector addition](#vector-addition-1) | "Hello world!" analogue to GPU programming |
| Example 2 | [Matmul w/ dynamic tile dim](#example-2) | Cooperative tiling for GEMM w/ dynamic tile dimensions |
| Example 3 | [Matmul w/ optimal tile dim](#example-3) | Cooperative tiling for GEMM w/ optimal tile dimensions based on hardware limitations |
| Exercise 1 | [Matrix addition](#exercise-1) | Tiling in matrix addition gives no benefits |
| Exercise 2 | [Tiling drawings](#exercise-2) | 8x8 matmul with 2x2 & 4x4 tiling drawings exemplify the efficient memory read factor |
| Exercise 3 | [Importance of `__syncthreads()`](#exercise-3) | Ommiting `__syncthreads()` results in read-after-write & write-after-read failures |
| Exercise 4 | [Benefits of shared memory](#exercise-4) | Even if registers would have near infinite capacity shared memory is still benefitial |
| Exercise 5 | [Reduction of memory bandwidth](#exercise-5) | Reduction of global memory reads factor is the tile dimension |
| Exercise 6 | [Local variable declaration in kernel](#exercise-6) | Local variable scope is at thread-level |
| Exercise 7 | [Shared memory variable declaration in kernel](#exercise-7) | Variable loaded to shared memory has thread-block-level scope |
| Exercise 8 | [Memory bandwidth w/ & w/o tiling](#exercise-8) | Reduction of global memory reads factor is the tile dimension |
| Exercise 9 | [Compute- or memory-bound](#exercise-9) | Roofline model for determining compute-bound and/or memory bound regimes |
| Exercise 10 | [`__syncthreads()` missing](#exercise-10) | Code example with no `__syncthreads()` forces tile dimension > 1 to be inefective |
| Exercise 11 | [Computational throughput and memory bandwidth](#exercise-11) | Different variable scopes and compute-intensity [OPerations/Byte] |
| Exercise 12 | [Occupancy, kernels & hardware specs](#exercise-12) | Occupancy dependence on kernel specs and hardware limitations |

---

## Book Examples

### Vector addition

[vecAdd.cu](vecAdd.cu) queries the properties of all GPU devices available as described in Section 4.8 of the book. In my case a humble RTX 4070 laptop GPU:

```
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
```

---

## Exercises

### Exercise 1

**c.** `i=blockIdx.x*blockDim.x + threadIdx.x` maps thread/block indices to data index (`i`). Assuming we want each thread to compute one vector addition element.

### Exercise 2

**c.** `i=(blockIdx.x*blockDim.x + threadIdx.x)*2` maps thread/block indices to data index (`i`). Assuming we want each thread to compute two adjacent elements of vector addition.

### Exercise 3

**d.** `i=blockIdx.x*blockDim.x*2 + threadIdx.x` maps thread/block indices to data index `i`. Assuming we want each thread to compute two elements of a vector addition where each thread-block process `2*blockDim.x` consecutive elements that form two sections (sections are processed sequentially).

### Exercise 4

**c.** Given a vector length of 8000 and a block dimension of 1024 we'd be launching a kernel of 8 blocks wich results in a total of 8192 invoked threads.

### Exercise 5

**d.** `v * sizeof(int)` gives the correct byte-size second argument to `cudaMalloc` for allocating `v` integer elements to the device global memory.

### Exercise 6

**d.** `(void **) &A_d` is a pointer to a pointer and is the correct first argument to `cudaMalloc` for allocating `n` floating-point elements to the device global memory.

### Exercise 7

**c.** `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);` is the correct API call to copy 3000 bytes of data from host `A_h` to device `A_d`.

### Exercise 8

**c.** `cudaError_t error` is the correct type for instantiating CUDA errors 

### Exercise 9

```cuda
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03     if (i < N) {
04         b[i] = 2.7f*a[i] - 4.3f;
05     }
06 }
07 void foo(float* a_d, float* b_d) {
08     unsigned int N = 200000;
09     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d, N);
10 }
```

**9.a.** - 128 threads/block.
**9.b.** - $1563\times 32=200064$ threads in grid.
**9.c.** - $(N+128-1)/128 = 1563$ blocks in grid.
**9.d.** - All 200064 threads are executed on code line `02`.
**9.e.** - Only 200000 threads are executed on code line `04`.

### Exercise 10

The student can simply use both `__host__ __device__` **function declarations** and the compiler will generate two versions of such functions one for the host and another for the device.
