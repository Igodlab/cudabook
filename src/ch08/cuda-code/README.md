# Chapter 8 — Memory architecture and data locality
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| Example 1 | [Tiled matmul](#example-1) | Cooperative tiling for GEMM w/ static tile dimensions |
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

### Example 1


- 
