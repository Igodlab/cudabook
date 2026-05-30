# Chapter 4 — Compute architecture and scheduling
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Concepts illustrated |
|------|----------------------|
| [Exercise 1](#exercise-1) | Number of warps per block, grid and SIMD efficiency |
| [Exercise 2](#exercise-2) | Dot product per thread, 1D grid design |
| [Exercise 3](#exercise-3) | Interpreting launch configs, counting total threads |
| [Exercise 4](#exercise-4) | Row-major vs column-major element addressing |
| [Exercise 5](#exercise-5) | Row-major addressing for rank-3 tensors |
| [Exercise 6](#exercise-6) | Row-major addressing for rank-3 tensors |
| [Exercise 7](#exercise-7) | Row-major addressing for rank-3 tensors |
| [Exercise 8](#exercise-8) | Row-major addressing for rank-3 tensors |
| [Exercise 9](#exercise-9) | Row-major addressing for rank-3 tensors |

---

## Exercises

### Exercise 1

```cuda
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if (i % 2 == 0) {
        a[i] = b[i] * 2;
    }
    for (unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<< (N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

Useful data is `gridDim(.x, .y, .z)=(8,1,1)` and `blockDim(.x, .y, .z)=(128,1,1)`
- **1.a.** There are 128 threads in each block thus, 4 warps per block.
- **1.b.** Four warps per block means there is 32 warps in the grid (total of 8 blocks).
- **1.c.** For the conditional of line `04`, divergence is any other path than the first 'then' path:
    - *i.* In every block:
        - Warp 0 (0-31) is active, all its threads indexes are < 40.
        - Warp 1 (32-63) has only threads 32-39 active (the rest is > 40).
        - Warp 2 (64-95) is completely inactive (all its indexes are < 104).
        - Warp 3 (96-127) has threads 104-127 active (fullfill >= 104).
        - There is 3 active warps per block which means there are 24 active warps in the grid.
    - *ii.* Following *1.c.i.* there is one divergent warp per block making a total of 8 divergent warps in the grid.
    - *iii.* Recall that SIMD efficiency is *active-threads/warp*. All 32 threads in warp 0 are active so 100%.
    - *iv.* Warp 1 has 8 active threads → 25%.
    - *v.* Warp 3 has 24 active thrads → 75%.
- **1.d.** For the conditional in line `07`
    - *i.* In every block:
        - Warp 0 (0-31) is active, all its threads indexes are < 40.
        - Warp 1 (32-63) has only threads 32-39 active (the rest is > 40).
        - Warp 2 (64-95) is completely inactive (all its indexes are < 104).
        - Warp 3 (96-127) has threads 104-127 active (fullfill >= 104).
        - There is 3 active warps per block which means there are 24 active warps in the grid.
    - *ii.* Following *1.d.i.* there is one divergent warp per block making a total of 8 divergent warps in the grid.
    - *iii.* Warp 0 of block 0 has 8 active threads → 25%.
- **1.e.** For the conditional in line `07`
    - *i.* In every block:
    - *ii.* Following *1.d.i.* there is one divergent warp per block making a total of 8 divergent warps in the grid.

### Exercise 2



### Exercise 3



### Exercise 4



### Exercise 5


### Exercise 6


### Exercise 7


### Exercise 8


### Exercise 9


