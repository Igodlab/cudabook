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
- **1.d.** For the conditional in line `07`:
    - *i.* All warps are active because half of its threads' indexes are even.
    - *ii.* All 32 warps in the grid are divergent because even numbered thread indexes take the then path at `07` whereas odd indexes do not.
    - *iii.* Warp 0 has 16 threads active (even indexes) → 50% SIMD efficiency.
- **1.e.** For the conditional in line `09`
    - *i.* The right-hand side conditional at line `09`:  `rhs@09 = 5 - (i%3)` yields values {5, 3, 4} that cycle around depending on the thread index of a warp. Which makes any thread index >= 5 divergent. That is:
        - All threads in warps 1-3 diverge in any of the 8 blocks.
        - And threads 4-31 of warp 0 diverge (x8 blocks). So only 4 iterations per block actually run → **32 iterations in total**.
        ```
        for all 8 blocks:

          threadIdx warp-0 = [0, 1, 2, 3, 4, 5, 6, ..., 30, 31]
         rhs@09 for warp-0 = [5, 4, 3, 5, 4, 3, 5, ...,  5,  4]
        -------------------------------------------------------
        forLoopBool warp-0 = [1, 1, 1, 1, 0, 0, 0, ...,  0,  0]  # (false, true)=(0, 1)

        ```
    - *ii.* Following *1.e.i.* there are 28 (warp 0) + 96 (warps 1-3) = 124 divergent threads per block → **992 divergent iterations**.

### Exercise 2

We'll have a 1D grid of $\text{ceil}(2000/512)=4$ thread blocks each containing 512 threads thus, there will be 2048 threads in the grid.

### Exercise 3

The last warp (block 3, warp 15, threadIdx 480-511, absolute threadIdx 2015-2047) is completely inactive because is larger than our vector size of 2000. The second to last warp (block 3, warp 14, threadIdx 448-479, absolute threadIdx 1982-2014) will be divergent due boundary conditions where absolute threadIdx 2000-2014 diverge.

### Exercise 4

The 8 hypothetical threads have execution times $\vec{t}=(2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9) \mu s$, respectively. The barrier for the entire block is set by the longest execution time $t_\text{barrier}=3.0\mu s$ until then most of the threads sit idle. The problem asks for the aggregate-waiting-time percentage 

$$
\begin{align*}
100\times\sum_i (t_\text{barrier} - t_i)/t_\text{barrier} &= \frac{100}{8t_\text{barrier}}\sum(1.0, 0.7, 0.0, 0.2, 0.6, 1.1, 0.4, 0.1) \\
&=100\times\frac{4.1}{24} \\
&=17.08 /%\% \textdiscount
\end{align*}
$$ 

### Exercise 5

Even if a kernel has only 32 threads is not a good idea to ommit a `__syncthreads()` despite it being redudant at a hardware level (where all threads in a warp execute in lockstep under the SIMD model) from a software perspective it forces a time barrier synchronization which helps with *transparent scalability* and makes the code more portable. 

### Exercise 6

If a SM can take up to 1536 and up to 4 thread blocks. The block configuration that gives the most threads in the SM is **(6.c.) 3 blocks with 512 threads/block**

| # | threads/block | 2 blocks | 3 blocks | 4 blocks |
|---|---|---|---|---|
| 6.a | 128  |  256 |      384 |  512 |
| 6.b | 256  |  512 |      768 | 1024 |
| 6.c | 512  | 1024 | **1536** | 2048 | 
| 6.d | 1024 | 2048 |     3072 | 4096 |

### Exercise 7

Assume a device with SMs that can take up to 64 blocks and 2048 threads/SM. All options are possible configurations given <= 2048 threads in the grid.
- **7.a.** 8 blocks with 128 threads each → 1024 threads in grid with an occupancy of $100\times 1024 / 2048 = 50$%.
- **7.b.** 16 blocks with 64 threads each → 1024 threads in grid with an occupancy of $100\times 1024 / 2048 = 50$%.
- **7.c.** 32 blocks with 32 threads each → 1024 threads in grid with an occupancy of $100\times 1024 / 2048 = 50$%.
- **7.d.** 64 blocks with 32 threads each → 2048 threads in grid with an occupancy of $100\times 2048 / 2048 = 50$%.
- **7.e.** 32 blocks with 64 threads each → 2048 threads in grid with an occupancy of $100\times 2048 / 2048 = 50$%.

### Exercise 8

A GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM and 64K (65536) registers/SM. Assess if the following specifications reach full occupancy and if not what is the limiting factor:
- **8.a.** The kernel uses 128 threads/block and 30 registers/thread.
- **8.b.** The kernel uses 32 threads/block and 29 registers/thread.
- **8.c.** The kernel uses 256 threads/block and 34 registers/thread.

### Exercise 9


