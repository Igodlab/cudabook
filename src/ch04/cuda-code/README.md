# Chapter 4 — Compute architecture and scheduling
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Concepts illustrated |
|------|----------------------|
| [Query Device Properties](#query-device-properties) | CUDA script to query device(s) properties |
| [Exercise 1](#exercise-1) | Number of warps per block, grid and SIMD efficiency |
| [Exercise 2](#exercise-2) | Threads in 1D grid |
| [Exercise 3](#exercise-3) | Divergent warps due to boundary conditions |
| [Exercise 4](#exercise-4) | Aggregate thread waiting time to barrier in warp |
| [Exercise 5](#exercise-5) | Conceptual understanding of `__syncthreads()` usage |
| [Exercise 6](#exercise-6) | Dynamic resource partitioning and occupancy |
| [Exercise 7](#exercise-7) | Resource partitioning and limits (block & thread level) |
| [Exercise 8](#exercise-8) | Resource partitioning and limits (block, thread & register level) |
| [Exercise 9](#exercise-9) | Resource allocation |

---

## Book Examples

### Query Device Properties

[device_properties.cu](device_properties.cu) queries the properties of all GPU devices available as described in Section 4.8 of the book. In my case a humble RTX 4070 laptop GPU:

```
./build/ch04_device_properties
Detected 1 CUDA capable device(s)

Device 0: "NVIDIA GeForce RTX 4070 Laptop GPU"
  Major revision number:         8
  Minor revision number:         9
  Total amount of global memory: 7.65 GB
  Number of multiprocessors:     36
  Total amount of constant memory: 65536 bytes
  Total amount of shared memory per block: 49152 bytes
  Total number of registers available per block: 65536
  Warp size:                     32
  Maximum number of threads per block: 1024
  Maximum sizes of each dimension of a block: 1024 x 1024 x 64
  Maximum sizes of each dimension of a grid: 2147483647 x 65535 x 65535
  Clock rate:                    1.98 GHz
  Memory clock rate:             8001.000488 MHz
  Memory bus width:              128-bit
  L2 cache size:                 33554432 bytes
```

---

## Exercises

### Exercise 1

```cuda
 1 __global__ void foo_kernel(int* a, int* b) {
 2     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 3     if (threadIdx.x < 40 || threadIdx.x >= 104) {
 4         b[i] = a[i] + 1;
 5     }
 6     if (i % 2 == 0) {
 7         a[i] = b[i] * 2;
 8     }
 9     for (unsigned int j = 0; j < 5 - (i%3); ++j) {
10         b[i] += j;
11     }
12 }
13 
14 void foo(int* a_d, int* b_d) {
15     unsigned int N = 1024;
16     foo_kernel<<< (N + 128 - 1)/128, 128>>>(a_d, b_d);
17 }
```

Useful data is `gridDim(.x, .y, .z)=(8,1,1)` and `blockDim(.x, .y, .z)=(128,1,1)`
- **1.a.** There are 128 threads in each block thus, 4 warps per block.
- **1.b.** Four warps per block means there is 32 warps in the grid (total of 8 blocks).
- **1.c.** For the statement of line `04`, divergence is any other path than the first 'then' path:
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
- **1.d.** For the statement in line `07`:
    - *i.* All warps are active because half of its threads' indexes are even.
    - *ii.* All 32 warps in the grid are divergent because even numbered thread indexes take the then path at `07` whereas odd indexes do not.
    - *iii.* Warp 0 has 16 threads active (even indexes) → 50% SIMD efficiency.
- **1.e.** For the statement in line `09`
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

The 8 hypothetical threads have execution times 

$$
\vec{t}=\left(2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9\right) \mu s
$$

, respectively. The barrier for the entire block is set by the longest execution time $t_\text{barrier}=3.0\mu s$ until then most of the threads sit idle. The problem asks for the aggregate-waiting-time percentage

$$
\begin{align*}
100\times\sum_i (t_\text{barrier} - t_i)/t_\text{barrier} &= \frac{100}{8t_\text{barrier}}\sum(1.0, 0.7, 0.0, 0.2, 0.6, 1.1, 0.4, 0.1) \\
&=100\times\frac{4.1}{24} \\
&=17.08 \\%
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
- **7.d.** 64 blocks with 32 threads each → 2048 threads in grid with an occupancy of $100\times 2048 / 2048 = 100$%.
- **7.e.** 32 blocks with 64 threads each → 2048 threads in grid with an occupancy of $100\times 2048 / 2048 = 100$%.

### Exercise 8

A GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM and 64K (65536) registers/SM. Assess if the following specifications reach full occupancy and if not what is the limiting factor:
- **8.a.** The kernel uses 128 threads/block and 30 registers/thread. Full occupancy would be $\text{num-threads/max-num-threads}=\text{num-threads}/2048=1$ here we have:

> The *real physical hardware constraint* is the number of registers per SM (65536) so we will check for it first.
> 
> The 32 blocks/SM and 2048 threads/SM are *architectural constraints* capped at these values according to other hardware limitations like warp scheduler that tracks block state (barrier counts, block IDs, synchronization status). 
> 
> That bookkeeping hardware has a fixed number of slots that cannot be exceeded, regardless of how lightweight the blocks are.

Lets compute max threads that can be distributed in 65536 registers/SM:

$$
65536\frac{\text{registers}}{\text{SM}} \times \frac{1}{30}\frac{\text{thread}}{\text{registers}} = \text{floor}(2184.5333) = 2184 \frac{\text{threads}}{\text{SM}} 
$$

since it exceeds the constraint it'll be automatically capped at 2048 threads/SM. Lets now check the block/SM constraint 

$$
\frac{1}{128}\frac{\text{block}}{\text{threads}} \times 2048\frac{\text{threads}}{\text{SM}} = 16\frac{\text{blocks}}{\text{SM}}
$$

this condition is met (less than 32 blocks/SM). So this configuration can achieve a throughput of

$$
128\frac{\text{threads}}{\text{block}} \times 16\frac{\text{blocks}}{\text{SM}} = 2048 \frac{\text{threads}}{\text{SM}} \rightarrow \text{occupancy} = 100\times\frac{2048}{2048} = 100\\%
$$

- **8.b.** The kernel uses 32 threads/block and 29 registers/thread. Same procedure as *8.a.* this configuration demands → 2259 threads/SM (capped at 2048) and 64 blocks/SM (capped at 32). Realistically this configuration can only operate at 

$$
32 \frac{\text{threads}}{\text{block}} \times 32\frac{\text{blocks}}{\text{SM}} = 1024 \frac{\text{threads}}{\text{SM}} \rightarrow \text{occupancy} = 100\times\frac{1024}{2048} = 50\\%
$$

- **8.c.** The kernel uses 256 threads/block and 34 registers/thread. This configuration demands 1927 threads/SM and 7 blocks/SM which meets hardware constraints. Reallistically it can achieve a throughput of

$$
256\frac{\text{threads}}{\text{block}} \times 7\frac{\text{blocks}}{\text{SM}} = 1792 \frac{\text{threads}}{\text{SM}} \rightarrow \text{occupancy} = 100\times\frac{1792}{2048} = 87.5\\%
$$

### Exercise 9

A student claims that he managed to multiply two 1024 x 1024 matrices using a matmul kernel of 32 x 32 blocks (one output matrix element per thread). He used a CUDA device with limits of 512 threads/block and 8 blocks/SM.

It is not possible to solve the matmul dimensions with the student's device. `blockDim = (32, 32, 1)` thus `gridDim = (32, 32, 1)`. One element per thread means that 1024 x 1024 = 1048576 threads need to be computed. The student's grid is well defined in terms of warps, it launches (32 x 32) x (32 x 32) = 1048576 threads. However, his device cannot support it since its limmit is 512 threads/block (much less than the intended 32 x 32 = 1024 threads/block requirement). The 8 blocks/SM constraint doesn't even come into relevance in this problem.
