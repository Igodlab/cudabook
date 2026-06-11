# Chapter 5 — Memory architecture and data locality
**Programming Massively Parallel Processors, 5th Edition**

---

## Summary

| # | Name | Concepts illustrated |
|---|------|----------------------|
| ... | [Notation](#notation) | We use *slow←fast* varying index (& corresponding dimension) notation | 
| Example 1 | [Color to greysclae](#color-to-greyscale) | Grid/block model, boundary conditions, per-thread pixel mapping |
| Example 2 | [Image blur](#image-blur) | 2D grid indexing, neighborhood access, RGB stride convention |
| Example 3 | [Matrix multiplication](#matrix-multiplication) | One output element per thread, row-major indexing, BLAS fundamentals |
| Exercise 1 | [Row/Column matmul variants](#exercise-1) | One output row or column per thread, coalescing tradeoffs |
| Exercise 2 | [Matrix-vector multiplication](#exercise-2) | Dot product per thread, 1D grid design |
| Exercise 3 | [Grid and block dimensions](#exercise-3) | Interpreting launch configs, counting total threads |
| Exercise 4 | [2D flat indexing](#exercise-4) | Row-major vs column-major element addressing |
| Exercise 5 | [3D flat indexing](#exercise-5) | Row-major addressing for rank-3 tensors |

---

## Book Examples

### Color to greyscale 


[color_to_grey.cu](color_to_grey.cu) shows how to get started with the grids and blocks GPU model using an example of converting a color image to greyscale. The problem introduces the use of boundary conditions in our kernel to account for excess threads larger than image pixels. We use [unspecific notation](#notation) with manual handling of the rgb channel dimension.

---

## Exercises

### Exercise 1

[ch03_ex01.cu](ch03_ex01.cu) shows two kernel variants for matrix multiplication where $M\in\mathbb{R}^{n\times p}$ and $N \in \mathbb{R}^{p\times m}$:

- **1.a** — one thread computes an entire output row-vector `A[row][:]` $\leftarrow \left[\sum_k^{p-1} M_{\text{row},k}N_{k,0}, \ldots, \sum_k^{p-1} M_{\text{row},k}N_{k,m-1}\right]$
- **1.b** — one thread computes an entire output column-vector `B[:][col]` $\leftarrow \left[\sum_k^{p-1} M_{0,k}N_{k,\text{col}}, \ldots, \sum_k^{p-1} M_{n-1,k}N_{k,\text{col}}\right]^T$
- **1.c** - Neither is optimal. Both carry one uncoalesced access pattern. The tiled shared memory approach (Ch. 5) resolves this


### Exercise 5

Given a 3D tensor $M\in\mathbb{R}^{p\times n\times m}$ in row-major order the rightmost index varies fastest, so the strides are:
- $k$ (depth) has stride $m \times p$
- $j$ (height) has stride $m$
- $i$ (width) has stride $1$

So the element `T[z=5][y=20][x=10]` of a $(p,n,m)=(300, 500, 400)$ tensor is accessed (in row-major) as `T[5*(400*500) + 20*400 + 10] = T[1008010]`

