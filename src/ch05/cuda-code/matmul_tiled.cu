#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "utils.hpp"

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
  float acc = 0;
  for (int ph = 0; ph < n/TILE; ++ph) {
    /* Collaborative loading of M, N */
    Mds[ty][tx] = M[row*n + ph*TILE + tx];
    Nds[ty][tx] = N[ph*TILE*n + ty*n + col];
    __syncthreads();

    for (int k = 0; k < n; ++k) {
      acc += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  P[row*n + col] = acc;
}

int main(void) {
  /* Matrices dim: 
   * m cols (i iterates over cols) fast index,
   * n rows (j iterates over rows) slow index
   */
  int m = 1000;
  int n = m;

  /* Iniitialize matrices as flat vectors */
  std::vector<float> M(m * n);
  std::vector<float> N(m * n);
  std::vector<float> P(m * n);

  /* random device and values */
  float u_min = -10.0f;
  float u_max = 10.0f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  /* populate M, N using row-major indexing */
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      M[j * m + i] = uniform_dist(rng);
      N[j * m + i] = uniform_dist(rng);
    }
  }

  /* Prep for kernel */
  float *M_d, *N_d, *P_d;
  size_t matByteDim = n * m * sizeof(float);
  cudaMalloc(&M_d, matByteDim);
  cudaMalloc(&N_d, matByteDim);
  cudaMalloc(&P_d, matByteDim);

  cudaMemcpy(M_d, M.data(), matByteDim, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N.data(), matByteDim, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 dimBlock(16, 16, 1);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, n, m);

  cudaMemcpy(P.data(), P_d, matByteDim, cudaMemcpyDeviceToHost);

  /* Print Matrices using utils.hpp:
   * printMatrixFlat(matrix, rows, cols, name, cap(optional)) 
   */
  printMatrixFlat(M, n, m, "M", 3);
  printMatrixFlat(N, n, m, "N", 3);
  printMatrixFlat(P, n, m, "P", 3);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);

  return 0;
}
