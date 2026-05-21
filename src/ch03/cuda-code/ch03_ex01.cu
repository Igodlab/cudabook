#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "utils.hpp"

// A - write a kernel that has each thread produce one output matrix row
__global__ void matmulRowKernel(
    float* M, // \in R^{m x k}
    float* N, // \in R^{k x n}
    float* A, // \in R^{m x n}
    int m,
    int k,
    int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    // Compute output row-vector A[row,:]
    // populate all j-th (\in n) elements of row-vector A[row,j] = \sum_{k_th}^k M[row,k_th] * N[k_th,j]
    for (int Nj = 0; Nj < n; ++Nj) {
      float acc = 0.0f;
      for (int k_th = 0; k_th < k; ++k_th) {
        acc += M[row * k + k_th] * N[k_th * n + Nj];
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
    int k,
    int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < n) {
    // Compute output col-vector A[row,:]
    // populate all i-th (\in m) elements of col-vector A[i,col] = \sum_{k_th}^k M[i,k_th] * N[k_th,col]
    for (int Mi = 0; Mi < m; ++Mi) {
      float acc = 0.0f;
      for (int k_th = 0; k_th < k; ++k_th){
        acc += M[Mi * k + k_th] * N[k_th * n + col];
      }
      // Populate Mi-th element of column-vector B[:,col]
      B[Mi * n + col] = acc;
    }
  }
}

int main(void) {
  // Matrices dim: i iterates over rows (height), j iterates over cols (width)
  int m = 3000;
  int k = 4000;
  int n = 5000;

  std::vector<float> M(m * k);
  std::vector<float> N(k * n);
  std::vector<float> A(m * n);
  std::vector<float> B(m * n);

  float u_min = -10.0f;
  float u_max = 10.0f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  // populate M, N 
  for (int i = 0; i < m; ++i) {
    for (int k_th = 0; k_th < k; ++k_th) {
      M[i * k + k_th] = uniform_dist(rng);
    }
  }
  for (int k_th = 0; k_th < k; ++k_th) {
    for (int j = 0; j < n; ++j) {
      N[k_th * n + j] = uniform_dist(rng);
    }
  }

  printMatrixFlat(M, m, k, "M", 3);
  printMatrixFlat(N, k, n, "N", 3);

  // GPU prep
  float *M_d, *N_d, *A_d, *B_d;
  size_t MByteSize = m * k * sizeof(float);
  size_t NByteSize = k * n * sizeof(float);
  size_t outByteSize = m * n * sizeof(float);

  cudaMalloc(&M_d, MByteSize);
  cudaMalloc(&N_d, NByteSize);
  cudaMalloc(&A_d, outByteSize);
  cudaMalloc(&B_d, outByteSize);

  cudaMemcpy(M_d, M.data(), MByteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N.data(), NByteSize, cudaMemcpyHostToDevice);

  // Kernel A - each thread computes a row
  dim3 gdA(ceil(m / 1024.0), 1, 1); // Fit height (m) in `blockDim.x` blocks
  dim3 bdA(1024, 1, 1);             // Each output matrix element computes an entire row-vector
  matmulRowKernel<<<gdA, bdA>>>(M_d, N_d, A_d, m, k, n);

  cudaMemcpy(A.data(), A_d, outByteSize, cudaMemcpyDeviceToHost);

  printMatrixFlat(A, m, n, "A (row-vector per thread)", 3);

  // Kernel B - each thread computes a column
  dim3 gdB(ceil(n / 1024.0), 1, 1); // Fit width (n) in `blockDim.x` blocks
  dim3 bdB(1024, 1, 1);             // Each output matrix element computes an entire column-vector
  matmulColKernel<<<gdB, bdB>>>(M_d, N_d, B_d, m, k, n);

  cudaMemcpy(B.data(), B_d, outByteSize, cudaMemcpyDeviceToHost);

  printMatrixFlat(B, m, n, "B (col-vector per thread)", 3);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(A_d);
  cudaFree(B_d);
  
  return 0;
}
