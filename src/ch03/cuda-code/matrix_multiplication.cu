#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <utils.hpp>

__global__ void MatrixMulKernel(
    float* M,
    float* N,
    float* A,
    int height,
    int width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float acc = 0;
    /* Compute A[j][i] = \sum_k^p M[j][k] * N[k][i] */
    for (int k = 0; k < width; ++k) {
      acc += M[row*width+k] * N[k*width+col];
    }
    A[row * width + col] = acc;
  }
}

int main(void) {
  /* Matrices dim: 
   * m cols (i iterates over cols) fast index,
   * n rows (j iterates over rows) slow index
   */
  int m = 1000;
  int n = m;

  /* Iniitialize matrices as flat vectors */
  std::vector<float> M(n * m);
  std::vector<float> N(n * m);
  std::vector<float> A(n * m);

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
  float *M_d, *N_d, *A_d;
  size_t matByteDim = n * m * sizeof(float);
  cudaMalloc(&M_d, matByteDim);
  cudaMalloc(&N_d, matByteDim);
  cudaMalloc(&A_d, matByteDim);

  cudaMemcpy(M_d, M.data(), matByteDim, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N.data(), matByteDim, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 dimBlock(16, 16, 1);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, A_d, n, m);

  cudaMemcpy(A.data(), A_d, matByteDim, cudaMemcpyDeviceToHost);

  /* Print Matrices using utils.hpp:
   * printMatrixFlat(matrix, rows, cols, name, cap(optional)) 
   */
  printMatrixFlat(M, n, m, "M", 3);
  printMatrixFlat(N, n, m, "N", 3);
  printMatrixFlat(A, n, m, "A", 3);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(A_d);

  return 0;
}
