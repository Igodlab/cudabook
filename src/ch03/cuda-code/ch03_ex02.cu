#include "utils.hpp"
#include <cuda_runtime.h>
#include <random>
#include <vector>

__global__ void matVecKernel(
    float* B,
    float* c,
    float* a,
    int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += B[row * n + j] * c[j];
    }
    a[row] = acc;
  }
}

int main(void) {
  /* Define matrix (square n x n) and vector (n x 1) */
  int m = 1000;
  int n = m;

  std::vector<float> B(n * m); /* n x n */
  std::vector<float> c(n);     /* n x 1 */
  std::vector<float> a(n);     /* n x 1 */

  float u_min = -10.0f;
  float u_max = 10.f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  /* Populate matrix B an vector c */
  for (int j = 0; j < n; ++j) {
    c[j] = uniform_dist(rng);
    for (int i = 0; i < m; ++i) {
      B[j * m + i] = uniform_dist(rng);
    }
  }

  printMatrixFlat(B, n, m, "B", 3);
  printVec(c.data(), n, 3);

  /* GPU prep */
  float *B_d, *c_d, *a_d;
  size_t matrixByteSize = n * m * sizeof(float);
  size_t vectorByteSize = n * sizeof(float);

  cudaMalloc(&B_d, matrixByteSize);
  cudaMalloc(&c_d, vectorByteSize);
  cudaMalloc(&a_d, vectorByteSize);

  cudaMemcpy(B_d, B.data(), matrixByteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c.data(), vectorByteSize, cudaMemcpyHostToDevice);

  /* Call kernel */
  dim3 gd(ceil(n/1024.0), 1, 1);
  dim3 bd(1024, 1, 1);
  matVecKernel<<<gd, bd>>>(B_d, c_d, a_d, n);

  cudaMemcpy(a.data(), a_d, vectorByteSize, cudaMemcpyDeviceToHost);
  printVec(a.data(), n, 3);

  return 0;
}
