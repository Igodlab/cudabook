#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <utils.hpp>

__global__ void MatrixMulKernel(
    float* M,
    float* N,
    float* P,
    int height,
    int width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float Pval = 0;
    // Compute P[i,j] = \sum_k M[i,k] * N[k,j]
    for (int k = 0; k < width; ++k) {
      Pval += M[row*width+k] * N[k*width+col];
    }
    P[row*width+col] = Pval;
  }
}

int main(void) {
  // Square matrices dim (i,j)=(n,m)
  int n = 10;
  int m = n;

  std::vector<std::vector<float>> M(m, std::vector<float>(n));
  std::vector<std::vector<float>> N(m, std::vector<float>(n));
  std::vector<std::vector<float>> P(m, std::vector<float>(n));

  // random device and values
  float u_min = -10.0f;
  float u_max = 10.0f;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  // populate M, N
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      M[i][j] = uniform_dist(rng);
      N[i][j] = uniform_dist(rng);
    }
  }

  // Prep for kernel
  float *M_d, *N_d, *P_d;
  size_t matByteDim = n * m * sizeof(float);
  cudaMalloc(&M_d, matByteDim);
  cudaMalloc(&N_d, matByteDim);
  cudaMalloc(&P_d, matByteDim);

  cudaMemcpy(M_d, M.data(), matByteDim, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N.data(), matByteDim, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 dimBlock(16, 16, 1);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, m, n);

  cudaMemcpy(P.data(), P_d, matByteDim, cudaMemcpyDeviceToHost);

  // printMatrix(P, "P");

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}
