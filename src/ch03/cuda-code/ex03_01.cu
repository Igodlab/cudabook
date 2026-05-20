#include <cuda_runtime.h>
#include <random>
#include <vector>

// A - write a kernel that has each thread produce on output matrix row
__global__ void matmulRowKernel(
    float* M, // \in R^{m x k}
    float* N, // \in R^{k x n}
    float* A, // \in R^{m x n}
    int m,
    int k,
    int n)
{
  int Ai = blockIdx.x * blockDim.x + threadIdx.x;

  if (Ai < m) {
    for (int Nj = 0; Nj < n; ++Nj) {
      float acc = 0.0f;
      for (int Mj = 0; Mj < k; ++Mj) {
        acc += M[Ai * m + Mj] * N[Mj * n + Nj];
      }
      A[Ai * n + Nj] = acc;
    }
  }
}


int main(void) {
  // Matrices dim: i iterates over rows (height), j iterates over cols (width)
  int m = 3;
  int k = 4;
  int n = 5;

  std::vector<float> M(m * k);
  std::vector<float> N(k * n);
  std::vector<float> A(m * n);
  std::vector<float> B(m * n);

  float u_min = -10.0f;
  float u_max = 10.0f;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  // populate M, N 
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      M[i * n + j] = uniform_dist(rng);
    }
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      N[i * n + j] = uniform_dist(rng);
    }
  }

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
  dim3 gdA(ceil(m / 32.0), 2, 1); // Fit height (m) in `blockDim.x` blocks
  dim3 bdA(32, 1, 1);             // Each output matrix element computes an entire row
  matmulRowKernel<<<gdA, bdA>>>(M_d, N_d, A_d, m, k, n);
  
  // Kernel B - each thread computes a column
  // dim3 gdB(1, ceil(std::get<0>(outDim) / 32.0), 1); // Fit each output matrix col in blocks
  // dim3 bdB(32, 1, 1);
  // matmulColKernel<<<gdA, bdA>>>(M_d, N_d, A_d, m, k, n);

  return 0;
}
