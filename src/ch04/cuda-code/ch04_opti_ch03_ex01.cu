#include <cuda_runtime.h>
#include <random>
#include <vector>

/* A - write a kernel that has each thread produce one output matrix row */
__global__ void matmulRowKernel(
    float* M, /* \in R^{n x p} */
    float* N, /* \in R^{p x m} */
    float* A, /* \in R^{n x m} */
    int m,
    int p,
    int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    /* Compute output row-vector A[row][:] */
    /* populate all i-th (\in m) elements in row-vector A[row][:] = \sum_k^p M[row][k] * N[k][i] */
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int k = 0; k < p; ++k) {
        acc += M[row * p + k] * N[k * m + i];
      }
      /* row-th row-vector A[row][:] */
      A[row * m + i] = acc;
    }
  }
}

/* B - write a kernel that has each thread produce one output matrix column */
__global__ void matmulColKernel(
    float* M,
    float* N,
    float* B,
    int m,
    int p,
    int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < m) {
    /* Compute output col-vector B[:][col] */
    /* populate all j-th (\in n) elements in col-vector B[:][col] = \sum_k^p M[j][k] * N[k][col] */
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < p; ++k){
        acc += M[j * p + k] * N[k * m + col];
      }
      /* col-th column-vector B[:][col] */
      B[j * m + col] = acc;
    }
  }
}

int main(void) {
  /* Matrices dim: 
   * m rows (i iterates over rows, fast ix),
   * n cols (j iterates over cols, slow ix)
   */
  int m = 3000;
  int p = 4000;
  int n = 5000;

  std::vector<float> M(n * p);
  std::vector<float> N(p * m);
  std::vector<float> A(n * m);
  std::vector<float> B(n * m);

  float u_min = -10.0f;
  float u_max = 10.0f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  /* populate inputs 
   * M is n x p
   * N is p x m
   */
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < p; ++k) {
      M[j * p + k] = uniform_dist(rng);
    }
  }
  for (int k = 0; k < p; ++k) {
    for (int i = 0; i < m; ++i) {
      N[k * m + i] = uniform_dist(rng);
    }
  }

  /* GPU prep */
  float *M_d, *N_d, *A_d, *B_d;
  size_t MByteSize = n * p * sizeof(float);
  size_t NByteSize = p * m * sizeof(float);
  size_t outByteSize = n * m * sizeof(float);

  cudaMalloc(&M_d, MByteSize);
  cudaMalloc(&N_d, NByteSize);
  cudaMalloc(&A_d, outByteSize);
  cudaMalloc(&B_d, outByteSize);

  cudaMemcpy(M_d, M.data(), MByteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N.data(), NByteSize, cudaMemcpyHostToDevice);

  /* Kernel A - each thread computes a row */
  dim3 gdA(36, 1, 1); /* Fit height (n) in `blockDim.x` blocks */
  dim3 bdA(160, 1, 1);             /* Each output matrix element computes an entire row-vector */
  matmulRowKernel<<<gdA, bdA>>>(M_d, N_d, A_d, m, p, n);

  cudaMemcpy(A.data(), A_d, outByteSize, cudaMemcpyDeviceToHost);

  /* Kernel B - each thread computes a column */
  dim3 gdB(36, 1, 1); /* Fit width (m) in `blockDim.x` blocks */
  dim3 bdB(96, 1, 1);             /* Each output matrix element computes an entire column-vector */
  matmulColKernel<<<gdB, bdB>>>(M_d, N_d, B_d, m, p, n);

  cudaMemcpy(B.data(), B_d, outByteSize, cudaMemcpyDeviceToHost);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(A_d);
  cudaFree(B_d);
  
  return 0;
}

