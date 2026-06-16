#include <cuda_runtime.h>
#include <random>
#include <vector>
#include "utils.hpp"

#define TILE 32

__global__ void matmulKernel(
    float* M,
    float* N,
    float* P,
    int d0,
    int _d,
    int d1)
{
  __shared__ float Msd[TILE * TILE];
  __shared__ float Nsd[TILE * TILE];

  /* relative indexes ty, tx (wrt blocks) 
   * absolute indexes y, x 
   */
  int tx = threadIdx.x; int bx = blockIdx.x;
  int ty = threadIdx.y; int by = blockIdx.y;

  int x = tx +  bx * blockDim.x;
  int y = ty +  by * blockDim.y;

  /* Iterate for each tile phase */
  float acc = 0.0f;
  for (int h = 0; h < ceil(_d/(float)TILE); ++h) {
    /* load to Msd, Nsd tile w/ boundary checks */
    if (y < d1 && (h*TILE + tx) < _d) {
      Msd[tx + ty*TILE] = M[tx + h*TILE + y*d0];
    } else Msd[tx + ty*TILE] = 0.0f;

    if ((h*TILE + ty) < _d && x < d0) {
      Nsd[tx + ty*TILE] = N[x + (ty + h*TILE)*d0];
    } else Nsd[tx + ty*TILE] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += Msd[k + ty*TILE] * Nsd[tx + k*TILE];
    }
    __syncthreads();

    /* Boundary check for output matrix */
    if (x < d0 && y < d1) {
      P[x + y*d0] = acc;
    }
  }
}

int main(void) {
  /* P = MN matmul
   * N \in _d x d0
   * M \in d1 x _d
   * P \in d1 x d0
   */
  int d0 = 1000;
  int _d = 1000;
  int d1 = 1000;

  /* random fill input matrices */
  std::vector<float> P(d1 * d0);
  std::vector<float> M(d1 * _d);
  std::vector<float> N(_d * d0);

  float u_min = -10.0f;
  float u_max = 10.0f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  /* random fill M */
  for (int _i = 0; _i < _d; ++_i) {
    for (int i1 = 0; i1 < d1; ++i1) {
      M[_i + i1*_d] = uniform_dist(rng);
    }
  }
  /* random fill N */
  for (int i0 = 0; i0 < d0; ++i0) {
    for (int _i = 0; _i < _d; ++_i) {
      N[i0 + _i*d0] = uniform_dist(rng);
    }
  }
  save_matrix_csv("inputM.csv", M, d1, _d);
  save_matrix_csv("inputN.csv", N, _d, d0);

  /* CUDA prep */
  float *M_d, *N_d, *P_d;
  cudaMalloc(&P_d, (size_t)(d1 * d0 * sizeof(float)));
  cudaMalloc(&N_d, (size_t)(_d * d0 * sizeof(float)));
  cudaMalloc(&M_d, (size_t)(d1 * _d * sizeof(float)));

  cudaMemcpy(N_d, N.data(), _d*d0*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(M_d, M.data(), d1*_d*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dg(ceil(d0/TILE), ceil(d1/TILE), 1);
  dim3 db(TILE, TILE, 1);
  matmulKernel<<<dg, db>>>(M_d, N_d, P_d, d0, _d, d1);

  cudaMemcpy(P.data(), P_d, d1*_d*sizeof(float), cudaMemcpyDeviceToHost);
  save_matrix_csv("outputP.csv", P, d1, d0);

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}
