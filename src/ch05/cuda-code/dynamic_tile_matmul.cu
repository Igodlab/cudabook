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
    int d_,
    int d1)
{
  extern __shared__ float Tld[];

  float *Mds = (float *) Tld;
  float *Nds = (float *) Tld + (TILE * TILE); /* Is valid for pointer arithmetic offset operands 
                                               * which basically is handled by the compiler as
                                               * Nds = address_of_Tld + (TILE * TILE * sizeof(float))
                                               */

  /* relative indexes ty, tx (wrt blocks) 
   * absolute indexes y, x 
   */
  int tx = threadIdx.x; int bx = blockIdx.x;
  int ty = threadIdx.y; int by = blockIdx.y;

  int x = tx +  bx * TILE;
  int y = ty +  by * TILE;

  /* Iterate for each tile phase */
  float acc = 0.0f;
  for (int h = 0; h < ceil(d_/(float)TILE); ++h) {
    /* load to Mds, Nds tile w/ boundary checks */
    if (y < d1 && (h*TILE + tx) < d_) {
      Mds[tx + ty*TILE] = M[tx + h*TILE + y*d_];
    } else Mds[tx + ty*TILE] = 0.0f;

    if ((h*TILE + ty) < d_ && x < d0) {
      Nds[tx + ty*TILE] = N[x + (ty + h*TILE)*d0];
    } else Nds[tx + ty*TILE] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += Mds[k + ty*TILE] * Nds[tx + k*TILE];
    }
    __syncthreads();
  }

  /* Boundary check for output matrix */
  if (x < d0 && y < d1) {
    P[x + y*d0] = acc;
  }
}

int main(void) {
  /* P = MN matmul
   * N \in d_ x d0
   * M \in d1 x d_
   * P \in d1 x d0
   */
  int d0 = 4000;
  int d_ = 3000;
  int d1 = 5000;

  /* random fill input matrices */
  std::vector<float> P(d1 * d0);
  std::vector<float> M(d1 * d_);
  std::vector<float> N(d_ * d0);

  float u_min = -10.0f;
  float u_max = 10.0f;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  /* random fill M */
  for (int _i = 0; _i < d_; ++_i) {
    for (int i1 = 0; i1 < d1; ++i1) {
      M[_i + i1*d_] = uniform_dist(rng);
    }
  }
  /* random fill N */
  for (int i0 = 0; i0 < d0; ++i0) {
    for (int _i = 0; _i < d_; ++_i) {
      N[i0 + _i*d0] = uniform_dist(rng);
    }
  }
  save_matrix_csv("inputM.csv", M, d1, d_);
  save_matrix_csv("inputN.csv", N, d_, d0);

  /* CUDA prep */
  float *Md_, *Nd_, *Pd_;
  size_t Psz = d1 * d0 * sizeof(float);
  size_t Nsz = d_ * d0 * sizeof(float);
  size_t Msz = d1 * d_ * sizeof(float);
  cudaMalloc(&Pd_, Psz);
  cudaMalloc(&Nd_, Nsz);
  cudaMalloc(&Md_, Msz);

  cudaMemcpy(Nd_, N.data(), Nsz, cudaMemcpyHostToDevice);
  cudaMemcpy(Md_, M.data(), Msz, cudaMemcpyHostToDevice);

  dim3 dg((d0 + TILE - 1)/TILE, (d1 + TILE - 1)/TILE, 1);
  dim3 db(TILE, TILE, 1);
  size_t sharedMemBytes = 2 * TILE * TILE * sizeof(float);
  matmulKernel<<<dg, db, sharedMemBytes>>>(Md_, Nd_, Pd_, d0, d_, d1);

  cudaMemcpy(P.data(), Pd_, Psz, cudaMemcpyDeviceToHost);
  save_matrix_csv("outputP.csv", P, d1, d0);

  cudaFree(Md_);
  cudaFree(Nd_);
  cudaFree(Pd_);

  return 0;
}
