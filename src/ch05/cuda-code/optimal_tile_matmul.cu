#include <cuda_runtime.h>

#include "utils.hpp"

const unsigned tileDim = 32;

__global__ void tileMatmulKernel(
    float *M,
    float *N,
    float *P,
    int d0,
    int _d,
    int d1,
    int tile)
{
  extern __shared__ float Tld[];

  float *Mds = (float *) Tld;
  float *Nds = (float *) Tld + tile * tile;

  int tx = threadIdx.x; int bx = blockIdx.x;
  int ty = threadIdx.y; int by = blockIdx.y;

  int x = tx + bx * tile;
  int y = ty + by * tile;

  float acc = 0.0f;
  for (int h = 0; h < (_d + tile - 1)/tile; ++h) {
    if (y < d1 && (tx + h*tile) < _d) {
      Mds[tx + ty*tile] = M[tx + h*tile + y*_d];
    } else Mds[tx + ty*tile] = 0.0f;

    if (x < d0 && (ty + h*tile) < _d) {
      Nds[tx + ty*tile] = N[x + (ty + h*tile)*d0];
    } else Nds[tx + ty*tile] = 0.0f;
    __syncthreads();

    for (int k = 0; k < tile; ++k) {
      acc += Mds[k + ty*tile] * Nds[tx + k*tile];
    }
    __syncthreads();
  }

  if (x < d0 && y < d1) {
    P[x + y*d0] = acc;
  }
}

int main(void) {
  /* P = MN matmul
   * N \in _d x d0
   * M \in d1 x _d
   * P \in d1 x d0
   */
  int d0 = 4000;
  int _d = 3000;
  int d1 = 5000;

  /* read csv input matrices */
  const std::string data_dir = "data/ch05";
  
  std::vector<float> M = read_matrix_csv("inputM.csv", d1, _d, data_dir);
  std::vector<float> N = read_matrix_csv("inputN.csv", _d, d0, data_dir);
  std::vector<float> P(d1 * d0);

  /* GPU */
  float *M_d, *N_d, *P_d;
  size_t Psz = d1 * d0 * sizeof(float);
  size_t Nsz = _d * d0 * sizeof(float);
  size_t Msz = d1 * _d * sizeof(float);

  cudaMalloc(&P_d, Psz);
  cudaMalloc(&N_d, Nsz);
  cudaMalloc(&M_d, Msz);

  cudaMemcpy(N_d, N.data(), Nsz, cudaMemcpyHostToDevice);
  cudaMemcpy(M_d, M.data(), Msz, cudaMemcpyHostToDevice);

  dim3 dg((d0 + tileDim - 1)/tileDim, (d1 + tileDim - 1)/tileDim, 1);
  dim3 db(tileDim, tileDim, 1);
  size_t sharedMemBytes = 2 * tileDim * tileDim * sizeof(float);
  tileMatmulKernel<<<dg, db, sharedMemBytes>>>(M_d, N_d, P_d, d0, _d, d1, tileDim);

  cudaMemcpy(P.data(), P_d, Psz, cudaMemcpyDeviceToHost);
  save_matrix_csv("outputP2.csv", P, d1, d0);

  cudaFree(P_d);
  cudaFree(N_d);
  cudaFree(M_d);

  return 0;
}
