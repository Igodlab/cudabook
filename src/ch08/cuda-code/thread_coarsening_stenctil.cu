#include <cuda_runtime.h>

__constant__ unsigned int IN_TILE = 6;
__constant__ unsigned int OUT_TILE = 4;

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
  int kStart = blockIdx.z*OUT_TILE;
  int j = blockIdx.y*OUT_TILE + threadIdx.y - 1;
  int i = blockIdx.x*OUT_TILE + threadIdx.x - 1;

  __shared__ float inPrev_s[IN_TILE][IN_TILE];
  __shared__ float inCurr_s[IN_TILE][IN_TILE];
  __shared__ float inNext_s[IN_TILE][IN_TILE];

  if (kStart-1 >= 0 && kStart-1 < N && j >= 0 && j < N && i >= 0 && i < N) {
    inPrev_s[threadIdx.y][threadIdx.x] = in[i + j*N + (kStart - 1)*N*N];
  }
  if (kStart >= 0 && kStart < N && j >= 0 && j < N && i >= 0 && i < N) {
    inCurr_s[threadIdx.y][threadIdx.x] = in[i + j*N + kStart*N*N];
  }
  for (int i kStart; k < kStart + OUT_TILE; ++k) {
    if (k+1 >= 0 && k+1 < N && j >= 0 && j < N && i >= 0 && i < N) {
      inNext_s[threadIdx.y][threadIdx.x] = in[i + j*N + (k + 1)*N*N];
    }
  }
  __syncthreads();

  if (k >= 1 && k < N-1 && j >= 1 && j < N-1 && i >= 1 && k < N-1) {
    if (threadIdx.y >= 1 && threadIdx.y < IN_TILE-1
        && threadIdx.x >= 1 && threadIdx.x < IN_TILE-1) {
      out[i + j*N + k*N*N] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                           + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                           + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                           + c3*inCurr_s[threadIdx.y-1][threadIdx.x]
                           + c4*inCurr_s[threadIdx.y+1][threadIdx.x]
                           + c5*inPrev_s[threadIdx.y][threadIdx.x]
                           + c6*inNext_s[threadIdx.y][threadIdx.x];
    }
  }
  __syncthreads();

  inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
  inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
}
