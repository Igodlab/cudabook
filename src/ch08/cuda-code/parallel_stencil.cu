#include <cuda_runtime.h>

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
  unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[k + j*N + i*N*N] = c0 * in[k + j*N + i*N*N]
                         + c1 * in[(k-1) + j*N + i*N*N]
                         + c2 * in[(k+1) + j*N + i*N*N]
                         + c3 * in[k + (j-1)*N + i*N*N]
                         + c4 * in[k + (j+1)*N + i*N*N]
                         + c5 * in[k + j*N + (i-1)*N*N]
                         + c6 * in[k + j*N + (i+1)*N*N];
  }
}
