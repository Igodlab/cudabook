#include <cuda_runtime.h>

__constant__ int CX;
__constant__ int CY;
__constant__ int CZ;
__constant__ int C0;

__global__ void stencil_kernel(float* U, float* Upp, unsigned int N) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    Upp[i + j*N + k*N*N] = C0 * U[i + j*N + k*N*N]
                         + CX * ( U[(i+1) + j*N + k*N*N]
                                  - 2 * U[i + j*N + k*N*N]
                                  + U[(i-1) + j*N + k*N*N])
                         + CY * ( U[i + (j+1)*N + k*N*N] 
                                  - 2 * U[i + j*N + k*N*N]
                                  + U[i + (j-1)*N + k*N*N])
                         + CZ * ( U[i + j*N + (k+1)*N*N]
                                  - 2 * U[i + j*N + k*N*N]
                                  + U[i + j*N + (k-1)*N*N]);
  }
}

int main(void) {
  /* define spatial values */
  int N = 16;

  /* CUDA prep */
  

  return 0;
}
