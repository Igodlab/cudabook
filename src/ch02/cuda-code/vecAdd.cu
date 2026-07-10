#include <cuda_runtime.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
  int i = blockIdx * blockDim.x + threadIdx;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
  int size = n * sizeof(float);

  // Initialize variables to be copied to GPU device
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **) &A_d, (size_t) size);
  cudaMalloc((void **) &B_d, (size_t) size);
  cudaMalloc((void **) &C_d, (size_t) size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Kernel invocation
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  // Copy result from device to host
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  // Free variables from device
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
