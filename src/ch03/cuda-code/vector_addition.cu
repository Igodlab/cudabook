#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include "utils.hpp"

// Kernel function for adding arrays
__global__
void vecAddKernel(float *A, float *B, float *C, int n){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// Compute vector addition sum C_h = A_h + B_h
void vecAdd_d(float* A, float* B, float* C, int n){
  // Part 1 - allocate memory for A, B, C
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_gpu;
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_gpu, size);

  // Copy arrays from Host to Device: A, C, B to global (device) memory
  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_gpu, n);

  // Copy result from device to host
  cudaMemcpy(C, C_gpu, size, cudaMemcpyDeviceToHost);

  // Free global memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_gpu);
}

// Compute vector addition sum in CPU only C_cpu = A_h + B_h
void vecAdd_h(float* A_h, float* B_h, float* C_cpu, int n){
  for (int i = 0; i < n; i++) {
    C_cpu[i] = A_h[i] + B_h[i];
  }
}

int main(void) {
  // Generate two random arrays A and B
  const int n = 1000;
  const float u_min = -100.0f;
  const float u_max = 100.0f;

  // Initialize inputs, ouputs and populate A, B w/ random values from uniform dist
  std::vector<float> A(n), B(n), C_cpu(n), C_gpu(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  for (int ni = 0; ni < n; ni++) {
    A[ni] = uniform_dist(gen);
    B[ni] = uniform_dist(gen);
  }

  int cap = 10; // Cap number of vector elements printed to screen
  // CPU
  auto start_h = std::chrono::high_resolution_clock::now();
  vecAdd_h(A.data(), B.data(), C_cpu.data(), n);
  auto end_h = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_h = end_h - start_h;
  printf("CPU:\n----\nC_cpu = "); printVec(C_cpu.data(), n, cap);
  printf("vecAdd_h took %.4f ms\n\n", elapsed_h.count());

  // GPU
  cudaEvent_t start_d, stop_d;
  cudaEventCreate(&start_d);
  cudaEventCreate(&stop_d);

  cudaEventRecord(start_d);
  vecAdd_d(A.data(), B.data(), C_gpu.data(), n);
  cudaEventRecord(stop_d);
  cudaEventSynchronize(stop_d);

  float elapsed_d = 0;
  cudaEventElapsedTime(&elapsed_d, start_d, stop_d);
  printf("GPU:\n----\nC_gpu = "); printVec(C_gpu.data(), n, cap);
  printf("vecAdd_d took %.4f ms\n", elapsed_d);

  cudaEventDestroy(start_d);
  cudaEventDestroy(stop_d);

  return 0;
}
