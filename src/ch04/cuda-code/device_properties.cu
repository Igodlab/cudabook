#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  printf("Detected %d CUDA capable device(s)\n", deviceCount);

  for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    printf("\nDevice %d: \"%s\"\n", deviceId, deviceProp.name);
    printf("  Major revision number:         %d\n", deviceProp.major);
    printf("  Minor revision number:         %d\n", deviceProp.minor);
    printf("  Total amount of global memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Number of multiprocessors:     %d\n", deviceProp.multiProcessorCount);
    printf("  Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Warp size:                     %d\n", deviceProp.warpSize);
    printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    /* Clock rate */
    int clockRateKHz;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, deviceId);
    float clockRateGHz = (float)clockRateKHz * 1e-6f ;
    printf("  Clock rate:                    %.2f GHz\n", clockRateGHz);

    /* Memory clock rate */
    int memoryClockRateKHz;
    cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, deviceId);
    float memoryClockRateMHz = (float)memoryClockRateKHz * 1e-3f;
    printf("  Memory clock rate:             %f MHz\n", memoryClockRateMHz);

    printf("  Memory bus width:              %d-bit\n", deviceProp.memoryBusWidth);
    printf("  L2 cache size:                 %d bytes\n", deviceProp.l2CacheSize);
  }

  return 0;
}
