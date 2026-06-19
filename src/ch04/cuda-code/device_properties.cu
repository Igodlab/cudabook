#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
  int devCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&devCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  printf("Detected %d CUDA capable device(s)\n", devCount);

  cudaDeviceProp devProp;
  for (unsigned int i = 0; i < devCount; ++i) {
    cudaGetDeviceProperties(&devProp, i);

    printf("\nDevice %d: \"%s\"\n", i, devProp.name);
    printf("  Major revision number:         %d\n", devProp.major);
    printf("  Minor revision number:         %d\n", devProp.minor);
    printf("  Total amount of global memory: %.2f GB\n", (float)devProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("  Total amount of constant memory: %lu bytes\n", devProp.totalConstMem);
    printf("  Total amount of shared memory per SM: %lu bytes\n", devProp.sharedMemPerBlock);
    printf("  Total number of registers available per SM: %d\n", devProp.regsPerBlock);
    printf("  Warp size:                     %d\n", devProp.warpSize);
    printf("  Maximum number of threads per block: %d\n", devProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n", devProp.maxThreadsDim[0],
        devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n", devProp.maxGridSize[0],
        devProp.maxGridSize[1], devProp.maxGridSize[2]);

    /* Clock rate */
    int clockRateKHz;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, i);
    float clockRateGHz = (float)clockRateKHz * 1e-6f ;
    printf("  Clock rate:                    %.2f GHz\n", clockRateGHz);

    /* Memory clock rate */
    int memoryClockRateKHz;
    cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, i);
    float memoryClockRateMHz = (float)memoryClockRateKHz * 1e-3f;
    printf("  Memory clock rate:             %f MHz\n", memoryClockRateMHz);

    printf("  Memory bus width:              %d-bit\n", devProp.memoryBusWidth);
    printf("  L2 cache size:                 %d bytes\n", devProp.l2CacheSize);
  }

  return 0;
}
