#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

/* Returns a struct with execution configuration params */
int getOptimalTileDim(int d1, int d_, int d0) {
  int devCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&devCount);

  if (error_id != cudaSuccess) {
    printf("`cudaDeviceCount` returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }
  printf("Detected %d CUDA capable device(s)", devCount);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);

  int maxThreadsBlock = devProp.maxThreadsPerBlock;
  int maxRegistersSM = devProp.regsPerBlock;
  dim3 maxThreadsDim = (
      devProp.maxThreadsDim[0], /* call as `.x` */
      devProp.maxThreadsDim[1], /* call as `.y` */
      devProp.maxThreadsDim[2]  /* call as `.z` */
      );
  int sharedMemSM = devProp.sharedMemPerBlock;

  /* Compute max dimensions of square tile */

  // 1. Based on max threads per block (square tiles)
  int tileDim = static_cast<int>(sqrt(maxThreadsBlock));

  // 2. Based on max block dimensions
  tileDim = min(tileDim, min(maxThreadsDim.x, maxThreadsDim.y));

  // 3. Based on shared memory (we need 2 tiles worth of shared memory)
  int maxTileWidthBySharedMem = static_cast<int>(sqrt(sharedMemSM / (2 * sizeof(float))));
  tileDim = min(tileDim, maxTileWidthBySharedMem);

  // 4. Based on matrix dimensions (no point in having tiles larger than matrices)
  tileDim = min(tileDim, min(d1, min(d_, d0)));

  // 5. Round down to nearest power of 2 for better memory alignment
  tileDim = 1 << static_cast<int>(log2(tileDim));

  // 6. Ensure minimum practical size
  tileDim = max(16, tileDim);  // minimum tile size of 16

  // Print diagnostic information
  std::cout << "Calculated optimal tile width: " << tileDim << std::endl;
  std::cout << "Based on:" << std::endl;
  std::cout << "- Max threads per block: " << maxThreadsBlock << std::endl;
  std::cout << "- Max block dimensions: " << maxThreadsDim.x << "x" << maxThreadsDim.y << std::endl;
  std::cout << "- Shared memory per block: " << sharedMemSM << " bytes" << std::endl;
  std::cout << "- Matrix dimensions: " << 1 << "x" << d_ << "x" << d0 << std::endl;

  return max(
      1 << static_cast<int>(log2(tileDim)),
      16);
}
