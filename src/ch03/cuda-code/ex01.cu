#include <cuda_runtime.h>
#include <random>
#include <tuple>
#include <vector>

__global__ matmulKernelA(
    float* M,
    float* N,
    float* A,
    std::tuple<int, int> Mdim,
    std::tuple<int, int> Ndim) 
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

}


int main(void) {
  // generate input matrices
  std::tuple<int, int> Mdim(3, 4);
  std::tuple<int, int> Ndim(4, 5);

  std::vector<std::vector<float>> M(std::get<0>(Mdim), std::get<1>(Mdim));
  std::vector<std::vector<float>> N(std::get<0>(Ndim), std::get<1>(Ndim));
  std::vector<std::vector<float>> A(std::get<0>(Mdim), std::get<1>(Ndim));
  std::vector<std::vector<float>> B(std::get<0>(Mdim), std::get<1>(Ndim));

  float u_min = -100.0f;
  float u_max = 100.0f;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<float> uniform_dist(u_min, u_max);

  // populate M, N

}
