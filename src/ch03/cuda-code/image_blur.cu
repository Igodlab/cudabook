#include <cassert>
#include <cuda_runtime.h>
#include <utils.hpp>

const float PI = std::numbers::pi_v<float>;
const int CHANNELS = 3;

__global__ void imageBlur(
    unsigned char* Pin,
    unsigned char* Pout,
    int W,
    int H,
    int blur_radii) 
{
  /* Deep Learning notation 
   * dimensions: H x W x C
   * indexes (slow←fast): (h, w, c)
   */
  int c = threadIdx.z; /* (b,g,r)=(0,1,2)$ */
  int w = blockIdx.x * blockDim.x + threadIdx.x; 
  int h = blockIdx.y * blockDim.y + threadIdx.y; 

  if (w < W && h < H) {
    int acc = 0;
    int pixels = 0; /* count number of pixels for averaging */

    for (int blurRow = -blur_radii; blurRow < blur_radii + 1; ++blurRow) {
      for (int blurCol = -blur_radii; blurCol < blur_radii + 1; ++blurCol) {
        int current_w = w + blurCol;
        int current_h = h + blurRow;

        if (current_w >= 0 && current_w < W && current_h >= 0 && current_h < H) {
          /* row-major indexing: Pin[h*(C*W) + w*C + c] */
          acc += Pin[current_h * (CHANNELS * W) + current_w * CHANNELS + c];
          ++pixels;
        }
      }
    }
    Pout[h * (CHANNELS * W) + w * CHANNELS + c] = (unsigned char)(acc / pixels);
  }
}

/* Neighbor pixels based on blur ratio */
int blurRadii(float br, int width, int height) {
  assert(0.0f <= br && br < 1.0f);
  /* Obtain radius from area: r = sqrt(area / pi) */
  return (int)(std::sqrt(width * height / PI) * br); 
}

int main(void) {
  cv::Mat img = loadImage("images/ch03/opeth-sorceress.png");

  size_t inBytes = img.cols * img.rows * CHANNELS;
  size_t outBytes = img.cols * img.rows * CHANNELS;

  unsigned char *Pin_d, *Pout_d;
  cudaMalloc(&Pin_d, inBytes);
  cudaMalloc(&Pout_d, outBytes);

  /* Copy data from host to device */
  cudaMemcpy(Pin_d, img.data, inBytes, cudaMemcpyHostToDevice);

  /* blur ratio */
  float br = 0.05;
  const int BLUR_RADII = (int)(floor(blurRadii(br, img.cols, img.rows)/2.0));

  dim3 dimGrid(ceil(img.rows/16.0), ceil(img.cols/16.0), 1);
  dim3 dimBlock(16, 16, 3);
  imageBlur<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, img.cols, img.rows, BLUR_RADII);

  /* Copy result back to a cv::Mat type */
  cv::Mat bluredMat(img.rows, img.cols, CV_8UC3);
  cudaMemcpy(bluredMat.data, Pout_d, outBytes, cudaMemcpyDeviceToHost);

  /* Save */
  saveImage("images/ch03/opeth-sorceress-blur.png", bluredMat);

  /* Cleanup */
  cudaFree(Pin_d);
  cudaFree(Pout_d);

  return 0;
}
