#include "utils.hpp"

const int CHANNELS = 3;

__global__
void coloToGrayscaleConvertion(
    unsigned char *Pout,
    unsigned char *Pin,
    int width,
    int height) 
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // Get 1D offset for the greyscale output
    int grayOffset = row * width + col;

    // Input has 3x more dimensions due to rgb color channels
    int rgbOffset = grayOffset * CHANNELS;

    // Each pixel requires three bytes (one for each color channel)
    // OpenCV reads images as bgr so we need to account for that
    unsigned char b = Pin[rgbOffset];     // blue
    unsigned char g = Pin[rgbOffset + 1]; // green
    unsigned char r = Pin[rgbOffset + 2]; // red

    // Compute greys
    Pout[grayOffset] = 0.299*r + 0.587*g + 0.114*b;
  }
}

int main(void) {
  cv::Mat img = loadImage("images/ch03/opeth-sorceress.png");

  size_t inBytes = img.cols * img.rows * CHANNELS;
  size_t outBytes = img.cols * img.rows;

  // Allocate memory
  unsigned char *Pin_d, *Pout_d;
  cudaMalloc((void **)&Pin_d, inBytes);
  cudaMalloc((void **)&Pout_d, outBytes);

  // Copy data from host to device
  cudaMemcpy(Pin_d, img.data, inBytes, cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 dimGrid(ceil(img.rows/16.0), ceil(img.cols/16.0), 1);
  dim3 dimBlock(16, 16, 1);
  coloToGrayscaleConvertion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, img.cols, img.rows);
  // cudaDeviceSynchronize();

  // Copy result back to a cv::Mat type
  cv::Mat greyMat(img.rows, img.cols, CV_8UC1);
  cudaMemcpy(greyMat.data, Pout_d, outBytes, cudaMemcpyDeviceToHost);

  // Save
  saveImage("images/ch03/opeth-sorceress-grey.png", greyMat);

  // Cleanup
  cudaFree(Pin_d);
  cudaFree(Pout_d);

  return 0;
}

