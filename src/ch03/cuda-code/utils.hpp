#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

cv::Mat loadImage(const std::string& path, int readMode = cv::IMREAD_COLOR);

void saveImage(const std::string& outputPath, const cv::Mat& image);

void printVec(const float* V, int n, std::optional<int> cap = std::nullopt);

// void printMatrix(const std::vector<std::vector<float>>& M, const std::string& name = "");
