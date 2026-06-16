#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

cv::Mat loadImage(const std::string& path, int readMode = cv::IMREAD_COLOR);

void saveImage(const std::string& outputPath, const cv::Mat& image);

void printVec(const float* V, int n, std::optional<int> cap = std::nullopt);

void printMatrix(const std::vector<std::vector<float>>& M, const std::string& name = "", std::optional<int> cap = std::nullopt);

/**
 * Print a row-major matrix.
 *
 * @param Matrix Matrix variable only, do not include `.data()` 
 * @param rows   Number of rows
 * @param cols   Number of columns
 * @param name   Matrix name for printing
 * @param cap    Cap number of elements being print (default None)
 */
void printMatrixFlat(const std::vector<float>& M, int rows, int cols, const std::string& name = "", std::optional<int> cap = std::nullopt);
