#include "utils.hpp"
#include <cassert>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <vector>

cv::Mat loadImage(const std::string& path, int readMode) {
    cv::Mat img = cv::imread(path, readMode);
    if (img.empty())
        throw std::runtime_error("Could not load image: " + path);
    if (!img.isContinuous())
        img = img.clone();
    return img;
}

void saveImage(const std::string& outputPath, const cv::Mat& image) {
    if (image.empty())
        throw std::runtime_error("Cannot save empty image to: " + outputPath);
    bool success = cv::imwrite(outputPath, image);
    if (!success)
        throw std::runtime_error("Failed to write image to: " + outputPath);
    std::cout << "Image saved to: " << outputPath << "\n";
}

void printVec(const float* V, int n, std::optional<int> cap) {
    assert(!cap || *cap <= n);
    int printCount = cap.value_or(n);

    printf("Vector [%d]:\n[", n);
    for (int i = 0; i < printCount; i++) {
        printf("%f", V[i]);
        if (i < printCount - 1) printf(", ");
    }
    if (cap) printf(", ...");
    printf("]\n");
}

// void printMatrix(const std::vector<std::vector<float>>& M, const std::string& name = "") {
//     if (!name.empty())
//         std::cout << name << " [" << M.size() << "x" << M[0].size() << "]:\n";
//     for (const auto& row : M) {
//         for (const auto& val : row)
//             std::cout << std::setw(8) << std::setprecision(3) << std::fixed << val << " ";
//         std::cout << "\n";
//     }
// }
