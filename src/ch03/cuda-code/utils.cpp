#include "utils.hpp"
#include <cassert>
#include <iomanip>
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

void printMatrix(const std::vector<std::vector<float>>& M, const std::string& name, std::optional<int> cap) {
    int totalRows = M.size();
    int totalCols = M[0].size();

    assert(!cap || (*cap <= totalRows && *cap <= totalCols));

    if (!name.empty())
        std::cout << name << " [" << totalRows << "x" << totalCols << "]:\n";

    auto printRow = [&](int i) {
        int cols = cap.value_or(totalCols);
        for (int j = 0; j < cols; j++)
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed << M[i][j] << " ";
        if (cap) {
            std::cout << "   ...";
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed << M[i][totalCols - 1];
        }
        std::cout << "\n";
    };

    if (!cap) {
        for (int i = 0; i < totalRows; i++)
            printRow(i);
    } else {
        for (int i = 0; i < *cap; i++)
            printRow(i);
        std::cout << "   ⋮\n";
        printRow(totalRows - 1);
    }
}
