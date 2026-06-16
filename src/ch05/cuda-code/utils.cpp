#include "utils.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

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
        for (int i = 0; i < totalRows; ++i)
            printRow(i);
    } else {
        for (int i = 0; i < *cap; ++i)
            printRow(i);
        std::cout << "   ⋮\n";
        printRow(totalRows - 1);
    }
}

void printMatrixFlat(const std::vector<float>& M, int rows, int cols, const std::string& name, std::optional<int> cap) {
    assert(!cap || (*cap <= rows && *cap <= cols));

    if (!name.empty())
        std::cout << name << " [" << rows << "x" << cols << "]:\n";

    auto printRow = [&](int i) {
        int printCols = cap.value_or(cols);
        for (int j = 0; j < printCols; j++)
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed << M[i*cols + j] << " ";
        if (cap) {
            std::cout << "   ...";
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed << M[i*cols + cols - 1];
        }
        std::cout << "\n";
    };

    if (!cap) {
        for (int i = 0; i < rows; i++)
            printRow(i);
    } else {
        for (int i = 0; i < *cap; i++)
            printRow(i);
        std::cout << "   ⋮\n";
        printRow(rows - 1);
    }
}

namespace fs = std::filesystem;
void save_matrix_csv(const std::string& filename,
                     const std::vector<float>& data,
                     int rows,
                     int cols,
                     const std::string& out_dir,
                     int precision)
{
    if (static_cast<int>(data.size()) != rows * cols)
        throw std::runtime_error(
            "save_matrix_csv: data.size() (" + std::to_string(data.size()) +
            ") != rows*cols (" + std::to_string(rows * cols) + ")");
 
    // create out/ (and any parents) if they don't exist yet
    fs::create_directories(out_dir);
 
    const std::string path = out_dir + "/" + filename;
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("save_matrix_csv: cannot open file: " + path);
 
    f << std::fixed << std::setprecision(precision);
 
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            f << data[r * cols + c];
            if (c + 1 < cols) f << ',';
        }
        f << '\n';
    }
}
