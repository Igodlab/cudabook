#pragma once
#include <optional>
#include <string>
#include <vector>

void printVec(const float* V, int n, std::optional<int> cap = std::nullopt);

void printMatrix(const std::vector<std::vector<float>>& M, const std::string& name = "", std::optional<int> cap = std::nullopt);

void printMatrixFlat(const std::vector<float>& M, int rows, int cols, const std::string& name = "", std::optional<int> cap = std::nullopt);

