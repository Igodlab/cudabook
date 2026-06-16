#pragma once
#include <optional>
#include <string>
#include <vector>

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


/**
 * Save a row-major matrix to CSV.
 *
 * @param filename  File name only, no directory (e.g. "M.csv")
 * @param data      Flat vector in row-major order: element (r,c) at data[r*cols+c]
 * @param rows      Number of rows
 * @param cols      Number of columns
 * @param out_dir   Output directory relative to root project (default "data/chX/")
 * @param precision Decimal places (default 6)
 */
void save_matrix_csv(const std::string& filename,
                     const std::vector<float>& data,
                     int rows,
                     int cols,
                     const std::string& out_dir = "data/ch05/",
                     int precision = 6);
