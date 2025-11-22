#include "sosnina_a_matrix_mult_horizontal/seq/include/ops_seq.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>
#include <utility>

namespace sosnina_a_matrix_mult_horizontal {

SosninaAMatrixMultHorizontalSEQ::SosninaAMatrixMultHorizontalSEQ(InTypeTriple in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // Инициализация вывода как пустой матрицы
  GetOutput() = std::vector<std::vector<double>>();
}

bool SosninaAMatrixMultHorizontalSEQ::ValidationImpl() {
  const auto& matrixA = input_.first;
  const auto& matrixB = input_.second;
  
  // Проверка: количество столбцов матрицы A должно равняться количеству строк матрицы B
  if (matrixA.empty() || matrixB.empty()) return false;
  return matrixA[0].size() == matrixB.size();
}

bool SosninaAMatrixMultHorizontalSEQ::PreProcessingImpl() {
  result_matrix_.clear();
  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::RunImpl() {
  const auto& matrixA = input_.first;
  const auto& matrixB = input_.second;
  
  size_t rowsA = matrixA.size();
  size_t colsA = matrixA[0].size();
  size_t colsB = matrixB[0].size();
  
  // Инициализация результирующей матрицы
  result_matrix_.resize(rowsA, std::vector<double>(colsB, 0.0));
  
  // Умножение матриц
  for (size_t i = 0; i < rowsA; i++) {
    for (size_t j = 0; j < colsB; j++) {
      for (size_t k = 0; k < colsA; k++) {
        result_matrix_[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
  }
  
  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::PostProcessingImpl() {
  GetOutput() = result_matrix_;
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
