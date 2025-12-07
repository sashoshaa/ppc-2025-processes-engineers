#include "sosnina_a_matrix_mult_horizontal/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace sosnina_a_matrix_mult_horizontal {

SosninaAMatrixMultHorizontalSEQ::SosninaAMatrixMultHorizontalSEQ(InTypeTriple in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // Инициализация вывода как пустой матрицы
  GetOutput() = std::vector<std::vector<double>>();
}

bool SosninaAMatrixMultHorizontalSEQ::ValidationImpl() {
  const auto &matrixA = input_.first;
  const auto &matrixB = input_.second;

  // Проверка: количество столбцов матрицы A должно равняться количеству строк матрицы B
  if (matrixA.empty() || matrixB.empty()) {
    return false;
  }
  return matrixA[0].size() == matrixB.size();
}

bool SosninaAMatrixMultHorizontalSEQ::PreProcessingImpl() {
  // Очищаем вывод (если нужно)
  GetOutput() = std::vector<std::vector<double>>();
  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::RunImpl() {
  const auto &matrixA = input_.first;
  const auto &matrixB = input_.second;

  size_t rowsA = matrixA.size();
  size_t colsA = matrixA[0].size();
  size_t colsB = matrixB[0].size();

  // Инициализация результирующей матрицы ПРЯМО в GetOutput()
  auto &output = GetOutput();
  output.resize(rowsA, std::vector<double>(colsB, 0.0));

  // Умножение матриц
  for (size_t i = 0; i < rowsA; i++) {
    for (size_t j = 0; j < colsB; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < colsA; k++) {
        sum += matrixA[i][k] * matrixB[k][j];
      }
      output[i][j] = sum;
    }
  }

  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
