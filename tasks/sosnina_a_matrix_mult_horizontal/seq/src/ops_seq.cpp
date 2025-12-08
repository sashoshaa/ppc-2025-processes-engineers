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

  // Проверка на пустые матрицы
  if (matrixA.empty() || matrixB.empty()) {
    return false;
  }

  // Проверка, что все строки матрицы A имеют одинаковый размер
  size_t colsA = matrixA[0].size();
  for (size_t i = 1; i < matrixA.size(); i++) {
    if (matrixA[i].size() != colsA) {
      return false;
    }
  }

  // Проверка, что все строки матрицы B имеют одинаковый размер
  size_t colsB = matrixB[0].size();
  for (size_t i = 1; i < matrixB.size(); i++) {
    if (matrixB[i].size() != colsB) {
      return false;
    }
  }
  return colsA == matrixB.size();
}

bool SosninaAMatrixMultHorizontalSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::RunImpl() {
  const auto &matrixA = input_.first;
  const auto &matrixB = input_.second;

  size_t rowsA = matrixA.size();
  size_t colsA = matrixA[0].size();
  size_t rowsB = matrixB.size();
  size_t colsB = matrixB[0].size();

  auto &output = GetOutput();

  output = std::vector<std::vector<double>>(rowsA, std::vector<double>(colsB, 0.0));

  for (size_t i = 0; i < rowsA; i++) {
    const auto &rowA = matrixA[i];
    auto &rowOutput = output[i];

    for (size_t k = 0; k < colsA; k++) {
      double aik = rowA[k];
      const auto &rowB = matrixB[k];

      for (size_t j = 0; j < colsB; j++) {
        rowOutput[j] += aik * rowB[j];
      }
    }
  }

  return true;
}

bool SosninaAMatrixMultHorizontalSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
