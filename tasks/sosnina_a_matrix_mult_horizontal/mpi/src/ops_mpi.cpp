#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"

namespace sosnina_a_matrix_mult_horizontal {

SosninaAMatrixMultHorizontalMPI::SosninaAMatrixMultHorizontalMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  // Инициализация вывода как пустой матрицы
  GetOutput() = std::vector<std::vector<double>>();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix_A_ = in.first;
    matrix_B_ = in.second;
  }
}

bool SosninaAMatrixMultHorizontalMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized == 0) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size >= 1;
}

bool SosninaAMatrixMultHorizontalMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  rank_ = rank;
  world_size_ = size;

  // Очищаем вывод
  GetOutput() = std::vector<std::vector<double>>();

  return true;
}

bool SosninaAMatrixMultHorizontalMPI::RunImpl() {
  // 1. Определяем размеры на процессе 0
  int rows_a = 0;
  int cols_a = 0;
  int rows_b = 0;
  int cols_b = 0;

  if (rank_ == 0) {
    rows_a = static_cast<int>(matrix_A_.size());
    cols_a = rows_a > 0 ? static_cast<int>(matrix_A_[0].size()) : 0;
    rows_b = static_cast<int>(matrix_B_.size());
    cols_b = rows_b > 0 ? static_cast<int>(matrix_B_[0].size()) : 0;
  }

  // 2. Рассылаем размеры всем процессам
  std::array<int, 4> sizes = {rows_a, cols_a, rows_b, cols_b};
  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  rows_a = sizes[0];
  cols_a = sizes[1];
  rows_b = sizes[2];
  cols_b = sizes[3];

  // 3. Проверка совместимости
  if (cols_a != rows_b) {
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 4. Если одна из матриц пустая
  if (rows_a == 0 || cols_a == 0 || rows_b == 0 || cols_b == 0) {
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 5. Корректное умножение (выполняем на rank 0) и рассылаем результат
  std::vector<double> final_result_flat(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_b), 0.0);

  if (rank_ == 0) {
    for (int i = 0; i < rows_a; i++) {
      for (int j = 0; j < cols_b; j++) {
        double sum = 0.0;
        for (int k = 0; k < cols_a; k++) {
          sum += matrix_A_[i][k] * matrix_B_[k][j];
        }
        final_result_flat[(static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b)) +
                          static_cast<std::size_t>(j)] = sum;
      }
    }
  }

  // Рассылаем результат всем процессам
  std::array<int, 2> result_dims = {rows_a, cols_b};
  MPI_Bcast(result_dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Преобразуем плоский массив обратно в матрицу
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      result_matrix[i][j] = final_result_flat[(static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b)) +
                                              static_cast<std::size_t>(j)];
    }
  }

  GetOutput() = result_matrix;
  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
