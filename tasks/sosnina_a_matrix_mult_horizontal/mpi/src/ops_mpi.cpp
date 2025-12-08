#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

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
  int rank = 0, size = 1;
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
  int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

  if (rank_ == 0) {
    rowsA = static_cast<int>(matrix_A_.size());
    colsA = rowsA > 0 ? static_cast<int>(matrix_A_[0].size()) : 0;
    rowsB = static_cast<int>(matrix_B_.size());
    colsB = rowsB > 0 ? static_cast<int>(matrix_B_[0].size()) : 0;
  }

  // 2. Рассылаем размеры всем процессам
  int sizes[4] = {rowsA, colsA, rowsB, colsB};
  MPI_Bcast(sizes, 4, MPI_INT, 0, MPI_COMM_WORLD);

  rowsA = sizes[0];
  colsA = sizes[1];
  rowsB = sizes[2];
  colsB = sizes[3];

  // 3. Проверка совместимости
  if (colsA != rowsB) {
    // Все процессы возвращают пустую матрицу
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 4. Рассылаем матрицу B всем процессам
  std::vector<double> B_flat(rowsB * colsB);
  if (rank_ == 0) {
    for (int i = 0; i < rowsB; i++) {
      for (int j = 0; j < colsB; j++) {
        B_flat[i * colsB + j] = matrix_B_[i][j];
      }
    }
  }
  MPI_Bcast(B_flat.data(), rowsB * colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 5. Распределяем матрицу A по строкам (горизонтальная схема)
  int rows_per_process = rowsA / world_size_;
  int rows_remainder = rowsA % world_size_;

  // Количество строк для текущего процесса
  int local_rows = (rank_ < rows_remainder) ? rows_per_process + 1 : rows_per_process;

  // Начальный индекс строк для текущего процесса
  int start_row = 0;
  for (int p = 0; p < rank_; p++) {
    start_row += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
  }

  // 6. Получаем локальную часть матрицы A
  std::vector<std::vector<double>> local_A(local_rows, std::vector<double>(colsA));

  if (rank_ == 0) {
    // Процесс 0 копирует свою часть
    for (int i = 0; i < local_rows; i++) {
      local_A[i] = matrix_A_[start_row + i];
    }

    // Отправляем части другим процессам
    for (int proc = 1; proc < world_size_; proc++) {
      int proc_rows = (proc < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      int proc_start = 0;
      for (int p = 0; p < proc; p++) {
        proc_start += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      }

      // Отправляем строки
      for (int i = 0; i < proc_rows; i++) {
        MPI_Send(matrix_A_[proc_start + i].data(), colsA, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // Получаем строки матрицы A от процесса 0
    for (int i = 0; i < local_rows; i++) {
      local_A[i].resize(colsA);
      MPI_Recv(local_A[i].data(), colsA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // 7. Локальное умножение (используем B_flat напрямую)
  std::vector<std::vector<double>> local_result(local_rows, std::vector<double>(colsB, 0.0));

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < colsB; j++) {
      double sum = 0.0;
      for (int k = 0; k < colsA; k++) {
        sum += local_A[i][k] * B_flat[k * colsB + j];
      }
      local_result[i][j] = sum;
    }
  }

  // 8. Только процесс 0 собирает полный результат
  if (rank_ == 0) {
    // Создаем результирующую матрицу
    std::vector<std::vector<double>> final_result(rowsA, std::vector<double>(colsB, 0.0));

    // Копируем свою часть
    for (int i = 0; i < local_rows; i++) {
      final_result[start_row + i] = local_result[i];
    }

    // Получаем от других процессов
    for (int proc = 1; proc < world_size_; proc++) {
      int proc_rows = (proc < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      int proc_start = 0;
      for (int p = 0; p < proc; p++) {
        proc_start += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      }

      for (int i = 0; i < proc_rows; i++) {
        MPI_Recv(final_result[proc_start + i].data(), colsB, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    // Процесс 0 устанавливает результат
    GetOutput() = final_result;
  } else {
    // Отправляем результаты процессу 0
    for (int i = 0; i < local_rows; i++) {
      MPI_Send(local_result[i].data(), colsB, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    // Процессы не-0 устанавливают пустой результат
    GetOutput() = std::vector<std::vector<double>>();
  }

  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
