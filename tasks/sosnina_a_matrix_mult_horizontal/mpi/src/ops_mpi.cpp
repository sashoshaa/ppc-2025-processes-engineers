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
    if (rank_ == 0) {
      GetOutput() = std::vector<std::vector<double>>();
    } else {
      GetOutput() = std::vector<std::vector<double>>();
    }
    return true;
  }

  // 4. Рассылаем матрицу B всем процессам ОДНИМ Bcast
  std::vector<double> B_linear;
  if (rank_ == 0) {
    // Процесс 0 готовит линейный массив
    B_linear.reserve(rowsB * colsB);
    for (int i = 0; i < rowsB; i++) {
      B_linear.insert(B_linear.end(), matrix_B_[i].begin(), matrix_B_[i].end());
    }
  } else {
    // Другие процессы резервируют память
    B_linear.resize(rowsB * colsB);
  }

  MPI_Bcast(B_linear.data(), rowsB * colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Преобразуем линейный массив обратно в матрицу для удобства умножения
  std::vector<std::vector<double>> local_B(rowsB, std::vector<double>(colsB));
  for (int i = 0; i < rowsB; i++) {
    std::copy(B_linear.begin() + i * colsB, B_linear.begin() + (i + 1) * colsB, local_B[i].begin());
  }

  // 5. Распределяем матрицу A по строкам
  int rows_per_process = rowsA / world_size_;
  int rows_remainder = rowsA % world_size_;

  // Количество строк для текущего процесса
  int local_rows = (rank_ < rows_remainder) ? rows_per_process + 1 : rows_per_process;

  // Начальный индекс строк для текущего процесса
  int start_row = 0;
  for (int p = 0; p < rank_; p++) {
    start_row += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
  }

  // 6. Подготавливаем буфер для локальной части A
  std::vector<std::vector<double>> local_A(local_rows, std::vector<double>(colsA));

  if (rank_ == 0) {
    // Процесс 0 копирует свою часть
    for (int i = 0; i < local_rows; i++) {
      local_A[i] = matrix_A_[start_row + i];
    }

    // Отправляем части другим процессам
    for (int proc = 1; proc < world_size_; proc++) {
      // Вычисляем параметры для процесса proc
      int proc_rows = (proc < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      int proc_start = 0;
      for (int p = 0; p < proc; p++) {
        proc_start += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      }

      // Отправляем количество строк
      MPI_Send(&proc_rows, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);

      // Отправляем start_row
      MPI_Send(&proc_start, 1, MPI_INT, proc, 1, MPI_COMM_WORLD);

      // Отправляем строки
      for (int i = 0; i < proc_rows; i++) {
        MPI_Send(matrix_A_[proc_start + i].data(), colsA, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
      }
    }
  } else {
    // Получаем количество строк
    MPI_Recv(&local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Получаем start_row
    MPI_Recv(&start_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Перераспределяем память
    local_A.resize(local_rows, std::vector<double>(colsA));

    // Получаем строки матрицы A
    for (int i = 0; i < local_rows; i++) {
      std::vector<double> row(colsA);
      MPI_Recv(row.data(), colsA, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_A[i] = row;
    }
  }

  // 7. Локальное умножение
  std::vector<std::vector<double>> local_result(local_rows, std::vector<double>(colsB, 0.0));

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < colsB; j++) {
      double sum = 0.0;
      for (int k = 0; k < colsA; k++) {
        sum += local_A[i][k] * local_B[k][j];
      }
      local_result[i][j] = sum;
    }
  }

  // 8. Сбор результатов в процессе 0
  std::vector<std::vector<double>> final_result;

  if (rank_ == 0) {
    final_result.resize(rowsA, std::vector<double>(colsB, 0.0));

    // Копируем свою часть
    for (int i = 0; i < local_rows; i++) {
      final_result[start_row + i] = local_result[i];
    }

    // Получаем от других процессов
    for (int proc = 1; proc < world_size_; proc++) {
      // Вычисляем параметры для процесса proc
      int proc_rows = (proc < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      int proc_start = 0;
      for (int p = 0; p < proc; p++) {
        proc_start += (p < rows_remainder) ? rows_per_process + 1 : rows_per_process;
      }

      // Получаем строки результата
      for (int i = 0; i < proc_rows; i++) {
        std::vector<double> row(colsB);
        MPI_Recv(row.data(), colsB, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        final_result[proc_start + i] = row;
      }
    }
  } else {
    // Отправляем свои результаты процессу 0
    for (int i = 0; i < local_rows; i++) {
      MPI_Send(local_result[i].data(), colsB, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
  }

  // 9. Рассылаем финальную матрицу всем процессам и сохраняем в GetOutput()
  if (rank_ == 0) {
    // Процесс 0 готовит линейный массив для рассылки
    std::vector<double> result_linear;
    result_linear.reserve(rowsA * colsB);
    for (int i = 0; i < rowsA; i++) {
      result_linear.insert(result_linear.end(), final_result[i].begin(), final_result[i].end());
    }

    // Рассылаем размеры всем процессам
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Рассылаем данные
    MPI_Bcast(result_linear.data(), rowsA * colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Сохраняем результат в GetOutput() (для процесса 0)
    GetOutput() = final_result;
  } else {
    // Получаем размеры
    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Получаем данные
    std::vector<double> result_linear(rowsA * colsB);
    MPI_Bcast(result_linear.data(), rowsA * colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Преобразуем в матрицу и сохраняем в GetOutput()
    std::vector<std::vector<double>> result_matrix(rowsA, std::vector<double>(colsB));
    for (int i = 0; i < rowsA; i++) {
      std::copy(result_linear.begin() + i * colsB, result_linear.begin() + (i + 1) * colsB, result_matrix[i].begin());
    }
    GetOutput() = result_matrix;  // ← сохраняем результат ПРЯМО здесь!
  }

  // Синхронизация
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
