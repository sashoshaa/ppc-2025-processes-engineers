#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
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

  // 5. Рассылаем матрицу B всем процессам
  std::vector<double> b_flat(static_cast<std::size_t>(rows_b) * static_cast<std::size_t>(cols_b));
  if (rank_ == 0) {
    for (int i = 0; i < rows_b; i++) {
      for (int j = 0; j < cols_b; j++) {
        b_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + j] = matrix_B_[i][j];
      }
    }
  }
  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 6. ЛЕНТОЧНОЕ РАСПРЕДЕЛЕНИЕ: вычисляем горизонтальные блоки строк
  int base_rows = rows_a / world_size_;
  int extra_rows = rows_a % world_size_;

  // Стартовая строка и количество строк, которые обрабатывает процесс rank_
  int start_row = rank_ * base_rows + std::min(rank_, extra_rows);
  int local_rows = base_rows + (rank_ < extra_rows ? 1 : 0);

  // 7. Получаем локальные строки матрицы A
  std::vector<double> local_a_flat(local_rows * cols_a);

  if (rank_ == 0) {
    // Процесс 0 копирует свои строки (непрерывный блок)
    for (int idx = 0; idx < local_rows; idx++) {
      int row_idx = start_row + idx;
      for (int j = 0; j < cols_a; j++) {
        local_a_flat[idx * cols_a + j] = matrix_A_[row_idx][j];
      }
    }

    // Отправляем непрерывные блоки другим процессам
    for (int proc = 1; proc < world_size_; proc++) {
      int proc_start_row = proc * base_rows + std::min(proc, extra_rows);
      int proc_rows = base_rows + (proc < extra_rows ? 1 : 0);

      if (proc_rows == 0) {
        continue;
      }

      std::vector<double> proc_A_flat(proc_rows * cols_a);
      for (int idx = 0; idx < proc_rows; idx++) {
        int row_idx = proc_start_row + idx;
        for (int j = 0; j < cols_a; j++) {
          proc_A_flat[idx * cols_a + j] = matrix_A_[row_idx][j];
        }
      }

      MPI_Send(proc_A_flat.data(), proc_rows * cols_a, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
    }
  } else {
    // Получаем данные от процесса 0 только если есть строки
    if (local_rows > 0) {
      MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // 8. Локальное умножение
  std::vector<double> local_result_flat(local_rows * cols_b, 0.0);

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < cols_b; j++) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; k++) {
        sum +=
            local_a_flat[i * cols_a + k] * b_flat[static_cast<std::size_t>(k) * static_cast<std::size_t>(cols_b) + j];
      }
      local_result_flat[i * cols_b + j] = sum;
    }
  }

  // 9. Собираем результаты на процессе 0
  std::vector<double> final_result_flat;
  if (rank_ == 0) {
    final_result_flat.resize(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_b), 0.0);
  }

  // Каждый процесс отправляет свои результаты процессу 0
  if (rank_ == 0) {
    // Копируем свои результаты в правильные позиции
    for (int idx = 0; idx < local_rows; idx++) {
      int row_idx = start_row + idx;
      for (int j = 0; j < cols_b; j++) {
        final_result_flat[static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(cols_b) + j] =
            local_result_flat[static_cast<std::size_t>(idx) * static_cast<std::size_t>(cols_b) + j];
      }
    }

    // Получаем результаты от других процессов (непрерывные блоки)
    for (int proc = 1; proc < world_size_; proc++) {
      int proc_start_row = proc * base_rows + std::min(proc, extra_rows);
      int proc_rows = base_rows + (proc < extra_rows ? 1 : 0);

      if (proc_rows > 0) {
        std::vector<double> proc_result_flat(proc_rows * cols_b);
        MPI_Recv(proc_result_flat.data(), proc_rows * cols_b, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int idx = 0; idx < proc_rows; idx++) {
          int row_idx = proc_start_row + idx;
          for (int j = 0; j < cols_b; j++) {
            final_result_flat[static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(cols_b) + j] =
                proc_result_flat[static_cast<std::size_t>(idx) * static_cast<std::size_t>(cols_b) + j];
          }
        }
      }
    }
  } else {
    // Отправляем результаты процессу 0 (если есть что отправлять)
    if (local_rows > 0) {
      MPI_Send(local_result_flat.data(), local_rows * cols_b, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
  }

  // 10. Рассылаем результат всем процессам
  if (rank_ == 0) {
    // Процесс 0 рассылает размеры и данные
    std::array<int, 2> result_dims = {rows_a, cols_b};
    MPI_Bcast(result_dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    // Получаем размеры и данные
    int result_dims[2];
    MPI_Bcast(result_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int result_rows = result_dims[0];
    int result_cols = result_dims[1];

    final_result_flat.resize(result_rows * result_cols);
    MPI_Bcast(final_result_flat.data(), result_rows * result_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // 11. Преобразуем плоский массив обратно в матрицу
  int result_rows = rows_a;
  int result_cols = cols_b;

  std::vector<std::vector<double>> result_matrix(result_rows, std::vector<double>(result_cols));
  for (int i = 0; i < result_rows; i++) {
    for (int j = 0; j < result_cols; j++) {
      result_matrix[i][j] = final_result_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(result_cols) + j];
    }
  }

  // 12. Устанавливаем результат
  GetOutput() = result_matrix;
  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
