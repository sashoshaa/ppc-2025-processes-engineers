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
// ВРЕМЕННО подключаем iostream для отладки
#include <iostream>

  // 0. Начало
  std::cout << "=== PROCESS " << rank_ << "/" << world_size_ << " STARTING ===" << std::endl;

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

    std::cout << "[RANK 0 DEBUG] Original sizes from input:" << std::endl;
    std::cout << "  A: " << rows_a << "x" << cols_a << std::endl;
    std::cout << "  B: " << rows_b << "x" << cols_b << std::endl;

    // Выводим матрицу A для проверки
    std::cout << "[RANK 0 DEBUG] Matrix A:" << std::endl;
    for (int i = 0; i < rows_a; i++) {
      std::cout << "  Row " << i << ": ";
      for (int j = 0; j < cols_a; j++) {
        std::cout << matrix_A_[i][j] << " ";
      }
      std::cout << std::endl;
    }

    // Выводим матрицу B для проверки
    std::cout << "[RANK 0 DEBUG] Matrix B:" << std::endl;
    for (int i = 0; i < rows_b; i++) {
      std::cout << "  Row " << i << ": ";
      for (int j = 0; j < cols_b; j++) {
        std::cout << matrix_B_[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // 2. Рассылаем размеры всем процессам
  std::array<int, 4> sizes = {rows_a, cols_a, rows_b, cols_b};
  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  rows_a = sizes[0];
  cols_a = sizes[1];
  rows_b = sizes[2];
  cols_b = sizes[3];

  std::cout << "[RANK " << rank_ << " DEBUG] After broadcast sizes:" << std::endl;
  std::cout << "  A: " << rows_a << "x" << cols_a << std::endl;
  std::cout << "  B: " << rows_b << "x" << cols_b << std::endl;

  // 3. Проверка совместимости
  if (cols_a != rows_b) {
    std::cout << "[RANK " << rank_ << " DEBUG] INCOMPATIBLE DIMENSIONS: cols_a=" << cols_a << " != rows_b=" << rows_b
              << std::endl;
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 4. Если одна из матриц пустая
  if (rows_a == 0 || cols_a == 0 || rows_b == 0 || cols_b == 0) {
    std::cout << "[RANK " << rank_ << " DEBUG] EMPTY MATRIX DETECTED" << std::endl;
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 5. Рассылаем матрицу B всем процессам
  std::vector<double> b_flat(rows_b * cols_b);
  if (rank_ == 0) {
    for (int i = 0; i < rows_b; i++) {
      for (int j = 0; j < cols_b; j++) {
        b_flat[i * cols_b + j] = matrix_B_[i][j];
      }
    }
    std::cout << "[RANK 0 DEBUG] B matrix flattened. First 5 elements: ";
    for (int i = 0; i < std::min(5, rows_b * cols_b); i++) {
      std::cout << b_flat[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::cout << "[RANK " << rank_ << " DEBUG] Received B matrix. First element: " << b_flat[0] << std::endl;

  // 6. Вычисляем распределение строк - КРИТИЧЕСКИЙ МОМЕНТ!
  int base_rows = rows_a / world_size_;
  int extra_rows = rows_a % world_size_;

  std::cout << "[RANK " << rank_ << " DEBUG] Distribution calculation:" << std::endl;
  std::cout << "  base_rows = " << base_rows << std::endl;
  std::cout << "  extra_rows = " << extra_rows << std::endl;

  // Вычисляем для ВСЕХ процессов чтобы было одинаково
  std::vector<int> all_rows_per_process(world_size_);
  std::vector<int> all_displacements(world_size_);

  int current_displ = 0;
  for (int i = 0; i < world_size_; i++) {
    all_rows_per_process[i] = base_rows + (i < extra_rows ? 1 : 0);
    all_displacements[i] = current_displ;
    current_displ += all_rows_per_process[i];
  }

  int local_rows = all_rows_per_process[rank_];
  int row_offset = all_displacements[rank_];

  std::cout << "[RANK " << rank_ << " DEBUG] My distribution:" << std::endl;
  std::cout << "  local_rows = " << local_rows << std::endl;
  std::cout << "  row_offset = " << row_offset << std::endl;
  std::cout << "  all_rows_per_process = ";
  for (int i = 0; i < world_size_; i++) {
    std::cout << all_rows_per_process[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  all_displacements = ";
  for (int i = 0; i < world_size_; i++) {
    std::cout << all_displacements[i] << " ";
  }
  std::cout << std::endl;

  // 7. Получаем локальные строки матрицы A
  std::vector<double> local_a_flat(local_rows * cols_a);

  if (rank_ == 0) {
    // Процесс 0 копирует свои строки
    std::cout << "[RANK 0 DEBUG] Copying my " << local_rows << " rows of A" << std::endl;
    for (int i = 0; i < local_rows; i++) {
      int global_row = row_offset + i;
      std::cout << "  Global row " << global_row << ": ";
      for (int j = 0; j < cols_a; j++) {
        local_a_flat[i * cols_a + j] = matrix_A_[global_row][j];
        std::cout << matrix_A_[global_row][j] << " ";
      }
      std::cout << std::endl;
    }

    // Отправляем другим процессам их части
    for (int dest = 1; dest < world_size_; dest++) {
      int dest_rows = all_rows_per_process[dest];
      if (dest_rows == 0) {
        std::cout << "[RANK 0 DEBUG] Process " << dest << " has 0 rows, skipping" << std::endl;
        continue;
      }

      std::vector<double> buffer(dest_rows * cols_a);
      int dest_offset = all_displacements[dest];

      std::cout << "[RANK 0 DEBUG] Sending " << dest_rows << " rows to process " << dest << " (offset " << dest_offset
                << ")" << std::endl;

      for (int i = 0; i < dest_rows; i++) {
        int global_row = dest_offset + i;
        std::cout << "  Global row " << global_row << ": ";
        for (int j = 0; j < cols_a; j++) {
          buffer[i * cols_a + j] = matrix_A_[global_row][j];
          std::cout << matrix_A_[global_row][j] << " ";
        }
        std::cout << std::endl;
      }

      MPI_Send(buffer.data(), dest_rows * cols_a, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      std::cout << "[RANK 0 DEBUG] Sent data to process " << dest << std::endl;
    }
  } else {
    // Получаем данные от процесса 0
    if (local_rows > 0) {
      std::cout << "[RANK " << rank_ << " DEBUG] Waiting for " << local_rows << " rows from process 0" << std::endl;
      MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::cout << "[RANK " << rank_ << " DEBUG] Received A data. First row: ";
      for (int j = 0; j < std::min(cols_a, 5); j++) {
        std::cout << local_a_flat[j] << " ";
      }
      std::cout << std::endl;
    } else {
      std::cout << "[RANK " << rank_ << " DEBUG] I have 0 rows, nothing to receive" << std::endl;
    }
  }

  // 8. Проверяем полученные данные ВСЕМИ процессами
  if (local_rows > 0) {
    std::cout << "[RANK " << rank_ << " DEBUG] My local A matrix (" << local_rows << "x" << cols_a << "):" << std::endl;
    for (int i = 0; i < std::min(local_rows, 3); i++) {
      std::cout << "  Row " << i << ": ";
      for (int j = 0; j < std::min(cols_a, 5); j++) {
        std::cout << local_a_flat[i * cols_a + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // 9. Локальное умножение
  std::vector<double> local_result_flat(local_rows * cols_b, 0.0);

  std::cout << "[RANK " << rank_ << " DEBUG] Starting local multiplication" << std::endl;

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < cols_b; j++) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; k++) {
        sum += local_a_flat[i * cols_a + k] * b_flat[k * cols_b + j];
      }
      local_result_flat[i * cols_b + j] = sum;
    }
  }

  std::cout << "[RANK " << rank_ << " DEBUG] Local multiplication done. First result: ";
  if (local_rows > 0 && cols_b > 0) {
    std::cout << local_result_flat[0] << std::endl;
  }

  // 10. Собираем результаты на процессе 0
  std::vector<double> final_result_flat;

  if (rank_ == 0) {
    final_result_flat.resize(rows_a * cols_b, 0.0);

    // Копируем свои результаты
    std::cout << "[RANK 0 DEBUG] Copying my results (offset " << row_offset << ")" << std::endl;
    for (int i = 0; i < local_rows; i++) {
      int global_row = row_offset + i;
      for (int j = 0; j < cols_b; j++) {
        final_result_flat[global_row * cols_b + j] = local_result_flat[i * cols_b + j];
      }
    }

    std::cout << "[RANK 0 DEBUG] My results copied. First 5 elements of final_result: ";
    for (int i = 0; i < std::min(5, rows_a * cols_b); i++) {
      std::cout << final_result_flat[i] << " ";
    }
    std::cout << std::endl;

    // Получаем результаты от других процессов
    for (int src = 1; src < world_size_; src++) {
      int src_rows = all_rows_per_process[src];
      if (src_rows == 0) {
        std::cout << "[RANK 0 DEBUG] Process " << src << " has 0 results, skipping" << std::endl;
        continue;
      }

      std::vector<double> buffer(src_rows * cols_b);
      int src_offset = all_displacements[src];

      std::cout << "[RANK 0 DEBUG] Waiting for " << src_rows << " results from process " << src << " (offset "
                << src_offset << ")" << std::endl;

      MPI_Recv(buffer.data(), src_rows * cols_b, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::cout << "[RANK 0 DEBUG] Received from process " << src << ". First element: " << buffer[0] << std::endl;

      // Копируем полученные результаты
      for (int i = 0; i < src_rows; i++) {
        int global_row = src_offset + i;
        for (int j = 0; j < cols_b; j++) {
          final_result_flat[global_row * cols_b + j] = buffer[i * cols_b + j];
        }
      }

      std::cout << "[RANK 0 DEBUG] After process " << src << ", first 5 elements: ";
      for (int i = 0; i < std::min(5, rows_a * cols_b); i++) {
        std::cout << final_result_flat[i] << " ";
      }
      std::cout << std::endl;
    }

    // ВЫВОД ИТОГОВОГО РЕЗУЛЬТАТА НА ПРОЦЕССЕ 0
    std::cout << "=== FINAL RESULT ON RANK 0 ===" << std::endl;
    std::cout << "Matrix " << rows_a << "x" << cols_b << ":" << std::endl;
    for (int i = 0; i < rows_a; i++) {
      std::cout << "  Row " << i << ": ";
      for (int j = 0; j < cols_b; j++) {
        std::cout << final_result_flat[i * cols_b + j] << " ";
      }
      std::cout << std::endl;
    }

    // Проверяем с ожидаемым результатом (для теста 2)
    std::cout << "=== EXPECTED (Test 2) ===" << std::endl;
    std::cout << "[[1, 2], [3, 4]]" << std::endl;

  } else {
    // Отправляем результаты процессу 0
    if (local_rows > 0) {
      std::cout << "[RANK " << rank_ << " DEBUG] Sending " << local_rows << " results to process 0" << std::endl;
      MPI_Send(local_result_flat.data(), local_rows * cols_b, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
  }

  // 11. Рассылаем окончательный результат всем процессам
  if (rank_ == 0) {
    std::cout << "[RANK 0 DEBUG] Broadcasting final result to all" << std::endl;
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    final_result_flat.resize(rows_a * cols_b);
    std::cout << "[RANK " << rank_ << " DEBUG] Waiting for final result broadcast" << std::endl;
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::cout << "[RANK " << rank_ << " DEBUG] Received final result. First element: " << final_result_flat[0]
              << std::endl;
  }

  // 12. Преобразуем в матрицу
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      result_matrix[i][j] = final_result_flat[i * cols_b + j];
    }
  }

  std::cout << "[RANK " << rank_ << " DEBUG] Final matrix ready. Setting output." << std::endl;
  GetOutput() = result_matrix;

  std::cout << "=== PROCESS " << rank_ << " FINISHED ===" << std::endl;
  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
