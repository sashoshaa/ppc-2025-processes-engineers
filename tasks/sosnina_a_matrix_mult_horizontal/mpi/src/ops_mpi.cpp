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
// ЖЕСТКАЯ ОТЛАДКА - на каждом шагу
#include <iomanip>
#include <iostream>

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
  std::cout << "PROCESS " << rank_ << "/" << world_size_ << " START - RunImpl" << std::endl;
  std::cout << "═══════════════════════════════════════════════════════" << std::endl;

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

    std::cout << "[RANK 0] INPUT MATRICES:" << std::endl;
    std::cout << "  Matrix A (" << rows_a << "x" << cols_a << "):" << std::endl;
    for (int i = 0; i < rows_a; i++) {
      std::cout << "    [";
      for (int j = 0; j < cols_a; j++) {
        std::cout << matrix_A_[i][j];
        if (j < cols_a - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }

    std::cout << "  Matrix B (" << rows_b << "x" << cols_b << "):" << std::endl;
    for (int i = 0; i < rows_b; i++) {
      std::cout << "    [";
      for (int j = 0; j < cols_b; j++) {
        std::cout << matrix_B_[i][j];
        if (j < cols_b - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  }

  // 2. Рассылаем размеры всем процессам
  std::array<int, 4> sizes = {rows_a, cols_a, rows_b, cols_b};
  std::cout << "[RANK " << rank_ << "] BEFORE BCAST sizes: [" << sizes[0] << ", " << sizes[1] << ", " << sizes[2]
            << ", " << sizes[3] << "]" << std::endl;

  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);

  rows_a = sizes[0];
  cols_a = sizes[1];
  rows_b = sizes[2];
  cols_b = sizes[3];

  std::cout << "[RANK " << rank_ << "] AFTER BCAST sizes: A(" << rows_a << "x" << cols_a << "), B(" << rows_b << "x"
            << cols_b << ")" << std::endl;

  // 3. Проверка совместимости
  if (cols_a != rows_b) {
    std::cout << "[RANK " << rank_ << "] ⚠️  INCOMPATIBLE: cols_a=" << cols_a << " != rows_b=" << rows_b << std::endl;
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 4. Если одна из матриц пустая
  if (rows_a == 0 || cols_a == 0 || rows_b == 0 || cols_b == 0) {
    std::cout << "[RANK " << rank_ << "] ⚠️  EMPTY MATRIX DETECTED" << std::endl;
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // 5. Рассылаем матрицу B всем процессам
  std::vector<double> b_flat(rows_b * cols_b);
  if (rank_ == 0) {
    std::cout << "[RANK 0] Flattening matrix B (" << rows_b << "x" << cols_b << "):" << std::endl;
    for (int i = 0; i < rows_b; i++) {
      for (int j = 0; j < cols_b; j++) {
        b_flat[i * cols_b + j] = matrix_B_[i][j];
        std::cout << "  B[" << i << "][" << j << "] = " << matrix_B_[i][j] << " -> b_flat[" << (i * cols_b + j) << "]"
                  << std::endl;
      }
    }

    std::cout << "[RANK 0] B_flat array: [";
    for (int i = 0; i < std::min(10, rows_b * cols_b); i++) {
      std::cout << b_flat[i];
      if (i < std::min(10, rows_b * cols_b) - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::cout << "[RANK " << rank_ << "] Received B_flat. First element = " << b_flat[0] << std::endl;

  // 6. ЛЕНТОЧНОЕ РАСПРЕДЕЛЕНИЕ - ДЕТАЛЬНАЯ ОТЛАДКА
  std::cout << "\n[RANK " << rank_ << "] ──── LENTOCHNOE RASPREDELENIE ────" << std::endl;

  int base_rows = rows_a / world_size_;
  int extra_rows = rows_a % world_size_;

  std::cout << "[RANK " << rank_ << "] base_rows = " << rows_a << " / " << world_size_ << " = " << base_rows
            << std::endl;
  std::cout << "[RANK " << rank_ << "] extra_rows = " << rows_a << " % " << world_size_ << " = " << extra_rows
            << std::endl;

  int local_rows = base_rows + (rank_ < extra_rows ? 1 : 0);
  std::cout << "[RANK " << rank_ << "] local_rows = " << base_rows << " + (" << rank_ << " < " << extra_rows
            << " ? 1 : 0) = " << local_rows << std::endl;

  // Определяем КАЖДУЮ строку, которую получает этот процесс
  std::vector<int> my_row_indices;
  std::cout << "[RANK " << rank_ << "] My rows (i % " << world_size_ << " == " << rank_ << "): ";
  for (int i = 0; i < rows_a; i++) {
    if (i % world_size_ == rank_) {
      my_row_indices.push_back(i);
      std::cout << i << " ";
    }
  }
  std::cout << std::endl;

  // Проверка
  if (local_rows != static_cast<int>(my_row_indices.size())) {
    std::cout << "[RANK " << rank_ << "] ⚠️  WARNING: local_rows(" << local_rows << ") != my_row_indices.size("
              << my_row_indices.size() << ")" << std::endl;
    local_rows = my_row_indices.size();
  }

  std::cout << "[RANK " << rank_ << "] Final: I will process " << local_rows << " rows" << std::endl;

  // 7. Получаем локальные строки матрицы A
  std::vector<double> local_a_flat(local_rows * cols_a, 0.0);

  if (rank_ == 0) {
    std::cout << "\n[RANK 0] ──── DISTRIBUTING MATRIX A ────" << std::endl;

    // Процесс 0 копирует свои строки
    std::cout << "[RANK 0] Copying MY rows from matrix A:" << std::endl;
    for (size_t idx = 0; idx < my_row_indices.size(); idx++) {
      int global_row = my_row_indices[idx];
      std::cout << "  Row " << global_row << " -> my index " << idx << ": [";
      for (int j = 0; j < cols_a; j++) {
        local_a_flat[idx * cols_a + j] = matrix_A_[global_row][j];
        std::cout << matrix_A_[global_row][j];
        if (j < cols_a - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }

    // Отправляем другим процессам их строки
    for (int dest = 1; dest < world_size_; dest++) {
      // Определяем строки для процесса dest
      std::vector<int> dest_rows;
      for (int i = 0; i < rows_a; i++) {
        if (i % world_size_ == dest) {
          dest_rows.push_back(i);
        }
      }

      int dest_row_count = dest_rows.size();
      std::cout << "[RANK 0] Process " << dest << " should get " << dest_row_count << " rows: ";
      for (int row : dest_rows) {
        std::cout << row << " ";
      }
      std::cout << std::endl;

      if (dest_row_count == 0) {
        std::cout << "[RANK 0] Process " << dest << " gets 0 rows, skipping" << std::endl;
        continue;
      }

      std::vector<double> buffer(dest_row_count * cols_a);

      std::cout << "[RANK 0] Preparing data for process " << dest << ":" << std::endl;
      for (int idx = 0; idx < dest_row_count; idx++) {
        int global_row = dest_rows[idx];
        std::cout << "  Row " << global_row << " -> buffer[" << idx << "]: [";
        for (int j = 0; j < cols_a; j++) {
          buffer[idx * cols_a + j] = matrix_A_[global_row][j];
          std::cout << matrix_A_[global_row][j];
          if (j < cols_a - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }

      std::cout << "[RANK 0] Sending " << (dest_row_count * cols_a) << " elements to process " << dest << std::endl;
      MPI_Send(buffer.data(), dest_row_count * cols_a, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      std::cout << "[RANK 0] Sent to process " << dest << std::endl;
    }
  } else {
    // Получаем данные от процесса 0
    std::cout << "\n[RANK " << rank_ << "] Waiting for " << local_rows << " rows from process 0" << std::endl;

    if (local_rows > 0) {
      MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::cout << "[RANK " << rank_ << "] Received data from process 0:" << std::endl;
      for (int i = 0; i < local_rows; i++) {
        std::cout << "  My row " << i << " (global row " << my_row_indices[i] << "): [";
        for (int j = 0; j < cols_a; j++) {
          std::cout << local_a_flat[i * cols_a + j];
          if (j < cols_a - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }
    } else {
      std::cout << "[RANK " << rank_ << "] I have 0 rows, nothing to receive" << std::endl;
    }
  }

  // 8. Локальное умножение - ДЕТАЛЬНЫЙ РАСЧЕТ
  std::cout << "\n[RANK " << rank_ << "] ──── LOCAL MULTIPLICATION ────" << std::endl;
  std::vector<double> local_result_flat(local_rows * cols_b, 0.0);

  if (local_rows > 0) {
    std::cout << "[RANK " << rank_ << "] Multiplying " << local_rows << "x" << cols_a << " with " << rows_b << "x"
              << cols_b << std::endl;

    for (int i = 0; i < local_rows; i++) {
      int global_row = my_row_indices[i];
      std::cout << "[RANK " << rank_ << "] Calculating row " << global_row << " (my index " << i << "):" << std::endl;

      for (int j = 0; j < cols_b; j++) {
        double sum = 0.0;
        std::cout << "  Column " << j << ": sum = ";

        for (int k = 0; k < cols_a; k++) {
          double a_val = local_a_flat[i * cols_a + k];
          double b_val = b_flat[k * cols_b + j];
          sum += a_val * b_val;

          std::cout << a_val << "*" << b_val;
          if (k < cols_a - 1) {
            std::cout << " + ";
          }
        }

        local_result_flat[i * cols_b + j] = sum;
        std::cout << " = " << sum << std::endl;
      }
    }

    std::cout << "[RANK " << rank_ << "] Local results:" << std::endl;
    for (int i = 0; i < local_rows; i++) {
      std::cout << "  Row " << my_row_indices[i] << ": [";
      for (int j = 0; j < cols_b; j++) {
        std::cout << local_result_flat[i * cols_b + j];
        if (j < cols_b - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  } else {
    std::cout << "[RANK " << rank_ << "] No rows to multiply" << std::endl;
  }

  // 9. Собираем результаты на процессе 0
  std::cout << "\n[RANK " << rank_ << "] ──── GATHERING RESULTS ────" << std::endl;
  std::vector<double> final_result_flat;

  if (rank_ == 0) {
    final_result_flat.resize(rows_a * cols_b, 0.0);

    std::cout << "[RANK 0] Initializing result matrix " << rows_a << "x" << cols_b << " with zeros" << std::endl;

    // Копируем свои результаты
    std::cout << "[RANK 0] Copying MY results:" << std::endl;
    for (size_t idx = 0; idx < my_row_indices.size(); idx++) {
      int global_row = my_row_indices[idx];
      std::cout << "  My row " << global_row << ": [";
      for (int j = 0; j < cols_b; j++) {
        final_result_flat[global_row * cols_b + j] = local_result_flat[idx * cols_b + j];
        std::cout << local_result_flat[idx * cols_b + j];
        if (j < cols_b - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }

    // Получаем результаты от других процессов
    for (int src = 1; src < world_size_; src++) {
      // Определяем строки для процесса src
      std::vector<int> src_rows;
      for (int i = 0; i < rows_a; i++) {
        if (i % world_size_ == src) {
          src_rows.push_back(i);
        }
      }

      int src_row_count = src_rows.size();
      std::cout << "[RANK 0] Waiting for " << src_row_count << " rows from process " << src << std::endl;

      if (src_row_count == 0) {
        std::cout << "[RANK 0] Process " << src << " has 0 rows, skipping" << std::endl;
        continue;
      }

      std::vector<double> buffer(src_row_count * cols_b);

      MPI_Recv(buffer.data(), src_row_count * cols_b, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::cout << "[RANK 0] Received from process " << src << ":" << std::endl;
      for (int idx = 0; idx < src_row_count; idx++) {
        int global_row = src_rows[idx];
        std::cout << "  Row " << global_row << ": [";
        for (int j = 0; j < cols_b; j++) {
          final_result_flat[global_row * cols_b + j] = buffer[idx * cols_b + j];
          std::cout << buffer[idx * cols_b + j];
          if (j < cols_b - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }
    }

    // ВЫВОД ФИНАЛЬНОГО РЕЗУЛЬТАТА
    std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "FINAL RESULT MATRIX " << rows_a << "x" << cols_b << " (Rank 0):" << std::endl;
    for (int i = 0; i < rows_a; i++) {
      std::cout << "  [";
      for (int j = 0; j < cols_b; j++) {
        std::cout << final_result_flat[i * cols_b + j];
        if (j < cols_b - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }

    // Проверка с ожидаемыми значениями (для теста 1)
    std::cout << "\nEXPECTED for test 1 (A=[[1,2],[3,4]], B=[[5,6],[7,8]]):" << std::endl;
    std::cout << "  [[19, 22], [43, 50]]" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;

  } else {
    // Отправляем результаты процессу 0
    if (local_rows > 0) {
      std::cout << "[RANK " << rank_ << "] Sending " << local_rows << " rows to process 0" << std::endl;
      MPI_Send(local_result_flat.data(), local_rows * cols_b, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      std::cout << "[RANK " << rank_ << "] Sent results to process 0" << std::endl;
    } else {
      std::cout << "[RANK " << rank_ << "] No results to send" << std::endl;
    }
  }

  // 10. Рассылаем окончательный результат всем процессам
  std::cout << "\n[RANK " << rank_ << "] ──── BROADCAST FINAL RESULT ────" << std::endl;

  if (rank_ == 0) {
    std::cout << "[RANK 0] Broadcasting final result (" << rows_a * cols_b << " elements)" << std::endl;
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    final_result_flat.resize(rows_a * cols_b);
    std::cout << "[RANK " << rank_ << "] Waiting for final result broadcast" << std::endl;
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << "[RANK " << rank_ << "] Received final result. First element = " << final_result_flat[0] << std::endl;
  }

  // 11. Преобразуем в матрицу
  std::cout << "\n[RANK " << rank_ << "] ──── CONVERTING TO MATRIX ────" << std::endl;
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));

  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      result_matrix[i][j] = final_result_flat[i * cols_b + j];
    }
  }

  std::cout << "[RANK " << rank_ << "] Setting output" << std::endl;
  GetOutput() = result_matrix;

  std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
  std::cout << "PROCESS " << rank_ << "/" << world_size_ << " FINISHED" << std::endl;
  std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;

  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  // Теперь здесь ничего не нужно делать, результат уже в GetOutput()
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
