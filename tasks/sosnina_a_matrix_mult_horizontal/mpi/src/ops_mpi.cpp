#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"

namespace sosnina_a_matrix_mult_horizontal {

SosninaAMatrixMultHorizontalMPI::SosninaAMatrixMultHorizontalMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
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
  GetOutput() = std::vector<std::vector<double>>();

  return true;
}

bool SosninaAMatrixMultHorizontalMPI::RunImpl() {
  int rows_a = 0, cols_a = 0, rows_b = 0, cols_b = 0;

  // Инициализация размеров на процессе 0
  if (rank_ == 0) {
    rows_a = static_cast<int>(matrix_A_.size());
    cols_a = rows_a > 0 ? static_cast<int>(matrix_A_[0].size()) : 0;
    rows_b = static_cast<int>(matrix_B_.size());
    cols_b = rows_b > 0 ? static_cast<int>(matrix_B_[0].size()) : 0;
  }

  // Передача размеров всем процессам
  std::array<int, 4> sizes = {rows_a, cols_a, rows_b, cols_b};
  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
  rows_a = sizes[0];
  cols_a = sizes[1];
  rows_b = sizes[2];
  cols_b = sizes[3];

  // Проверка корректности размеров
  if (cols_a != rows_b || rows_a == 0 || cols_a == 0 || rows_b == 0 || cols_b == 0) {
    GetOutput() = std::vector<std::vector<double>>();
    return true;
  }

  // Подготовка и рассылка матрицы B
  std::vector<double> b_flat(static_cast<size_t>(rows_b) * static_cast<size_t>(cols_b));
  if (rank_ == 0) {
    for (int i = 0; i < rows_b; ++i) {
      for (int j = 0; j < cols_b; ++j) {
        b_flat[(static_cast<size_t>(i) * static_cast<size_t>(cols_b)) + static_cast<size_t>(j)] = matrix_B_[i][j];
      }
    }
  }
  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Распределение строк матрицы A
  int local_rows = (rows_a / world_size_) + (rank_ < (rows_a % world_size_) ? 1 : 0);
  std::vector<int> my_row_indices;

  for (int i = 0; i < rows_a; ++i) {
    if (i % world_size_ == rank_) {
      my_row_indices.push_back(i);
    }
  }

  if (static_cast<int>(my_row_indices.size()) != local_rows) {
    local_rows = static_cast<int>(my_row_indices.size());
  }

  // Передача данных матрицы A
  std::vector<double> local_a_flat(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_a));

  if (rank_ == 0) {
    // Обработка локальных данных
    for (size_t idx = 0; idx < my_row_indices.size(); ++idx) {
      for (int j = 0; j < cols_a; ++j) {
        local_a_flat[(idx * static_cast<size_t>(cols_a)) + static_cast<size_t>(j)] = matrix_A_[my_row_indices[idx]][j];
      }
    }

    // Отправка данных другим процессам
    for (int dest = 1; dest < world_size_; ++dest) {
      std::vector<int> dest_rows;
      for (int i = 0; i < rows_a; ++i) {
        if (i % world_size_ == dest) {
          dest_rows.push_back(i);
        }
      }

      int dest_row_count = static_cast<int>(dest_rows.size());
      MPI_Send(&dest_row_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

      if (dest_row_count > 0) {
        MPI_Send(dest_rows.data(), dest_row_count, MPI_INT, dest, 1, MPI_COMM_WORLD);
        std::vector<double> buffer(static_cast<size_t>(dest_row_count) * static_cast<size_t>(cols_a));

        for (int idx = 0; idx < dest_row_count; ++idx) {
          for (int j = 0; j < cols_a; ++j) {
            buffer[(idx * static_cast<size_t>(cols_a)) + static_cast<size_t>(j)] = matrix_A_[dest_rows[idx]][j];
          }
        }

        MPI_Send(buffer.data(), dest_row_count * cols_a, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Recv(&local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (local_rows > 0) {
      my_row_indices.resize(local_rows);
      local_a_flat.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_a));
      MPI_Recv(my_row_indices.data(), local_rows, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Локальное умножение
  std::vector<double> local_result_flat(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_b), 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      for (int k = 0; k < cols_a; ++k) {
        local_result_flat[(i * static_cast<size_t>(cols_b)) + static_cast<size_t>(j)] +=
            local_a_flat[(i * static_cast<size_t>(cols_a)) + static_cast<size_t>(k)] *
            b_flat[(k * static_cast<size_t>(cols_b)) + static_cast<size_t>(j)];
      }
    }
  }

  // Сбор результатов
  std::vector<double> final_result_flat;

  if (rank_ == 0) {
    final_result_flat.resize(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0.0);

    // Локальные результаты
    for (size_t idx = 0; idx < my_row_indices.size(); ++idx) {
      for (int j = 0; j < cols_b; ++j) {
        final_result_flat[(my_row_indices[idx] * static_cast<size_t>(cols_b)) + j] =
            local_result_flat[(idx * static_cast<size_t>(cols_b)) + j];
      }
    }

    // Получение результатов от других процессов
    for (int src = 1; src < world_size_; ++src) {
      std::vector<int> src_rows;
      for (int i = 0; i < rows_a; ++i) {
        if (i % world_size_ == src) {
          src_rows.push_back(i);
        }
      }

      int src_row_count = static_cast<int>(src_rows.size());
      if (src_row_count > 0) {
        std::vector<double> buffer(static_cast<size_t>(src_row_count) * static_cast<size_t>(cols_b));
        MPI_Recv(buffer.data(), src_row_count * cols_b, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int idx = 0; idx < src_row_count; ++idx) {
          for (int j = 0; j < cols_b; ++j) {
            final_result_flat[(src_rows[idx] * static_cast<size_t>(cols_b)) + j] =
                buffer[(idx * static_cast<size_t>(cols_b)) + j];
          }
        }
      }
    }

    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    if (local_rows > 0) {
      MPI_Send(local_result_flat.data(), local_rows * cols_b, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }

    final_result_flat.resize(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b));
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Формирование результата
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      result_matrix[i][j] = final_result_flat[(i * static_cast<size_t>(cols_b)) + j];
    }
  }

  GetOutput() = result_matrix;
  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal
