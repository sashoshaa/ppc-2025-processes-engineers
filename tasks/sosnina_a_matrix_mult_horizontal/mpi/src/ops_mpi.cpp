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
  int rows_a = 0, cols_a = 0, rows_b = 0, cols_b = 0;

  if (rank_ == 0) {
    rows_a = static_cast<int>(matrix_A_.size());
    cols_a = rows_a > 0 ? static_cast<int>(matrix_A_[0].size()) : 0;
    rows_b = static_cast<int>(matrix_B_.size());
    cols_b = rows_b > 0 ? static_cast<int>(matrix_B_[0].size()) : 0;
  }

  // 2. Рассылаем размеры всем процессам
  int sizes[4] = {rows_a, cols_a, rows_b, cols_b};
  MPI_Bcast(sizes, 4, MPI_INT, 0, MPI_COMM_WORLD);

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
  std::vector<double> b_flat(rows_b * cols_b);
  if (rank_ == 0) {
    for (int i = 0; i < rows_b; i++) {
      for (int j = 0; j < cols_b; j++) {
        b_flat[i * cols_b + j] = matrix_B_[i][j];
      }
    }
  }
  MPI_Bcast(b_flat.data(), rows_b * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 6. ЛЕНТОЧНОЕ распределение матрицы A
  // Определяем какие строки получает каждый процесс
  // Строка i идет процессу (i % world_size_)
  
  // Сначала определим сколько строк у каждого процесса
  int base_rows = rows_a / world_size_;
  int extra_rows = rows_a % world_size_;
  
  // Количество строк для текущего процесса
  int local_rows = base_rows + (rank_ < extra_rows ? 1 : 0);
  
  // Создаем массив с номерами строк для этого процесса
  std::vector<int> my_row_indices;
  for (int i = 0; i < rows_a; i++) {
    if (i % world_size_ == rank_) {
      my_row_indices.push_back(i);
    }
  }
  
  // Должно совпадать с local_rows
  if (static_cast<int>(my_row_indices.size()) != local_rows) {
    local_rows = my_row_indices.size();
  }

  // 7. Получаем локальные строки матрицы A (ЛЕНТОЧНО!)
  std::vector<double> local_a_flat(local_rows * cols_a);

  if (rank_ == 0) {
    // Процесс 0 копирует СВОИ строки (ленточное распределение)
    for (size_t idx = 0; idx < my_row_indices.size(); idx++) {
      int global_row = my_row_indices[idx];
      for (int j = 0; j < cols_a; j++) {
        local_a_flat[idx * cols_a + j] = matrix_A_[global_row][j];
      }
    }

    // Отправляем другим процессам ИХ строки (ленточное распределение)
    for (int dest = 1; dest < world_size_; dest++) {
      // Определяем строки для процесса dest
      std::vector<int> dest_rows;
      for (int i = 0; i < rows_a; i++) {
        if (i % world_size_ == dest) {
          dest_rows.push_back(i);
        }
      }
      
      int dest_row_count = dest_rows.size();
      
      // ВАЖНО: отправляем даже если 0 строк!
      // Сначала отправляем количество строк
      MPI_Send(&dest_row_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
      
      if (dest_row_count > 0) {
        // Отправляем номера строк
        MPI_Send(dest_rows.data(), dest_row_count, MPI_INT, dest, 1, MPI_COMM_WORLD);
        
        // Отправляем данные строк
        std::vector<double> buffer(dest_row_count * cols_a);
        for (int idx = 0; idx < dest_row_count; idx++) {
          int global_row = dest_rows[idx];
          for (int j = 0; j < cols_a; j++) {
            buffer[idx * cols_a + j] = matrix_A_[global_row][j];
          }
        }
        MPI_Send(buffer.data(), dest_row_count * cols_a, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
      }
    }
  } else {
    // Получаем количество строк
    MPI_Recv(&local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Перераспределяем память если нужно
    if (local_rows > 0) {
      my_row_indices.resize(local_rows);
      local_a_flat.resize(local_rows * cols_a);
      
      // Получаем номера строк
      MPI_Recv(my_row_indices.data(), local_rows, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      // Получаем данные
      MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 
               0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // 8. Локальное умножение
  std::vector<double> local_result_flat(local_rows * cols_b, 0.0);
  
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < cols_b; j++) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; k++) {
        sum += local_a_flat[i * cols_a + k] * b_flat[k * cols_b + j];
      }
      local_result_flat[i * cols_b + j] = sum;
    }
  }

  // 9. Собираем результаты на процессе 0
  std::vector<double> final_result_flat;
  
  if (rank_ == 0) {
    final_result_flat.resize(rows_a * cols_b, 0.0);
    
    // Копируем свои результаты
    for (size_t idx = 0; idx < my_row_indices.size(); idx++) {
      int global_row = my_row_indices[idx];
      for (int j = 0; j < cols_b; j++) {
        final_result_flat[global_row * cols_b + j] = local_result_flat[idx * cols_b + j];
      }
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
      
      // ВАЖНО: всегда пытаемся получить, даже если 0 строк
      if (src_row_count > 0) {
        std::vector<double> buffer(src_row_count * cols_b);
        MPI_Recv(buffer.data(), src_row_count * cols_b, MPI_DOUBLE,
                 src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Копируем полученные результаты
        for (int idx = 0; idx < src_row_count; idx++) {
          int global_row = src_rows[idx];
          for (int j = 0; j < cols_b; j++) {
            final_result_flat[global_row * cols_b + j] = buffer[idx * cols_b + j];
          }
        }
      } else {
        // Процесс не имеет строк, но нам нужно проверить не отправил ли он пустое сообщение
        MPI_Status status;
        int flag;
        MPI_Iprobe(src, 3, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
          // Если отправил (даже пустое), получаем
          double dummy;
          MPI_Recv(&dummy, 0, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, &status);
        }
      }
    }
  } else {
    // Отправляем результаты процессу 0
    // ВАЖНО: отправляем даже если local_rows == 0!
    MPI_Send(local_result_flat.data(), local_rows * cols_b, MPI_DOUBLE,
             0, 3, MPI_COMM_WORLD);
  }

  // 10. Рассылаем финальный результат всем процессам
  if (rank_ == 0) {
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    final_result_flat.resize(rows_a * cols_b);
    MPI_Bcast(final_result_flat.data(), rows_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // 11. Преобразуем в матрицу
  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b));
  for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      result_matrix[i][j] = final_result_flat[i * cols_b + j];
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
