#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <utility>

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"

namespace sosnina_a_matrix_mult_horizontal {

SosninaAMatrixMultHorizontalMPI::SosninaAMatrixMultHorizontalMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::vector<std::vector<double>>();
  
  // Сохраняем входные данные
  matrix_A_ = in.first;
  matrix_B_ = in.second;
}

bool SosninaAMatrixMultHorizontalMPI::ValidationImpl() {
  if (matrix_A_.empty() || matrix_B_.empty()) return false;
  return matrix_A_[0].size() == matrix_B_.size();
}

bool SosninaAMatrixMultHorizontalMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  result_matrix_.clear();
  return true;
}

std::vector<std::vector<double>> SosninaAMatrixMultHorizontalMPI::MultiplyLocalPart(
    const std::vector<std::vector<double>> &local_A, 
    const std::vector<std::vector<double>> &matrix_B) {
  
  size_t local_rows = local_A.size();
  size_t colsA = local_A[0].size();
  size_t colsB = matrix_B[0].size();
  
  std::vector<std::vector<double>> local_result(local_rows, std::vector<double>(colsB, 0.0));
  
  for (size_t i = 0; i < local_rows; i++) {
    for (size_t j = 0; j < colsB; j++) {
      for (size_t k = 0; k < colsA; k++) {
        local_result[i][j] += local_A[i][k] * matrix_B[k][j];
      }
    }
  }
  
  return local_result;
}

bool SosninaAMatrixMultHorizontalMPI::RunImpl() {
  size_t rowsA = matrix_A_.size();
  size_t colsA = matrix_A_[0].size();
  size_t rowsB = matrix_B_.size();
  size_t colsB = matrix_B_[0].size();

  // Рассылаем размеры матрицы B всем процессам
  int dims[2] = {static_cast<int>(rowsB), static_cast<int>(colsB)};
  MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Процессы кроме 0 получают матрицу B
  if (rank_ != 0) {
    matrix_B_.resize(dims[0], std::vector<double>(dims[1]));
  }
  
  // Рассылаем матрицу B всем процессам
  for (size_t i = 0; i < rowsB; i++) {
    MPI_Bcast(matrix_B_[i].data(), colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Вычисление количества строк матрицы A для каждого процесса
  size_t rows_per_process = rowsA / world_size_;
  size_t remainder = rowsA % world_size_;
  
  // Определение диапазона строк для текущего процесса
  size_t start_row, local_rows;
  if (rank_ < remainder) {
    local_rows = rows_per_process + 1;
    start_row = rank_ * local_rows;
  } else {
    local_rows = rows_per_process;
    start_row = remainder * (rows_per_process + 1) + (rank_ - remainder) * rows_per_process;
  }

  // Процесс 0 рассылает части матрицы A
  if (rank_ == 0) {
    // Отправляем части матрицы A другим процессам
    for (int proc = 1; proc < world_size_; proc++) {
      size_t proc_start_row, proc_rows;
      if (proc < remainder) {
        proc_rows = rows_per_process + 1;
        proc_start_row = proc * proc_rows;
      } else {
        proc_rows = rows_per_process;
        proc_start_row = remainder * (rows_per_process + 1) + (proc - remainder) * rows_per_process;
      }
      
      // Отправляем количество строк
      int send_rows = static_cast<int>(proc_rows);
      MPI_Send(&send_rows, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
      
      // Отправляем каждую строку матрицы A
      for (size_t i = 0; i < proc_rows; i++) {
        MPI_Send(matrix_A_[proc_start_row + i].data(), colsA, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      }
    }
    
    // Локальная часть для процесса 0
    local_rows = (remainder > 0) ? rows_per_process + 1 : rows_per_process;
    start_row = 0;
  } else {
    // Получаем количество строк
    int recv_rows;
    MPI_Recv(&recv_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_rows = recv_rows;
    
    // Получаем строки матрицы A
    matrix_A_.resize(local_rows, std::vector<double>(colsA));
    for (size_t i = 0; i < local_rows; i++) {
      MPI_Recv(matrix_A_[i].data(), colsA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Локальное умножение
  std::vector<std::vector<double>> local_result = MultiplyLocalPart(matrix_A_, matrix_B_);

  // Сбор результатов в процессе 0
  if (rank_ == 0) {
    result_matrix_.resize(rowsA, std::vector<double>(colsB, 0.0));
    
    // Копируем локальный результат процесса 0
    for (size_t i = 0; i < local_rows; i++) {
      result_matrix_[i] = local_result[i];
    }
    
    // Получаем результаты от других процессов
    for (int proc = 1; proc < world_size_; proc++) {
      size_t proc_start_row, proc_rows;
      if (proc < remainder) {
        proc_rows = rows_per_process + 1;
        proc_start_row = proc * proc_rows;
      } else {
        proc_rows = rows_per_process;
        proc_start_row = remainder * (rows_per_process + 1) + (proc - remainder) * rows_per_process;
      }
      
      for (size_t i = 0; i < proc_rows; i++) {
        MPI_Recv(result_matrix_[proc_start_row + i].data(), colsB, MPI_DOUBLE,
                proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    // Отправляем локальные результаты процессу 0
    for (size_t i = 0; i < local_rows; i++) {
      MPI_Send(local_result[i].data(), colsB, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }
  
  return true;
}

bool SosninaAMatrixMultHorizontalMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    GetOutput() = result_matrix_;
  } else {
    GetOutput() = std::vector<std::vector<double>>();
  }
  return true;
}

}  // namespace sosnina_a_matrix_mult_horizontal