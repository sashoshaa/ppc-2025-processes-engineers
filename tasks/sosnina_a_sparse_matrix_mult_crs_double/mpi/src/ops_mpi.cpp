#include "sosnina_a_sparse_matrix_mult_crs_double/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

SosninaAMatrixMultCRSMPI::SosninaAMatrixMultCRSMPI(const InType &in)
    : values_A_(std::get<0>(in)),
      col_indices_A_(std::get<1>(in)),
      row_ptr_A_(std::get<2>(in)),
      n_rows_A_(std::get<6>(in)),
      n_cols_A_(std::get<7>(in)),
      values_B_(std::get<3>(in)),
      col_indices_B_(std::get<4>(in)),
      row_ptr_B_(std::get<5>(in)),
      n_cols_B_(std::get<8>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool SosninaAMatrixMultCRSMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized == 0) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size >= 1;
}

bool SosninaAMatrixMultCRSMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  rank_ = rank;
  world_size_ = size;

  local_rows_.clear();
  local_values_A_.clear();
  local_col_indices_A_.clear();
  local_row_ptr_A_.clear();
  local_values_C_.clear();
  local_col_indices_C_.clear();
  local_row_ptr_C_.clear();

  values_C_.clear();
  col_indices_C_.clear();
  row_ptr_C_.clear();

  return true;
}

bool SosninaAMatrixMultCRSMPI::RunImpl() {
  if (world_size_ == 1) {
    return RunSequential();
  }

  int n_rows_a = 0;
  int n_cols_a = 0;
  int n_cols_b = 0;

  if (!PrepareAndValidateSizes(n_rows_a, n_cols_a, n_cols_b)) {
    return true;
  }

  BroadcastMatrixB();

  DistributeMatrixAData();

  ComputeLocalMultiplication();

  GatherResults();

  return true;
}

bool SosninaAMatrixMultCRSMPI::RunSequential() {
  if (rank_ != 0) {
    return true;
  }

  row_ptr_C_.resize(n_rows_A_ + 1, 0);
  row_ptr_C_[0] = 0;

  std::vector<std::vector<double>> row_values(n_rows_A_);
  std::vector<std::vector<int>> row_cols(n_rows_A_);

  for (int i = 0; i < n_rows_A_; i++) {
    ProcessRowForSequential(i, row_values[i], row_cols[i]);
    row_ptr_C_[i + 1] = row_ptr_C_[i] + static_cast<int>(row_cols[i].size());
  }

  for (int i = 0; i < n_rows_A_; i++) {
    values_C_.insert(values_C_.end(), row_values[i].begin(), row_values[i].end());
    col_indices_C_.insert(col_indices_C_.end(), row_cols[i].begin(), row_cols[i].end());
  }

  OutType result = std::make_tuple(values_C_, col_indices_C_, row_ptr_C_);
  GetOutput() = result;

  return true;
}

void SosninaAMatrixMultCRSMPI::ProcessRowForSequential(int row_idx, std::vector<double> &row_values,
                                                       std::vector<int> &row_cols) {
  int row_start_a = row_ptr_A_[row_idx];
  int row_end_a = row_ptr_A_[row_idx + 1];

  std::vector<double> temp_row(n_cols_B_, 0.0);

  for (int k_idx = row_start_a; k_idx < row_end_a; k_idx++) {
    double a_val = values_A_[k_idx];
    int k = col_indices_A_[k_idx];

    int row_start_b = row_ptr_B_[k];
    int row_end_b = row_ptr_B_[k + 1];

    for (int j_idx = row_start_b; j_idx < row_end_b; j_idx++) {
      double b_val = values_B_[j_idx];
      int j = col_indices_B_[j_idx];

      temp_row[j] += a_val * b_val;
    }
  }

  for (int j = 0; j < n_cols_B_; j++) {
    if (std::abs(temp_row[j]) > 1e-12) {
      row_values.push_back(temp_row[j]);
      row_cols.push_back(j);
    }
  }

  if (!row_cols.empty()) {
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(row_cols.size());
    for (size_t idx = 0; idx < row_cols.size(); idx++) {
      pairs.emplace_back(row_cols[idx], row_values[idx]);
    }

    std::ranges::sort(pairs);
    static_cast<void>(std::ranges::begin(pairs));

    for (size_t idx = 0; idx < pairs.size(); idx++) {
      row_cols[idx] = pairs[idx].first;
      row_values[idx] = pairs[idx].second;
    }
  }
}

bool SosninaAMatrixMultCRSMPI::PrepareAndValidateSizes(int &n_rows_a, int &n_cols_a, int &n_cols_b) {
  if (rank_ == 0) {
    n_rows_a = n_rows_A_;
    n_cols_a = n_cols_A_;
    n_cols_b = n_cols_B_;
  }

  std::array<int, 3> sizes = {n_rows_a, n_cols_a, n_cols_b};
  MPI_Bcast(sizes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  n_rows_a = sizes[0];
  n_cols_a = sizes[1];
  n_cols_b = sizes[2];

  n_rows_A_ = n_rows_a;
  n_cols_A_ = n_cols_a;
  n_cols_B_ = n_cols_b;

  return n_rows_a > 0 && n_cols_a > 0 && n_cols_b > 0;
}

void SosninaAMatrixMultCRSMPI::BroadcastMatrixB() {
  std::array<int, 3> b_sizes = {0, 0, 0};

  if (rank_ == 0) {
    b_sizes[0] = static_cast<int>(values_B_.size());
    b_sizes[1] = static_cast<int>(col_indices_B_.size());
    b_sizes[2] = static_cast<int>(row_ptr_B_.size());
  }

  MPI_Bcast(b_sizes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  int values_size = b_sizes[0];
  int indices_size = b_sizes[1];
  int row_ptr_size = b_sizes[2];

  if (rank_ != 0) {
    values_B_.resize(values_size);
    col_indices_B_.resize(indices_size);
    row_ptr_B_.resize(row_ptr_size);
  }

  MPI_Bcast(values_B_.data(), values_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(col_indices_B_.data(), indices_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(row_ptr_B_.data(), row_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void SosninaAMatrixMultCRSMPI::DistributeMatrixAData() {
  local_rows_.clear();
  for (int i = 0; i < n_rows_A_; ++i) {
    if (i % world_size_ == rank_) {
      local_rows_.push_back(i);
    }
  }

  if (rank_ == 0) {
    for (int dest = 1; dest < world_size_; ++dest) {
      SendMatrixADataToProcess(dest);
    }

    local_values_A_.clear();
    local_col_indices_A_.clear();
    local_row_ptr_A_.resize(local_rows_.size() + 1, 0);

    for (size_t idx = 0; idx < local_rows_.size(); ++idx) {
      int row = local_rows_[idx];
      int row_start = row_ptr_A_[row];
      int row_end = row_ptr_A_[row + 1];
      int row_nnz = row_end - row_start;

      local_values_A_.insert(local_values_A_.end(), values_A_.begin() + row_start, values_A_.begin() + row_end);

      local_col_indices_A_.insert(local_col_indices_A_.end(), col_indices_A_.begin() + row_start,
                                  col_indices_A_.begin() + row_end);

      local_row_ptr_A_[idx + 1] = local_row_ptr_A_[idx] + row_nnz;
    }
  } else {
    ReceiveMatrixAData();
  }
}

void SosninaAMatrixMultCRSMPI::SendMatrixADataToProcess(int dest) {
  std::vector<int> dest_rows;
  for (int i = 0; i < n_rows_A_; ++i) {
    if (i % world_size_ == dest) {
      dest_rows.push_back(i);
    }
  }

  int dest_row_count = static_cast<int>(dest_rows.size());
  MPI_Send(&dest_row_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

  if (dest_row_count > 0) {
    MPI_Send(dest_rows.data(), dest_row_count, MPI_INT, dest, 1, MPI_COMM_WORLD);

    for (int row : dest_rows) {
      int row_start = row_ptr_A_[row];
      int row_end = row_ptr_A_[row + 1];
      int row_nnz = row_end - row_start;

      MPI_Send(&row_nnz, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);

      if (row_nnz > 0) {
        std::vector<double> row_values(values_A_.begin() + row_start, values_A_.begin() + row_end);
        MPI_Send(row_values.data(), row_nnz, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);

        std::vector<int> row_cols(col_indices_A_.begin() + row_start, col_indices_A_.begin() + row_end);
        MPI_Send(row_cols.data(), row_nnz, MPI_INT, dest, 4, MPI_COMM_WORLD);
      }
    }
  }
}

void SosninaAMatrixMultCRSMPI::ReceiveMatrixAData() {
  int local_row_count = 0;
  MPI_Recv(&local_row_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (local_row_count > 0) {
    local_rows_.resize(local_row_count);
    MPI_Recv(local_rows_.data(), local_row_count, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_values_A_.clear();
    local_col_indices_A_.clear();
    local_row_ptr_A_.resize(local_row_count + 1, 0);

    for (int i = 0; i < local_row_count; ++i) {
      int row_nnz = 0;
      MPI_Recv(&row_nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (row_nnz > 0) {
        std::vector<double> row_values(row_nnz);
        MPI_Recv(row_values.data(), row_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> row_cols(row_nnz);
        MPI_Recv(row_cols.data(), row_nnz, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_values_A_.insert(local_values_A_.end(), row_values.begin(), row_values.end());
        local_col_indices_A_.insert(local_col_indices_A_.end(), row_cols.begin(), row_cols.end());
      }

      local_row_ptr_A_[i + 1] = local_row_ptr_A_[i] + row_nnz;
    }
  }
}

void SosninaAMatrixMultCRSMPI::ComputeLocalMultiplication() {
  int local_row_count = static_cast<int>(local_rows_.size());

  std::vector<std::vector<double>> local_row_values(local_row_count);
  std::vector<std::vector<int>> local_row_cols(local_row_count);
  local_row_ptr_C_.resize(local_row_count + 1, 0);

  for (int local_idx = 0; local_idx < local_row_count; ++local_idx) {
    ProcessLocalRow(local_idx, local_row_values[local_idx], local_row_cols[local_idx]);
    local_row_ptr_C_[local_idx + 1] = local_row_ptr_C_[local_idx] + static_cast<int>(local_row_cols[local_idx].size());
  }

  local_values_C_.clear();
  local_col_indices_C_.clear();

  for (int i = 0; i < local_row_count; ++i) {
    local_values_C_.insert(local_values_C_.end(), local_row_values[i].begin(), local_row_values[i].end());
    local_col_indices_C_.insert(local_col_indices_C_.end(), local_row_cols[i].begin(), local_row_cols[i].end());
  }
}

void SosninaAMatrixMultCRSMPI::ProcessLocalRow(int local_idx, std::vector<double> &row_values,
                                               std::vector<int> &row_cols) {
  int row_start = local_row_ptr_A_[local_idx];
  int row_end = local_row_ptr_A_[local_idx + 1];

  std::vector<double> temp_row(n_cols_B_, 0.0);

  MultiplyRowByMatrixB(row_start, row_end, temp_row);

  CollectNonZeroElements(temp_row, n_cols_B_, row_values, row_cols);

  SortRowElements(row_values, row_cols);
}

void SosninaAMatrixMultCRSMPI::MultiplyRowByMatrixB(int row_start, int row_end, std::vector<double> &temp_row) {
  for (int k_idx = row_start; k_idx < row_end; ++k_idx) {
    ProcessElementA(k_idx, temp_row);
  }
}

void SosninaAMatrixMultCRSMPI::ProcessElementA(int k_idx, std::vector<double> &temp_row) {
  if (k_idx < 0 || static_cast<size_t>(k_idx) >= local_values_A_.size() ||
      static_cast<size_t>(k_idx) >= local_col_indices_A_.size()) {
    return;
  }

  double a_val = local_values_A_[k_idx];
  int k = local_col_indices_A_[k_idx];

  if (k < 0 || k >= n_cols_A_ || k >= static_cast<int>(row_ptr_B_.size()) - 1) {
    return;
  }

  MultiplyByRowB(k, a_val, temp_row);
}

void SosninaAMatrixMultCRSMPI::MultiplyByRowB(int k, double a_val, std::vector<double> &temp_row) {
  int b_row_start = row_ptr_B_[k];
  int b_row_end = row_ptr_B_[k + 1];

  if (b_row_start < 0 || static_cast<size_t>(b_row_end) > values_B_.size() || b_row_start > b_row_end) {
    return;
  }
  for (int j_idx = b_row_start; j_idx < b_row_end; ++j_idx) {
    if (j_idx < 0 || static_cast<size_t>(j_idx) >= values_B_.size() ||
        static_cast<size_t>(j_idx) >= col_indices_B_.size()) {
      continue;
    }

    double b_val = values_B_[j_idx];
    int j = col_indices_B_[j_idx];

    if (j >= 0 && j < n_cols_B_) {
      temp_row[j] += a_val * b_val;
    }
  }
}

void SosninaAMatrixMultCRSMPI::CollectNonZeroElements(const std::vector<double> &temp_row, int n_cols_b,
                                                      std::vector<double> &row_values, std::vector<int> &row_cols) {
  for (int j = 0; j < n_cols_b; ++j) {
    if (std::abs(temp_row[j]) > 1e-12) {
      row_values.push_back(temp_row[j]);
      row_cols.push_back(j);
    }
  }
}

void SosninaAMatrixMultCRSMPI::SortRowElements(std::vector<double> &row_values, std::vector<int> &row_cols) {
  if (!row_cols.empty()) {
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(row_cols.size());
    for (size_t idx = 0; idx < row_cols.size(); ++idx) {
      pairs.emplace_back(row_cols[idx], row_values[idx]);
    }

    std::ranges::sort(pairs);
    static_cast<void>(std::ranges::begin(pairs));

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
      row_cols[idx] = pairs[idx].first;
      row_values[idx] = pairs[idx].second;
    }
  }
}

void SosninaAMatrixMultCRSMPI::GatherResults() {
  if (rank_ == 0) {
    std::vector<std::vector<double>> row_values(n_rows_A_);
    std::vector<std::vector<int>> row_cols(n_rows_A_);

    ProcessLocalResults(row_values, row_cols);

    for (int src = 1; src < world_size_; ++src) {
      ReceiveResultsFromProcess(src, row_values, row_cols);
    }

    CollectAllResults(row_values, row_cols);
  } else {
    int local_row_count = static_cast<int>(local_rows_.size());

    MPI_Send(&local_row_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    if (local_row_count > 0) {
      std::vector<int> rows_to_send = local_rows_;
      MPI_Send(rows_to_send.data(), local_row_count, MPI_INT, 0, 1, MPI_COMM_WORLD);

      std::vector<int> row_ptr_to_send = local_row_ptr_C_;
      MPI_Send(row_ptr_to_send.data(), local_row_count + 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

      int total_nnz = static_cast<int>(local_values_C_.size());

      if (total_nnz > 0) {
        std::vector<double> values_to_send = local_values_C_;
        std::vector<int> indices_to_send = local_col_indices_C_;

        MPI_Send(values_to_send.data(), total_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        MPI_Send(indices_to_send.data(), total_nnz, MPI_INT, 0, 4, MPI_COMM_WORLD);
      }
    }
  }

  if (rank_ == 0) {
    OutType result = std::make_tuple(values_C_, col_indices_C_, row_ptr_C_);
    GetOutput() = result;
  } else {
    OutType empty_result = std::make_tuple(std::vector<double>(), std::vector<int>(), std::vector<int>());
    GetOutput() = empty_result;
  }
}

void SosninaAMatrixMultCRSMPI::ProcessLocalResults(std::vector<std::vector<double>> &row_values,
                                                   std::vector<std::vector<int>> &row_cols) {
  for (size_t i = 0; i < local_rows_.size(); ++i) {
    int global_row = local_rows_[i];
    int local_start = local_row_ptr_C_[i];
    int local_end = local_row_ptr_C_[i + 1];

    for (int j = local_start; j < local_end; ++j) {
      row_values[global_row].push_back(local_values_C_[j]);
      row_cols[global_row].push_back(local_col_indices_C_[j]);
    }
  }
}

void SosninaAMatrixMultCRSMPI::ReceiveResultsFromProcess(int src, std::vector<std::vector<double>> &row_values,
                                                         std::vector<std::vector<int>> &row_cols) {
  int received_row_count = 0;
  MPI_Recv(&received_row_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (received_row_count > 0) {
    std::vector<int> received_rows(received_row_count);
    MPI_Recv(received_rows.data(), received_row_count, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<int> src_local_row_ptr(received_row_count + 1);
    MPI_Recv(src_local_row_ptr.data(), received_row_count + 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int src_total_nnz = src_local_row_ptr[received_row_count];

    std::vector<double> src_values(src_total_nnz);
    std::vector<int> src_col_indices(src_total_nnz);

    if (src_total_nnz > 0) {
      MPI_Recv(src_values.data(), src_total_nnz, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(src_col_indices.data(), src_total_nnz, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < received_row_count; ++i) {
      int global_row = received_rows[i];
      int src_start = src_local_row_ptr[i];
      int src_end = src_local_row_ptr[i + 1];

      for (int j = src_start; j < src_end; ++j) {
        row_values[global_row].push_back(src_values[j]);
        row_cols[global_row].push_back(src_col_indices[j]);
      }
    }
  }
}

void SosninaAMatrixMultCRSMPI::CollectAllResults(std::vector<std::vector<double>> &row_values,
                                                 std::vector<std::vector<int>> &row_cols) {
  values_C_.clear();
  col_indices_C_.clear();
  row_ptr_C_.resize(n_rows_A_ + 1, 0);
  row_ptr_C_[0] = 0;

  for (int i = 0; i < n_rows_A_; ++i) {
    SortAndPackRow(i, row_values, row_cols);
    values_C_.insert(values_C_.end(), row_values[i].begin(), row_values[i].end());
    col_indices_C_.insert(col_indices_C_.end(), row_cols[i].begin(), row_cols[i].end());
    row_ptr_C_[i + 1] = row_ptr_C_[i] + static_cast<int>(row_values[i].size());
  }
}

void SosninaAMatrixMultCRSMPI::SortAndPackRow(int row_idx, std::vector<std::vector<double>> &row_values,
                                              std::vector<std::vector<int>> &row_cols) {
  if (!row_cols[row_idx].empty()) {
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(row_cols[row_idx].size());
    for (size_t idx = 0; idx < row_cols[row_idx].size(); ++idx) {
      pairs.emplace_back(row_cols[row_idx][idx], row_values[row_idx][idx]);
    }
    std::ranges::sort(pairs);
    static_cast<void>(std::ranges::begin(pairs));

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
      row_cols[row_idx][idx] = pairs[idx].first;
      row_values[row_idx][idx] = pairs[idx].second;
    }
  }
}

bool SosninaAMatrixMultCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
