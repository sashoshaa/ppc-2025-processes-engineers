#include "sosnina_a_sparse_matrix_mult_crs_double/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

SosninaAMatrixMultCRSSEQ::SosninaAMatrixMultCRSSEQ(InType in)
    : input_(std::move(in)),
      n_rows_A_(std::get<6>(input_)),
      n_cols_A_(std::get<7>(input_)),
      n_cols_B_(std::get<8>(input_)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = OutType();
}

bool SosninaAMatrixMultCRSSEQ::ValidationImpl() {
  values_A_ = std::get<0>(input_);
  col_indices_A_ = std::get<1>(input_);
  row_ptr_A_ = std::get<2>(input_);
  values_B_ = std::get<3>(input_);
  col_indices_B_ = std::get<4>(input_);
  row_ptr_B_ = std::get<5>(input_);
  n_rows_A_ = std::get<6>(input_);
  n_cols_A_ = std::get<7>(input_);
  n_cols_B_ = std::get<8>(input_);

  if (n_rows_A_ <= 0 || n_cols_A_ <= 0 || n_cols_B_ <= 0) {
    return false;
  }

  if (!ValidateMatrixA()) {
    return false;
  }

  if (!ValidateMatrixB()) {
    return false;
  }

  return true;
}

bool SosninaAMatrixMultCRSSEQ::ValidateMatrixA() const {
  if (row_ptr_A_.size() != static_cast<size_t>(n_rows_A_) + 1U) {
    return false;
  }

  if (row_ptr_A_[0] != 0) {
    return false;
  }

  for (size_t i = 0; i < row_ptr_A_.size() - 1; i++) {
    if (row_ptr_A_[i] > row_ptr_A_[i + 1]) {
      return false;
    }
  }

  if (values_A_.size() != col_indices_A_.size()) {
    return false;
  }

  static_cast<void>(std::ranges::begin(col_indices_A_));
  return std::ranges::all_of(col_indices_A_,
                             [n_cols_a = n_cols_A_](int col_idx) { return col_idx >= 0 && col_idx < n_cols_a; });
}

bool SosninaAMatrixMultCRSSEQ::ValidateMatrixB() const {
  int n_rows_b = n_cols_A_;

  if (row_ptr_B_.size() != static_cast<size_t>(n_rows_b) + 1U) {
    return false;
  }

  if (row_ptr_B_[0] != 0) {
    return false;
  }

  for (size_t i = 0; i < row_ptr_B_.size() - 1; i++) {
    if (row_ptr_B_[i] > row_ptr_B_[i + 1]) {
      return false;
    }
  }

  if (values_B_.size() != col_indices_B_.size()) {
    return false;
  }
  static_cast<void>(std::ranges::begin(col_indices_B_));
  return std::ranges::all_of(col_indices_B_,
                             [n_cols_b = n_cols_B_](int col_idx) { return col_idx >= 0 && col_idx < n_cols_b; });
}

bool SosninaAMatrixMultCRSSEQ::PreProcessingImpl() {
  values_C_.clear();
  col_indices_C_.clear();
  row_ptr_C_.clear();

  return true;
}

bool SosninaAMatrixMultCRSSEQ::RunImpl() {
  row_ptr_C_.resize(n_rows_A_ + 1, 0);
  row_ptr_C_[0] = 0;

  std::vector<std::vector<double>> row_values(n_rows_A_);
  std::vector<std::vector<int>> row_cols(n_rows_A_);

  for (int i = 0; i < n_rows_A_; i++) {
    ProcessRow(i, row_values[i], row_cols[i]);
    row_ptr_C_[i + 1] = row_ptr_C_[i] + static_cast<int>(row_cols[i].size());
  }

  for (int i = 0; i < n_rows_A_; i++) {
    values_C_.insert(values_C_.end(), row_values[i].begin(), row_values[i].end());
    col_indices_C_.insert(col_indices_C_.end(), row_cols[i].begin(), row_cols[i].end());
  }

  return true;
}

void SosninaAMatrixMultCRSSEQ::ProcessRow(int row_idx, std::vector<double> &row_values, std::vector<int> &row_cols) {
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

bool SosninaAMatrixMultCRSSEQ::PostProcessingImpl() {
  OutType result = std::make_tuple(values_C_, col_indices_C_, row_ptr_C_);
  GetOutput() = result;

  return true;
}

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
