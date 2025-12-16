#pragma once

#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

class SosninaAMatrixMultCRSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SosninaAMatrixMultCRSSEQ(InType in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool ValidateMatrixA() const;
  bool ValidateMatrixB() const;
  void ProcessRow(int row_idx, std::vector<double> &row_values, std::vector<int> &row_cols);

  InType input_;

  // Данные матрицы A
  std::vector<double> values_A_;
  std::vector<int> col_indices_A_;
  std::vector<int> row_ptr_A_;
  int n_rows_A_;
  int n_cols_A_;

  // Данные матрицы B
  std::vector<double> values_B_;
  std::vector<int> col_indices_B_;
  std::vector<int> row_ptr_B_;
  int n_cols_B_;

  // Результат (матрица C = A * B)
  std::vector<double> values_C_;
  std::vector<int> col_indices_C_;
  std::vector<int> row_ptr_C_;
};

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
