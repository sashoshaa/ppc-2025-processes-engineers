#pragma once

#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

class SosninaAMatrixMultCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SosninaAMatrixMultCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool RunSequential();

  bool PrepareAndValidateSizes(int &n_rows_a, int &n_cols_a, int &n_cols_b);

  void BroadcastMatrixB();

  void DistributeMatrixAData();
  void SendMatrixADataToProcess(int dest);
  void ReceiveMatrixAData();

  void ComputeLocalMultiplication();
  void ProcessRowForSequential(int row_idx, std::vector<double> &row_values, std::vector<int> &row_cols);
  void ProcessLocalRow(int local_idx, std::vector<double> &row_values, std::vector<int> &row_cols);
  void MultiplyRowByMatrixB(int row_start, int row_end, std::vector<double> &temp_row);
  void ProcessElementA(int k_idx, std::vector<double> &temp_row);
  void MultiplyByRowB(int k, double a_val, std::vector<double> &temp_row);
  static void CollectNonZeroElements(const std::vector<double> &temp_row, int n_cols_b, std::vector<double> &row_values,
                                     std::vector<int> &row_cols);
  static void SortRowElements(std::vector<double> &row_values, std::vector<int> &row_cols);

  void GatherResults();
  void ProcessLocalResults(std::vector<std::vector<double>> &row_values, std::vector<std::vector<int>> &row_cols);
  static void ReceiveResultsFromProcess(int src, std::vector<std::vector<double>> &row_values,
                                        std::vector<std::vector<int>> &row_cols);
  void CollectAllResults(std::vector<std::vector<double>> &row_values, std::vector<std::vector<int>> &row_cols);
  static void SortAndPackRow(int row_idx, std::vector<std::vector<double>> &row_values,
                             std::vector<std::vector<int>> &row_cols);

  std::vector<double> values_A_;
  std::vector<int> col_indices_A_;
  std::vector<int> row_ptr_A_;
  int n_rows_A_;
  int n_cols_A_;

  std::vector<double> values_B_;
  std::vector<int> col_indices_B_;
  std::vector<int> row_ptr_B_;
  int n_cols_B_;

  std::vector<double> values_C_;
  std::vector<int> col_indices_C_;
  std::vector<int> row_ptr_C_;

  std::vector<int> local_rows_;
  std::vector<double> local_values_A_;
  std::vector<int> local_col_indices_A_;
  std::vector<int> local_row_ptr_A_;
  std::vector<double> local_values_C_;
  std::vector<int> local_col_indices_C_;
  std::vector<int> local_row_ptr_C_;

  int rank_ = 0;
  int world_size_ = 1;
};

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
