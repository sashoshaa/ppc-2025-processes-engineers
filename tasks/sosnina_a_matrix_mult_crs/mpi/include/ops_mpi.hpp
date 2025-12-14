#pragma once

#include <tuple>
#include <vector>

#include "sosnina_a_matrix_mult_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_matrix_mult_crs {

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

  bool PrepareAndValidateSizes(int &n_rows_A, int &n_cols_A, int &n_cols_B);

  void BroadcastMatrixB();

  void DistributeMatrixAData();

  void ComputeLocalMultiplication();

  void GatherResults();

  // Вспомогательные функции для работы с CRS
  void ConvertToDenseRow(const std::vector<double> &values, const std::vector<int> &col_indices,
                         const std::vector<int> &row_ptr, int row_index, int n_cols, std::vector<double> &dense_row);

  void ConvertDenseToCRS(const std::vector<double> &dense_row, std::vector<double> &values,
                         std::vector<int> &col_indices, int &row_start, int row_end, double epsilon = 1e-12);

  // Данные в формате CRS
  std::vector<double> values_A_;
  std::vector<int> col_indices_A_;
  std::vector<int> row_ptr_A_;
  int n_rows_A_;
  int n_cols_A_;

  std::vector<double> values_B_;
  std::vector<int> col_indices_B_;
  std::vector<int> row_ptr_B_;
  int n_cols_B_;

  // Результат
  std::vector<double> values_C_;
  std::vector<int> col_indices_C_;
  std::vector<int> row_ptr_C_;

  // Локальные данные для процесса
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

}  // namespace sosnina_a_matrix_mult_crs
