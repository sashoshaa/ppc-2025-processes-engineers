#pragma once

#include <cstddef>
#include <vector>

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_matrix_mult_horizontal {

class SosninaAMatrixMultHorizontalMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SosninaAMatrixMultHorizontalMPI(const InType &in);

 private:
  std::vector<std::vector<double>> MultiplyLocalPart(const std::vector<std::vector<double>> &local_A, 
                                                    const std::vector<std::vector<double>> &matrix_B);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<std::vector<double>> matrix_A_;
  std::vector<std::vector<double>> matrix_B_;
  std::vector<std::vector<double>> result_matrix_;
  int rank_ = 0;
  int world_size_ = 1;
};

}  // namespace sosnina_a_matrix_mult_horizontal