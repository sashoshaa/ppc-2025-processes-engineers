#pragma once

#include <mpi.h>

#include <algorithm>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SosninaADiffCountMPI(const InType &in);

  int GetDiffCount() const;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string str1_;
  std::string str2_;
  int diff_counter_ = 0;
};

}  // namespace sosnina_a_diff_count
