#pragma once

#include <string>
#include <utility>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_diff_count {

using InTypePair = std::pair<std::string, std::string>;  // входная пара строк

class SosninaADiffCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SosninaADiffCountSEQ(const InTypePair &in);

  int GetDiffCount() const;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  InTypePair input_;
  int diff_counter = 0;
};

}  // namespace sosnina_a_diff_count
