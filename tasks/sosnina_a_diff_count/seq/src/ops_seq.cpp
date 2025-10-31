#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sosnina_a_diff_count {

SosninaADiffCountSEQ::SosninaADiffCountSEQ(const InTypePair &in) : input_(in), diff_counter(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = 0;
}

bool SosninaADiffCountSEQ::ValidationImpl() {
  return true;
}

bool SosninaADiffCountSEQ::PreProcessingImpl() {
  diff_counter = 0;
  return true;
}

bool SosninaADiffCountSEQ::RunImpl() {
  const std::string &str1 = input_.first;
  const std::string &str2 = input_.second;

  size_t min_len = std::min(str1.size(), str2.size());
  diff_counter = 0;

  for (size_t i = 0; i < min_len; i++) {
    if (str1[i] != str2[i]) {
      diff_counter++;
    }
  }

  diff_counter += static_cast<int>(std::max(str1.size(), str2.size()) - min_len);
  return true;
}

bool SosninaADiffCountSEQ::PostProcessingImpl() {
  GetOutput() = diff_counter;
  return true;
}

int SosninaADiffCountSEQ::GetDiffCount() const {
  return diff_counter;
}

}  // namespace sosnina_a_diff_count
