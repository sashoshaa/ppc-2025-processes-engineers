#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 200000000;

 protected:
  void SetUp() override {
    str1 = std::string(kSize, 'z');
    str2 = std::string(kSize, 'v');

    expected_res_ = static_cast<int>(kSize);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_res_ == output_data;
  }

  InType GetTestInputData() final {
    return std::make_pair(str1, str2);
  }

 private:
  std::string str1;
  std::string str2;
  OutType expected_res_{};
};

TEST_P(SosninaADiffCountRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SosninaADiffCountMPI, SosninaADiffCountSEQ>(PPC_SETTINGS_sosnina_a_diff_count);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SosninaADiffCountRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SosninaADiffCountRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace sosnina_a_diff_count
