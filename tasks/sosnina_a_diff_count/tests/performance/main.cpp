#include <gtest/gtest.h>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SosninaADiffCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SosninaADiffCountMPI, SosninaADiffCountSEQ>(PPC_SETTINGS_sosnina_a_diff_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SosninaADiffCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SosninaADiffCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace sosnina_a_diff_count
