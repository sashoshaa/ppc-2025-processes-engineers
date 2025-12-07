#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"
#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"
#include "sosnina_a_matrix_mult_horizontal/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sosnina_a_matrix_mult_horizontal {

class SosninaAMatrixMultHorizontalRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 800;  // Большая матрица 1000x1000

 protected:
  void SetUp() override {
    matrixA_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));
    matrixB_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        matrixA_[i][j] = (i * kSize + j) * 0.001;
        matrixB_[i][j] = (i + j) * 0.002;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return std::make_pair(matrixA_, matrixB_);
  }

 private:
  std::vector<std::vector<double>> matrixA_;
  std::vector<std::vector<double>> matrixB_;
};

TEST_P(SosninaAMatrixMultHorizontalRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SosninaAMatrixMultHorizontalMPI, SosninaAMatrixMultHorizontalSEQ>(
        PPC_SETTINGS_sosnina_a_matrix_mult_horizontal);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SosninaAMatrixMultHorizontalRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SosninaAMatrixMultHorizontalRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace sosnina_a_matrix_mult_horizontal
