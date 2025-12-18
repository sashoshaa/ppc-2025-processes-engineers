#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "sosnina_a_sparse_matrix_mult_crs_double/mpi/include/ops_mpi.hpp"
#include "sosnina_a_sparse_matrix_mult_crs_double/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

class SosninaAMatrixMultCRSRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kSize = 50000;

 protected:
  void SetUp() override {
    CreateDiagonalMatrix(values_A_, col_indices_A_, row_ptr_A_, kSize, kSize, 1.0);
    CreateDiagonalMatrix(values_B_, col_indices_B_, row_ptr_B_, kSize, kSize, 2.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto &[values, col_indices, row_ptr] = output_data;

    if (row_ptr.empty()) {
      return true;
    }

    if (row_ptr[0] != 0) {
      return false;
    }

    if (values.size() != col_indices.size()) {
      return false;
    }

    for (size_t i = 0; i < row_ptr.size() - 1; i++) {
      if (row_ptr[i] > row_ptr[i + 1]) {
        return false;
      }
    }

    if (row_ptr.empty()) {
      return false;
    }

    if (!col_indices.empty()) {
      for (size_t i = 0; i < col_indices.size(); i++) {
        if (col_indices[i] < 0 || col_indices[i] >= kSize) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return std::make_tuple(values_A_, col_indices_A_, row_ptr_A_, values_B_, col_indices_B_, row_ptr_B_, kSize, kSize,
                           kSize);
  }

 private:
  static void CreateDiagonalMatrix(std::vector<double> &values, std::vector<int> &col_indices,
                                   std::vector<int> &row_ptr, int n_rows, int n_cols, double diag_value) {
    values.clear();
    col_indices.clear();
    row_ptr.clear();
    row_ptr.push_back(0);

    for (int i = 0; i < n_rows; i++) {
      values.push_back(diag_value);
      col_indices.push_back(i);

      if (i % 10 == 0 && i + 1 < n_cols) {
        values.push_back(0.5);
        col_indices.push_back(i + 1);
      }

      row_ptr.push_back(static_cast<int>(values.size()));
    }
  }

  static void AddOffDiagonalElements(std::vector<double> &values, std::vector<int> &col_indices,
                                     std::vector<int> &row_ptr, int n, int count) {
    for (int k = 0; k < count; k++) {
      int i = (k * 13) % n;
      int j = (k * 17) % n;

      if (i != j) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        int insert_pos = row_start;
        while (insert_pos < row_end && col_indices[insert_pos] < j) {
          insert_pos++;
        }

        values.insert(values.begin() + insert_pos, 0.3);
        col_indices.insert(col_indices.begin() + insert_pos, j);

        for (size_t idx = i + 1; idx < row_ptr.size(); idx++) {
          row_ptr[idx]++;
        }
      }
    }
  }

  std::vector<double> values_A_;
  std::vector<int> col_indices_A_;
  std::vector<int> row_ptr_A_;

  std::vector<double> values_B_;
  std::vector<int> col_indices_B_;
  std::vector<int> row_ptr_B_;
};

TEST_P(SosninaAMatrixMultCRSRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SosninaAMatrixMultCRSMPI, SosninaAMatrixMultCRSSEQ>(
    PPC_SETTINGS_sosnina_a_sparse_matrix_mult_crs_double);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SosninaAMatrixMultCRSRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SosninaAMatrixMultCRSRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
