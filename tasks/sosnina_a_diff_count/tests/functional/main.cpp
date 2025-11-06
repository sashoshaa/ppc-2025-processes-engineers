#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <algorithm>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string combined = std::get<1>(test_param);
    std::replace(combined.begin(), combined.end(), ' ', '_');
    return std::to_string(std::get<0>(test_param)) + "_" + combined;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string combined = std::get<1>(params);
    auto pos = combined.find('_');
    str1_ = combined.substr(0, pos);
    str2_ = combined.substr(pos + 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (size_t i = 0; i < str1_.size() && i < str2_.size(); i++) {
      if (str1_[i] != str2_[i]) {
        expected++;
      }
    }
    expected += static_cast<int>(std::max(str1_.size(), str2_.size()) - std::min(str1_.size(), str2_.size()));
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return std::make_pair(str1_, str2_);
  }

 private:
  std::string str1_;
  std::string str2_;
};

namespace {

// Functional Tests
TEST_P(SosninaADiffCountFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

// Coverage Tests
TEST_P(SosninaADiffCountFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 19> kFunctionalTests = {
    std::make_tuple(1, "happy_heppy"),
    std::make_tuple(2, "abcdef_abzzef"),
    std::make_tuple(3, "baby_baby"),
    std::make_tuple(4, "abc_defgh"),
    std::make_tuple(5, "_mpi"),
    std::make_tuple(6, "longstring_longstring"),
    std::make_tuple(7, "abcd_efgh"),
    std::make_tuple(8, "__"),
    std::make_tuple(9, "z_"),
    std::make_tuple(10, "_v"),
    std::make_tuple(11, "z_v"),
    std::make_tuple(12, "z_z"),
    std::make_tuple(13, "zv_vz"),
    std::make_tuple(14, "prizet_PRIZET"),
    std::make_tuple(15, "zzz_vvv"),
    std::make_tuple(16, "54321_09876"),
    std::make_tuple(17, "TEST_TEST"),
    std::make_tuple(18, "z_v_z_v"),
    std::make_tuple(19, "veryvery_long_string_one_veryvery_long_string_two")};

const std::array<TestType, 5> kCoverageTests = {
    std::make_tuple(20, "__"), std::make_tuple(21, "short_very_long_string"), std::make_tuple(22, "a_bbb"),
    std::make_tuple(23, "hello_hxllo"), std::make_tuple(24, "test_text")};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountMPI, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_diff_count),
                   ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountSEQ, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_diff_count));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountMPI, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_diff_count),
                   ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountSEQ, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_diff_count));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

inline const auto kPerfTestName = SosninaADiffCountFuncTests::PrintFuncTestName<SosninaADiffCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, SosninaADiffCountFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, SosninaADiffCountFuncTests, kCoverageGtestValues, kPerfTestName);

}  // namespace

}  // namespace sosnina_a_diff_count
