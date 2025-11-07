#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"

namespace sosnina_a_diff_count {

static int CalcOfDiff(const std::string &s1, const std::string &s2) {
  int diff_count = 0;
  size_t len = std::max(s1.size(), s2.size());
  for (size_t i = 0; i < len; i++) {
    char c1 = i < s1.size() ? s1[i] : '\0';
    char c2 = i < s2.size() ? s2[i] : '\0';
    if (c1 != c2) {
      diff_count++;
    }
  }
  return diff_count;
}

class SosninaADiffCountPerfTests : public ::testing::TestWithParam<std::tuple<std::string, std::string, size_t>> {
  protected:
   void SetUp() override {
     auto params = GetParam();
     str1_pattern = std::get<0>(params);
     str2_pattern = std::get<1>(params);
     str_size = std::get<2>(params);
 
     str1 = std::string(str_size, str1_pattern[0]);  
     str2 = std::string(str_size, str2_pattern[0]);
   }
 
   std::string str1;         
   std::string str2;         
   std::string str1_pattern; 
   std::string str2_pattern; 
   size_t str_size = 0;      
 };

static std::string PrintTestParam(const testing::TestParamInfo<SosninaADiffCountPerfTests::ParamType> &info) {
  std::string str1 = std::get<0>(info.param);
  std::string str2 = std::get<1>(info.param);
  size_t size = std::get<2>(info.param);

  for (char& c : str1) if (c == ' ') c = '_';
  for (char& c : str2) if (c == ' ') c = '_';

  return str1 + "_vs_" + str2 + "_size_" + std::to_string(size);
}

TEST_P(SosninaADiffCountPerfTests, TestPipelineRun) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int expected = CalcOfDiff(str1, str2);

  // mpi
  InType mpi_input = std::make_pair(str1, str2);
  SosninaADiffCountMPI mpi_task(mpi_input);
  mpi_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  auto start_mpi = std::chrono::high_resolution_clock::now();
  bool mpi_success = mpi_task.Validation() && mpi_task.PreProcessing() && mpi_task.Run() && mpi_task.PostProcessing();
  auto end_mpi = std::chrono::high_resolution_clock::now();
  double mpi_time = std::chrono::duration<double>(end_mpi - start_mpi).count();
  ASSERT_TRUE(mpi_success) << "mpi pipeline failed";
  ASSERT_EQ(mpi_task.GetOutput(), expected) << "mpi pipeline result incorrect";

  // seq
  InType seq_input = std::make_pair(str1, str2);
  SosninaADiffCountSEQ seq_task(seq_input);
  seq_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_success = seq_task.Validation() && seq_task.PreProcessing() && seq_task.Run() && seq_task.PostProcessing();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  ASSERT_TRUE(seq_success) << "seq pipeline failed";
  ASSERT_EQ(seq_task.GetOutput(), expected) << "seq pipeline result incorrect";

  if (rank == 0) {
    std::cout << "sosnina_a_diff_count_seq_enabled:pipeline:" << seq_time << '\n';
    std::cout << "sosnina_a_diff_count_mpi_enabled:pipeline:" << mpi_time << '\n';
  }
}

TEST_P(SosninaADiffCountPerfTests, TestTaskRun) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int expected = CalcOfDiff(str1, str2);

  // mpi
  InType mpi_input = std::make_pair(str1, str2);
  SosninaADiffCountMPI mpi_task(mpi_input);
  mpi_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  ASSERT_TRUE(mpi_task.Validation() && mpi_task.PreProcessing());
  auto start_mpi = std::chrono::high_resolution_clock::now();
  bool mpi_run = mpi_task.Run();
  auto end_mpi = std::chrono::high_resolution_clock::now();
  double mpi_time = std::chrono::duration<double>(end_mpi - start_mpi).count();
  ASSERT_TRUE(mpi_run && mpi_task.PostProcessing());
  ASSERT_EQ(mpi_task.GetOutput(), expected) << "mpi task run incorrect";

  // seq
  InType seq_input = std::make_pair(str1, str2);
  SosninaADiffCountSEQ seq_task(seq_input);
  seq_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  ASSERT_TRUE(seq_task.Validation() && seq_task.PreProcessing());
  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_run = seq_task.Run();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  ASSERT_TRUE(seq_run && seq_task.PostProcessing());
  ASSERT_EQ(seq_task.GetOutput(), expected) << "seq task run incorrect";

  if (rank == 0) {
    std::cout << "sosnina_a_diff_count_seq_enabled:task_run:" << seq_time << '\n';
    std::cout << "sosnina_a_diff_count_mpi_enabled:task_run:" << mpi_time << '\n';
  }
}

const std::vector<std::tuple<std::string, std::string, size_t>> kTestParams = {
    std::make_tuple("z", "v", 200000000),

};

INSTANTIATE_TEST_SUITE_P(PerfTests, SosninaADiffCountPerfTests, ::testing::ValuesIn(kTestParams), PrintTestParam);

}  // namespace sosnina_a_diff_count
