#include <gtest/gtest.h>
#include <mpi.h>

#include <string>

#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sosnina_a_diff_count {

static int CalcOfDiff(const std::string &s1, const std::string &s2) {
  int diff_count = 0;
  size_t len = std::max(s1.size(), s2.size());
  for (size_t i = 0; i < len; i++) {
    char c1 = i < s1.size() ? s1[i] : 0;
    char c2 = i < s2.size() ? s2[i] : 0;
    if (c1 != c2) {
      diff_count++;
    }
  }
  return diff_count;
}

TEST(sosnina_a_diff_count_mpi, test_pipeline_run) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string str1(200000000, 'z');
  std::string str2(200000000, 'v');
  int expected = CalcOfDiff(str1, str2);

  // mpi
  InType mpi_input = std::make_pair(str1, str2);
  SosninaADiffCountMPI mpi_task(mpi_input);
  mpi_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  auto start_mpi = std::chrono::high_resolution_clock::now();
  bool mpi_success = mpi_task.Validation() && mpi_task.PreProcessing() && mpi_task.Run() && mpi_task.PostProcessing();
  auto end_mpi = std::chrono::high_resolution_clock::now();
  double mpi_time = std::chrono::duration<double>(end_mpi - start_mpi).count();
  ASSERT_TRUE(mpi_success) << "MPI pipeline failed";
  ASSERT_EQ(mpi_task.GetOutput(), expected) << "MPI pipeline result incorrect";

  // seq
  InTypePair seq_input = std::make_pair(str1, str2);
  SosninaADiffCountSEQ seq_task(seq_input);
  seq_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_success = seq_task.Validation() && seq_task.PreProcessing() && seq_task.Run() && seq_task.PostProcessing();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  ASSERT_TRUE(seq_success) << "SEQ pipeline failed";
  ASSERT_EQ(seq_task.GetOutput(), expected) << "SEQ pipeline result incorrect";

  if (rank == 0) {
    std::cout << "sosnina_a_diff_count_seq_enabled:pipeline:" << seq_time << std::endl;
    std::cout << "sosnina_a_diff_count_mpi_enabled:pipeline:" << mpi_time << std::endl;
  }
}

TEST(sosnina_a_diff_count_mpi, test_task_run) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string str1(200000000, 'z');
  std::string str2(200000000, 'v');
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
  ASSERT_EQ(mpi_task.GetOutput(), expected) << "MPI task run result incorrect";

  // seq
  InTypePair seq_input = std::make_pair(str1, str2);
  SosninaADiffCountSEQ seq_task(seq_input);
  seq_task.GetStateOfTesting() = ppc::task::StateOfTesting::kPerf;

  ASSERT_TRUE(seq_task.Validation() && seq_task.PreProcessing());
  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_run = seq_task.Run();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  ASSERT_TRUE(seq_run && seq_task.PostProcessing());
  ASSERT_EQ(seq_task.GetOutput(), expected) << "SEQ task run result incorrect";

  if (rank == 0) {
    std::cout << "sosnina_a_diff_count_seq_enabled:task_run:" << seq_time << std::endl;
    std::cout << "sosnina_a_diff_count_mpi_enabled:task_run:" << mpi_time << std::endl;
  }
}

}  // namespace sosnina_a_diff_count
