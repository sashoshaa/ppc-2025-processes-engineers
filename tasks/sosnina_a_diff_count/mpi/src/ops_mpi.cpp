#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "sosnina_a_diff_count/common/include/common.hpp"

namespace sosnina_a_diff_count {

SosninaADiffCountMPI::SosninaADiffCountMPI(const InType &in) : str1_(in.first), str2_(in.second) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = 0;
}

bool SosninaADiffCountMPI::ValidationImpl() {
  return true;
}

bool SosninaADiffCountMPI::PreProcessingImpl() {
  return true;
}

bool SosninaADiffCountMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t str1_len = str1_.size();
  std::size_t str2_len = str2_.size();
  std::size_t total_len = std::max(str1_len, str2_len);
  std::size_t min_len = std::min(str1_len, str2_len);

  if (total_len == 0) {
    diff_counter_ = 0;
    return true;
  }

  std::size_t block_size = total_len / size;
  std::size_t remainder = total_len % size;

  std::size_t start =
      (static_cast<std::size_t>(rank) * block_size) + std::min(static_cast<std::size_t>(rank), remainder);
  std::size_t end = start + block_size;
  if (std::cmp_less(rank, remainder)) {
    end += 1;
  }
  end = std::min(end, total_len);

  int local_diff_count = 0;
  for (std::size_t i = start; i < end; i++) {
    if (i < min_len) {
      if (str1_[i] != str2_[i]) {
        local_diff_count++;
      }
    } else {
      local_diff_count++;
    }
  }

  diff_counter_ = local_diff_count;

  if (size > 1) {
    if (rank == 0) {
      for (int i = 1; i < size; i++) {
        int received_count = 0;
        MPI_Recv(&received_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        diff_counter_ += received_count;
      }
    } else {
      MPI_Send(&local_diff_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool SosninaADiffCountMPI::PostProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size > 1) {
    int end_result = diff_counter_;
    MPI_Bcast(&end_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    diff_counter_ = end_result;
  }

  GetOutput() = diff_counter_;
  return true;
}

}  // namespace sosnina_a_diff_count
