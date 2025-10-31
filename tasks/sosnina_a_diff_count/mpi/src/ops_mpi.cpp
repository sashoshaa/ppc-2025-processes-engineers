#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sosnina_a_diff_count {

SosninaADiffCountMPI::SosninaADiffCountMPI(const InType &in) : str1_(in.first), str2_(in.second), diff_counter(0) {
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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t str1_len = str1_.size();
  size_t str2_len = str2_.size();
  size_t total_len = std::max(str1_len, str2_len);
  size_t min_len = std::min(str1_len, str2_len);

  if (total_len == 0) {
    diff_counter = 0;
    return true;
  }

  size_t block_size = total_len / size;
  size_t remainder = total_len % size;

  size_t start = rank * block_size + std::min((size_t)rank, remainder);
  size_t end = start + block_size + (rank < (int)remainder ? 1 : 0);
  end = std::min(end, total_len);

  int local_diff_count = 0;

  for (size_t i = start; i < end; i++) {
    if (i < min_len) {
      if (str1_[i] != str2_[i]) {
        local_diff_count++;
      }
    } else {
      local_diff_count++;
    }
  }

  if (size > 1) {
    if (rank == 0) {
      diff_counter = local_diff_count;

      for (int i = 1; i < size; i++) {
        int received_count;
        MPI_Recv(&received_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        diff_counter += received_count;
      }
    } else {
      MPI_Send(&local_diff_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  } else {
    diff_counter = local_diff_count;
  }

  return true;
}

bool SosninaADiffCountMPI::PostProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size > 1) {
    int end_result = diff_counter;
    MPI_Bcast(&end_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    diff_counter = end_result;
  }

  GetOutput() = diff_counter;
  return true;
}

int SosninaADiffCountMPI::GetDiffCount() const {
  return diff_counter;
}

}  // namespace sosnina_a_diff_count
