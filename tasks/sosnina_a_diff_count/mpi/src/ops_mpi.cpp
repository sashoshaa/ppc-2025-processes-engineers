#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>

#include "sosnina_a_diff_count/common/include/common.hpp"

namespace sosnina_a_diff_count {

SosninaADiffCountMPI::SosninaADiffCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = 0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    str1_ = in.first;
    str2_ = in.second;
  }
}

bool SosninaADiffCountMPI::ValidationImpl() {
  return true;
}

bool SosninaADiffCountMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::array<int, 2> lengths{};
  if (rank == 0) {
    lengths[0] = static_cast<int>(str1_.size());
    lengths[1] = static_cast<int>(str2_.size());
  }

  MPI_Bcast(lengths.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    str1_.resize(lengths[0]);
    str2_.resize(lengths[1]);
  }

  MPI_Bcast(str1_.data(), lengths[0], MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(str2_.data(), lengths[1], MPI_CHAR, 0, MPI_COMM_WORLD);

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

  if (total_len == 0) {
    diff_counter_ = 0;
    return true;
  }

  std::size_t chunk_size = total_len / size;
  std::size_t remainder = total_len % size;

  std::size_t start = (rank * chunk_size) + std::min(static_cast<std::size_t>(rank), remainder);
  std::size_t end = start + chunk_size;
  if (std::cmp_less(rank, remainder)) {
    end += 1;
  }

  // подсчёт несовпадений на своём отрезке
  int local_diff_count = 0;
  for (std::size_t i = start; i < end && i < total_len; i++) {
    if (i >= str1_len || i >= str2_len || str1_[i] != str2_[i]) {
      local_diff_count++;
    }
  }

  if (size > 1) {
    MPI_Reduce(&local_diff_count, &diff_counter_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    diff_counter_ = local_diff_count;
  }

  return true;
}

bool SosninaADiffCountMPI::PostProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size > 1) {
    MPI_Bcast(&diff_counter_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  GetOutput() = diff_counter_;
  return true;
}

}  // namespace sosnina_a_diff_count
