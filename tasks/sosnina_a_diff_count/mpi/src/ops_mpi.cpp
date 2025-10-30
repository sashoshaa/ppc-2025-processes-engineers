#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include <algorithm>
#include <mpi.h>

namespace sosnina_a_diff_count {

SosninaADiffCountMPI::SosninaADiffCountMPI(const InType &in)
    : str1_(in.first), str2_(in.second), diff_counter(0) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetOutput() = 0;
}

bool SosninaADiffCountMPI::ValidationImpl() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
    if (rank == 0) {}   
    int validation_result = 1; 
    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    return validation_result != 0;
}

bool SosninaADiffCountMPI::PreProcessingImpl() {
    diff_counter = 0;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    if (rank == 0) {
        int str1_size = str1_.size();
        int str2_size = str2_.size();
        
        for (int i = 1; i < size; i++) {
            MPI_Send(&str1_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&str2_size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
        
        for (int i = 1; i < size; i++) {
            if (str1_size > 0) {
                MPI_Send(str1_.data(), str1_size, MPI_CHAR, i, 2, MPI_COMM_WORLD);
            }
            if (str2_size > 0) {
                MPI_Send(str2_.data(), str2_size, MPI_CHAR, i, 3, MPI_COMM_WORLD);
            }
        }
    } else {
        int str1_size, str2_size;
        MPI_Recv(&str1_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str2_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        str1_.resize(str1_size);
        str2_.resize(str2_size);
        
        if (str1_size > 0) {
            MPI_Recv(&str1_[0], str1_size, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (str2_size > 0) {
            MPI_Recv(&str2_[0], str2_size, MPI_CHAR, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    return true;
}

bool SosninaADiffCountMPI::RunImpl() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t min_len = std::min(str1_.size(), str2_.size());
    size_t max_len = std::max(str1_.size(), str2_.size());

    if (max_len == 0) {
        diff_counter = 0;
        return true;
    }

    size_t chunk_size = max_len / size;
    size_t remainder = max_len % size;
    size_t start = rank * chunk_size + std::min((size_t)rank, remainder);
    size_t end = start + chunk_size + (rank < (int)remainder ? 1 : 0);
    end = std::min(end, max_len);

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
        MPI_Reduce(&local_diff_count, &diff_counter, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
        MPI_Bcast(&diff_counter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    GetOutput() = diff_counter;
    
    return true;
}

int SosninaADiffCountMPI::GetDiffCount() const {
    return diff_counter;
}

}  // namespace sosnina_a_diff_count