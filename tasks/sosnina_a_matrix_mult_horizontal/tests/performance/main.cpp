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

#include "sosnina_a_matrix_mult_horizontal/common/include/common.hpp"
#include "sosnina_a_matrix_mult_horizontal/mpi/include/ops_mpi.hpp"
#include "sosnina_a_matrix_mult_horizontal/seq/include/ops_seq.hpp"

namespace sosnina_a_matrix_mult_horizontal {

// ЗАРАНЕЕ подготовленные матрицы для перфоманс тестов
static std::vector<std::vector<double>> CreatePrecomputedMatrixA_100x100() {
    std::vector<std::vector<double>> matrix(100, std::vector<double>(100));
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            matrix[i][j] = i + j + 1.0;  // Простая детерминированная формула
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixB_100x100() {
    std::vector<std::vector<double>> matrix(100, std::vector<double>(100));
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            matrix[i][j] = (i + 1) * (j + 1) * 0.1;  // Простая детерминированная формула
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixA_500x500() {
    std::vector<std::vector<double>> matrix(500, std::vector<double>(500));
    for (int i = 0; i < 500; i++) {
        for (int j = 0; j < 500; j++) {
            matrix[i][j] = (i * 500 + j) * 0.01 + 1.0;
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixB_500x500() {
    std::vector<std::vector<double>> matrix(500, std::vector<double>(500));
    for (int i = 0; i < 500; i++) {
        for (int j = 0; j < 500; j++) {
            matrix[i][j] = (i + j * 0.5) * 0.02 + 0.5;
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixA_1000x1000() {
    std::vector<std::vector<double>> matrix(1000, std::vector<double>(1000));
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            matrix[i][j] = (i * 1000 + j) * 0.001 + 0.1;
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixB_1000x1000() {
    std::vector<std::vector<double>> matrix(1000, std::vector<double>(1000));
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            matrix[i][j] = (i * 0.7 + j * 0.3) * 0.005 + 0.2;
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixA_200x300() {
    std::vector<std::vector<double>> matrix(200, std::vector<double>(300));
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 300; j++) {
            matrix[i][j] = (i * 300 + j) * 0.01 + 0.5;
        }
    }
    return matrix;
}

static std::vector<std::vector<double>> CreatePrecomputedMatrixB_300x400() {
    std::vector<std::vector<double>> matrix(300, std::vector<double>(400));
    for (int i = 0; i < 300; i++) {
        for (int j = 0; j < 400; j++) {
            matrix[i][j] = (i * 400 + j) * 0.008 + 0.3;
        }
    }
    return matrix;
}

class SosninaAMatrixMultHorizontalPerfTests : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {
 protected:
  void SetUp() override {
    auto params = GetParam();
    rowsA = std::get<0>(params);
    colsA = std::get<1>(params);
    colsB = std::get<2>(params);

    // Используем ЗАРАНЕЕ подготовленные матрицы вместо генерации
    if (rowsA == 100 && colsA == 100 && colsB == 100) {
        matrixA = CreatePrecomputedMatrixA_100x100();
        matrixB = CreatePrecomputedMatrixB_100x100();
    } else if (rowsA == 500 && colsA == 500 && colsB == 500) {
        matrixA = CreatePrecomputedMatrixA_500x500();
        matrixB = CreatePrecomputedMatrixB_500x500();
    } else if (rowsA == 1000 && colsA == 1000 && colsB == 1000) {
        matrixA = CreatePrecomputedMatrixA_1000x1000();
        matrixB = CreatePrecomputedMatrixB_1000x1000();
    } else if (rowsA == 200 && colsA == 300 && colsB == 400) {
        matrixA = CreatePrecomputedMatrixA_200x300();
        matrixB = CreatePrecomputedMatrixB_300x400();
    }
    // Матрицы уже готовы - никакой генерации во время тестов!
  }

  std::vector<std::vector<double>> matrixA;
  std::vector<std::vector<double>> matrixB;
  size_t rowsA = 0;
  size_t colsA = 0;
  size_t colsB = 0;
};

static std::string PrintTestParam(const testing::TestParamInfo<SosninaAMatrixMultHorizontalPerfTests::ParamType> &info) {
  size_t rowsA = std::get<0>(info.param);
  size_t colsA = std::get<1>(info.param);
  size_t colsB = std::get<2>(info.param);

  return std::to_string(rowsA) + "x" + std::to_string(colsA) + "_times_" + 
         std::to_string(colsA) + "x" + std::to_string(colsB);
}

TEST_P(SosninaAMatrixMultHorizontalPerfTests, TestPipelineRun) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // MPI версия
  InType mpi_input = std::make_pair(matrixA, matrixB);
  SosninaAMatrixMultHorizontalMPI mpi_task(mpi_input);

  auto start_mpi = std::chrono::high_resolution_clock::now();
  bool mpi_success = mpi_task.Validation() && mpi_task.PreProcessing() && mpi_task.Run() && mpi_task.PostProcessing();
  auto end_mpi = std::chrono::high_resolution_clock::now();
  double mpi_time = std::chrono::duration<double>(end_mpi - start_mpi).count();
  
  ASSERT_TRUE(mpi_success) << "MPI pipeline failed";

  // SEQ версия
  InType seq_input = std::make_pair(matrixA, matrixB);
  SosninaAMatrixMultHorizontalSEQ seq_task(seq_input);

  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_success = seq_task.Validation() && seq_task.PreProcessing() && seq_task.Run() && seq_task.PostProcessing();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  
  ASSERT_TRUE(seq_success) << "SEQ pipeline failed";

  if (rank == 0) {
    std::cout << "sosnina_a_matrix_mult_horizontal_seq_enabled:pipeline:" << seq_time << '\n';
    std::cout << "sosnina_a_matrix_mult_horizontal_mpi_enabled:pipeline:" << mpi_time << '\n';
    std::cout << "Matrix size: " << rowsA << "x" << colsA << " * " << colsA << "x" << colsB << '\n';
  }
}

TEST_P(SosninaAMatrixMultHorizontalPerfTests, TestTaskRun) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // MPI версия
  InType mpi_input = std::make_pair(matrixA, matrixB);
  SosninaAMatrixMultHorizontalMPI mpi_task(mpi_input);

  ASSERT_TRUE(mpi_task.Validation() && mpi_task.PreProcessing());
  auto start_mpi = std::chrono::high_resolution_clock::now();
  bool mpi_run = mpi_task.Run();
  auto end_mpi = std::chrono::high_resolution_clock::now();
  double mpi_time = std::chrono::duration<double>(end_mpi - start_mpi).count();
  ASSERT_TRUE(mpi_run && mpi_task.PostProcessing());

  // SEQ версия
  InType seq_input = std::make_pair(matrixA, matrixB);
  SosninaAMatrixMultHorizontalSEQ seq_task(seq_input);

  ASSERT_TRUE(seq_task.Validation() && seq_task.PreProcessing());
  auto start_seq = std::chrono::high_resolution_clock::now();
  bool seq_run = seq_task.Run();
  auto end_seq = std::chrono::high_resolution_clock::now();
  double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();
  ASSERT_TRUE(seq_run && seq_task.PostProcessing());

  if (rank == 0) {
    std::cout << "sosnina_a_matrix_mult_horizontal_seq_enabled:task_run:" << seq_time << '\n';
    std::cout << "sosnina_a_matrix_mult_horizontal_mpi_enabled:task_run:" << mpi_time << '\n';
  }
}

// Параметры тестов: (rowsA, colsA, colsB)
const std::vector<std::tuple<size_t, size_t, size_t>> kTestParams = {
    std::make_tuple(100, 100, 100),
    std::make_tuple(500, 500, 500),
    std::make_tuple(1000, 1000, 1000),
    std::make_tuple(200, 300, 400),  // Неквадратные матрицы
};

INSTANTIATE_TEST_SUITE_P(PerfTests, SosninaAMatrixMultHorizontalPerfTests, 
                         ::testing::ValuesIn(kTestParams), PrintTestParam);

}  // namespace sosnina_a_matrix_mult_horizontal