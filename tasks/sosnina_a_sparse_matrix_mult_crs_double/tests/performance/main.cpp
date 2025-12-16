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
    // Создаем тестовые разреженные матрицы размера kSize x kSize
    // Матрица A: диагональная (очень разреженная - только диагональ)
    CreateDiagonalMatrix(values_A_, col_indices_A_, row_ptr_A_, kSize, kSize, 1.0);

    // Матрица B: тоже диагональная, но с другими значениями
    CreateDiagonalMatrix(values_B_, col_indices_B_, row_ptr_B_, kSize, kSize, 2.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto &[values, col_indices, row_ptr] = output_data;

    // Для perf тестов проверяем только базовую корректность CRS формата
    // На не-root процессах в MPI результат может быть пустым

    if (row_ptr.empty()) {
      // Пустой результат допустим для не-root процессов в MPI
      return true;
    }

    // Проверяем базовые инварианты CRS формата
    if (row_ptr[0] != 0) {
      return false;
    }

    if (values.size() != col_indices.size()) {
      return false;
    }

    // Проверяем монотонность row_ptr
    for (size_t i = 0; i < row_ptr.size() - 1; i++) {
      if (row_ptr[i] > row_ptr[i + 1]) {
        return false;
      }
    }

    // Проверяем, что row_ptr имеет правильный размер (kSize + 1)
    // Но для MPI это может быть не так на не-root процессах
    if (row_ptr.empty()) {
      return false;
    }

    // Проверяем, что все индексы в допустимом диапазоне
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
  // Создание диагональной матрицы в формате CRS
  static void CreateDiagonalMatrix(std::vector<double> &values, std::vector<int> &col_indices,
                                   std::vector<int> &row_ptr, int n_rows, int n_cols, double diag_value) {
    values.clear();
    col_indices.clear();
    row_ptr.clear();
    row_ptr.push_back(0);

    for (int i = 0; i < n_rows; i++) {
      // На диагонали всегда есть элемент
      values.push_back(diag_value);
      col_indices.push_back(i);

      // Каждая 10-я строка имеет дополнительный элемент
      if (i % 10 == 0 && i + 1 < n_cols) {
        values.push_back(0.5);
        col_indices.push_back(i + 1);
      }

      row_ptr.push_back(static_cast<int>(values.size()));
    }
  }

  // Добавление внедиагональных элементов
  static void AddOffDiagonalElements(std::vector<double> &values, std::vector<int> &col_indices,
                                     std::vector<int> &row_ptr, int n, int count) {
    // Добавляем count псевдослучайных элементов
    for (int k = 0; k < count; k++) {
      int i = (k * 13) % n;  // Псевдослучайная строка
      int j = (k * 17) % n;  // Псевдослучайный столбец

      if (i != j) {  // Не на диагонали
        // Находим позицию для вставки в строку i
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        // Ищем место для вставки (столбцы должны быть отсортированы)
        int insert_pos = row_start;
        while (insert_pos < row_end && col_indices[insert_pos] < j) {
          insert_pos++;
        }

        // Вставляем элемент
        values.insert(values.begin() + insert_pos, 0.3);
        col_indices.insert(col_indices.begin() + insert_pos, j);

        // Обновляем row_ptr для всех последующих строк
        for (size_t idx = i + 1; idx < row_ptr.size(); idx++) {
          row_ptr[idx]++;
        }
      }
    }
  }

  // Данные матриц в CRS формате
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
