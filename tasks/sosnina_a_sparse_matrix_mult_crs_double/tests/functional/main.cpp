#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sosnina_a_sparse_matrix_mult_crs_double/common/include/common.hpp"
#include "sosnina_a_sparse_matrix_mult_crs_double/mpi/include/ops_mpi.hpp"
#include "sosnina_a_sparse_matrix_mult_crs_double/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

class SosninaAMatrixMultCRSFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_matrix_mult";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    dense_A_ = std::get<1>(params);
    dense_B_ = std::get<2>(params);
    dense_expected_ = std::get<3>(params);

    if (dense_A_.empty()) {
      row_ptr_A_ = {0};
      values_A_.clear();
      col_indices_A_.clear();
    } else {
      ConvertDenseToCRS(dense_A_, values_A_, col_indices_A_, row_ptr_A_);
    }

    if (dense_B_.empty()) {
      row_ptr_B_ = {0};
      values_B_.clear();
      col_indices_B_.clear();
    } else {
      ConvertDenseToCRS(dense_B_, values_B_, col_indices_B_, row_ptr_B_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto &[values, col_indices, row_ptr] = output_data;

    if (row_ptr.empty()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
      bool all_zeros = true;
      for (const auto &row : dense_expected_) {
        for (double val : row) {
          if (std::abs(val) > 1e-12) {
            all_zeros = false;
            break;
          }
        }
        if (!all_zeros) {
          break;
        }
      }
      return all_zeros;
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

    if (values.empty() && row_ptr.back() == 0) {
      bool all_zeros = true;
      for (const auto &row : dense_expected_) {
        for (double val : row) {
          if (std::abs(val) > 1e-12) {
            all_zeros = false;
            break;
          }
        }
        if (!all_zeros) {
          break;
        }
      }
      return all_zeros;
    }

    std::vector<std::vector<double>> dense_result;
    int n_rows = static_cast<int>(dense_expected_.size());
    int n_cols = dense_expected_.empty() ? 0 : static_cast<int>(dense_expected_[0].size());

    if (!ConvertCRSToDense(values, col_indices, row_ptr, n_rows, n_cols, dense_result)) {
      return false;
    }

    if (dense_result.size() != dense_expected_.size()) {
      return false;
    }
    if (!dense_result.empty() && dense_result[0].size() != dense_expected_[0].size()) {
      return false;
    }

    const double tolerance = 1e-10;

    for (size_t i = 0; i < dense_expected_.size(); i++) {
      for (size_t j = 0; j < dense_expected_[i].size(); j++) {
        if (std::abs(dense_result[i][j] - dense_expected_[i][j]) > tolerance) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    int n_rows = static_cast<int>(dense_A_.size());
    int n_cols_a = dense_A_.empty() ? 0 : static_cast<int>(dense_A_[0].size());
    int n_cols_b = dense_B_.empty() ? 0 : static_cast<int>(dense_B_[0].size());
    return std::make_tuple(values_A_, col_indices_A_, row_ptr_A_, values_B_, col_indices_B_, row_ptr_B_, n_rows,
                           n_cols_a, n_cols_b);
  }

 private:
  static void ConvertDenseToCRS(const std::vector<std::vector<double>> &dense, std::vector<double> &values,
                                std::vector<int> &col_indices, std::vector<int> &row_ptr) {
    values.clear();
    col_indices.clear();
    row_ptr.clear();
    row_ptr.push_back(0);

    for (const auto &row : dense) {
      std::vector<std::pair<int, double>> row_elements;
      for (size_t j = 0; j < row.size(); j++) {
        if (std::abs(row[j]) > 1e-12) {
          row_elements.emplace_back(static_cast<int>(j), row[j]);
        }
      }

      std::ranges::sort(row_elements, [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
        return a.first < b.first;
      });

      for (const auto &elem : row_elements) {
        col_indices.push_back(elem.first);
        values.push_back(elem.second);
      }

      row_ptr.push_back(static_cast<int>(values.size()));
    }
  }

  static bool ConvertCRSToDense(const std::vector<double> &values, const std::vector<int> &col_indices,
                                const std::vector<int> &row_ptr, int n_rows, int n_cols,
                                std::vector<std::vector<double>> &dense) {
    if (row_ptr.empty()) {
      dense.assign(n_rows, std::vector<double>(n_cols, 0.0));
      return true;
    }

    if (static_cast<int>(row_ptr.size()) != n_rows + 1) {
      dense.assign(n_rows, std::vector<double>(n_cols, 0.0));
      return false;
    }

    if (row_ptr[0] != 0) {
      dense.assign(n_rows, std::vector<double>(n_cols, 0.0));
      return false;
    }

    for (size_t i = 0; i < row_ptr.size() - 1; i++) {
      if (row_ptr[i] < 0 || static_cast<size_t>(row_ptr[i]) > values.size() || row_ptr[i + 1] < 0 ||
          static_cast<size_t>(row_ptr[i + 1]) > values.size() || row_ptr[i] > row_ptr[i + 1]) {
        dense.assign(n_rows, std::vector<double>(n_cols, 0.0));
        return false;
      }
    }

    dense.assign(n_rows, std::vector<double>(n_cols, 0.0));

    for (int i = 0; i < n_rows; i++) {
      int start = row_ptr[i];
      int end = row_ptr[i + 1];

      if (start < 0 || static_cast<size_t>(end) > values.size() || start > end) {
        continue;
      }

      for (int idx = start; idx < end; idx++) {
        if (idx < 0 || static_cast<size_t>(idx) >= col_indices.size()) {
          continue;
        }
        int j = col_indices[idx];
        if (j >= 0 && j < n_cols && static_cast<size_t>(idx) < values.size()) {
          double val = values[idx];
          dense[i][j] = val;
        }
      }
    }

    return true;
  }

  std::vector<std::vector<double>> dense_A_;
  std::vector<std::vector<double>> dense_B_;
  std::vector<std::vector<double>> dense_expected_;

  std::vector<double> values_A_;
  std::vector<int> col_indices_A_;
  std::vector<int> row_ptr_A_;

  std::vector<double> values_B_;
  std::vector<int> col_indices_B_;
  std::vector<int> row_ptr_B_;
};

namespace {

TEST_P(SosninaAMatrixMultCRSFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(SosninaAMatrixMultCRSFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 34> kFunctionalTests = {
    // 1. Базовое умножение 2x2
    std::make_tuple(1, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{19, 22}, {43, 50}}),

    // 2. Единичная матрица
    std::make_tuple(2, std::vector<std::vector<double>>{{1, 0}, {0, 1}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<std::vector<double>>{{1, 2}, {3, 4}}),

    // 3. Матрица из единиц
    std::make_tuple(3, std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{1, 1}, {1, 1}}, std::vector<std::vector<double>>{{2, 2}, {2, 2}}),

    // 4. Диагональные матрицы
    std::make_tuple(4, std::vector<std::vector<double>>{{2, 0}, {0, 2}},
                    std::vector<std::vector<double>>{{3, 0}, {0, 3}}, std::vector<std::vector<double>>{{6, 0}, {0, 6}}),

    // 5. Вектор-строка на вектор-столбец
    std::make_tuple(5, std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<std::vector<double>>{{4}, {5}, {6}},
                    std::vector<std::vector<double>>{{32}}),

    // 6. Вектор-столбец на вектор-строку
    std::make_tuple(6, std::vector<std::vector<double>>{{1}, {2}, {3}}, std::vector<std::vector<double>>{{4, 5, 6}},
                    std::vector<std::vector<double>>{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}}),

    // 7. Неквадратные матрицы
    std::make_tuple(7, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{19, 22}, {43, 50}}),

    // 8. 1x1 матрицы
    std::make_tuple(8, std::vector<std::vector<double>>{{1}}, std::vector<std::vector<double>>{{1}},
                    std::vector<std::vector<double>>{{1}}),

    // 9. Нулевая матрица
    std::make_tuple(9, std::vector<std::vector<double>>{{0, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    // 10. Скалярное умножение
    std::make_tuple(10, std::vector<std::vector<double>>{{2}}, std::vector<std::vector<double>>{{3}},
                    std::vector<std::vector<double>>{{6}}),

    // 11. 2x3 на 3x2
    std::make_tuple(11, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}},
                    std::vector<std::vector<double>>{{7, 8}, {9, 10}, {11, 12}},
                    std::vector<std::vector<double>>{{58, 64}, {139, 154}}),

    // 12. Единичная матрица 3x3
    std::make_tuple(12, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}),

    // 13. Дробные числа
    std::make_tuple(13, std::vector<std::vector<double>>{{0.5, 0.5}, {0.5, 0.5}},
                    std::vector<std::vector<double>>{{2, 4}, {6, 8}}, std::vector<std::vector<double>>{{4, 6}, {4, 6}}),

    // 14. 3x2 на 2x3
    std::make_tuple(14, std::vector<std::vector<double>>{{1, 1}, {1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{1, 1, 1}, {1, 1, 1}},
                    std::vector<std::vector<double>>{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}),

    // 15. Обратное умножение
    std::make_tuple(15, std::vector<std::vector<double>>{{2, 4}, {6, 8}},
                    std::vector<std::vector<double>>{{1, 0}, {0, 1}}, std::vector<std::vector<double>>{{2, 4}, {6, 8}}),

    // 16. 3x2 на 2x3
    std::make_tuple(16, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{7, 8, 9}, {10, 11, 12}},
                    std::vector<std::vector<double>>{{27, 30, 33}, {61, 68, 75}, {95, 106, 117}}),

    // 17. Дробные числа 2
    std::make_tuple(17, std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}},
                    std::vector<std::vector<double>>{{5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{1.9, 2.2}, {4.3, 5.0}}),

    // 18. Матрица из двоек
    std::make_tuple(18, std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{2, 2}, {2, 2}}, std::vector<std::vector<double>>{{4, 4}, {4, 4}}),

    // 19. Скаляр 3x4
    std::make_tuple(19, std::vector<std::vector<double>>{{3}}, std::vector<std::vector<double>>{{4}},
                    std::vector<std::vector<double>>{{12}}),

    // 20. Частично нулевая матрица
    std::make_tuple(20, std::vector<std::vector<double>>{{1, 0}, {0, 0}},
                    std::vector<std::vector<double>>{{0, 1}, {0, 0}}, std::vector<std::vector<double>>{{0, 1}, {0, 0}}),

    // 21. Обратная единичная
    std::make_tuple(21, std::vector<std::vector<double>>{{2, 3}, {4, 5}},
                    std::vector<std::vector<double>>{{1, 0}, {0, 1}}, std::vector<std::vector<double>>{{2, 3}, {4, 5}}),

    // 22. Умножение на нулевую матрицу
    std::make_tuple(22, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}, std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    // 23. Отрицательные числа
    std::make_tuple(23, std::vector<std::vector<double>>{{-1, -2}, {-3, -4}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{-7, -10}, {-15, -22}}),

    // 24. Смешанные положительные и отрицательные
    std::make_tuple(24, std::vector<std::vector<double>>{{1, -2}, {3, -4}},
                    std::vector<std::vector<double>>{{-1, 2}, {-3, 4}},
                    std::vector<std::vector<double>>{{5, -6}, {9, -10}}),

    // 25. Маленькие дробные числа
    std::make_tuple(25, std::vector<std::vector<double>>{{0.001, 0.002}, {0.003, 0.004}},
                    std::vector<std::vector<double>>{{1000, 2000}, {3000, 4000}},
                    std::vector<std::vector<double>>{{7, 10}, {15, 22}}),

    // 26. Высокая точность дробных
    std::make_tuple(26, std::vector<std::vector<double>>{{0.123456789, 0.987654321}, {0.555555555, 0.111111111}},
                    std::vector<std::vector<double>>{{1.0, 2.0}, {3.0, 4.0}},
                    std::vector<std::vector<double>>{{3.086419752, 4.197530862}, {0.888888888, 1.555555554}}),

    // 27. Разные порядки величин
    std::make_tuple(27, std::vector<std::vector<double>>{{1.5, 2.5}, {3.5, 4.5}},
                    std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}},
                    std::vector<std::vector<double>>{{0.9, 1.3}, {1.7, 2.5}}),

    // 28. Дробные с периодической частью
    std::make_tuple(28, std::vector<std::vector<double>>{{1.0 / 3.0, 2.0 / 3.0}, {1.0 / 7.0, 2.0 / 7.0}},
                    std::vector<std::vector<double>>{{3.0, 6.0}, {7.0, 14.0}},
                    std::vector<std::vector<double>>{{17.0 / 3.0, 34.0 / 3.0}, {17.0 / 7.0, 34.0 / 7.0}}),

    // 29. Разреженная матрица 4x4 (75% нулей)
    std::make_tuple(29, std::vector<std::vector<double>>{{1, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}},
                    std::vector<std::vector<double>>{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}},
                    std::vector<std::vector<double>>{{1, 1, 1, 1}, {4, 4, 4, 4}, {9, 9, 9, 9}, {16, 16, 16, 16}}),

    // 30. Разреженная матрица 5x5 (80% нулей)
    std::make_tuple(30,
                    std::vector<std::vector<double>>{
                        {1, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 0, 0, 3}, {0, 4, 0, 0, 0}, {5, 0, 0, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}},
                    std::vector<std::vector<double>>{{1, 2}, {10, 12}, {27, 30}, {12, 16}, {5, 10}}),

    // 31. Разреженная матрица с несколькими ненулевыми в строке
    std::make_tuple(31, std::vector<std::vector<double>>{{1, 0, 2, 0}, {0, 3, 0, 4}, {5, 0, 0, 6}, {0, 7, 8, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
                    std::vector<std::vector<double>>{{11, 14}, {37, 44}, {47, 58}, {61, 76}}),

    // 32. Разреженная матрица 3x4 (67% нулей)
    std::make_tuple(32, std::vector<std::vector<double>>{{1, 0, 0, 2}, {0, 3, 0, 0}, {0, 0, 4, 0}},
                    std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
                    std::vector<std::vector<double>>{{21, 24, 27}, {12, 15, 18}, {28, 32, 36}}),

    // 33. Разреженная матрица с одной ненулевой строкой
    std::make_tuple(33, std::vector<std::vector<double>>{{0, 0, 0}, {1, 2, 3}, {0, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{0, 0}, {22, 28}, {0, 0}}),

    // 34. Разреженная матрица с одной ненулевой колонкой
    std::make_tuple(34, std::vector<std::vector<double>>{{1, 0, 0}, {2, 0, 0}, {3, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2, 3, 4}, {2, 4, 6, 8}, {3, 6, 9, 12}})};

const std::array<TestType, 19> kCoverageTests = {
    std::make_tuple(31, std::vector<std::vector<double>>{{1}}, std::vector<std::vector<double>>{{1}},
                    std::vector<std::vector<double>>{{1}}),

    std::make_tuple(32, std::vector<std::vector<double>>{{0}}, std::vector<std::vector<double>>{{0}},
                    std::vector<std::vector<double>>{{0}}),

    std::make_tuple(33, std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{1, 0}, {0, 1}}, std::vector<std::vector<double>>{{1, 2}, {3, 4}}),

    std::make_tuple(34, std::vector<std::vector<double>>{{1, 1}, {1, 1}},
                    std::vector<std::vector<double>>{{2, 2}, {2, 2}}, std::vector<std::vector<double>>{{4, 4}, {4, 4}}),

    std::make_tuple(35, std::vector<std::vector<double>>{{0.5}}, std::vector<std::vector<double>>{{2}},
                    std::vector<std::vector<double>>{{1}}),

    std::make_tuple(36, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{5, 6, 7}, {8, 9, 10}, {11, 12, 13}},
                    std::vector<std::vector<double>>{{5, 6, 7}, {8, 9, 10}, {11, 12, 13}}),

    std::make_tuple(37, std::vector<std::vector<double>>{{2, 4}, {6, 8}},
                    std::vector<std::vector<double>>{{0.5, 0}, {0, 0.5}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}}),

    // 38. 2x3 на 3x2
    std::make_tuple(38, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}},
                    std::vector<std::vector<double>>{{7, 8}, {9, 10}, {11, 12}},
                    std::vector<std::vector<double>>{{58, 64}, {139, 154}}),

    std::make_tuple(39, std::vector<std::vector<double>>{{1}, {2}, {3}}, std::vector<std::vector<double>>{{4, 5, 6}},
                    std::vector<std::vector<double>>{{4, 5, 6}, {8, 10, 12}, {12, 15, 18}}),

    std::make_tuple(40, std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<std::vector<double>>{{4}, {5}, {6}},
                    std::vector<std::vector<double>>{{32}}),

    // 41. Отрицательные дробные
    std::make_tuple(41, std::vector<std::vector<double>>{{-0.5, -0.25}, {-0.75, -0.125}},
                    std::vector<std::vector<double>>{{2, 4}, {8, 16}},
                    std::vector<std::vector<double>>{{-3, -6}, {-2.5, -5}}),

    // 42. Большие отрицательные
    std::make_tuple(42, std::vector<std::vector<double>>{{-1e6, -2e6}, {-3e6, -4e6}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}},
                    std::vector<std::vector<double>>{{-7e6, -10e6}, {-15e6, -22e6}}),

    // 43. Смешанные порядки (упрощенные)
    std::make_tuple(43, std::vector<std::vector<double>>{{1.23456789, 9.87654321}, {0.001, 0.002}},
                    std::vector<std::vector<double>>{{1000, 2000}, {3000, 4000}},
                    std::vector<std::vector<double>>{{30864.19752, 41975.30862}, {7, 10}}),

    // 44. Периодические дроби
    std::make_tuple(44, std::vector<std::vector<double>>{{1.0 / 9.0, 2.0 / 9.0}, {1.0 / 11.0, 2.0 / 11.0}},
                    std::vector<std::vector<double>>{{9, 18}, {11, 22}},
                    std::vector<std::vector<double>>{{31.0 / 9.0, 62.0 / 9.0}, {31.0 / 11.0, 62.0 / 11.0}}),

    // 45. Разные знаки и порядки
    std::make_tuple(45, std::vector<std::vector<double>>{{-1.5, 2.5}, {-3.5, 4.5}},
                    std::vector<std::vector<double>>{{-0.1, 0.2}, {0.3, -0.4}},
                    std::vector<std::vector<double>>{{0.9, -1.3}, {1.7, -2.5}}),

    // 46. Степени двойки
    std::make_tuple(46, std::vector<std::vector<double>>{{0.125, 0.25}, {0.5, 1.0}},
                    std::vector<std::vector<double>>{{8, 4}, {2, 1}},
                    std::vector<std::vector<double>>{{1.5, 0.75}, {6, 3}}),

    // 47. Числа с высокой точностью
    std::make_tuple(
        47,
        std::vector<std::vector<double>>{{0.123456789012345, 0.987654321098765},
                                         {0.555555555555555, 0.111111111111111}},
        std::vector<std::vector<double>>{{1.0, 2.0}, {3.0, 4.0}},
        std::vector<std::vector<double>>{{3.08641975230864, 4.19753086241975}, {0.888888888888888, 1.555555555555554}}),

    // 48. Смешанные порядки (упрощенные)
    std::make_tuple(48, std::vector<std::vector<double>>{{10, 0.02}, {0.03, 40}},
                    std::vector<std::vector<double>>{{0.1, 20}, {30, 0.4}},
                    std::vector<std::vector<double>>{{1.6, 200.008}, {1200.003, 16.6}}),

    // 49. Разреженная с дробными
    std::make_tuple(49, std::vector<std::vector<double>>{{0.1, 0, 0.2}, {0, 0.3, 0}, {0.4, 0, 0.5}},
                    std::vector<std::vector<double>>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
                    std::vector<std::vector<double>>{{1.1, 1.4}, {0.9, 1.2}, {2.9, 3.8}})};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_sparse_matrix_mult_crs_double::SosninaAMatrixMultCRSMPI, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_sparse_matrix_mult_crs_double),
                   ppc::util::AddFuncTask<sosnina_a_sparse_matrix_mult_crs_double::SosninaAMatrixMultCRSSEQ, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_sparse_matrix_mult_crs_double));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_sparse_matrix_mult_crs_double::SosninaAMatrixMultCRSMPI, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_sparse_matrix_mult_crs_double),
                   ppc::util::AddFuncTask<sosnina_a_sparse_matrix_mult_crs_double::SosninaAMatrixMultCRSSEQ, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_sparse_matrix_mult_crs_double));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

inline const auto kPerfTestName = SosninaAMatrixMultCRSFuncTests::PrintFuncTestName<SosninaAMatrixMultCRSFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, SosninaAMatrixMultCRSFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, SosninaAMatrixMultCRSFuncTests, kCoverageGtestValues, kPerfTestName);

}  // namespace

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
