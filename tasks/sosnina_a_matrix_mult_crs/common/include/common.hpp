// common.hpp
#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace sosnina_a_matrix_mult_crs {

// Входной тип: пара разреженных матриц в формате CRS
using InType = std::tuple<std::vector<double>,  // values_A
                          std::vector<int>,     // col_indices_A
                          std::vector<int>,     // row_ptr_A
                          std::vector<double>,  // values_B
                          std::vector<int>,     // col_indices_B
                          std::vector<int>,     // row_ptr_B
                          int,                  // n_rows_A
                          int,                  // n_cols_A (совпадает с n_rows_B для умножения)
                          int                   // n_cols_B
                          >;

// Выходной тип: результат умножения в формате CRS
using OutType = std::tuple<std::vector<double>,  // values_C
                           std::vector<int>,     // col_indices_C
                           std::vector<int>      // row_ptr_C
                           >;

// Тестовый тип: id + полные матрицы для тестирования
using TestType = std::tuple<int,                               // id теста
                            std::vector<std::vector<double>>,  // Матрица A (плотная, для тестов)
                            std::vector<std::vector<double>>,  // Матрица B (плотная, для тестов)
                            std::vector<std::vector<double>>   // Ожидаемый результат (плотная матрица)
                            >;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sosnina_a_matrix_mult_crs
