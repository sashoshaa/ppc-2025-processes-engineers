#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sosnina_a_sparse_matrix_mult_crs_double {

using InType = std::tuple<std::vector<double>, std::vector<int>, std::vector<int>, std::vector<double>,
                          std::vector<int>, std::vector<int>, int, int, int>;

using OutType = std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>;

using TestType = std::tuple<int, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
                            std::vector<std::vector<double>>>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sosnina_a_sparse_matrix_mult_crs_double
