#include "sosnina_a_diff_count/common/include/common.hpp"
#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

#include <string>
#include <tuple>
#include <array>
#include <algorithm>
#include <utility>

namespace sosnina_a_diff_count {

class SosninaADiffCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    
    std::string combined = std::get<1>(params);
    auto pos = combined.find('_');
    str1_ = combined.substr(0, pos);
    str2_ = combined.substr(pos + 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (size_t i = 0; i < str1_.size() && i < str2_.size(); i++) {
      if (str1_[i] != str2_[i])
        expected++;
    }
    expected += static_cast<int>(std::max(str1_.size(), str2_.size()) - std::min(str1_.size(), str2_.size()));
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return std::make_pair(str1_, str2_);
  }

 private:
  std::string str1_;
  std::string str2_;
};

namespace {

  //проверяем
  void DiffFindResult(const std::string& str1, const std::string& str2, int result) {
    int expected = 0;
    for (size_t i = 0; i < str1.size() && i < str2.size(); i++) {
      if (str1[i] != str2[i])
        expected++;
    }
    expected += static_cast<int>(std::max(str1.size(), str2.size()) - std::min(str1.size(), str2.size()));
    EXPECT_EQ(result, expected) << "Failed for strings: " << str1 << " and " << str2;
  }
  
 
  
  //mpi
  TEST(sosnina_a_diff_count_mpi, one_char_difference) {
      InType input = std::make_pair("happy", "heppy");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("happy", "heppy", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, multiple_chars_difference) {
      InType input = std::make_pair("abcdef", "abzzef");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("abcdef", "abzzef", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, identical_strings) {
      InType input = std::make_pair("baby", "baby");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("baby", "baby", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, different_length_strings) {
      InType input = std::make_pair("abc", "defgh");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("abc", "defgh", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, empty_vs_non_empty) {
      InType input = std::make_pair("", "mpi");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("", "mpi", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, long_identical_strings) {
      InType input = std::make_pair("longstring", "longstring");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("longstring", "longstring", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, basic_strings_comparison) {
      InType input = std::make_pair("abcd", "efgh");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("abcd", "efgh", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, both_strings_empty) {
      InType input = std::make_pair("", "");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("", "", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, single_char_empty_first) {
      InType input = std::make_pair("z", "");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("z", "", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, single_char_empty_second) {
      InType input = std::make_pair("", "v");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("", "v", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, single_char_difference) {
      InType input = std::make_pair("z", "v");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("z", "v", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, single_char_identical) {
      InType input = std::make_pair("z", "z");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("z", "z", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, two_chars_swapped) {
      InType input = std::make_pair("zv", "vz");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("zv", "vz", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, case_sensitive_comparison) {
      InType input = std::make_pair("prizet", "PRIZET");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("prizet", "PRIZET", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, triple_chars_difference) {
      InType input = std::make_pair("zzz", "vvv");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("zzz", "vvv", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, numeric_strings) {
      InType input = std::make_pair("54321", "09876");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("54321", "09876", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, uppercase_identical) {
      InType input = std::make_pair("TEST", "TEST");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("TEST", "TEST", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, repeated_pattern) {
      InType input = std::make_pair("z v", "z v");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("z v", "z v", task.GetOutput());
  }
  
  TEST(sosnina_a_diff_count_mpi, very_long_strings) {
      InType input = std::make_pair("veryvery_long_string_one", "veryvery_long_string_two");
      SosninaADiffCountMPI task(input);  
      bool success = task.Validation() && task.PreProcessing() &&
                    task.Run() && task.PostProcessing();
      ASSERT_TRUE(success);
      DiffFindResult("veryvery_long_string_one", "veryvery_long_string_two", task.GetOutput());
  }

TEST(sosnina_a_diff_count_mpi, coverage_empty_strings) {
    std::pair<std::string, std::string> empty_input = {"", ""};
    SosninaADiffCountMPI empty_task(empty_input);
    
    bool empty_success = empty_task.Validation() && empty_task.PreProcessing() &&
                        empty_task.Run() && empty_task.PostProcessing();
    
    ASSERT_TRUE(empty_success);
    ASSERT_EQ(empty_task.GetOutput(), 0);
    ASSERT_EQ(empty_task.GetDiffCount(), 0);
}

TEST(sosnina_a_diff_count_mpi, coverage_different_lengths) {
    std::pair<std::string, std::string> diff_len_input = {"short", "very_long_string"};
    SosninaADiffCountMPI diff_len_task(diff_len_input);
    
    bool diff_len_success = diff_len_task.Validation() && diff_len_task.PreProcessing() &&
                           diff_len_task.Run() && diff_len_task.PostProcessing();
    
    ASSERT_TRUE(diff_len_success);
    ASSERT_GT(diff_len_task.GetOutput(), 0);
}

//seq
TEST(sosnina_a_diff_count_seq, one_char_difference) {
  InType input = std::make_pair("happy", "heppy");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("happy", "heppy", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, multiple_chars_difference) {
  InType input = std::make_pair("abcdef", "abzzef");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("abcdef", "abzzef", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, identical_strings) {
  InType input = std::make_pair("baby", "baby");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("baby", "baby", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, different_length_strings) {
  InType input = std::make_pair("abc", "defgh");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("abc", "defgh", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, empty_vs_non_empty) {
  InType input = std::make_pair("", "mpi");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("", "mpi", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, long_identical_strings) {
  InType input = std::make_pair("longstring", "longstring");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("longstring", "longstring", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, basic_strings_comparison) {
  InType input = std::make_pair("abcd", "efgh");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("abcd", "efgh", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, both_strings_empty) {
  InType input = std::make_pair("", "");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("", "", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, single_char_empty_first) {
  InType input = std::make_pair("z", "");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("z", "", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, single_char_empty_second) {
  InType input = std::make_pair("", "v");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("", "v", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, single_char_difference) {
  InType input = std::make_pair("z", "v");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("z", "v", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, single_char_identical) {
  InType input = std::make_pair("z", "z");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("z", "z", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, two_chars_swapped) {
  InType input = std::make_pair("zv", "vz");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("zv", "vz", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, case_sensitive_comparison) {
  InType input = std::make_pair("prizet", "PRIZET");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("prizet", "PRIZET", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, triple_chars_difference) {
  InType input = std::make_pair("zzz", "vvv");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("zzz", "vvv", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, numeric_strings) {
  InType input = std::make_pair("54321", "09876");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("54321", "09876", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, uppercase_identical) {
  InType input = std::make_pair("TEST", "TEST");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("TEST", "TEST", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, repeated_pattern) {
  InType input = std::make_pair("z v", "z v");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("z v", "z v", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, very_long_strings) {
  InType input = std::make_pair("veryvery_long_string_one", "veryvery_long_string_two");
  SosninaADiffCountSEQ task(input);  
  bool success = task.Validation() && task.PreProcessing() &&
                task.Run() && task.PostProcessing();
  ASSERT_TRUE(success);
  DiffFindResult("veryvery_long_string_one", "veryvery_long_string_two", task.GetOutput());
}

TEST(sosnina_a_diff_count_seq, coverage_empty_strings) {
    std::pair<std::string, std::string> empty_input = {"", ""};
    SosninaADiffCountSEQ empty_task(empty_input);
    
    bool empty_success = empty_task.Validation() && empty_task.PreProcessing() &&
                        empty_task.Run() && empty_task.PostProcessing();
    
    ASSERT_TRUE(empty_success);
    ASSERT_EQ(empty_task.GetOutput(), 0);
    ASSERT_EQ(empty_task.GetDiffCount(), 0);
}

TEST(sosnina_a_diff_count_seq, coverage_different_lengths) {
    std::pair<std::string, std::string> diff_len_input = {"a", "bbb"};
    SosninaADiffCountSEQ diff_len_task(diff_len_input);
    
    bool diff_len_success = diff_len_task.Validation() && diff_len_task.PreProcessing() &&
                           diff_len_task.Run() && diff_len_task.PostProcessing();
    
    ASSERT_TRUE(diff_len_success);
    ASSERT_GT(diff_len_task.GetOutput(), 0);
}

TEST(sosnina_a_diff_count_mpi, get_diff_count_method) {
    std::pair<std::string, std::string> input = {"hello", "hxllo"};
    SosninaADiffCountMPI task(input);
    
    bool success = task.Validation() && task.PreProcessing() &&
                  task.Run() && task.PostProcessing();
    
    ASSERT_TRUE(success);
    ASSERT_EQ(task.GetOutput(), 1);
    ASSERT_EQ(task.GetDiffCount(), 1);
}

TEST(sosnina_a_diff_count_seq, get_diff_count_method) {
    std::pair<std::string, std::string> input = {"test", "text"};
    SosninaADiffCountSEQ task(input);
    
    bool success = task.Validation() && task.PreProcessing() &&
                  task.Run() && task.PostProcessing();
    
    ASSERT_TRUE(success);
    ASSERT_EQ(task.GetOutput(), 1);
    ASSERT_EQ(task.GetDiffCount(), 1);
}

} 

}  // namespace sosnina_a_diff_count
