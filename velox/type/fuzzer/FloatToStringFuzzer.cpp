/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/type/Conversions.h"

#include <filesystem>
#include <iostream>
#include <tuple>
#include <vector>
#include "boost/process.hpp"
#include "fmt/core.h"
#include "gtest/gtest.h"

namespace bp = boost::process;

namespace {
template <typename T>
std::tuple<bool, std::vector<T>, std::vector<std::string>>
generateFloatTestCases(int count) {
  auto workDir = std::filesystem::canonical("/proc/self/exe").parent_path();
  bp::ipstream pipeStream;
  bp::child c(
      fmt::format(
          "java FloatGenerator {} {}",
          std::is_same_v<T, double> ? "double" : "float",
          count),
      bp::start_dir(workDir.string()),
      bp::std_out > pipeStream);

  std::string buggyJavaVersion;
  std::vector<T> values;
  std::vector<std::string> expects;
  values.reserve(count);
  expects.reserve(count);

  pipeStream >> buggyJavaVersion;

  std::conditional_t<std::is_same_v<T, double>, int64_t, int32_t> carrierInt;
  std::string expect;

  while (pipeStream >> expect >> carrierInt) {
    values.emplace_back(reinterpret_cast<const T&>(carrierInt));
    expects.emplace_back(expect);
  }

  c.wait();
  int result = c.exit_code();
  if (result != 0) {
    throw std::runtime_error(
        fmt::format("Process exited with code: {}", result));
  }

  return {buggyJavaVersion == "true", values, expects};
}

template <typename T>
void testCastToVarchar(
    const std::vector<T> values,
    const std::vector<std::string> expects,
    bool buggyJavaVersion) {
  using namespace facebook::velox;
  util::Converter<TypeKind::VARCHAR> convertor;

  ASSERT_EQ(values.size(), expects.size());

  for (int i = 0; i < values.size(); i++) {
    const auto& value = values[i];
    const auto& expect = expects[i];
    auto actual = convertor.tryCast(value).value_or("");

    // Old java (< 19) may produce longer or incorrect decimal.
    // See https://bugs.openjdk.org/browse/JDK-4511638.
    // e.g.
    // Actual       | JDK <= 18            | JDK 19   
    // 7.5371334E25 | 7.5371335E25         | 7.5371334E25  # incorrect
    // 1.0E23       | 9.999999999999999E22 | 1.0E23        # longer
    if (buggyJavaVersion) {
      EXPECT_TRUE(
          actual == expect ||
          // Shorter but same decimal
          (actual.size() <= expect.size() &&
           (std::is_same_v<T, double> ? std::stod(actual)
                                      : std::stof(actual)) == value));

      if (actual != expect) {
        std::cerr << "Warning: " << actual << " != " << expect << std::endl;
      }
    } else {
      EXPECT_EQ(actual, expect);
    }
  }
}
} // namespace

TEST(FloatToString, float) {
  auto [buggyJavaVersion, values, expects] =
      generateFloatTestCases<float>(10'000);
  testCastToVarchar(values, expects, buggyJavaVersion);
}

TEST(FloatToString, double) {
  auto [buggyJavaVersion, values, expects] =
      generateFloatTestCases<double>(10'000);
  testCastToVarchar(values, expects, buggyJavaVersion);
}
