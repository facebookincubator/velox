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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

static constexpr auto kNaNDouble = std::numeric_limits<double>::quiet_NaN();
static constexpr auto kNaNFloat = std::numeric_limits<float>::quiet_NaN();
static constexpr auto kMaxDouble = std::numeric_limits<double>::max();
static constexpr auto kMaxFloat = std::numeric_limits<float>::max();

class AtLeastNNonNullsTest : public SparkFunctionBaseTest {
 protected:
  template <typename... Arguments>
  std::optional<bool> atLeastNNonNulls(
      const std::optional<int32_t>& n,
      const std::optional<Arguments>&... args) {
    constexpr auto numArgs{sizeof...(Arguments)};
    std::string func = "at_least_n_non_nulls(c0";
    for (auto i = 1; i <= numArgs; ++i) {
      func += fmt::format(", c{}", i);
    }
    func += ")";
    return evaluateOnce<bool>(func, n, args...);
  }
};

TEST_F(AtLeastNNonNullsTest, basic) {
  auto result =
      atLeastNNonNulls<std::string, std::string, double, double, float>(
          2, "x", std::nullopt, std::nullopt, kNaNDouble, 0.5f);
  EXPECT_EQ(result, true);

  result = atLeastNNonNulls<std::string, std::string, double, double, float>(
      3, "x", std::nullopt, std::nullopt, kNaNDouble, 0.5f);
  EXPECT_EQ(result, false);

  result = atLeastNNonNulls<std::string, double, float, double, double>(
      3, "x", 10.0, kNaNFloat, std::log(-2.0), kMaxDouble);
  EXPECT_EQ(result, true);

  result = atLeastNNonNulls<std::string, double, float, double, double>(
      4, "x", 10.0, kNaNFloat, std::log(-2.0), kMaxDouble);
  EXPECT_EQ(result, false);

  result = atLeastNNonNulls<std::string, double, int64_t, float, bool>(
      3, "x", std::nullopt, std::nullopt, kMaxFloat, false);
  EXPECT_EQ(result, true);

  result = atLeastNNonNulls<std::string, double, int64_t, float, bool>(
      4, "x", std::nullopt, std::nullopt, kMaxFloat, false);
  EXPECT_EQ(result, false);

  result = atLeastNNonNulls<std::string, Timestamp, int32_t, float, bool>(
      4, "x", std::nullopt, std::nullopt, kMaxFloat, false);
  EXPECT_EQ(result, false);

  result = atLeastNNonNulls<std::string, Timestamp, int8_t, float, bool>(
      4, "x", std::nullopt, std::nullopt, kMaxFloat, false);
  EXPECT_EQ(result, false);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
