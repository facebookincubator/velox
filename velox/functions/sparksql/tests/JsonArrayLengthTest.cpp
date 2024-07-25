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
#include <utility>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class JsonArrayLengthTest : public SparkFunctionBaseTest {
 protected:
  std::optional<int32_t> jsonArrayLength(
      const std::optional<std::string>& json) {
    return evaluateOnce<int32_t>("json_array_length(c0)", json);
  }
};

TEST_F(JsonArrayLengthTest, basic) {
  EXPECT_EQ(jsonArrayLength(R"([])"), 0);
  EXPECT_EQ(jsonArrayLength(R"([1])"), 1);
  EXPECT_EQ(jsonArrayLength(R"([1, 2, 3])"), 3);
  EXPECT_EQ(
      jsonArrayLength(
          R"([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])"),
      20);

  EXPECT_EQ(jsonArrayLength(R"(1)"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"("hello")"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"("")"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"(true)"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":"v1"})"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":[0,1,2]})"), std::nullopt);
  EXPECT_EQ(jsonArrayLength(R"({"k1":[0,1,2], "k2":"v1"})"), std::nullopt);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
