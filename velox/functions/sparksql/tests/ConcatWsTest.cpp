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

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ConcatWsFunctionsTest : public SparkFunctionBaseTest {
 protected:
  template <class... Args>
  void testVariadic(const char* result, Args... args) {
    int size = sizeof...(args);
    std::string expr("concat_ws(");
    std::string c("c");
    for (int i = 0; i < size; ++i) {
      expr += c + std::to_string(i) + " ,";
    }
    expr.size() > 10               ? expr = expr.substr(0, expr.size() - 2),
                       expr += ")" : expr += ")";
    EXPECT_EQ(
        result,
        evaluateOnce<std::string>(expr, std::optional<std::string>(args)...));
  }
  void testArray(
      const char* result,
      const std::vector<std::vector<std::optional<std::string>>>& data) {
    auto row = makeRowVector({makeNullableArrayVector<std::string>(data)});
    EXPECT_EQ(result, evaluateOnce<std::string>("concat_ws(c0)", row));
  }
};
TEST_F(ConcatWsFunctionsTest, ConcatWs) {
  testVariadic("1+2+3", "+", "1", "2", "3");
  testVariadic("1_2_3_4_5", "_", "1", "2", "3", "4", "5");

  testArray("121_221", {{"_", "121", "221"}});
  testArray("121+221+43+17", {{"+", "121", "221", "43", "17"}});
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
