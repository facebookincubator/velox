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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayInsertTest : public SparkFunctionBaseTest {
 protected:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    // For parsing literal integers as INTEGER, not BIGINT.
    options_.parseIntegerAsBigint = false;
  }

  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    const auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  void evaluateExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    evaluate(expression, makeRowVector(input));
  }
};

TEST_F(ArrayInsertTest, nullSrcArrays) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"null"});
  const auto expected = makeArrayVectorFromJson<int64_t>({"null"});
  testExpression("array_insert(c0, 1, 1, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, nullPosition) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"[1, 1]"});
  const auto expected = makeArrayVectorFromJson<int64_t>({"null"});
  testExpression(
      "array_insert(c0, null::INTEGER, 1, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, basic) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]"});

  auto expected = makeArrayVectorFromJson<int64_t>({"[0, 1]", "[0, 2, 2]"});
  testExpression("array_insert(c0, 1, 0, false)", {arrays}, expected);

  expected = makeArrayVectorFromJson<int64_t>({"[null, 1]", "[null, 2, 2]"});
  testExpression(
      "array_insert(c0, 1, null::INTEGER, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, posGTArraySize) {
  const auto arrays = makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]"});

  auto expected =
      makeArrayVectorFromJson<int64_t>({"[1, null, 0]", "[2, 2, 0]"});
  testExpression("array_insert(c0, 3, 0, false)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[1, null, null]", "[2, 2, null]"});
  testExpression(
      "array_insert(c0, 3, null::INTEGER, false)", {arrays}, expected);

  // Insert into position INT_MAX
  EXPECT_THROW(evaluateExpression(
      "array_insert(c0, 2147483647, 0, false)", {arrays}), VeloxUserError);
  // Insert into position INT_MIN
  EXPECT_THROW(evaluateExpression(
      "array_insert(c0, cast(-2147483648 as integer), 0, false)", {arrays}), VeloxUserError);
}

TEST_F(ArrayInsertTest, negativePos) {
  const auto arrays =
      makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]", "[3, 3, 3]"});

  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[0, null, 1]", "[0, 2, 2]", "[3, 0, 3, 3]"});
  testExpression("array_insert(c0, -3, 0, false)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[1, 0]", "[2, 2, 0]", "[3, 3, 3, 0]"});
  testExpression("array_insert(c0, -1, 0, false)", {arrays}, expected);

  expected = makeArrayVectorFromJson<int64_t>(
      {"[null, null, 1]", "[null, 2, 2]", "[3, null, 3, 3]"});
  testExpression(
      "array_insert(c0, -3, null::INTEGER, false)", {arrays}, expected);
}

TEST_F(ArrayInsertTest, negativePosLegacy) {
  const auto arrays =
      makeArrayVectorFromJson<int64_t>({"[1]", "[2, 2]", "[3, 3, 3]"});

  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[0, null, null, 1]", "[0, null, 2, 2]", "[0, 3, 3, 3]"});
  testExpression("array_insert(c0, -3, 0, true)", {arrays}, expected);

  expected =
      makeArrayVectorFromJson<int64_t>({"[0, 1]", "[2, 0, 2]", "[3, 3, 0, 3]"});
  testExpression("array_insert(c0, -1, 0, true)", {arrays}, expected);

  expected = makeArrayVectorFromJson<int64_t>(
      {"[null, null, null, 1]", "[null, null, 2, 2]", "[null, 3, 3, 3]"});
  testExpression(
      "array_insert(c0, -3, null::INTEGER, true)", {arrays}, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
