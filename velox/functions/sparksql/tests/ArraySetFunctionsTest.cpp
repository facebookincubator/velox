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
#include "velox/functions/sparksql/tests/CanonicalFloatingPoint.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

// Tests that array_distinct, array_intersect and array_except return -0.0 as
// 0.0 and every NaN as the canonical NaN, as Spark does.
class ArraySetFunctionsTest : public SparkFunctionBaseTest {
 protected:
  static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
    expectCanonicalFloatingPoint(*result);
  }

  VectorPtr makeNestedDoubleArray(
      const std::vector<std::vector<std::vector<double>>>& data) {
    std::vector<std::optional<
        std::vector<std::optional<std::vector<std::optional<double>>>>>>
        rows;
    for (const auto& row : data) {
      std::vector<std::optional<std::vector<std::optional<double>>>> arrays;
      for (const auto& array : row) {
        arrays.push_back(
            std::vector<std::optional<double>>(array.begin(), array.end()));
      }
      rows.push_back(arrays);
    }
    return makeNullableNestedArrayVector<double>(rows);
  }
};

TEST_F(ArraySetFunctionsTest, arrayDistinct) {
  testExpression(
      "array_distinct(c0)",
      {makeNullableArrayVector<double>({
          {-0.0, 0.0, kNonCanonicalNaN, std::nullopt, kNaN, std::nullopt},
          {1.0, -0.0},
          {-0.0},
      })},
      makeNullableArrayVector<double>({
          {0.0, kNaN, std::nullopt},
          {1.0, 0.0},
          {0.0},
      }));

  testExpression(
      "array_distinct(c0)",
      {makeArrayVector<float>({{-0.0f, 0.0f, std::nanf("")}})},
      makeArrayVector<float>({{0.0f, std::nanf("")}}));
}

TEST_F(ArraySetFunctionsTest, arrayIntersect) {
  testExpression(
      "array_intersect(c0, c1)",
      {makeArrayVector<double>({{-0.0, kNonCanonicalNaN, 2.0}, {-0.0}}),
       makeArrayVector<double>({{0.0, kNaN}, {0.0}})},
      makeArrayVector<double>({{0.0, kNaN}, {0.0}}));
}

TEST_F(ArraySetFunctionsTest, arrayExcept) {
  testExpression(
      "array_except(c0, c1)",
      {makeArrayVector<double>(
           {{-0.0, 1.0}, {-0.0, 1.0, kNonCanonicalNaN}, {-0.0}}),
       makeArrayVector<double>({{2.0}, {0.0}, {0.0}})},
      makeArrayVector<double>({{0.0, 1.0}, {1.0, kNaN}, {}}));
}

TEST_F(ArraySetFunctionsTest, nested) {
  testExpression(
      "array_distinct(c0)",
      {makeNestedDoubleArray({{{-0.0}, {0.0}}, {{1.0, -0.0}}})},
      makeNestedDoubleArray({{{0.0}}, {{1.0, 0.0}}}));

  testExpression(
      "array_intersect(c0, c1)",
      {makeNestedDoubleArray({{{-0.0}}}), makeNestedDoubleArray({{{0.0}}})},
      makeNestedDoubleArray({{{0.0}}}));

  testExpression(
      "array_except(c0, c1)",
      {makeNestedDoubleArray({{{-0.0, 1.0}}}),
       makeNestedDoubleArray({{{2.0}}})},
      makeNestedDoubleArray({{{0.0, 1.0}}}));

  auto structs = makeRowVector({
      makeFlatVector<double>({-0.0, 0.0, -0.0, 1.0}),
      makeFlatVector<int32_t>({1, 1, 2, 2}),
  });
  testExpression(
      "array_distinct(c0)",
      {makeArrayVector({0, 2}, structs)},
      makeArrayVector(
          {0, 1},
          makeRowVector({
              makeFlatVector<double>({0.0, 0.0, 1.0}),
              makeFlatVector<int32_t>({1, 2, 2}),
          })));
}

TEST_F(ArraySetFunctionsTest, noFloatingPoint) {
  testExpression(
      "array_distinct(c0)",
      {makeArrayVector<int64_t>({{1, 2, 1}})},
      makeArrayVector<int64_t>({{1, 2}}));
  testExpression(
      "array_except(c0, c1)",
      {makeArrayVector<int64_t>({{1, 2, 3}}), makeArrayVector<int64_t>({{2}})},
      makeArrayVector<int64_t>({{1, 3}}));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
