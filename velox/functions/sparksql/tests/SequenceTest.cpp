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
#include <optional>

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
using namespace facebook::velox::test;

class SequenceTest : public SparkFunctionBaseTest {};

TEST_F(SequenceTest, ascending) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({1, 3}),
          makeFlatVector<int64_t>({5, 6}),
      }));
  auto expected = makeArrayVector<int64_t>({{1, 2, 3, 4, 5}, {3, 4, 5, 6}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, descending) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({5}),
          makeFlatVector<int64_t>({1}),
      }));
  auto expected = makeArrayVector<int64_t>({{5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, withStep) {
  auto result = evaluate(
      "sequence(c0, c1, c2)",
      makeRowVector({
          makeFlatVector<int64_t>({1, 5}),
          makeFlatVector<int64_t>({10, 1}),
          makeFlatVector<int64_t>({3, -2}),
      }));
  auto expected = makeArrayVector<int64_t>({{1, 4, 7, 10}, {5, 3, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, singleElement) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({5}),
          makeFlatVector<int64_t>({5}),
      }));
  auto expected = makeArrayVector<int64_t>({{5}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, integerType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int32_t>({1, 5}),
          makeFlatVector<int32_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int32_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, integerWithStep) {
  auto result = evaluate(
      "sequence(c0, c1, c2)",
      makeRowVector({
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int32_t>({9}),
          makeFlatVector<int32_t>({2}),
      }));
  auto expected = makeArrayVector<int32_t>({{1, 3, 5, 7, 9}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, smallintType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int16_t>({1, 5}),
          makeFlatVector<int16_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int16_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, tinyintType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int8_t>({1, 5}),
          makeFlatVector<int8_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int8_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, nullInputs) {
  auto start = makeNullableFlatVector<int64_t>({std::nullopt, 1, 1});
  auto stop = makeNullableFlatVector<int64_t>({5, std::nullopt, 3});
  auto result = evaluate("sequence(c0, c1)", makeRowVector({start, stop}));

  auto expected = makeNullableArrayVector<int64_t>(
      {std::nullopt,
       std::nullopt,
       std::optional<std::vector<std::optional<int64_t>>>({{1, 2, 3}})});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, nullStep) {
  auto start = makeNullableFlatVector<int64_t>({1});
  auto stop = makeNullableFlatVector<int64_t>({5});
  auto step = makeNullableFlatVector<int64_t>({std::nullopt});
  auto result =
      evaluate("sequence(c0, c1, c2)", makeRowVector({start, stop, step}));

  auto expected = makeNullableArrayVector<int64_t>(
      {std::optional<std::vector<std::optional<int64_t>>>(std::nullopt)});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, stepZeroError) {
  VELOX_ASSERT_THROW(
      evaluate(
          "sequence(c0, c1, c2)",
          makeRowVector({
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({5}),
              makeFlatVector<int64_t>({0}),
          })),
      "step must not be zero");
}

TEST_F(SequenceTest, wrongDirectionError) {
  VELOX_ASSERT_THROW(
      evaluate(
          "sequence(c0, c1, c2)",
          makeRowVector({
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({5}),
              makeFlatVector<int64_t>({-1}),
          })),
      "sequence stop value should be greater than or equal to start value if "
      "step is greater than zero otherwise stop should be less than or equal "
      "to start");
}

} // namespace facebook::velox::functions::sparksql::test
