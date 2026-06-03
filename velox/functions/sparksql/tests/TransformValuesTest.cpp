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

using namespace facebook::velox::test;

class TransformValuesTest : public SparkFunctionBaseTest {};

TEST_F(TransformValuesTest, basic) {
  auto input =
      makeMapVector<int32_t, int64_t>({{{1, 10}, {2, 20}}, {{3, 30}, {4, 40}}});
  auto result = evaluate<MapVector>(
      "transform_values(c0, (k, v) -> v * 2)", makeRowVector({input}));
  auto expected =
      makeMapVector<int32_t, int64_t>({{{1, 20}, {2, 40}}, {{3, 60}, {4, 80}}});
  assertEqualVectors(expected, result);
}

TEST_F(TransformValuesTest, keyAndValueInLambda) {
  // Uses multiply instead of + because the Spark registry registers addition
  // as "add" while velox's expression parser resolves + as "plus", which is
  // not registered in the Spark context.
  auto input = makeMapVector<int32_t, int64_t>({{{1, 10}, {2, 20}}, {{3, 30}}});
  auto result = evaluate<MapVector>(
      "transform_values(c0, (k, v) -> v * k)", makeRowVector({input}));
  auto expected =
      makeMapVector<int32_t, int64_t>({{{1, 10}, {2, 40}}, {{3, 90}}});
  assertEqualVectors(expected, result);
}

TEST_F(TransformValuesTest, nullMap) {
  // Null input map rows pass through as null output rows.
  vector_size_t size = 4;
  auto input = makeMapVector<int32_t, int64_t>(
      size,
      [](auto /* row */) { return 2; },
      [](auto row) { return row % 4; },
      [](auto row) { return row * 10LL; },
      nullEvery(2));
  auto result = evaluate<MapVector>(
      "transform_values(c0, (k, v) -> v * 2)", makeRowVector({input}));
  auto expected = makeMapVector<int32_t, int64_t>(
      size,
      [](auto /* row */) { return 2; },
      [](auto row) { return row % 4; },
      [](auto row) { return row * 20LL; },
      nullEvery(2));
  assertEqualVectors(expected, result);
}

TEST_F(TransformValuesTest, nullValueInMap) {
  // Null values inside a non-null map produce null outputs; the lambda is
  // applied to every entry regardless.
  auto input =
      makeMapVectorFromJson<int32_t, int64_t>({"{1: 10, 2: null, 3: 30}"});
  auto result = evaluate<MapVector>(
      "transform_values(c0, (k, v) -> v * 2)", makeRowVector({input}));
  auto expected =
      makeMapVectorFromJson<int32_t, int64_t>({"{1: 20, 2: null, 3: 60}"});
  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
