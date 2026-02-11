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
          makeFlatVector<int64_t>({1}),
          makeFlatVector<int64_t>({10}),
          makeFlatVector<int64_t>({3}),
      }));
  auto expected = makeArrayVector<int64_t>({{1, 4, 7, 10}});
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

} // namespace facebook::velox::functions::sparksql::test
