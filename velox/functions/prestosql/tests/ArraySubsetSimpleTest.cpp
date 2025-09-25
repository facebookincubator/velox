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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class ArraySubsetSimpleTest : public test::FunctionBaseTest {};

TEST_F(ArraySubsetSimpleTest, basicTest) {
  auto inputArray = makeArrayVector<int32_t>({{1, 2, 3, 4, 5}});
  auto indices = makeArrayVector<int32_t>({{1, 3, 5}});
  auto expected = makeArrayVector<int32_t>({{1, 3, 5}});

  auto result =
      evaluate("array_subset(c0, c1)", makeRowVector({inputArray, indices}));
  assertEqualVectors(expected, result);
}

TEST_F(ArraySubsetSimpleTest, nullTest) {
  // Test array_subset(array[null,1], array[1]) = array[null]
  std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {std::nullopt, 1}};
  auto inputArray = makeNullableArrayVector(inputData);
  auto indices = makeArrayVector<int32_t>({{1}});
  std::vector<std::vector<std::optional<int32_t>>> expectedData = {
      {std::nullopt}};
  auto expected = makeNullableArrayVector(expectedData);

  auto result =
      evaluate("array_subset(c0, c1)", makeRowVector({inputArray, indices}));
  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions
