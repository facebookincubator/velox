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

#include <stdint.h>

namespace facebook::velox::functions::sparksql::test {
namespace {

class MapFromArraysTest : public SparkFunctionBaseTest {
 protected:
  template <typename K = int64_t, typename V = std::string>
  void testMapFromArrays(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters,
      const VectorPtr& expected) {
    auto result = evaluate<MapVector>(expression, makeRowVector(parameters));
    ::facebook::velox::test::assertEqualVectors(expected, result);
  }

  void testMapFromArraysFails(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters,
      const std::string errorMsg) {
    VELOX_ASSERT_USER_THROW(
        evaluate<MapVector>(expression, makeRowVector(parameters)), errorMsg);
  }
};

TEST_F(MapFromArraysTest, Basics) {
  auto inputVector1 = makeArrayVector<int32_t>({
    {1, 2},
    {3, 4, 5}
  });
  auto inputVector2 = makeArrayVector<StringView>({
    {"a", "b"},
    {"c", "d", "e"}
  });

  auto mapVector = makeMapVector<int32_t, StringView>({
    {{1, "a"}, {2, "b"}},
    {{3, "c"}, {4, "d"}, {5, "e"}}
  });

  testMapFromArrays("map_from_arrays(c0, c1)", {inputVector1, inputVector2}, mapVector);
}

TEST_F(MapFromArraysTest, DuplicateKeysLastWin) {
  setSparkMapKeyDupPolicy("LAST_WIN");
  auto inputVector1 = makeArrayVector<int32_t>({
    {1, 1},
    {3, 4, 3}
  });
  auto inputVector2 = makeArrayVector<StringView>({
    {"a", "b"},
    {"c", "d", "e"}
  });

  auto mapVector = makeMapVector<int32_t, StringView>({
    {{1, "b"}},
    {{3, "e"}, {4, "d"}}
  });

  testMapFromArrays("map_from_arrays(c0, c1)", {inputVector1, inputVector2}, mapVector);
}

TEST_F(MapFromArraysTest, DuplicateKeysThrowException) {
  setSparkMapKeyDupPolicy("EXCEPTION");
  auto inputVector1 = makeArrayVector<int32_t>({
    {1, 1},
    {3, 4, 3}
  });
  auto inputVector2 = makeArrayVector<StringView>({
    {"a", "b"},
    {"c", "d", "e"}
  });

  auto errorMsg = "Duplicate map keys (1) are not allowed";

  testMapFromArraysFails("map_from_arrays(c0, c1)", {inputVector1, inputVector2}, errorMsg);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
