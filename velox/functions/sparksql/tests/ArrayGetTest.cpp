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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayGetTest : public SparkFunctionBaseTest {
 protected:
  template <typename T, typename IndexType>
  std::optional<T> arrayGet(
      const ArrayVectorPtr& arrayVector,
      const std::optional<IndexType>& index,
      const std::string& functionName = "get") {
    auto input =
        makeRowVector({arrayVector, makeConstant(index, arrayVector->size())});
    return evaluateOnce<T>(fmt::format("{}(c0, c1)", functionName), input);
  }

  template <typename IndexType>
  void testArrayGet() {
    auto arrayVector = makeNullableArrayVector<int32_t>({{1, std::nullopt, 2}});

    auto result = arrayGet<int32_t, IndexType>(arrayVector, 0);
    EXPECT_EQ(result, 1);
    result = arrayGet<int32_t, IndexType>(arrayVector, 1);
    EXPECT_EQ(result, std::nullopt);
    result = arrayGet<int32_t, IndexType>(arrayVector, 2);
    EXPECT_EQ(result, 2);

    result = arrayGet<int32_t, IndexType>(arrayVector, -1);
    EXPECT_EQ(result, std::nullopt);
    result = arrayGet<int32_t, IndexType>(arrayVector, 3);
    EXPECT_EQ(result, std::nullopt);
    result = arrayGet<int32_t, IndexType>(arrayVector, std::nullopt);
    EXPECT_EQ(result, std::nullopt);
  }
};

TEST_F(ArrayGetTest, basic) {
  testArrayGet<int8_t>();
  testArrayGet<int16_t>();
  testArrayGet<int32_t>();
  testArrayGet<int64_t>();
}

TEST_F(ArrayGetTest, getArrayItem) {
  auto arrayVector = makeNullableArrayVector<int32_t>({{1, std::nullopt, 2}});
  auto getArrayItem = [&](std::optional<int64_t> index) {
    return arrayGet<int32_t, int64_t>(arrayVector, index, "get_array_item");
  };

  for (const auto& ansiEnabled : {"false", "true"}) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          ansiEnabled}});
    EXPECT_EQ(getArrayItem(0), 1);
    EXPECT_EQ(getArrayItem(1), std::nullopt);
    EXPECT_EQ(getArrayItem(2), 2);
    EXPECT_EQ(getArrayItem(std::nullopt), std::nullopt);

    // All integer index types are supported.
    EXPECT_EQ((arrayGet<int32_t, int8_t>(arrayVector, 2, "get_array_item")), 2);
    EXPECT_EQ(
        (arrayGet<int32_t, int16_t>(arrayVector, 2, "get_array_item")), 2);
    EXPECT_EQ(
        (arrayGet<int32_t, int32_t>(arrayVector, 2, "get_array_item")), 2);
  }

  // ANSI disabled: a negative or out-of-bounds index returns NULL.
  queryCtx_->testingOverrideConfigUnsafe(
      {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled), "false"}});
  EXPECT_EQ(getArrayItem(-1), std::nullopt);
  EXPECT_EQ(getArrayItem(3), std::nullopt);

  // ANSI enabled: throws. get keeps returning NULL.
  queryCtx_->testingOverrideConfigUnsafe(
      {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled), "true"}});
  VELOX_ASSERT_THROW(
      getArrayItem(-1), "Array subscript index cannot be negative");
  VELOX_ASSERT_THROW(getArrayItem(3), "Array subscript index out of bounds");
  EXPECT_EQ((arrayGet<int32_t, int64_t>(arrayVector, -1)), std::nullopt);
  EXPECT_EQ((arrayGet<int32_t, int64_t>(arrayVector, 3)), std::nullopt);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
