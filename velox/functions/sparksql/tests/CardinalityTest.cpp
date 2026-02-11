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
class CardinalityTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<int32_t> cardinalityArray(
      const std::vector<std::optional<T>>& input) {
    auto row = makeRowVector({makeNullableArrayVector(
        std::vector<std::vector<std::optional<T>>>{input})});
    return evaluateOnce<int32_t>("cardinality(c0)", row);
  }
};

TEST_F(CardinalityTest, arrayBasic) {
  EXPECT_EQ(cardinalityArray<int32_t>({1, 2, 3}), 3);
  EXPECT_EQ(cardinalityArray<int32_t>({1}), 1);
  EXPECT_EQ(cardinalityArray<int32_t>({}), 0);
}

TEST_F(CardinalityTest, arrayWithNullElements) {
  // NULL elements are counted.
  EXPECT_EQ(cardinalityArray<int32_t>({1, std::nullopt, 3}), 3);
  EXPECT_EQ(cardinalityArray<int32_t>({std::nullopt}), 1);
  EXPECT_EQ(
      cardinalityArray<int32_t>({std::nullopt, std::nullopt, std::nullopt}), 3);
}

TEST_F(CardinalityTest, arrayNullInput) {
  // NULL array returns NULL.
  auto result = evaluateOnce<int32_t>(
      "cardinality(c0)",
      makeRowVector(
          {BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool())}));
  EXPECT_FALSE(result.has_value());
}

TEST_F(CardinalityTest, arrayStringElements) {
  EXPECT_EQ(cardinalityArray<std::string>({"a", "b", "c"}), 3);
  EXPECT_EQ(cardinalityArray<std::string>({}), 0);
}

TEST_F(CardinalityTest, mapBasic) {
  auto mapVector = makeMapVector<int32_t, int32_t>({{{1, 10}, {2, 20}}});
  auto result =
      evaluateOnce<int32_t>("cardinality(c0)", makeRowVector({mapVector}));
  EXPECT_EQ(result, 2);

  auto emptyMap = makeMapVector<int32_t, int32_t>({{}});
  result = evaluateOnce<int32_t>("cardinality(c0)", makeRowVector({emptyMap}));
  EXPECT_EQ(result, 0);
}

TEST_F(CardinalityTest, mapNullInput) {
  // NULL map returns NULL.
  auto result = evaluateOnce<int32_t>(
      "cardinality(c0)",
      makeRowVector({BaseVector::createNullConstant(
          MAP(INTEGER(), INTEGER()), 1, pool())}));
  EXPECT_FALSE(result.has_value());
}

} // namespace facebook::velox::functions::sparksql::test
