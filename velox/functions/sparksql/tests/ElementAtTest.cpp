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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class ElementAtTest : public SparkFunctionBaseTest {
 protected:
  template <typename T = int64_t>
  std::optional<T> elementAtSimple(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters) {
    auto result =
        evaluate<SimpleVector<T>>(expression, makeRowVector(parameters));
    if (result->size() != 1) {
      throw std::invalid_argument(
          "elementAtSimple expects a single output row.");
    }
    if (result->isNullAt(0)) {
      return std::nullopt;
    }
    return result->valueAt(0);
  }
};

} // namespace

// Spark's element_at ("a[1]") behavior:
// This behavior is only when spark.sql.ansi.enabled = false.
// #1 - start indices at 1. If Index is 0 will throw an error.
// #2 - allow out of bounds access for arrays (return null).
// #3 - allow negative indices (return elements from the last to the first).
TEST_F(ElementAtTest, allFlavors2) {
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});

  // Create a simple vector containing a single map ([10=>10, 11=>11, 12=>12]).
  auto keyAt = [](auto idx) { return idx + 10; };
  auto sizeAt = [](auto) { return 3; };
  auto mapValueAt = [](auto idx) { return idx + 10; };
  auto mapVector =
      makeMapVector<int64_t, int64_t>(1, sizeAt, keyAt, mapValueAt);

  // #1
  EXPECT_EQ(elementAtSimple("element_at(C0, 1)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("element_at(C0, 2)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("element_at(C0, 3)", {arrayVector}), 12);
  VELOX_ASSERT_THROW(
      elementAtSimple("element_at(C0, 0)", {arrayVector}),
      "SQL array indices start at 1");
  // #2
  EXPECT_EQ(elementAtSimple("element_at(C0, 4)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("element_at(C0, 5)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("element_at(C0, 1001)", {mapVector}), std::nullopt);

  // #3
  EXPECT_EQ(elementAtSimple("element_at(C0, -1)", {arrayVector}), 12);
  EXPECT_EQ(elementAtSimple("element_at(C0, -2)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("element_at(C0, -3)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("element_at(C0, -4)", {arrayVector}), std::nullopt);
}

// Spark's checked_element_at behavior (spark.sql.ansi.enabled = true):
// #1 - start indices at 1. If Index is 0 will throw an error.
// #2 - throw on out of bounds access for arrays (instead of returning null).
// #3 - throw on missing map key (instead of returning null).
// #4 - allow negative indices (return elements from the last to the first).
// #5 - throw when negative index is out of bounds.
TEST_F(ElementAtTest, checkedElementAt) {
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});

  // Create a simple vector containing a single map ([10=>10, 11=>11, 12=>12]).
  auto keyAt = [](auto idx) { return idx + 10; };
  auto sizeAt = [](auto) { return 3; };
  auto mapValueAt = [](auto idx) { return idx + 10; };
  auto mapVector =
      makeMapVector<int64_t, int64_t>(1, sizeAt, keyAt, mapValueAt);

  // #1 - Normal array access works.
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 1)", {arrayVector}), 10);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 2)", {arrayVector}), 11);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 3)", {arrayVector}), 12);

  // Index 0 throws.
  VELOX_ASSERT_THROW(
      elementAtSimple("checked_element_at(C0, 0)", {arrayVector}),
      "SQL array indices start at 1");

  // #2 - Out of bounds throws instead of returning null.
  VELOX_ASSERT_THROW(
      elementAtSimple("checked_element_at(C0, 4)", {arrayVector}),
      "Array subscript index out of bounds");
  VELOX_ASSERT_THROW(
      elementAtSimple("checked_element_at(C0, 5)", {arrayVector}),
      "Array subscript index out of bounds");

  // #3 - Missing map key throws instead of returning null.
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 10)", {mapVector}), 10);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 11)", {mapVector}), 11);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, 12)", {mapVector}), 12);
  VELOX_ASSERT_THROW(
      elementAtSimple("checked_element_at(C0, 1001)", {mapVector}),
      "Key not found in map");

  // #4 - Negative indices still work.
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, -1)", {arrayVector}), 12);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, -2)", {arrayVector}), 11);
  EXPECT_EQ(
      elementAtSimple("checked_element_at(C0, -3)", {arrayVector}), 10);

  // #5 - Negative index out of bounds throws.
  VELOX_ASSERT_THROW(
      elementAtSimple("checked_element_at(C0, -4)", {arrayVector}),
      "Array subscript index out of bounds");
}

} // namespace facebook::velox::functions::sparksql::test
