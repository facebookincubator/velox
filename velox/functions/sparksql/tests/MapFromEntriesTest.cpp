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
#include <bit>
#include <cstdint>
#include <limits>

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/ArrayConstructor.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
std::optional<std::vector<std::pair<int32_t, std::optional<int32_t>>>>
makeOptional(
    const std::vector<std::pair<int32_t, std::optional<int32_t>>>& vector) {
  return std::make_optional(vector);
}

class MapFromEntriesTest : public SparkFunctionBaseTest {
 protected:
  // Create an MAP vector of size 1 using specified 'keys' and 'values' vector.
  VectorPtr makeSingleRowMapVector(
      const VectorPtr& keys,
      const VectorPtr& values) {
    BufferPtr offsets = allocateOffsets(1, pool());
    BufferPtr sizes = allocateSizes(1, pool());
    sizes->asMutable<vector_size_t>()[0] = keys->size();

    return std::make_shared<MapVector>(
        pool(),
        MAP(keys->type(), values->type()),
        nullptr,
        1,
        offsets,
        sizes,
        keys,
        values);
  }

  void verifyMapFromEntries(const VectorPtr& input, const VectorPtr& expected) {
    const std::string expr = fmt::format("map_from_entries({})", "c0");
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expected, result);
  }

  // Checks the resulting keys and values in entry order, which comparing whole
  // maps ignores.
  void verifyEntryOrder(
      const VectorPtr& input,
      const VectorPtr& expectedKeys,
      const VectorPtr& expectedValues) {
    auto data = makeRowVector({input});
    assertEqualVectors(
        expectedKeys, evaluate("map_keys(map_from_entries(c0))", data));
    assertEqualVectors(
        expectedValues, evaluate("map_values(map_from_entries(c0))", data));
  }
};
} // namespace

TEST_F(MapFromEntriesTest, nullMapEntries) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  {
    std::vector<std::vector<std::optional<std::tuple<int32_t, int32_t>>>> data =
        {
            {std::nullopt},
            {{{1, 11}}},
        };
    auto input = makeArrayOfRowVector(data, rowType);
    auto expected = makeNullableMapVector<int32_t, int32_t>(
        {std::nullopt, makeOptional({{1, 11}})});
    verifyMapFromEntries(input, expected);
  }
  {
    // Create array(row(a,b)) where a, b sizes are 0 because all row(a, b)
    // values are null.
    std::vector<std::vector<std::optional<std::tuple<int32_t, int32_t>>>> data =
        {
            {std::nullopt, std::nullopt, std::nullopt},
            {std::nullopt},
        };
    auto input = makeArrayOfRowVector(data, rowType);
    auto rowInput = input->as<ArrayVector>();
    rowInput->elements()->as<RowVector>()->childAt(0)->resize(0);
    rowInput->elements()->as<RowVector>()->childAt(1)->resize(0);

    auto expected =
        makeNullableMapVector<int32_t, int32_t>({std::nullopt, std::nullopt});
    verifyMapFromEntries(input, expected);
  }
}

TEST_F(MapFromEntriesTest, nullKeys) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({variant::null(TypeKind::INTEGER), 0})},
      {variant::row({1, 11})}};
  auto input = makeArrayOfRowVector(rowType, data);
  VELOX_ASSERT_THROW(
      evaluate("map_from_entries(c0)", makeRowVector({input})),
      "Cannot use null as map key!");
}

TEST_F(MapFromEntriesTest, arrayOfConstantRowOfNulls) {
  RowVectorPtr rowVector =
      makeRowVector({makeFlatVector<int32_t>(0), makeFlatVector<int32_t>(0)});
  rowVector->resize(1);
  rowVector->setNull(0, true);
  rowVector->childAt(0)->resize(0);
  rowVector->childAt(1)->resize(0);
  EXPECT_EQ(rowVector->childAt(0)->size(), 0);
  EXPECT_EQ(rowVector->childAt(1)->size(), 0);

  VectorPtr rowVectorConstant = BaseVector::wrapInConstant(4, 0, rowVector);

  auto offsets = makeIndices({0, 2});
  auto sizes = makeIndices({2, 2});

  auto arrayVector = std::make_shared<ArrayVector>(
      pool(),
      ARRAY(ROW({INTEGER(), INTEGER()})),
      nullptr,
      2,
      offsets,
      sizes,
      rowVectorConstant);
  VectorPtr result =
      evaluate("map_from_entries(c0)", makeRowVector({arrayVector}));
  for (int i = 0; i < result->size(); i++) {
    EXPECT_TRUE(result->isNullAt(i));
  }
}

TEST_F(MapFromEntriesTest, unknownInputs) {
  facebook::velox::functions::registerArrayConstructor("array_constructor");
  VELOX_ASSERT_THROW(
      evaluate(
          "map_from_entries(array_constructor(row_constructor(null, null)))",
          makeRowVector({makeFlatVector<int32_t>(2)})),
      "Cannot use null as map key!");
  VELOX_ASSERT_THROW(
      evaluate(
          "map_from_entries(array_constructor(null))",
          makeRowVector({makeFlatVector<int32_t>(2)})),
      "Scalar function signature is not supported: map_from_entries(ARRAY<UNKNOWN>)");
  VELOX_ASSERT_THROW(
      evaluate(
          "map_from_entries(null)",
          makeRowVector({makeFlatVector<int32_t>(2)})),
      "Scalar function signature is not supported: map_from_entries(UNKNOWN)");
}

TEST_F(MapFromEntriesTest, nullRowEntriesWithSmallerChildren) {
  // Row vector is of size 3, children are of size 2 since row 2 is null.
  auto rowVector = makeRowVector(
      {makeNullableFlatVector<int32_t>({std::nullopt, 2}),
       makeFlatVector<int32_t>({1, 2})});
  rowVector->appendNulls(1);
  rowVector->setNull(2, true);

  // Array [(null,1), (2,2), null]
  auto arrayVector = makeArrayVector({0}, rowVector);
  auto result = evaluate("map_from_entries(c0)", makeRowVector({arrayVector}));
  EXPECT_TRUE(result->isNullAt(0));
}

TEST_F(MapFromEntriesTest, nullAndRepeatedKeysFailInKeyOrder) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  {
    auto input = makeRowVector({makeArrayOfRowVector(
        rowType,
        {{variant::row({1, 1}),
          variant::row({1, 2}),
          variant::row({variant::null(TypeKind::INTEGER), 3})}})});
    setThrowExceptionOnDuplicateMapKeys(true);
    VELOX_ASSERT_THROW(
        evaluate("map_from_entries(c0)", input),
        "Duplicate map key was found: 1");
    setThrowExceptionOnDuplicateMapKeys(false);
    VELOX_ASSERT_THROW(
        evaluate("map_from_entries(c0)", input), "Cannot use null as map key!");
  }
  {
    auto input = makeRowVector({makeArrayOfRowVector(
        rowType,
        {{variant::row({1, 1}),
          variant::row({variant::null(TypeKind::INTEGER), 2}),
          variant::row({1, 3})}})});
    for (const bool throwOnDuplicate : {true, false}) {
      setThrowExceptionOnDuplicateMapKeys(throwOnDuplicate);
      VELOX_ASSERT_THROW(
          evaluate("map_from_entries(c0)", input),
          "Cannot use null as map key!");
    }
  }
}
TEST_F(MapFromEntriesTest, lastWinKeepsFirstPositionAndLastValue) {
  setThrowExceptionOnDuplicateMapKeys(false);
  auto input = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {{variant::row({1, "a"}),
        variant::row({2, "b"}),
        variant::row({1, "c"})}});
  verifyEntryOrder(
      input,
      makeArrayVector<int32_t>({{1, 2}}),
      makeArrayVector<std::string>({{"c", "b"}}));
}

TEST_F(MapFromEntriesTest, exceptionPolicyThrowsOnRepeatedKey) {
  setThrowExceptionOnDuplicateMapKeys(true);
  auto input = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {{variant::row({1, "a"}),
        variant::row({2, "b"}),
        variant::row({1, "c"})}});
  VELOX_ASSERT_THROW(
      evaluate("map_from_entries(c0)", makeRowVector({input})),
      "Duplicate map key was found: 1");
}

TEST_F(MapFromEntriesTest, entriesKeepInputOrder) {
  auto input = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {{variant::row({3, "p"}), variant::row({1, "q"}), variant::row({2, "r"})},
       {variant::row({2, "x"}), variant::row({1, "y"})}});
  for (const bool throwOnDuplicate : {false, true}) {
    SCOPED_TRACE(fmt::format("throwOnDuplicate: {}", throwOnDuplicate));
    setThrowExceptionOnDuplicateMapKeys(throwOnDuplicate);
    verifyEntryOrder(
        input,
        makeArrayVector<int32_t>({{3, 1, 2}, {2, 1}}),
        makeArrayVector<std::string>({{"p", "q", "r"}, {"x", "y"}}));
  }
}

TEST_F(MapFromEntriesTest, topLevelSignedZeroKeysStayDistinct) {
  auto input = makeArrayOfRowVector(
      ROW({DOUBLE(), INTEGER()}),
      {{variant::row({-0.0, 1}), variant::row({0.0, 2})}});
  auto data = makeRowVector({input});
  for (const bool throwOnDuplicate : {false, true}) {
    SCOPED_TRACE(fmt::format("throwOnDuplicate: {}", throwOnDuplicate));
    setThrowExceptionOnDuplicateMapKeys(throwOnDuplicate);
    EXPECT_THAT(
        elementBitPatterns(evaluate("map_keys(map_from_entries(c0))", data)),
        testing::ElementsAre(
            std::bit_cast<uint64_t>(-0.0), std::bit_cast<uint64_t>(0.0)));
    assertEqualVectors(
        makeArrayVector<int32_t>({{1, 2}}),
        evaluate("map_values(map_from_entries(c0))", data));
  }
}

TEST_F(MapFromEntriesTest, nanKeysAreOneKey) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  auto input = makeArrayOfRowVector(
      ROW({DOUBLE(), INTEGER()}),
      {{variant::row({kNaN, 1}), variant::row({kNaN, 2})}});
  {
    setThrowExceptionOnDuplicateMapKeys(false);
    verifyEntryOrder(
        input,
        makeArrayVector<double>({{kNaN}}),
        makeArrayVector<int32_t>({{2}}));
  }
  {
    setThrowExceptionOnDuplicateMapKeys(true);
    VELOX_ASSERT_THROW(
        evaluate("map_from_entries(c0)", makeRowVector({input})),
        "Duplicate map key was found: NaN");
  }
}

TEST_F(MapFromEntriesTest, nestedSignedZeroKeysAreOneKey) {
  setThrowExceptionOnDuplicateMapKeys(false);
  auto entries = makeRowVector({
      makeArrayVector<double>({{-0.0}, {0.0}}),
      makeFlatVector<int32_t>({1, 2}),
  });
  auto input = makeArrayVector({0}, entries);
  auto data = makeRowVector({input});
  auto keys = evaluate("map_keys(map_from_entries(c0))", data);
  ASSERT_EQ(keys->as<ArrayVector>()->sizeAt(0), 1);
  assertEqualVectors(
      makeArrayVector<int32_t>({{2}}),
      evaluate("map_values(map_from_entries(c0))", data));
  EXPECT_THAT(
      elementBitPatterns(
          evaluate("flatten(map_keys(map_from_entries(c0)))", data)),
      testing::ElementsAre(std::bit_cast<uint64_t>(-0.0)));
}

} // namespace facebook::velox::functions::sparksql::test
