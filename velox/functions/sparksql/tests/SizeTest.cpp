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
#include <string>

#include <gtest/gtest.h>
#include <velox/core/QueryConfig.h>
#include <optional>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/Timestamp.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
class SizeTest : public SparkFunctionBaseTest {
 protected:
  std::function<vector_size_t(vector_size_t /* row */)> sizeAt =
      [](vector_size_t row) { return 1 + row % 7; };

  void testLegacySizeOfNull(VectorPtr vector, vector_size_t numRows) {
    auto result = evaluate<SimpleVector<int32_t>>(
        "size(c0, true)", makeRowVector({vector}));
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_FALSE(result->isNullAt(i)) << "at " << i;
      if (vector->isNullAt(i)) {
        EXPECT_EQ(result->valueAt(i), -1) << "at " << i;
      } else {
        EXPECT_EQ(result->valueAt(i), sizeAt(i)) << "at " << i;
      }
    }
  }

  void testSize(VectorPtr vector, vector_size_t numRows) {
    auto result = evaluate<SimpleVector<int32_t>>(
        "size(c0, false)", makeRowVector({vector}));
    for (vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(result->isNullAt(i), vector->isNullAt(i)) << "at " << i;
      if (!vector->isNullAt(i)) {
        EXPECT_EQ(result->valueAt(i), vector->as<ArrayVectorBase>()->sizeAt(i))
            << "at " << i;
      }
    }
  }

  template <typename T>
  int32_t testArraySize(const std::vector<std::optional<T>>& input) {
    auto row = makeRowVector({makeNullableArrayVector(
        std::vector<std::vector<std::optional<T>>>{input})});
    return evaluateOnce<int32_t>("size(c0, false)", row).value();
  }

  static inline vector_size_t valueAt(vector_size_t idx) {
    return idx + 1;
  }
};

// Ensure that out is set to -1 for null input if legacySizeOfNull = true.
TEST_F(SizeTest, legacySizeOfNull) {
  vector_size_t numRows = 100;
  auto arrayVector =
      makeArrayVector<int64_t>(numRows, sizeAt, valueAt, nullptr);
  testLegacySizeOfNull(arrayVector, numRows);
  arrayVector =
      makeArrayVector<int64_t>(numRows, sizeAt, valueAt, nullEvery(5));
  testLegacySizeOfNull(arrayVector, numRows);
  auto mapVector = makeMapVector<int64_t, int64_t>(
      numRows, sizeAt, valueAt, valueAt, nullptr);
  testLegacySizeOfNull(mapVector, numRows);
  mapVector = makeMapVector<int64_t, int64_t>(
      numRows, sizeAt, valueAt, valueAt, nullEvery(5));
  testLegacySizeOfNull(mapVector, numRows);
}

// Ensure that out is set to null for null input if legacySizeOfNull = false.
TEST_F(SizeTest, size) {
  vector_size_t numRows = 100;
  auto arrayVector =
      makeArrayVector<int64_t>(numRows, sizeAt, valueAt, nullEvery(1));
  testSize(arrayVector, numRows);
  arrayVector =
      makeArrayVector<int64_t>(numRows, sizeAt, valueAt, nullEvery(5));
  testSize(arrayVector, numRows);
  auto mapVector = makeMapVector<int64_t, int64_t>(
      numRows, sizeAt, valueAt, valueAt, nullEvery(1));
  testSize(mapVector, numRows);
  mapVector = makeMapVector<int64_t, int64_t>(
      numRows, sizeAt, valueAt, valueAt, nullEvery(5));
  testSize(mapVector, numRows);
}

TEST_F(SizeTest, encodedArrays) {
  std::vector<std::optional<std::vector<std::optional<int32_t>>>> data = {
      std::vector<std::optional<int32_t>>{1, 2},
      std::nullopt,
      std::vector<std::optional<int32_t>>{},
      std::vector<std::optional<int32_t>>{3, 4, 5}};
  auto base = makeNullableArrayVector<int32_t>(data);
  auto dictionary = wrapInDictionary(makeIndices({3, 1, 0, 2, 1}), base);

  auto result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({dictionary}));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int32_t>({3, std::nullopt, 2, 0, std::nullopt}),
      result);

  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, true)", makeRowVector({dictionary}));
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({3, -1, 2, 0, -1}), result);

  auto dictionaryWithNulls = BaseVector::wrapInDictionary(
      makeNulls(5, [](vector_size_t row) { return row == 0 || row == 3; }),
      makeIndices({3, 1, 0, 2, 3}),
      5,
      base);
  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({dictionaryWithNulls}));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, 2, std::nullopt, 3}),
      result);

  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, true)", makeRowVector({dictionaryWithNulls}));
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({-1, -1, 2, -1, 3}), result);

  auto constant = BaseVector::wrapInConstant(5, 3, base);
  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({constant}));
  facebook::velox::test::assertEqualVectors(
      makeConstant<int32_t>(3, 5), result);

  auto nullConstant = BaseVector::wrapInConstant(5, 1, base);
  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, true)", makeRowVector({nullConstant}));
  facebook::velox::test::assertEqualVectors(
      makeConstant<int32_t>(-1, 5), result);
}

TEST_F(SizeTest, typedNullConstants) {
  const std::vector<TypePtr> types{ARRAY(INTEGER()), MAP(INTEGER(), BIGINT())};
  for (const auto& type : types) {
    SCOPED_TRACE(type->toString());
    auto nullConstant = BaseVector::createNullConstant(type, 5, pool());
    const std::vector<VectorPtr> inputs{
        nullConstant,
        wrapInDictionary(makeIndices({4, 3, 2, 1, 0}), nullConstant)};
    for (const auto& input : inputs) {
      auto result = evaluate<SimpleVector<int32_t>>(
          "size(c0, false)", makeRowVector({input}));
      facebook::velox::test::assertEqualVectors(
          makeAllNullFlatVector<int32_t>(5), result);

      result = evaluate<SimpleVector<int32_t>>(
          "size(c0, true)", makeRowVector({input}));
      facebook::velox::test::assertEqualVectors(
          makeConstant<int32_t>(-1, 5), result);
    }
  }
}

TEST_F(SizeTest, flatMap) {
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  std::vector<std::optional<MapEntries>> maps = {
      MapEntries{{1, 10}, {2, 20}},
      std::nullopt,
      MapEntries{},
      MapEntries{{1, 30}, {3, std::nullopt}, {4, 40}}};
  auto flatMap = makeNullableFlatMapVector<int32_t, int32_t>(maps);

  auto result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({flatMap}));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int32_t>({2, std::nullopt, 0, 3}), result);

  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, true)", makeRowVector({flatMap}));
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({2, -1, 0, 3}), result);

  auto dictionary = BaseVector::wrapInDictionary(
      makeNulls(5, [](vector_size_t row) { return row == 2; }),
      makeIndices({3, 1, 0, 2, 3}),
      5,
      flatMap);
  result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({dictionary}));
  facebook::velox::test::assertEqualVectors(
      makeNullableFlatVector<int32_t>({3, std::nullopt, std::nullopt, 0, 3}),
      result);
}

TEST_F(SizeTest, selectedRows) {
  auto arrays = makeArrayVector<int32_t>(
      6, [](auto row) { return row; }, [](auto row) { return row; });
  SelectivityVector rows(6, false);
  rows.setValid(1, true);
  rows.setValid(4, true);
  rows.updateBounds();

  auto result = evaluate<SimpleVector<int32_t>>(
      "size(c0, false)", makeRowVector({arrays}), rows);
  EXPECT_EQ(result->valueAt(1), 1);
  EXPECT_EQ(result->valueAt(4), 4);

  auto conditions =
      makeFlatVector<bool>({false, true, false, false, true, false});
  result = evaluate<SimpleVector<int32_t>>(
      "if(c1, size(c0, false), 99::integer)",
      makeRowVector({arrays, conditions}));
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({99, 1, 99, 99, 4, 99}), result);
}

TEST_F(SizeTest, invalidLegacySizeOfNullIsDeferred) {
  auto arrays = makeArrayVector<int32_t>(
      4, [](auto row) { return row; }, [](auto row) { return row; });
  auto flags = makeFlatVector<bool>({false, true, false, true});
  auto conditions = makeConstant(true, 4);
  auto data = makeRowVector({arrays, flags, conditions});

  const std::string errorMessage =
      "requires legacySizeOfNull to be a non-null constant boolean";
  VELOX_ASSERT_THROW(evaluate("size(c0, c1)", data), errorMessage);
  VELOX_ASSERT_THROW(evaluate("size(c0, null::boolean)", data), errorMessage);

  auto result = evaluate<SimpleVector<int32_t>>("try(size(c0, c1))", data);
  facebook::velox::test::assertEqualVectors(
      makeAllNullFlatVector<int32_t>(4), result);

  result =
      evaluate<SimpleVector<int32_t>>("if(c2, 0::integer, size(c0, c1))", data);
  facebook::velox::test::assertEqualVectors(
      makeConstant<int32_t>(0, 4), result);

  result =
      evaluate<SimpleVector<int32_t>>("try(size(c0, null::boolean))", data);
  facebook::velox::test::assertEqualVectors(
      makeAllNullFlatVector<int32_t>(4), result);

  result = evaluate<SimpleVector<int32_t>>(
      "if(c2, 0::integer, size(c0, null::boolean))", data);
  facebook::velox::test::assertEqualVectors(
      makeConstant<int32_t>(0, 4), result);
}

TEST_F(SizeTest, boolean) {
  EXPECT_EQ(testArraySize<bool>({true, false}), 2);
  EXPECT_EQ(testArraySize<bool>({true}), 1);
  EXPECT_EQ(testArraySize<bool>({}), 0);
  EXPECT_EQ(testArraySize<bool>({true, false, true, std::nullopt}), 4);
}

TEST_F(SizeTest, smallint) {
  EXPECT_EQ(testArraySize<int8_t>({}), 0);
  EXPECT_EQ(testArraySize<int8_t>({1}), 1);
  EXPECT_EQ(testArraySize<int8_t>({std::nullopt}), 1);
  EXPECT_EQ(testArraySize<int8_t>({std::nullopt, 1}), 2);
}

TEST_F(SizeTest, real) {
  EXPECT_EQ(testArraySize<float>({}), 0);
  EXPECT_EQ(testArraySize<float>({1.1}), 1);
  EXPECT_EQ(testArraySize<float>({std::nullopt}), 1);
  EXPECT_EQ(testArraySize<float>({std::nullopt, 1.1}), 2);
}

TEST_F(SizeTest, varchar) {
  EXPECT_EQ(testArraySize<std::string>({"red", "blue"}), 2);
  EXPECT_EQ(
      testArraySize<std::string>({std::nullopt, "blue", "yellow", "orange"}),
      4);
  EXPECT_EQ(testArraySize<std::string>({}), 0);
  EXPECT_EQ(testArraySize<std::string>({std::nullopt}), 1);
}

TEST_F(SizeTest, integer) {
  EXPECT_EQ(testArraySize<int32_t>({1, 2}), 2);
}

TEST_F(SizeTest, timestamp) {
  auto ts = [](int64_t micros) { return Timestamp::fromMicros(micros); };
  EXPECT_EQ(testArraySize<Timestamp>({}), 0);
  EXPECT_EQ(testArraySize<Timestamp>({std::nullopt}), 1);
  EXPECT_EQ(testArraySize<Timestamp>({ts(0), ts(1)}), 2);
}

} // namespace facebook::velox::functions::sparksql::test
