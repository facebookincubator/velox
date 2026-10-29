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
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/Cardinality.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/FlatMapVector.h"

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

  result = evaluate<SimpleVector<int32_t>>(
      "if(c1, 99::integer, size(c0, false))",
      makeRowVector({arrays, conditions}));
  facebook::velox::test::assertEqualVectors(
      makeFlatVector<int32_t>({0, 99, 2, 3, 99, 5}), result);
}

TEST_F(SizeTest, differentialEncodings) {
  registerFunction<CardinalityFunction, int32_t, Array<Generic<T1>>>(
      {"size_reference"});
  registerFunction<CardinalityFunction, int32_t, Map<Generic<T1>, Generic<T2>>>(
      {"size_reference"});

  const auto check = [&](const VectorPtr& input, const VectorPtr& reference) {
    SCOPED_TRACE(input->toSummaryString());
    const auto numRows = input->size();
    auto data = makeRowVector({input});
    for (const auto legacy : {false, true}) {
      SCOPED_TRACE(legacy);
      auto expected = evaluate<SimpleVector<int32_t>>(
          legacy ? "coalesce(size_reference(c0), -1::integer)"
                 : "size_reference(c0)",
          makeRowVector({reference}));
      facebook::velox::test::assertEqualVectors(
          expected,
          evaluate(legacy ? "size(c0, true)" : "size(c0, false)", data));

      std::vector<VectorPtr> args{input, makeConstant(legacy, numRows)};
      auto function = getVectorFunction(
          "size",
          {input->type(), BOOLEAN()},
          {nullptr, args[1]},
          queryCtx_->queryConfig());
      ASSERT_NE(function, nullptr);
      ExprSet exprSet({}, &execCtx_);
      EvalCtx context(&execCtx_, &exprSet, data.get());
      SelectivityVector finalRows(numRows);
      ScopedFinalSelectionSetter finalSelection(context, &finalRows);
      for (const auto sparse : {false, true}) {
        SelectivityVector rows(numRows, !sparse);
        if (sparse) {
          for (vector_size_t row = 1; row + 1 < numRows; ++row) {
            rows.setValid(row, row % 3 != 0);
          }
          rows.updateBounds();
          if (!rows.hasSelections()) {
            continue;
          }
        }

        std::vector<std::optional<int32_t>> initial(numRows);
        for (vector_size_t row = 0; row < numRows; ++row) {
          if (row % 2 == 0) {
            initial[row] = 99;
          }
        }
        auto selectedExpected = initial;
        rows.applyToSelected([&](vector_size_t row) {
          selectedExpected[row] = expected->isNullAt(row)
              ? std::nullopt
              : std::optional<int32_t>(expected->valueAt(row));
        });

        auto expectedResult = makeNullableFlatVector<int32_t>(selectedExpected);
        for (const auto shared : {false, true}) {
          SCOPED_TRACE(shared);
          VectorPtr result = makeNullableFlatVector<int32_t>(initial);
          VectorPtr preserved = shared ? result : nullptr;
          // Bypass expression peeling to exercise the encoded input paths.
          function->apply(rows, args, INTEGER(), context, result);
          facebook::velox::test::assertEqualVectors(expectedResult, result);
          if (preserved) {
            facebook::velox::test::assertEqualVectors(
                makeNullableFlatVector<int32_t>(initial), preserved);
          }
        }
      }
    }
  };

  using ArrayElements = std::vector<std::optional<int32_t>>;
  using MapEntries = std::vector<std::pair<int32_t, std::optional<int32_t>>>;
  for (const vector_size_t numRows : {1, 63, 64, 65, 127, 128, 129, 257}) {
    SCOPED_TRACE(numRows);
    auto indices =
        makeIndices(numRows, [numRows](auto row) { return numRows - row - 1; });
    auto repeated = makeIndices(numRows, [](auto row) { return row / 2; });
    auto nulls = makeNulls(numRows, nullEvery(5));
    for (const auto nullable : {false, true}) {
      SCOPED_TRACE(nullable);
      std::vector<std::optional<ArrayElements>> arrays(numRows);
      std::vector<std::optional<MapEntries>> maps(numRows);
      for (vector_size_t row = 0; row < numRows; ++row) {
        if (nullable && row % 7 == 0) {
          continue;
        }
        arrays[row].emplace();
        maps[row].emplace();
        for (int32_t entry = 0; entry < row % 33; ++entry) {
          const auto value = entry % 5 == 0
              ? std::nullopt
              : std::optional<int32_t>(row + entry);
          arrays[row]->push_back(value);
          maps[row]->emplace_back(entry, value);
        }
      }
      auto array = makeNullableArrayVector<int32_t>(arrays);
      auto map = makeNullableMapVector<int32_t, int32_t>(maps);
      auto flatMap = FlatMapVector::fromMapVector(*map);
      for (const auto& [input, reference] :
           std::vector<std::pair<VectorPtr, VectorPtr>>{
               {array, array}, {map, map}, {flatMap, map}}) {
        check(input, reference);
        check(
            wrapInDictionary(indices, input),
            wrapInDictionary(indices, reference));
        auto dictionary =
            BaseVector::wrapInDictionary(nulls, indices, numRows, input);
        auto referenceDictionary =
            BaseVector::wrapInDictionary(nulls, indices, numRows, reference);
        check(dictionary, referenceDictionary);
        check(
            wrapInDictionary(repeated, dictionary),
            wrapInDictionary(repeated, referenceDictionary));

        const auto constantRow = numRows > 1 ? 1 : 0;
        auto constant = BaseVector::wrapInConstant(numRows, constantRow, input);
        auto referenceConstant =
            BaseVector::wrapInConstant(numRows, constantRow, reference);
        check(constant, referenceConstant);
        check(
            BaseVector::wrapInDictionary(nulls, indices, numRows, constant),
            BaseVector::wrapInDictionary(
                nulls, indices, numRows, referenceConstant));

        auto typedNull =
            BaseVector::createNullConstant(input->type(), numRows, pool());
        auto nullReference = BaseVector::create(input->type(), numRows, pool());
        for (vector_size_t row = 0; row < numRows; ++row) {
          nullReference->setNull(row, true);
        }
        check(typedNull, nullReference);
        check(
            wrapInDictionary(indices, typedNull),
            wrapInDictionary(indices, nullReference));
      }
    }
  }
}

TEST_F(SizeTest, maximumSizeWithoutLoadingChildren) {
  constexpr auto kMaxSize = std::numeric_limits<vector_size_t>::max();
  auto elements = std::make_shared<LazyVector>(
      pool(),
      INTEGER(),
      kMaxSize,
      std::make_unique<facebook::velox::test::SimpleVectorLoader>(
          [](RowSet /*rows*/) -> VectorPtr {
            VELOX_FAIL("Spark size must not load collection children");
          }));
  const std::vector<VectorPtr> inputs{
      std::make_shared<ArrayVector>(
          pool(),
          ARRAY(INTEGER()),
          nullptr,
          1,
          makeIndices({0}),
          makeIndices({kMaxSize}),
          elements),
      std::make_shared<MapVector>(
          pool(),
          MAP(INTEGER(), INTEGER()),
          nullptr,
          1,
          makeIndices({0}),
          makeIndices({kMaxSize}),
          elements,
          elements)};
  for (const auto& input : inputs) {
    for (const auto legacy : {false, true}) {
      std::vector<VectorPtr> args{input, makeConstant(legacy, 1)};
      auto function = getVectorFunction(
          "size",
          {input->type(), BOOLEAN()},
          {nullptr, args[1]},
          queryCtx_->queryConfig());
      ASSERT_NE(function, nullptr);
      ExprSet exprSet({}, &execCtx_);
      auto data = makeRowVector({input});
      EvalCtx context(&execCtx_, &exprSet, data.get());
      SelectivityVector rows(1);
      VectorPtr result;
      function->apply(rows, args, INTEGER(), context, result);
      facebook::velox::test::assertEqualVectors(
          makeFlatVector<int32_t>({kMaxSize}), result);
      EXPECT_FALSE(elements->isLoaded());
    }
  }
}

TEST_F(SizeTest, invalidLegacySizeOfNullIsDeferred) {
  auto arrays = makeArrayVector<int32_t>(
      4, [](auto row) { return row; }, [](auto row) { return row; });
  auto flags = makeFlatVector<bool>({false, true, false, true});
  auto conditions = makeConstant(true, 4);
  auto data = makeRowVector({arrays, flags, conditions});

  const std::string errorMessage =
      "size requires legacySizeOfNull to be a non-null constant boolean";
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
