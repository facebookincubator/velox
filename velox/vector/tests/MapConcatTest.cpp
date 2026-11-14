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
#include "velox/vector/MapConcat.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox {
namespace {

class MapConcatTest : public testing::Test, public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Decode a vector and return the DecodedVector.  Caller must keep the
  // returned object alive while using the pointer.
  std::unique_ptr<DecodedVector> decode(const VectorPtr& vector) {
    auto decoded = std::make_unique<DecodedVector>(*vector);
    return decoded;
  }

  // Helper to call mapConcat with VectorPtrs (decodes internally).
  template <typename T = MapVector>
  std::shared_ptr<T> concat(
      const std::vector<VectorPtr>& inputs,
      const SelectivityVector& rows,
      const MapConcatConfig& config = {}) {
    std::vector<std::unique_ptr<DecodedVector>> decodedOwners;
    std::vector<DecodedVector*> decodedPtrs;
    decodedOwners.reserve(inputs.size());
    decodedPtrs.reserve(inputs.size());
    for (const auto& input : inputs) {
      decodedOwners.push_back(decode(input));
      decodedPtrs.push_back(decodedOwners.back().get());
    }
    auto result = std::dynamic_pointer_cast<T>(
        mapConcat(pool(), inputs[0]->type(), decodedPtrs, rows, config));
    VELOX_CHECK_NOT_NULL(result);
    return result;
  }

  // Checks that mapConcat over 'flatMapInputs' gives a flat map equal to
  // 'expected' and to mapConcat over 'mapInputs', the same maps as MapVectors.
  void testFlatMapConcat(
      const std::vector<VectorPtr>& flatMapInputs,
      const std::vector<VectorPtr>& mapInputs,
      const MapConcatConfig& config,
      const VectorPtr& expected) {
    SelectivityVector rows(flatMapInputs[0]->size());
    auto result = concat<FlatMapVector>(flatMapInputs, rows, config);
    ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
    result->validate();
    test::assertEqualVectors(expected, result);
    test::assertEqualVectors(concat(mapInputs, rows, config), result);
  }
};

TEST_F(MapConcatTest, basic) {
  // Two maps with no overlapping keys.
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{3, 30}},
      {{4, 40}, {5, 50}, {6, 60}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{7, 70}},
      {{8, 80}, {9, 90}},
      {{10, 100}},
  });

  SelectivityVector rows(3);
  auto result = concat({map1, map2}, rows);

  // Row 0: {1->10, 2->20, 7->70}
  ASSERT_EQ(result->sizeAt(0), 3);
  // Row 1: {3->30, 8->80, 9->90}
  ASSERT_EQ(result->sizeAt(1), 3);
  // Row 2: {4->40, 5->50, 6->60, 10->100}
  ASSERT_EQ(result->sizeAt(2), 4);

  // Verify by comparing with expected.
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {7, 70}},
      {{3, 30}, {8, 80}, {9, 90}},
      {{4, 40}, {5, 50}, {6, 60}, {10, 100}},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, duplicateKeys) {
  // Keys 2 and 3 overlap.  Last input wins.
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{2, 200}, {3, 300}, {4, 400}},
  });

  SelectivityVector rows(1);
  auto result = concat({map1, map2}, rows);

  // Key 2 -> 200 (from map2), key 3 -> 300 (from map2).
  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 200}, {3, 300}, {4, 400}},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, throwOnDuplicateKeys) {
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{2, 200}, {3, 300}},
  });

  SelectivityVector rows(1);
  MapConcatConfig config;
  config.throwOnDuplicateKeys = true;
  VELOX_ASSERT_USER_THROW(
      concat({map1, map2}, rows, config), "Duplicate map key");
}

TEST_F(MapConcatTest, emptyForNull) {
  // When emptyForNull is true, null inputs are treated as empty maps.
  auto map1 = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {2, 20}}},
      std::nullopt,
      {{{3, 30}}},
  });
  auto map2 = makeNullableMapVector<int64_t, int64_t>({
      std::nullopt,
      {{{4, 40}}},
      {{{5, 50}}},
  });

  SelectivityVector rows(3);
  MapConcatConfig config;
  config.emptyForNull = true;
  auto result = concat({map1, map2}, rows, config);

  // Row 0: map1 has {1->10, 2->20}, map2 is null (empty).  Result: {1->10,
  // 2->20}.
  ASSERT_FALSE(result->isNullAt(0));
  ASSERT_EQ(result->sizeAt(0), 2);
  // Row 1: map1 is null (empty), map2 has {4->40}.  Result: {4->40}.
  ASSERT_FALSE(result->isNullAt(1));
  ASSERT_EQ(result->sizeAt(1), 1);
  // Row 2: both non-null.  Result: {3->30, 5->50}.
  ASSERT_FALSE(result->isNullAt(2));
  ASSERT_EQ(result->sizeAt(2), 2);
}

TEST_F(MapConcatTest, nullPropagation) {
  // Default behavior: null in any input => null row.
  auto map1 = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}}},
      std::nullopt,
      {{{3, 30}}},
  });
  auto map2 = makeNullableMapVector<int64_t, int64_t>({
      std::nullopt,
      {{{4, 40}}},
      {{{5, 50}}},
  });

  SelectivityVector rows(3);
  auto result = concat({map1, map2}, rows);

  // Row 0: map2 is null -> null.
  ASSERT_TRUE(result->isNullAt(0));
  // Row 1: map1 is null -> null.
  ASSERT_TRUE(result->isNullAt(1));
  // Row 2: both non-null.
  ASSERT_FALSE(result->isNullAt(2));
  ASSERT_EQ(result->sizeAt(2), 2);
}

TEST_F(MapConcatTest, partialRows) {
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}},
      {{2, 20}},
      {{3, 30}},
      {{4, 40}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{5, 50}},
      {{6, 60}},
      {{7, 70}},
      {{8, 80}},
  });

  // Only process rows 1 and 3.
  SelectivityVector rows(4, false);
  rows.setValid(1, true);
  rows.setValid(3, true);
  rows.updateBounds();

  auto result = concat({map1, map2}, rows);

  // Unselected rows get size 0.
  ASSERT_EQ(result->sizeAt(0), 0);
  ASSERT_EQ(result->sizeAt(2), 0);
  // Selected rows are merged.
  ASSERT_EQ(result->sizeAt(1), 2);
  ASSERT_EQ(result->sizeAt(3), 2);
}

TEST_F(MapConcatTest, multipleInputs) {
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{2, 20}},
  });
  auto map3 = makeMapVector<int64_t, int64_t>({
      {{3, 30}},
  });

  SelectivityVector rows(1);
  auto result = concat({map1, map2, map3}, rows);

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, multipleInputsDuplicateKeys) {
  // Key 1 appears in all three.  Last input wins.
  auto map1 = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{1, 100}, {3, 30}},
  });
  auto map3 = makeMapVector<int64_t, int64_t>({
      {{1, 1000}, {4, 40}},
  });

  SelectivityVector rows(1);
  auto result = concat({map1, map2, map3}, rows);

  auto expected = makeMapVector<int64_t, int64_t>({
      {{1, 1000}, {2, 20}, {3, 30}, {4, 40}},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, varcharKeys) {
  auto map1 = makeMapVector<StringView, int64_t>({
      {{StringView("a"), 1}, {StringView("b"), 2}},
  });
  auto map2 = makeMapVector<StringView, int64_t>({
      {{StringView("b"), 20}, {StringView("c"), 3}},
  });

  SelectivityVector rows(1);
  auto result = concat({map1, map2}, rows);

  auto expected = makeMapVector<StringView, int64_t>({
      {{StringView("a"), 1}, {StringView("b"), 20}, {StringView("c"), 3}},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, emptyMaps) {
  auto map1 = makeMapVector<int64_t, int64_t>({
      {},
      {{1, 10}},
      {},
  });
  auto map2 = makeMapVector<int64_t, int64_t>({
      {{2, 20}},
      {},
      {},
  });

  SelectivityVector rows(3);
  auto result = concat({map1, map2}, rows);

  auto expected = makeMapVector<int64_t, int64_t>({
      {{2, 20}},
      {{1, 10}},
      {},
  });
  test::assertEqualVectors(expected, result);
}

TEST_F(MapConcatTest, dictionaryEncoded) {
  auto base = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{3, 30}},
      {{4, 40}, {5, 50}},
  });

  // Dictionary that reverses the order: [2, 1, 0].
  auto indices = makeIndices({2, 1, 0});
  auto dict = wrapInDictionary(indices, 3, base);

  auto map2 = makeMapVector<int64_t, int64_t>({
      {{6, 60}},
      {{7, 70}},
      {{8, 80}},
  });

  SelectivityVector rows(3);
  auto result = concat({dict, map2}, rows);

  // Row 0: dict[0] = base[2] = {4->40, 5->50} + {6->60}.
  ASSERT_EQ(result->sizeAt(0), 3);
  // Row 1: dict[1] = base[1] = {3->30} + {7->70}.
  ASSERT_EQ(result->sizeAt(1), 2);
  // Row 2: dict[2] = base[0] = {1->10, 2->20} + {8->80}.
  ASSERT_EQ(result->sizeAt(2), 3);
}

TEST_F(MapConcatTest, constantEncoded) {
  auto base = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}},
  });

  // Constant vector repeating base[0] for 3 rows.
  auto constant = BaseVector::wrapInConstant(3, 0, base);

  auto map2 = makeMapVector<int64_t, int64_t>({
      {{3, 30}},
      {{4, 40}},
      {{5, 50}},
  });

  SelectivityVector rows(3);
  auto result = concat({constant, map2}, rows);

  // Each row: {1->10, 2->20} + one new entry.
  ASSERT_EQ(result->sizeAt(0), 3);
  ASSERT_EQ(result->sizeAt(1), 3);
  ASSERT_EQ(result->sizeAt(2), 3);
}

TEST_F(MapConcatTest, emptyForNullAllNull) {
  // When all inputs are null for a row and emptyForNull is true, result is
  // an empty map (not null).
  auto map1 = makeNullableMapVector<int64_t, int64_t>({
      std::nullopt,
  });
  auto map2 = makeNullableMapVector<int64_t, int64_t>({
      std::nullopt,
  });

  SelectivityVector rows(1);
  MapConcatConfig config;
  config.emptyForNull = true;
  auto result = concat({map1, map2}, rows, config);

  ASSERT_FALSE(result->isNullAt(0));
  ASSERT_EQ(result->sizeAt(0), 0);
}

// Custom type where keys are compared modulo 100.  So 1 and 101 are considered
// equal, but native int64_t == treats them as different.
class Mod100Type : public BigintType {
 public:
  constexpr Mod100Type() : BigintType{ProvideCustomComparison{}} {}

  int32_t compare(const int64_t& left, const int64_t& right) const override {
    return static_cast<int32_t>((left % 100) - (right % 100));
  }

  uint64_t hash(const int64_t& value) const override {
    return folly::hasher<int64_t>()(value % 100);
  }

  bool equivalent(const Type& other) const override {
    return this == &other;
  }

  const char* name() const override {
    return "MOD100";
  }

  std::string toString() const override {
    return name();
  }
};

TEST_F(MapConcatTest, customComparisonType) {
  // Keys 1 and 101 are different by native ==, but equal by Mod100Type
  // comparison (both are 1 mod 100).  Dedup should treat them as the same key.
  static const Mod100Type kMod100Type;
  auto mod100TypePtr =
      std::shared_ptr<const Type>(std::shared_ptr<const Type>{}, &kMod100Type);
  auto mapType = MAP(mod100TypePtr, BIGINT());

  // Build map1 with key 1, map2 with key 101.
  auto map1Keys = makeFlatVector<int64_t>({1}, mod100TypePtr);
  auto map1Values = makeFlatVector<int64_t>({10});
  auto map1 = std::make_shared<MapVector>(
      pool(),
      mapType,
      BufferPtr(nullptr),
      1,
      allocateOffsets(1, pool()),
      allocateSizes(1, pool()),
      map1Keys,
      map1Values);
  // Set offset=0, size=1 for row 0.
  map1->setOffsetAndSize(0, 0, 1);

  auto map2Keys = makeFlatVector<int64_t>({101}, mod100TypePtr);
  auto map2Values = makeFlatVector<int64_t>({20});
  auto map2 = std::make_shared<MapVector>(
      pool(),
      mapType,
      BufferPtr(nullptr),
      1,
      allocateOffsets(1, pool()),
      allocateSizes(1, pool()),
      map2Keys,
      map2Values);
  map2->setOffsetAndSize(0, 0, 1);

  SelectivityVector rows(1);
  auto result = concat({map1, map2}, rows);

  // With custom comparison, 1 and 101 are the same key.  Last input wins,
  // so the result should have 1 entry with value 20 (from map2).
  ASSERT_EQ(result->sizeAt(0), 1);
}

TEST_F(MapConcatTest, flatMapWithMap) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{3:30}",
  });
  auto map = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:99}",
      "{7:70}",
  });
  SelectivityVector rows(2);

  VELOX_ASSERT_THROW(concat({flatMap, map}, rows), "Wrong type cast expected");
  VELOX_ASSERT_THROW(
      concat({flatMap, BaseVector::wrapInConstant(2, 0, map)}, rows),
      "Wrong type cast expected");
  VELOX_ASSERT_THROW(concat({map, flatMap}, rows), "Wrong type cast expected");
  VELOX_ASSERT_THROW(
      concat({map, BaseVector::wrapInConstant(2, 0, flatMap)}, rows),
      "Wrong type cast expected");
  VELOX_ASSERT_THROW(
      concat({map, map, flatMap}, rows), "Wrong type cast expected");
}

TEST_F(MapConcatTest, twoFlatMaps) {
  auto first = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "null",
      "{1:11, 3:31}",
      "{}",
  });
  auto second = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{2:22, 4:40}",
      "{4:41}",
      "{}",
      "{5:53}",
  });

  testFlatMapConcat(
      {first, second},
      {first->toMapVector(), second->toMapVector()},
      {},
      makeMapVectorFromJson<int64_t, int64_t>({
          "{1:10, 2:22, 4:40}",
          "null",
          "{1:11, 3:31}",
          "{5:53}",
      }));

  testFlatMapConcat(
      {first, second, first},
      {first->toMapVector(), second->toMapVector(), first->toMapVector()},
      {.emptyForNull = true},
      makeMapVectorFromJson<int64_t, int64_t>({
          "{1:10, 2:20, 4:40}",
          "{4:41}",
          "{1:11, 3:31}",
          "{5:53}",
      }));
}

TEST_F(MapConcatTest, flatMapDuplicateKeys) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{3:30}",
  });
  auto other = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{4:40}",
      "{3:31}",
  });
  const MapConcatConfig config{.throwOnDuplicateKeys = true};
  SelectivityVector rows(2);

  VELOX_ASSERT_THROW(
      concat<FlatMapVector>({flatMap, other}, rows, config),
      "Duplicate map key 3 was found.");
  VELOX_ASSERT_THROW(
      concat({flatMap->toMapVector(), other->toMapVector()}, rows, config),
      "Duplicate map key 3 was found.");
  VELOX_ASSERT_THROW(
      concat<FlatMapVector>({flatMap, flatMap}, rows, config),
      "Duplicate map key");

  {
    auto nullInSecondRow = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{5:50}",
        "null",
    });
    testFlatMapConcat(
        {flatMap, other, nullInSecondRow},
        {flatMap->toMapVector(),
         other->toMapVector(),
         nullInSecondRow->toMapVector()},
        config,
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:10, 2:20, 4:40, 5:50}",
            "null",
        }));
  }

  {
    auto noDuplicates = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{4:40}",
        "{5:50}",
    });
    testFlatMapConcat(
        {flatMap, noDuplicates},
        {flatMap->toMapVector(), noDuplicates->toMapVector()},
        config,
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:10, 2:20, 4:40}",
            "{3:30, 5:50}",
        }));
  }
}

TEST_F(MapConcatTest, flatMapDuplicateKeysInUnselectedRows) {
  auto inMapAt = [&](const std::vector<vector_size_t>& rowsWithKey) {
    auto inMap = AlignedBuffer::allocate<bool>(3, pool(), false);
    for (auto row : rowsWithKey) {
      bits::setBit(inMap->asMutable<uint64_t>(), row);
    }
    return inMap;
  };
  auto makeInput = [&](std::vector<BufferPtr> inMaps) {
    return std::make_shared<FlatMapVector>(
        pool(),
        MAP(BIGINT(), BIGINT()),
        nullptr,
        3,
        makeFlatVector<int64_t>({3, 4}),
        std::vector<VectorPtr>{
            makeFlatVector<int64_t>({30, 31, 32}),
            makeFlatVector<int64_t>({40, 41, 42}),
        },
        std::move(inMaps));
  };
  auto first = makeInput({inMapAt({0, 1}), inMapAt({2})});
  auto second = makeInput({inMapAt({1, 2}), inMapAt({0})});
  SelectivityVector rows(3);
  rows.setValid(1, false);
  rows.updateBounds();
  const MapConcatConfig config{.throwOnDuplicateKeys = true};

  auto result = concat<FlatMapVector>({first, second}, rows, config);
  test::assertEqualVectors(
      concat({first->toMapVector(), second->toMapVector()}, rows, config),
      result,
      rows);
}

TEST_F(MapConcatTest, flatMapSharesCoveringValues) {
  auto inMapAt = [&](const std::vector<vector_size_t>& rowsWithKey) {
    auto inMap = AlignedBuffer::allocate<bool>(3, pool(), false);
    for (auto row : rowsWithKey) {
      bits::setBit(inMap->asMutable<uint64_t>(), row);
    }
    return inMap;
  };
  auto first = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      3,
      makeFlatVector<int64_t>({1, 2}),
      std::vector<VectorPtr>{
          makeFlatVector<int64_t>({10, 11, 12}),
          makeFlatVector<int64_t>({20, 21, 22}),
      },
      std::vector<BufferPtr>{nullptr, inMapAt({0, 2})});
  auto second = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      3,
      makeFlatVector<int64_t>({2}),
      std::vector<VectorPtr>{makeFlatVector<int64_t>({30, 31, 32})},
      std::vector<BufferPtr>{nullptr});
  SelectivityVector rows(3);

  auto result = concat<FlatMapVector>({first, second}, rows);
  test::assertEqualVectors(
      concat({first->toMapVector(), second->toMapVector()}, rows), result);
  EXPECT_EQ(result->mapValuesAt(1), second->mapValuesAt(0));
  EXPECT_EQ(result->inMapsAt(1), nullptr);
}

TEST_F(MapConcatTest, flatMapDoesNotModifyInputs) {
  constexpr vector_size_t kSize{3};
  auto inMapAt = [&](const std::vector<vector_size_t>& rowsWithKey) {
    auto inMap = AlignedBuffer::allocate<bool>(kSize, pool(), false);
    for (auto row : rowsWithKey) {
      bits::setBit(inMap->asMutable<uint64_t>(), row);
    }
    return inMap;
  };
  auto makeInput = [&](BufferPtr nulls,
                       const std::vector<int64_t>& keys,
                       int64_t firstValue,
                       std::vector<BufferPtr> inMaps) {
    std::vector<VectorPtr> values;
    for (size_t i = 0; i < keys.size(); ++i) {
      values.push_back(makeFlatVector<int64_t>(kSize, [&](auto row) {
        return firstValue + i * 10 + row;
      }));
    }
    return std::make_shared<FlatMapVector>(
        pool(),
        MAP(BIGINT(), BIGINT()),
        std::move(nulls),
        kSize,
        makeFlatVector<int64_t>(keys),
        std::move(values),
        std::move(inMaps));
  };
  auto testConcat = [&](const std::vector<VectorPtr>& inputs,
                        const MapConcatConfig& config) {
    std::vector<VectorPtr> mapInputs;
    for (const auto& input : inputs) {
      mapInputs.push_back(test::withMapBase(input));
    }
    SelectivityVector rows(kSize);
    auto result = concat<FlatMapVector>(inputs, rows, config);
    result->validate();
    test::assertEqualVectors(concat(mapInputs, rows, config), result);
    for (size_t i = 0; i < inputs.size(); ++i) {
      test::assertEqualVectors(mapInputs[i], inputs[i]);
    }
  };

  {
    auto first = makeInput(nullptr, {1}, 10, {inMapAt({0})});
    auto second = makeInput(nullptr, {1}, 20, {inMapAt({1})});
    testConcat({first, second}, {});
  }

  for (const auto& secondInMaps :
       {std::vector<BufferPtr>{nullptr, nullptr}, std::vector<BufferPtr>{}}) {
    auto first = makeInput(nullptr, {1}, 10, {inMapAt({2})});
    auto second =
        makeInput(makeNulls({false, false, true}), {1, 2}, 20, secondInMaps);
    testConcat({first, second}, {.emptyForNull = true});
  }

  {
    auto first = makeInput(nullptr, {1}, 10, {inMapAt({0})});
    auto second = makeInput(nullptr, {1}, 20, {inMapAt({1})});
    auto third = makeInput(nullptr, {1}, 30, {inMapAt({0, 1, 2})});
    auto fourth = makeInput(nullptr, {1}, 40, {inMapAt({2})});
    testConcat({first, second, third, fourth}, {});
  }
}

TEST_F(MapConcatTest, flatMapVarcharKeys) {
  auto mapType = MAP(VARCHAR(), ARRAY(BIGINT()));
  auto flatMap = std::make_shared<FlatMapVector>(
      pool(),
      mapType,
      nullptr,
      3,
      makeFlatVector<StringView>({"a", "a long key that is not inlined"}),
      std::vector<VectorPtr>{
          makeArrayVectorFromJson<int64_t>({"[1, 2]", "[3]", "null"}),
          makeArrayVectorFromJson<int64_t>({"[10]", "[]", "[30, 31]"}),
      },
      std::vector<BufferPtr>{makeNulls({false, true, false})});
  auto map = makeMapVector(
      {0, 1, 1},
      makeFlatVector<StringView>({"a long key that is not inlined", "b"}),
      makeArrayVectorFromJson<int64_t>({"[99]", "[7, 8]"}));

  testFlatMapConcat(
      {flatMap, test::withFlatMapBase(map)},
      {flatMap->toMapVector(), map},
      {},
      makeMapVector(
          {0, 2, 3},
          makeFlatVector<StringView>({
              "a",
              "a long key that is not inlined",
              "a long key that is not inlined",
              "a",
              "b",
              "a long key that is not inlined",
          }),
          makeArrayVectorFromJson<int64_t>({
              "[1, 2]",
              "[99]",
              "[]",
              "null",
              "[7, 8]",
              "[30, 31]",
          })));
}

TEST_F(MapConcatTest, flatMapCustomComparisonKeys) {
  static const Mod100Type kMod100Type;
  auto mod100Type =
      std::shared_ptr<const Type>(std::shared_ptr<const Type>{}, &kMod100Type);
  auto makeInput = [&](int64_t key, int64_t value) {
    return std::make_shared<FlatMapVector>(
        pool(),
        MAP(mod100Type, BIGINT()),
        nullptr,
        1,
        makeFlatVector<int64_t>({key}, mod100Type),
        std::vector<VectorPtr>{makeFlatVector<int64_t>({value})},
        std::vector<BufferPtr>{nullptr});
  };
  auto first = makeInput(1, 10);
  auto second = makeInput(101, 20);
  SelectivityVector rows(1);

  auto result = concat<FlatMapVector>({first, second}, rows);
  ASSERT_EQ(result->numDistinctKeys(), 1);
  test::assertEqualVectors(result->mapValuesAt(0), second->mapValuesAt(0));
  test::assertEqualVectors(
      concat({first->toMapVector(), second->toMapVector()}, rows), result);
}

TEST_F(MapConcatTest, flatMapArrayKeys) {
  auto mapType = MAP(ARRAY(BIGINT()), BIGINT());
  auto makeInput = [&](const std::vector<std::string>& keys,
                       std::vector<VectorPtr> values) {
    return std::make_shared<FlatMapVector>(
        pool(),
        mapType,
        nullptr,
        2,
        makeArrayVectorFromJson<int64_t>(keys),
        std::move(values),
        std::vector<BufferPtr>{});
  };
  auto first = makeInput(
      {"[1, 2]", "[3]"},
      {makeFlatVector<int64_t>({10, 11}), makeFlatVector<int64_t>({20, 21})});
  auto second = makeInput(
      {"[3]", "[1]"},
      {makeFlatVector<int64_t>({30, 31}), makeFlatVector<int64_t>({40, 41})});

  testFlatMapConcat(
      {first, second},
      {first->toMapVector(), second->toMapVector()},
      {},
      makeMapVector(
          {0, 3},
          makeArrayVectorFromJson<int64_t>({
              "[1, 2]",
              "[3]",
              "[1]",
              "[1, 2]",
              "[3]",
              "[1]",
          }),
          makeFlatVector<int64_t>({10, 30, 40, 11, 31, 41})));
}

TEST_F(MapConcatTest, flatMapRowsShorterThanInputs) {
  auto first = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{1:11}",
      "{3:32}",
      "{2:23}",
  });
  auto second = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{2:21}",
      "{4:41}",
      "{4:42}",
      "{5:53}",
  });
  SelectivityVector rows(2);

  auto result = concat<FlatMapVector>({first, second}, rows);
  ASSERT_EQ(result->size(), 2);
  result->validate();
  test::assertEqualVectors(
      makeMapVectorFromJson<int64_t, int64_t>({
          "{1:10, 2:21}",
          "{1:11, 4:41}",
      }),
      result);
}

TEST_F(MapConcatTest, flatMapSharesValues) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{1:11}",
      "null",
  });
  auto other = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{2:21}",
      "{3:30}",
      "{3:32}",
  });
  SelectivityVector rows(3);

  auto valuesOf = [](const FlatMapVector& vector, int64_t key) {
    return vector.mapValuesAt(vector.getKeyChannel(key).value());
  };

  auto result =
      concat<FlatMapVector>({flatMap, other}, rows, {.emptyForNull = true});
  ASSERT_EQ(result->numDistinctKeys(), 3);
  EXPECT_EQ(valuesOf(*result, 1), valuesOf(*flatMap, 1));
  EXPECT_NE(valuesOf(*result, 2), valuesOf(*flatMap, 2));
  EXPECT_EQ(valuesOf(*result, 2), valuesOf(*other, 2));
  EXPECT_EQ(valuesOf(*result, 3), valuesOf(*other, 3));

  result = concat<FlatMapVector>({flatMap}, rows, {.emptyForNull = true});
  EXPECT_EQ(result->distinctKeys(), flatMap->distinctKeys());
  EXPECT_TRUE(result->testingSharesKeyIndex(*flatMap));
  EXPECT_EQ(valuesOf(*result, 1), valuesOf(*flatMap, 1));
  EXPECT_EQ(valuesOf(*result, 2), valuesOf(*flatMap, 2));
}

TEST_F(MapConcatTest, flatMapEncoded) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{1:11}",
      "null",
      "{2:23}",
  });

  {
    auto indices = makeIndices({3, 0, 0, 2, 1});
    auto nulls = makeNulls({false, false, true, false, false});
    auto wrap = [&](const VectorPtr& vector) {
      return BaseVector::wrapInDictionary(nulls, indices, 5, vector);
    };
    auto other = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{5:50}",
        "{}",
        "{1:12}",
        "{1:13}",
        "{2:24}",
    });

    testFlatMapConcat(
        {wrap(flatMap), other},
        {wrap(flatMap->toMapVector()), other->toMapVector()},
        {},
        makeMapVectorFromJson<int64_t, int64_t>({
            "{2:23, 5:50}",
            "{1:10, 2:20}",
            "null",
            "null",
            "{1:11, 2:24}",
        }));
    testFlatMapConcat(
        {wrap(flatMap), other},
        {wrap(flatMap->toMapVector()), other->toMapVector()},
        {.emptyForNull = true},
        makeMapVectorFromJson<int64_t, int64_t>({
            "{2:23, 5:50}",
            "{1:10, 2:20}",
            "{1:12}",
            "{1:13}",
            "{1:11, 2:24}",
        }));
  }

  {
    auto other = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{5:50}",
        "{2:21}",
        "null",
    });
    testFlatMapConcat(
        {BaseVector::wrapInConstant(3, 0, flatMap), other},
        {BaseVector::wrapInConstant(3, 0, flatMap->toMapVector()),
         other->toMapVector()},
        {},
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:10, 2:20, 5:50}",
            "{1:10, 2:21}",
            "null",
        }));
  }
}

TEST_F(MapConcatTest, flatMapConstantNotFirst) {
  auto first = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{1:11}",
      "null",
      "{2:23}",
  });

  {
    auto constantBase = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{2:200, 3:300}",
        "{4:400}",
    });
    auto last = makeFlatMapVectorFromJson<int64_t, int64_t>({
        "{3:30}",
        "{5:50}",
        "{3:32}",
        "{}",
    });
    testFlatMapConcat(
        {first, BaseVector::wrapInConstant(4, 0, constantBase), last},
        {first->toMapVector(),
         BaseVector::wrapInConstant(4, 0, constantBase->toMapVector()),
         last->toMapVector()},
        {},
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:10, 2:200, 3:30}",
            "{1:11, 2:200, 3:300, 5:50}",
            "null",
            "{2:200, 3:300}",
        }));
    testFlatMapConcat(
        {first, BaseVector::wrapInConstant(4, 0, constantBase), last},
        {first->toMapVector(),
         BaseVector::wrapInConstant(4, 0, constantBase->toMapVector()),
         last->toMapVector()},
        {.emptyForNull = true},
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1:10, 2:200, 3:30}",
            "{1:11, 2:200, 3:300, 5:50}",
            "{2:200, 3:32}",
            "{2:200, 3:300}",
        }));
  }
}

TEST_F(MapConcatTest, flatMapDictionaryWrapped) {
  VectorFuzzer::Options options;
  options.allowFlatMapVector = true;
  options.nullRatio = 0.1;
  VectorFuzzer fuzzer(options, pool());

  constexpr vector_size_t kSize{1'003};
  auto flatMap = fuzzer.fuzzFlatMap(BIGINT(), INTEGER(), kSize);
  auto indices = makeIndices(kSize, [](auto row) { return (row * 7) % kSize; });
  auto nulls = makeNulls(kSize, [](auto row) { return row % 11 == 0; });
  auto wrap = [&](const VectorPtr& vector) {
    return BaseVector::wrapInDictionary(nulls, indices, kSize, vector);
  };
  auto map = makeMapVector<int64_t, int32_t>(
      kSize,
      [](auto row) { return row % 3; },
      [](auto row) { return std::numeric_limits<int64_t>::max() - row % 5; },
      [](auto row) { return row; });

  for (const auto& config :
       {MapConcatConfig{}, MapConcatConfig{.emptyForNull = true}}) {
    SelectivityVector rows(kSize);
    auto result = concat<FlatMapVector>(
        {wrap(flatMap), test::withFlatMapBase(map)}, rows, config);
    ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
    result->validate();
    test::assertEqualVectors(
        concat({wrap(flatMap->toMapVector()), map}, rows, config), result);

    std::vector<const BaseVector*> dictionaries;
    for (const auto& values : result->mapValues()) {
      if (values->encoding() == VectorEncoding::Simple::DICTIONARY) {
        dictionaries.push_back(values.get());
      }
    }
    ASSERT_EQ(dictionaries.size(), flatMap->numDistinctKeys());
    for (const auto* dictionary : dictionaries) {
      EXPECT_EQ(dictionary->wrapInfo(), dictionaries[0]->wrapInfo());
    }
  }
}

TEST_F(MapConcatTest, flatMapDictionaryWrappedWithoutNulls) {
  constexpr vector_size_t kSize{1'003};
  auto first = makeMapVector<int64_t, int32_t>(
      kSize,
      [](auto row) { return row % 4; },
      [](auto row) { return row % 7; },
      [](auto row) { return row; });
  auto second = makeMapVector<int64_t, int32_t>(
      kSize,
      [](auto row) { return row % 3; },
      [](auto row) { return row % 5; },
      [](auto row) { return -row; });
  auto indices = makeIndices(kSize, [](auto row) { return (row * 7) % kSize; });
  auto wrap = [&](const VectorPtr& vector) {
    return BaseVector::wrapInDictionary(nullptr, indices, kSize, vector);
  };

  SelectivityVector rows(kSize);
  auto result = concat<FlatMapVector>(
      {wrap(test::withFlatMapBase(first)), test::withFlatMapBase(second)},
      rows,
      {});
  result->validate();
  test::assertEqualVectors(concat({wrap(first), second}, rows, {}), result);
}

TEST_F(MapConcatTest, flatMapPartialRows) {
  auto flatMap = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20}",
      "{1:11}",
      "null",
      "{2:23}",
  });
  auto other = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{5:50}",
      "{6:60}",
      "{1:12}",
      "{2:24}",
  });
  SelectivityVector rows(4);
  rows.setValid(1, false);
  rows.setValid(3, false);
  rows.updateBounds();

  for (const auto& config :
       {MapConcatConfig{}, MapConcatConfig{.emptyForNull = true}}) {
    auto result = concat<FlatMapVector>({flatMap, other}, rows, config);
    ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
    test::assertEqualVectors(
        concat({flatMap->toMapVector(), other->toMapVector()}, rows, config),
        result,
        rows);
  }
}

TEST_F(MapConcatTest, flatMapDropsKeys) {
  constexpr vector_size_t kSize{4};
  auto inMapAt = [&](const std::vector<vector_size_t>& rowsWithKey) {
    auto inMap = AlignedBuffer::allocate<bool>(kSize, pool(), false);
    for (auto row : rowsWithKey) {
      bits::setBit(inMap->asMutable<uint64_t>(), row);
    }
    return inMap;
  };
  auto first = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      kSize,
      makeFlatVector<int64_t>({1, 2, 3, 7}),
      std::vector<VectorPtr>{
          makeFlatVector<int64_t>({10, 11, 12, 13}),
          makeFlatVector<int64_t>({20, 21, 22, 23}),
          makeFlatVector<int64_t>({30, 31, 32, 33}),
          makeFlatVector<int64_t>({70, 71, 72, 73}),
      },
      std::vector<BufferPtr>{nullptr, inMapAt({1}), inMapAt({}), inMapAt({3})});
  auto second = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{5:50}",
      "{2:22, 6:60}",
      "{}",
      "null",
  });
  SelectivityVector rows(kSize);
  rows.setValid(1, false);
  rows.updateBounds();

  auto keysOf = [](const FlatMapVector& vector) {
    DecodedVector decoded(*vector.distinctKeys());
    std::vector<int64_t> keys;
    for (vector_size_t channel = 0; channel < vector.numDistinctKeys();
         ++channel) {
      keys.push_back(decoded.valueAt<int64_t>(channel));
    }
    return keys;
  };
  auto testConcat = [&](const std::vector<VectorPtr>& inputs,
                        const SelectivityVector& rows,
                        const MapConcatConfig& config) {
    auto result = concat<FlatMapVector>(inputs, rows, config);
    result->validate();
    std::vector<VectorPtr> mapInputs;
    for (const auto& input : inputs) {
      mapInputs.push_back(test::withMapBase(input));
    }
    test::assertEqualVectors(concat(mapInputs, rows, config), result, rows);
    return result;
  };

  {
    auto result = testConcat({first, second}, rows, {});
    EXPECT_THAT(keysOf(*result), testing::ElementsAre(1, 5));
    EXPECT_FALSE(result->testingSharesKeyIndex(*first));
  }

  {
    auto result = testConcat({first, second}, rows, {.emptyForNull = true});
    EXPECT_THAT(keysOf(*result), testing::ElementsAre(1, 7, 5));
  }

  {
    auto result = testConcat({first}, rows, {});
    EXPECT_THAT(keysOf(*result), testing::ElementsAre(1, 7));
    EXPECT_FALSE(result->testingSharesKeyIndex(*first));
    EXPECT_EQ(result->mapValuesAt(0), first->mapValuesAt(0));
    EXPECT_EQ(result->mapValuesAt(1), first->mapValuesAt(3));
  }

  {
    auto result = testConcat({first}, SelectivityVector(kSize), {});
    EXPECT_THAT(keysOf(*result), testing::ElementsAre(1, 2, 7));
    EXPECT_EQ(result->mapValuesAt(2), first->mapValuesAt(3));
    EXPECT_EQ(result->inMapsAt(2), first->inMapsAt(3));
  }

  {
    auto result = testConcat({first}, SelectivityVector(kSize, false), {});
    EXPECT_EQ(result->numDistinctKeys(), 0);
  }
}

TEST_F(MapConcatTest, flatMapDroppedKeyReplacedByNewKey) {
  auto first = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      2,
      makeFlatVector<int64_t>({1, 2}),
      std::vector<VectorPtr>{
          makeFlatVector<int64_t>({10, 11}),
          makeFlatVector<int64_t>({20, 21}),
      },
      std::vector<BufferPtr>{nullptr, makeNulls({true, true})});
  auto second = makeFlatMapVectorFromJson<int64_t, int64_t>({
      "{3:30}",
      "{}",
  });
  SelectivityVector rows(2);

  auto result = concat<FlatMapVector>({first, second}, rows);
  result->validate();
  EXPECT_FALSE(result->testingSharesKeyIndex(*first));
  test::assertEqualVectors(
      makeMapVectorFromJson<int64_t, int64_t>({
          "{1:10, 3:30}",
          "{1:11}",
      }),
      result);
}

TEST_F(MapConcatTest, flatMapSparseKeys) {
  constexpr vector_size_t kSize{1'000};
  using Map = std::vector<std::pair<int64_t, std::optional<int64_t>>>;
  std::vector<std::optional<Map>> firstMaps;
  std::vector<std::optional<Map>> secondMaps;
  for (vector_size_t row = 0; row < kSize; ++row) {
    if (row % 13 == 0) {
      firstMaps.push_back(std::nullopt);
    } else {
      firstMaps.push_back(Map{{row % 200, row}});
    }
    if (row % 17 == 0) {
      secondMaps.push_back(std::nullopt);
    } else {
      secondMaps.push_back(
          Map{{(row + 100) % 200, -row}, {200 + row % 50, row * 10}});
    }
  }
  auto first = makeNullableFlatMapVector<int64_t, int64_t>(firstMaps);
  auto second = makeNullableFlatMapVector<int64_t, int64_t>(secondMaps);
  auto reversed = BaseVector::wrapInDictionary(
      makeNulls(kSize, [](auto row) { return row % 19 == 0; }),
      makeIndices(kSize, [](auto row) { return kSize - 1 - row; }),
      kSize,
      second);
  auto duplicate = test::withFlatMapBase(
      makeMapVector<int64_t, int64_t>(
          kSize,
          [](auto row) { return row == 7 ? 1 : 0; },
          [](auto) { return 7; },
          [](auto) { return 70; }));

  SelectivityVector allRows(kSize);
  SelectivityVector partialRows(kSize);
  for (vector_size_t row = 0; row < kSize; ++row) {
    if (row < 10 || row % 3 == 0) {
      partialRows.setValid(row, false);
    }
  }
  partialRows.updateBounds();

  auto testConcat = [&](const std::vector<VectorPtr>& inputs,
                        const SelectivityVector& rows,
                        const MapConcatConfig& config) {
    SCOPED_TRACE(
        fmt::format(
            "rows: {}, emptyForNull: {}, throwOnDuplicateKeys: {}",
            rows.countSelected(),
            config.emptyForNull,
            config.throwOnDuplicateKeys));
    auto result = concat<FlatMapVector>(inputs, rows, config);
    result->validate();
    std::vector<VectorPtr> mapInputs;
    for (const auto& input : inputs) {
      mapInputs.push_back(test::withMapBase(input));
    }
    test::assertEqualVectors(concat(mapInputs, rows, config), result, rows);
  };

  for (const auto& config : {
           MapConcatConfig{},
           MapConcatConfig{.emptyForNull = true},
           MapConcatConfig{.throwOnDuplicateKeys = true},
       }) {
    testConcat({first, reversed}, allRows, config);
    for (const auto* rows : {&allRows, &partialRows}) {
      testConcat({first, second}, *rows, config);
      testConcat({second, first}, *rows, config);
      testConcat(
          {first, second, duplicate},
          *rows,
          {.emptyForNull = config.emptyForNull});
    }
  }

  const MapConcatConfig config{.throwOnDuplicateKeys = true};
  VELOX_ASSERT_THROW(
      concat<FlatMapVector>({first, second, duplicate}, allRows, config),
      "Duplicate map key 7 was found.");
  VELOX_ASSERT_THROW(
      concat(
          {
              first->toMapVector(),
              second->toMapVector(),
              test::withMapBase(duplicate),
          },
          allRows,
          config),
      "Duplicate map key 7 was found.");
  testConcat({first, second, duplicate}, partialRows, config);
}

TEST_F(MapConcatTest, flatMapFuzz) {
  VectorFuzzer::Options options;
  options.allowFlatMapVector = true;
  options.nullRatio = 0.1;
  VectorFuzzer fuzzer(options, pool());

  constexpr vector_size_t kSize{1'000};
  for (auto iteration = 0; iteration < 20; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration: {}", iteration));
    auto first = fuzzer.fuzzFlatMap(BIGINT(), INTEGER(), kSize);
    auto second = fuzzer.fuzzFlatMap(BIGINT(), INTEGER(), kSize);
    auto third = fuzzer.fuzzFlatMap(BIGINT(), INTEGER(), kSize);
    for (const auto& config :
         {MapConcatConfig{}, MapConcatConfig{.emptyForNull = true}}) {
      SelectivityVector rows(kSize);
      auto result = concat<FlatMapVector>({first, third, second}, rows, config);
      result->validate();
      test::assertEqualVectors(
          concat(
              {first->toMapVector(),
               third->toMapVector(),
               second->toMapVector()},
              rows,
              config),
          result);
    }
  }
}

} // namespace
} // namespace facebook::velox
