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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
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
  MapVectorPtr concat(
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
    return mapConcat(pool(), inputs[0]->type(), decodedPtrs, rows, config);
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
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "at " << i << ": expected " << expected->toString(i) << ", got "
        << result->toString(i);
  }
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
  ASSERT_TRUE(expected->equalValueAt(result.get(), 0, 0))
      << "expected " << expected->toString(0) << ", got "
      << result->toString(0);
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
  ASSERT_TRUE(expected->equalValueAt(result.get(), 0, 0))
      << "expected " << expected->toString(0) << ", got "
      << result->toString(0);
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
  ASSERT_TRUE(expected->equalValueAt(result.get(), 0, 0))
      << "expected " << expected->toString(0) << ", got "
      << result->toString(0);
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
  ASSERT_TRUE(expected->equalValueAt(result.get(), 0, 0))
      << "expected " << expected->toString(0) << ", got "
      << result->toString(0);
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
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "at " << i << ": expected " << expected->toString(i) << ", got "
        << result->toString(i);
  }
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

} // namespace
} // namespace facebook::velox
