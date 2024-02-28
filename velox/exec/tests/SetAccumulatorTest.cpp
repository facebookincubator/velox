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
#include "velox/exec/SetAccumulator.h"

#include <gtest/gtest.h>
#include "velox/exec/AddressableNonNullValueList.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

// The tests in this class validate the following
// (for both Primitive and Complex types) :
// i) Builds a SetAccumulator from the input data.
// ii) Tracks the unique values in the input data for validation.
// iii) Serializes the SetAccumulator and de-serializes the result in a second
//      accumulator.
// The test validates that both accumulators have the same contents and the
// contents of the deserialized accumulator comprise the unique values from
// the input data.
class SetAccumulatorTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  template <typename T>
  void testPrimitive(const VectorPtr& data) {
    std::unordered_set<T> uniqueValues;
    SetAccumulator<T> accumulator(data->type(), allocator());
    DecodedVector decodedVector(*data);
    vector_size_t nullPosition = -1;
    for (auto i = 0; i < data->size(); ++i) {
      if (decodedVector.isNullAt(i)) {
        nullPosition = i;
      }
      accumulator.addValue(decodedVector, i, allocator());
      uniqueValues.insert(decodedVector.valueAt<T>(i));
    }
    ASSERT_EQ(accumulator.size(), uniqueValues.size());

    auto serializedSize =
        nullPosition == -1 ? uniqueValues.size() + 1 : uniqueValues.size();
    auto serialized = BaseVector::create(VARBINARY(), serializedSize, pool());
    accumulator.serialize(serialized, 0);

    // Initialize another accumulator from the serialized vector.
    SetAccumulator<T> accumulator2(data->type(), allocator());
    auto flatSerialized = serialized->template asFlatVector<StringView>();
    accumulator2.deserialize(
        *flatSerialized, 0, serialized->size(), allocator());

    // Extract the contents of the accumulator. The contents should match
    // all the uniqueValues.
    auto copy = BaseVector::create(data->type(), accumulator2.size(), pool());
    auto copyFlat = copy->template asFlatVector<T>();
    accumulator2.extractValues(*copyFlat, 0);

    ASSERT_EQ(copy->size(), accumulator.size());
    for (auto i = 0; i < copy->size(); i++) {
      if (copyFlat->isNullAt(i)) {
        ASSERT_EQ(i, nullPosition);
      } else {
        ASSERT_TRUE(uniqueValues.count(copyFlat->valueAt(i)) != 0);
      }
    }
  }

  void testComplexType(const VectorPtr& data) {
    using T = AddressableNonNullValueList::Entry;
    using Set = folly::F14FastSet<
        T,
        AddressableNonNullValueList::Hash,
        AddressableNonNullValueList::EqualTo,
        AlignedStlAllocator<T, 16>>;

    // Unique values set used for validation in the tests.
    AddressableNonNullValueList values;
    Set uniqueValues{
        0,
        AddressableNonNullValueList::Hash{},
        AddressableNonNullValueList::EqualTo{data->type()},
        AlignedStlAllocator<T, 16>(allocator())};

    // Build an accumulator from the input data. Also create a set of the
    // unique values for validation.
    SetAccumulator<ComplexType> accumulator1(data->type(), allocator());
    DecodedVector decodedVector(*data);
    vector_size_t nullPosition = -1;
    for (auto i = 0; i < data->size(); ++i) {
      accumulator1.addValue(decodedVector, i, allocator());
      if (!decodedVector.isNullAt(i)) {
        auto entry = values.append(decodedVector, i, allocator());
        if (uniqueValues.contains(entry)) {
          values.removeLast(entry);
          continue;
        }
        ASSERT_TRUE(uniqueValues.insert(entry).second);
        ASSERT_TRUE(uniqueValues.contains(entry));
        ASSERT_FALSE(uniqueValues.insert(entry).second);
      } else {
        nullPosition = i;
      }
    }

    auto accumulatorSizeCheck =
        [&](const SetAccumulator<ComplexType>& accumulator) {
          if (nullPosition != -1) {
            ASSERT_EQ(accumulator.size(), uniqueValues.size() + 1);
          } else {
            ASSERT_EQ(accumulator.size(), uniqueValues.size());
          }
        };
    accumulatorSizeCheck(accumulator1);

    // Serialize the accumulator.
    auto serialized =
        BaseVector::create(VARBINARY(), uniqueValues.size() + 1, pool());
    accumulator1.serialize(serialized, 0);

    // Initialize another accumulator from the serialized vector.
    SetAccumulator<ComplexType> accumulator2(data->type(), allocator());
    auto serializedFlat = serialized->asFlatVector<StringView>();
    accumulator2.deserialize(
        *serializedFlat, 0, serialized->size(), allocator());
    ASSERT_EQ(accumulator2.size(), accumulator1.size());
    accumulatorSizeCheck(accumulator2);

    // Extract the contents of the deserialized accumulator.
    // All the values extracted are in the uniqueValues set already.
    auto copy = BaseVector::create(data->type(), accumulator2.size(), pool());
    accumulator2.extractValues(*copy, 0);
    DecodedVector copyDecoded(*copy);
    for (auto i = 0; i < copy->size(); ++i) {
      if (copyDecoded.isNullAt(i)) {
        ASSERT_EQ(i, nullPosition);
      } else {
        auto position = values.append(copyDecoded, i, allocator());
        ASSERT_TRUE(uniqueValues.contains(position));
        values.removeLast(position);
      }
    }
  }

  HashStringAllocator* allocator() {
    return allocator_.get();
  }

  std::unique_ptr<HashStringAllocator> allocator_{
      std::make_unique<HashStringAllocator>(pool())};
};

TEST_F(SetAccumulatorTest, integral) {
  auto data1 = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  testPrimitive<int32_t>(data1);
  auto data2 = makeFlatVector<int16_t>({1, 2, 2, 3, 3, 4, 5, 5});
  testPrimitive<int16_t>(data2);
  auto data3 = makeFlatVector<int64_t>({1, 2, 2, 3, 4, 5, 3, 1, 4});
  testPrimitive<int64_t>(data3);
  auto data4 = makeNullableFlatVector<int32_t>({std::nullopt, 1, 2});
  testPrimitive<int32_t>(data4);
}

TEST_F(SetAccumulatorTest, date) {
  auto data = makeFlatVector<int32_t>({1, 2, 3, 4, 5}, DATE());
  testPrimitive<int32_t>(data);
  data = makeFlatVector<int32_t>({1, 2, 2, 3, 3, 4, 5, 5}, DATE());
  testPrimitive<int32_t>(data);
  data = makeFlatVector<int32_t>({1, 2, 2, 3, 4, 5, 3, 1, 4}, DATE());
  testPrimitive<int32_t>(data);
  data = makeNullableFlatVector<int32_t>({1, 2, std::nullopt}, DATE());
  testPrimitive<int32_t>(data);
}

TEST_F(SetAccumulatorTest, strings) {
  auto data =
      makeFlatVector<StringView>({"abc", "non-inline string", "1234!@#$"});
  testPrimitive<StringView>(data);

  data = makeFlatVector<StringView>(
      {"abc",
       "non-inline string",
       "non-inline string",
       "reallylongstringreallylongstringreallylongstring",
       "1234!@#$",
       "abc"});
  testPrimitive<StringView>(data);

  data = makeNullableFlatVector<StringView>({"abc", std::nullopt, "def"});
  testPrimitive<StringView>(data);
}

TEST_F(SetAccumulatorTest, array) {
  auto data = makeArrayVector<int32_t>({
      {1, 2, 3},
      {4, 5},
      {6, 7, 8, 9},
      {},
  });
  testComplexType(data);

  data = makeNullableArrayVector<int32_t>({
      {1, 2, 3},
      {4, 5},
      {std::nullopt},
      {6, 7, 8, 9},
      {},
  });
  testComplexType(data);

  data = makeArrayVector<int32_t>({
      {1, 2, 3},
      {1, 2, 3},
      {4, 5},
      {6, 7, 8, 9},
      {},
      {4, 5},
      {1, 2, 3},
      {},
  });
  testComplexType(data);
}

TEST_F(SetAccumulatorTest, map) {
  auto data = makeMapVector<int32_t, float>({
      {{1, 10.1213}, {2, 20}},
      {{3, 30}, {4, 40.258703570235497205}, {5, 50}},
      {{1, 10.4324}, {3, 30}, {4, 40.45209809}, {6, 60}},
      {},
  });
  testComplexType(data);

  data = makeNullableMapVector<int32_t, StringView>({
      {{{1, "abc"}, {2, "this is a non-inline string"}}},
      std::nullopt,
      {{{3, "qrs"}, {4, "m"}, {5, "%&^%&^af489372843"}}},
      {{}},
  });
  testComplexType(data);

  // Has non-unique rows.
  data = makeMapVector<int16_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{3, 30}, {4, 40}, {5, 50}},
      {{3, 30}, {4, 40}, {5, 50}},
      {{1, 10}, {2, 20}},
      {{1, 10}, {3, 30}, {4, 40}, {6, 60}},
      {},
      {{1, 10}, {2, 20}},
      {},
      {{3, 30}, {4, 40}, {5, 50}},
  });
  testComplexType(data);
}

TEST_F(SetAccumulatorTest, row) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<StringView>(
          {"abc", "this is a non-inline string", "def", "ghij", "klm"}),
      makeFlatVector<int64_t>({11, 22, 33, 44, 55}),
  });
  testComplexType(data);

  // Has non-unique rows.
  data = makeRowVector({
      makeFlatVector<int16_t>({1, 2, 3, 4, 2, 5, 3}),
      makeFlatVector<float>(
          {10.1, 20.1234567, 30.35, 40, 20.1234567, 50.42309234, 30}),
      makeFlatVector<int32_t>({11, 22, 33, 44, 22, 55, 33}, DATE()),
  });
  testComplexType(data);

  data = makeRowVector({
      makeNullableFlatVector<int16_t>({1, 2, std::nullopt, 4, 5}),
      makeNullableFlatVector<int32_t>({10, 20, 30, std::nullopt, 50}),
      makeNullableFlatVector<int64_t>({std::nullopt, 22, 33, std::nullopt, 55}),
  });
  testComplexType(data);
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql
