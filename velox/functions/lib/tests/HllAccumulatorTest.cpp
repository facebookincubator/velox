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
#include "velox/functions/lib/HllAccumulator.h"

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::common::hll;

namespace {
const int8_t kDefaultIndexBitLength = 11;
const double kDefaultStandardError =
    1.04 / std::sqrt(1 << kDefaultIndexBitLength);
} // namespace

template <typename TAllocator>
class HllAccumulatorTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    if constexpr (std::is_same_v<TAllocator, HashStringAllocator>) {
      allocator_ = &hsa_;
    } else {
      allocator_ = pool_.get();
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator hsa_{pool_.get()};
  TAllocator* allocator_{};
};

using AllocatorTypes =
    ::testing::Types<HashStringAllocator, memory::MemoryPool>;

class NameGenerator {
 public:
  template <typename TAllocator>
  static std::string GetName(int) {
    if constexpr (std::is_same_v<TAllocator, HashStringAllocator>) {
      return "hsa";
    } else if constexpr (std::is_same_v<TAllocator, memory::MemoryPool>) {
      return "pool";
    } else {
      VELOX_UNREACHABLE(
          "Only HashStringAllocator and MemoryPool are supported allocator types.");
    }
  }
};

TYPED_TEST_SUITE(HllAccumulatorTest, AllocatorTypes, NameGenerator);

TYPED_TEST(HllAccumulatorTest, basicInt64) {
  // Test SparseHLL.
  HllAccumulator<int64_t, false, TypeParam> accumulator(
      kDefaultIndexBitLength, this->allocator_);

  constexpr int64_t numValues = 100;
  for (int64_t i = 0; i < numValues; i++) {
    accumulator.append(i);
  }

  // Sparse HLL should be exact.
  EXPECT_EQ(accumulator.cardinality(), numValues);

  // Test DenseHLL.
  constexpr int64_t numValuesDense = 10000;
  for (int64_t i = 0; i < numValuesDense; i++) {
    accumulator.append(i);
  }
  EXPECT_NEAR(
      accumulator.cardinality(),
      numValuesDense,
      numValuesDense * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, basicDouble) {
  // Test SparseHLL.
  HllAccumulator<double, true, TypeParam> accumulator(
      kDefaultIndexBitLength, this->allocator_);

  constexpr int numValues = 150;
  for (int i = 0; i < numValues; i++) {
    accumulator.append(static_cast<double>(i) * 1.5);
  }
  EXPECT_EQ(accumulator.cardinality(), numValues);

  // Test DenseHLL.
  constexpr int numValuesDense = 15000;
  for (int i = numValues; i < numValuesDense; i++) {
    accumulator.append(static_cast<double>(i) * 1.5);
  }
  EXPECT_NEAR(
      accumulator.cardinality(),
      numValuesDense,
      numValuesDense * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, basicStringView) {
  // Test SparseHLL.
  HllAccumulator<StringView, true, TypeParam> accumulator(
      kDefaultIndexBitLength, this->allocator_);

  constexpr int numValues = 100;
  std::vector<std::string> strings;
  strings.reserve(numValues);
  for (int i = 0; i < numValues; i++) {
    strings.push_back("value_" + std::to_string(i));
  }
  for (const auto& str : strings) {
    accumulator.append(StringView(str));
  }
  EXPECT_EQ(accumulator.cardinality(), numValues);

  // Test DenseHLL.
  constexpr int numValuesDense = 10000;
  for (int i = numValues; i < numValuesDense; i++) {
    strings.push_back("value_" + std::to_string(i));
  }
  for (int i = numValues; i < numValuesDense; i++) {
    accumulator.append(StringView(strings[i]));
  }
  EXPECT_NEAR(
      accumulator.cardinality(),
      numValuesDense,
      numValuesDense * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, serde) {
  HllAccumulator<int64_t, false, TypeParam> accumulator(
      kDefaultIndexBitLength, this->allocator_);

  constexpr int64_t numValues = 200;
  for (int64_t i = 0; i < numValues; i++) {
    accumulator.append(i);
  }

  auto size = accumulator.serializedSize();
  std::string serialized(size, '\0');
  accumulator.serialize(serialized.data());

  auto deserialized = HllAccumulator<int64_t, false, TypeParam>::deserialize(
      serialized.data(), this->allocator_);

  EXPECT_EQ(deserialized->cardinality(), numValues);

  // Test round trip
  std::string reserialized(deserialized->serializedSize(), '\0');
  deserialized->serialize(reserialized.data());
  EXPECT_EQ(reserialized, serialized);
}

TYPED_TEST(HllAccumulatorTest, mergeWithBothSparse) {
  HllAccumulator<int64_t, false, TypeParam> accumulator1(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> accumulator2(
      kDefaultIndexBitLength, this->allocator_);

  // Add non-overlapping values that keep both sparse.
  for (int64_t i = 0; i < 100; i++) {
    accumulator1.append(i);
  }
  for (int64_t i = 100; i < 200; i++) {
    accumulator2.append(i);
  }

  accumulator1.mergeWith(accumulator2);

  // Resulting accumulator should be sparse and exact.
  EXPECT_TRUE(accumulator1.isSparse());
  EXPECT_EQ(accumulator1.cardinality(), 200);
}

TYPED_TEST(HllAccumulatorTest, mergeWithBothDense) {
  HllAccumulator<int64_t, false, TypeParam> accumulator1(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> accumulator2(
      kDefaultIndexBitLength, this->allocator_);

  // Add non-overlapping values that trigger dense mode.
  constexpr int64_t numValues = 10000;
  for (int64_t i = 0; i < 5000; i++) {
    accumulator1.append(i);
  }
  for (int64_t i = 5000; i < numValues; i++) {
    accumulator2.append(i);
  }

  accumulator1.mergeWith(accumulator2);
  EXPECT_FALSE(accumulator1.isSparse());

  EXPECT_NEAR(
      accumulator1.cardinality(), numValues, numValues * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, mergeSparseWithDense) {
  HllAccumulator<int64_t, false, TypeParam> sparseAccumulator(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> denseAccumulator(
      kDefaultIndexBitLength, this->allocator_);

  for (int64_t i = 0; i < 100; i++) {
    sparseAccumulator.append(i);
  }

  constexpr int64_t numValuesDense = 5000;
  for (int64_t i = 100; i < numValuesDense; i++) {
    denseAccumulator.append(i);
  }

  sparseAccumulator.mergeWith(denseAccumulator);

  // mergeWith should convert any sparse accumulator to dense if either is
  // dense. Result should be dense and approximate.

  EXPECT_FALSE(sparseAccumulator.isSparse());
  EXPECT_NEAR(
      sparseAccumulator.cardinality(),
      numValuesDense,
      numValuesDense * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, mergeDenseWithSparse) {
  HllAccumulator<int64_t, false, TypeParam> sparseAccumulator(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> denseAccumulator(
      kDefaultIndexBitLength, this->allocator_);

  for (int64_t i = 0; i < 100; i++) {
    sparseAccumulator.append(i);
  }

  constexpr int64_t numValuesDense = 5000;
  for (int64_t i = 100; i < numValuesDense; i++) {
    denseAccumulator.append(i);
  }

  denseAccumulator.mergeWith(sparseAccumulator);

  // Result should be dense and approximate.
  EXPECT_FALSE(denseAccumulator.isSparse());
  EXPECT_NEAR(
      denseAccumulator.cardinality(),
      numValuesDense,
      numValuesDense * kDefaultStandardError);
}

TYPED_TEST(HllAccumulatorTest, mergeWithSerializedDataSparse) {
  HllAccumulator<int64_t, false, TypeParam> accumulator1(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> accumulator2(
      kDefaultIndexBitLength, this->allocator_);

  for (int64_t i = 0; i < 100; i++) {
    accumulator1.append(i);
  }
  for (int64_t i = 100; i < 200; i++) {
    accumulator2.append(i);
  }

  auto size = accumulator2.serializedSize();
  std::string buffer(size, '\0');
  accumulator2.serialize(buffer.data());

  // Merge with serialized data (should remain sparse).
  accumulator1.mergeWith(StringView(buffer), this->allocator_);

  // Should be sparse and exact.
  EXPECT_TRUE(accumulator1.isSparse());
  EXPECT_EQ(accumulator1.cardinality(), 200);
}

TYPED_TEST(HllAccumulatorTest, mergeWithOverlappingDataSparse) {
  HllAccumulator<int64_t, false, TypeParam> accumulator1(
      kDefaultIndexBitLength, this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> accumulator2(
      kDefaultIndexBitLength, this->allocator_);

  for (int64_t i = 0; i < 100; i++) {
    accumulator1.append(i);
  }
  for (int64_t i = 50; i < 150; i++) {
    accumulator2.append(i);
  }

  accumulator1.mergeWith(accumulator2);

  // Sparse HLL should be sparse and exact: union of [0, 100) and [50, 150) =
  // [0, 150)
  EXPECT_TRUE(accumulator1.isSparse());
  EXPECT_EQ(accumulator1.cardinality(), 150);
}

TYPED_TEST(HllAccumulatorTest, mergeUninitializedAccumulator) {
  HllAccumulator<int64_t, false, TypeParam> accumulator(this->allocator_);
  HllAccumulator<int64_t, false, TypeParam> initialized(
      kDefaultIndexBitLength, this->allocator_);

  constexpr int64_t numValues = 100;
  for (int64_t i = 0; i < numValues; i++) {
    initialized.append(i);
  }

  auto size = initialized.serializedSize();
  std::string buffer(size, '\0');
  initialized.serialize(buffer.data());

  accumulator.mergeWith(StringView(buffer), this->allocator_);

  EXPECT_EQ(accumulator.cardinality(), numValues);
}
