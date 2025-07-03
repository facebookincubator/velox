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

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/BitMap.h"
#include "gtest/gtest.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {
// In tests, we use the default allocator (StlAllocator) which works with
// HashStringAllocator.
using Allocator = facebook::velox::StlAllocator<int8_t>;

class BitMapTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

 protected:
  BitMap CreateBitMap(uint32_t length) {
    return BitMap(length, &allocator_);
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

class RandomizationStrategyForTest : RandomizationStrategy {
 public:
  RandomizationStrategyForTest() : RandomizationStrategy() {}
  bool nextBoolean(double p) override {
    return p >= 0.5;
  }
};

TEST_F(BitMapTest, basicBitMapTest) {
  // Create a bit map with 983232 bits.
  BitMap bitMap = CreateBitMap(983232);

  for (uint32_t i = 0; i < bitMap.length(); i++) {
    // All bits should be false initially.
    ASSERT_TRUE(bitMap.bitAt(i) == false);
  }

  // Set the bits at indices vector to true.
  std::vector<uint32_t> indices = {
      1, 4, 6, 8, 32, 323, 873, 9887, 33778, 787666};

  for (const auto& index : indices) {
    bitMap.setBit(index, true);
  }

  // Verify that the bits at indices vector are set to true,
  // and all other bits are set to false.
  for (uint32_t i = 0; i < bitMap.length(); i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      ASSERT_TRUE(bitMap.bitAt(i) == true);
    } else {
      ASSERT_TRUE(bitMap.bitAt(i) == false);
    }
  }

  // Count the number of bits set to true.
  ASSERT_EQ(bitMap.countBits(), indices.size());

  // Flip the bits back to false at the indices vector.
  for (const auto& index : indices) {
    bitMap.flipBit(index);
  }

  // Verify that the all bits are set to false,
  for (uint32_t i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.bitAt(i) == false);
  }
}

TEST_F(BitMapTest, randomFlipTest) {
  BitMap bitMap = CreateBitMap(24);

  // Flip the bit if p >= 0.5.
  RandomizationStrategyForTest randomizationStrategy;

  bitMap.flipBit(0, 0.6, randomizationStrategy);
  ASSERT_TRUE(bitMap.bitAt(0));

  bitMap.flipBit(0, 0.6, randomizationStrategy);
  ASSERT_FALSE(bitMap.bitAt(0));

  bitMap.flipBit(0, 0.4, randomizationStrategy);
  ASSERT_FALSE(bitMap.bitAt(0));

  bitMap.flipAll(0.7, randomizationStrategy);
  for (uint32_t i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.bitAt(i));
  }

  bitMap.flipAll(0.3, randomizationStrategy);
  for (uint32_t i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.bitAt(i));
  }
}

TEST_F(BitMapTest, bitwiseOrTest) {
  std::vector<int8_t> bits1 = {1, 4, 6, 32, 21, 56, 99, 8, 32, 45};
  std::vector<int8_t> bits2 = {33, 23, 11, 56, 99, 8, 32, 32, 87, 98};

  BitMap bitMap1 = CreateBitMap(static_cast<uint32_t>(bits1.size()) * 8);
  BitMap bitMap2 = CreateBitMap(static_cast<uint32_t>(bits2.size()) * 8);

  // Initialize bitMap1 with bits1 data
  for (auto i = 0; i < bits1.size(); ++i) {
    for (auto bit = 0; bit < 8; ++bit) {
      if (bits1[i] & (1 << bit)) {
        bitMap1.setBit(static_cast<uint32_t>(i * 8 + bit), true);
      }
    }
  }

  // Initialize bitMap2 with bits2 data
  for (auto i = 0; i < bits2.size(); ++i) {
    for (auto bit = 0; bit < 8; ++bit) {
      if (bits2[i] & (1 << bit)) {
        bitMap2.setBit(static_cast<uint32_t>(i * 8 + bit), true);
      }
    }
  }

  BitMap bitMap3 = CreateBitMap(bitMap1.length());
  // Copy bitMap1 to bitMap3
  for (uint32_t i = 0; i < bitMap1.length(); ++i) {
    bitMap3.setBit(i, bitMap1.bitAt(i));
  }
  bitMap1.bitwiseOr(bitMap2);

  for (uint32_t i = 0; i < bitMap1.length(); i++) {
    ASSERT_EQ(bitMap1.bitAt(i), bitMap3.bitAt(i) || bitMap2.bitAt(i));
  }
}

// Test the serialization and deserialization of a bit map.
// we can deserialize an integer vector to a bit map.
// We can also serialize a bit map to an integer vector.
// During the serialization, we drop the trailing zeros to save space.
TEST_F(BitMapTest, truncateSerializeDeserializeTest) {
  // Create a bit map from a bigint integer vector.
  std::vector<int8_t> bits = {100, 4, 6, 33, 21, 56, 99, 8, 32, 32};
  BitMap bitMap = CreateBitMap(static_cast<uint32_t>(bits.size()) * 8);

  // Initialize bitMap with bits data
  for (auto i = 0; i < bits.size(); ++i) {
    for (auto bit = 0; bit < 8; ++bit) {
      if (bits[i] & (1 << bit)) {
        bitMap.setBit(static_cast<uint32_t>(i * 8 + bit), true);
      }
    }
  }

  // Intentionally set the last 40 bits to false.
  for (uint32_t i = 0; i < 40; i++) {
    bitMap.setBit(i + bitMap.length() - 40, false);
  }

  // Serialize the bit map. the expected behavior is that the trailing zeros
  // will be dropped.
  auto serialized = bitMap.toCompactIntVector();
  ASSERT_EQ(serialized.size(), 5);

  // Assert that the serialized vector is the same as the original vector.
  for (uint32_t i = 0; i < 5; i++) {
    ASSERT_EQ(serialized[i], bits[i]);
  }
}

TEST_F(BitMapTest, fullLengthSerializeDeserializeTest) {
  // Create a bitmap with 1024 bits
  BitMap bitMap = CreateBitMap(1024);

  // Set specific bits, make shure the last bit is not 0.
  std::vector<uint32_t> indices = {1, 4, 6, 8, 32, 323, 1023};
  for (const auto& index : indices) {
    bitMap.setBit(index, true);
  }

  // Serialize the bitmap
  auto serialized = bitMap.toCompactIntVector();

  // Create new bitmap from serialized data
  BitMap newBitMap = CreateBitMap(1024);

  // Initialize newBitMap with serialized data
  for (auto i = 0; i < serialized.size(); ++i) {
    for (auto bit = 0; bit < 8; ++bit) {
      if (serialized[i] & (1 << bit)) {
        newBitMap.setBit(static_cast<uint32_t>(i * 8 + bit), true);
      }
    }
  }

  // Verify that only originally set bits are set in new bitmap
  for (uint32_t i = 0; i < newBitMap.length(); i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      ASSERT_TRUE(newBitMap.bitAt(i));
    } else {
      ASSERT_FALSE(newBitMap.bitAt(i));
    }
  }
}

} // namespace facebook::velox::functions::aggregate
