// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/BitMap.h"
#include "gtest/gtest.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::function::aggregate {
class BitMapTest : public ::testing::Test {
 protected:
  BitMap CreateBitMap(int64_t length) {
    return BitMap(length);
  }
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

  for (int64_t i = 0; i < bitMap.length(); i++) {
    // All bits should be false initially.
    ASSERT_TRUE(bitMap.getBit(i) == false);
  }

  // Set the bits at indices vector to true.
  std::vector<int64_t> indices = {
      1, 4, 6, 8, 32, 323, 873, 9887, 33778, 787666};

  for (const auto& index : indices) {
    bitMap.setBit(index, true);
  }

  // Verify that the bits at indices vector are set to true,
  // and all other bits are set to false.
  for (int64_t i = 0; i < bitMap.length(); i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      ASSERT_TRUE(bitMap.getBit(i) == true);
    } else {
      ASSERT_TRUE(bitMap.getBit(i) == false);
    }
  }

  // Count the number of bits set to true.
  ASSERT_EQ(bitMap.cardinality(), indices.size());

  // Flip the bits back to false at the indices vector.
  for (const auto& index : indices) {
    bitMap.flipBit(index);
  }

  // Verify that the all bits are set to false,
  for (int64_t i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.getBit(i) == false);
  }
}

TEST_F(BitMapTest, randomFlipTest) {
  BitMap bitMap = CreateBitMap(24);

  // Flip the bit if p >= 0.5.
  RandomizationStrategyForTest randomizationStrategy;

  bitMap.flipBit(0, 0.6, randomizationStrategy);
  ASSERT_TRUE(bitMap.getBit(0));

  bitMap.flipBit(0, 0.6, randomizationStrategy);
  ASSERT_FALSE(bitMap.getBit(0));

  bitMap.flipBit(0, 0.4, randomizationStrategy);
  ASSERT_FALSE(bitMap.getBit(0));

  bitMap.flipAll(0.7, randomizationStrategy);
  for (int i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.getBit(i));
  }

  bitMap.flipAll(0.3, randomizationStrategy);
  for (int i = 0; i < bitMap.length(); i++) {
    ASSERT_TRUE(bitMap.getBit(i));
  }
}

TEST_F(BitMapTest, bitwiseOrTest) {
  std::vector<int8_t> bits1 = {1, 4, 6, 32, 21, 56, 99, 8, 32, 45};
  std::vector<int8_t> bits2 = {33, 23, 11, 56, 99, 8, 32, 32, 87, 98};

  BitMap bitMap1 = BitMap(static_cast<int64_t>(bits1.size()) * 8, bits1);
  BitMap bitMap2 = BitMap(static_cast<int64_t>(bits2.size()) * 8, bits2);

  BitMap bitMap3 = BitMap(bitMap1.length(), bitMap1);
  bitMap1.bitwiseOr(bitMap2);

  for (int64_t i = 0; i < bitMap1.length(); i++) {
    ASSERT_EQ(bitMap1.getBit(i), bitMap3.getBit(i) || bitMap2.getBit(i));
  }
}

// Test the serialization and deserialization of a bit map.
// we can deserialize an integer vector to a bit map.
// We can also serialize a bit map to an integer vector.
// During the serialization, we drop the trailing zeros to save space.
TEST_F(BitMapTest, truncateSerializeDeserializeTest) {
  // Create a bit map from a bigint integer vector.
  std::vector<int8_t> bits = {100, 4, 6, 33, 21, 56, 99, 8, 32, 32};
  BitMap bitMap = BitMap(static_cast<int64_t>(bits.size()) * 8, bits);

  // Intentionally set the last 40 bits to false.
  for (int64_t i = 0; i < 40; i++) {
    bitMap.setBit(i + bitMap.length() - 40, false);
  }

  // Serialize the bit map. the expected behavior is that the trailing zeros
  // will be dropped.
  auto serialized = bitMap.toIntVector();
  ASSERT_EQ(serialized.size(), 5);

  // Assert that the serialized vector is the same as the original vector.
  for (int64_t i = 0; i < 5; i++) {
    ASSERT_EQ(serialized[i], bits[i]);
  }
}

TEST_F(BitMapTest, fullLengthserializeDeserializeTest) {
  // Create a bitmap with 1024 bits
  BitMap bitMap = CreateBitMap(1024);

  // Set specific bits, make shure the last bit is not 0.
  std::vector<int64_t> indices = {1, 4, 6, 8, 32, 323, 1023};
  for (const auto& index : indices) {
    bitMap.setBit(index, true);
  }

  // Serialize the bitmap
  auto serialized = bitMap.toIntVector();

  // Create new bitmap from serialized data
  BitMap newBitMap = BitMap(1024, serialized);

  // Verify that only originally set bits are set in new bitmap
  for (int i = 0; i < newBitMap.length(); i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      ASSERT_TRUE(newBitMap.getBit(i));
    } else {
      ASSERT_FALSE(newBitMap.getBit(i));
    }
  }
}

} // namespace facebook::velox::function::aggregate
