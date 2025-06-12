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

  for (int i = 0; i < bitMap.length(); i++) {
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
  for (int i = 0; i < bitMap.length(); i++) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      ASSERT_TRUE(bitMap.getBit(i) == true);
    } else {
      ASSERT_TRUE(bitMap.getBit(i) == false);
    }
  }

  // Count the number of bits set to true.
  ASSERT_TRUE(bitMap.cardinality() == indices.size());

  // Flip the bits back to false at the indices vector.
  for (const auto& index : indices) {
    bitMap.flipBit(index);
  }

  // Verify that the all bits are set to false,
  for (int i = 0; i < bitMap.length(); i++) {
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
  std::vector<int64_t> bits1 = {1, 4, 6, 323, 211, 56, 99, 8, 32, 323};
  std::vector<int64_t> bits2 = {332, 323, 211, 56, 99, 8, 32, 323, 873, 9887};

  BitMap bitMap1 = BitMap(bits1.size() * 64, bits1);
  BitMap bitMap2 = BitMap(bits2.size() * 64, bits2);

  BitMap bitMap3 = BitMap(bitMap1.length(), bitMap1);
  bitMap1.bitwiseOr(bitMap2);

  for (int i = 0; i < bitMap1.length(); i++) {
    ASSERT_EQ(bitMap1.getBit(i), bitMap3.getBit(i) || bitMap2.getBit(i));
  }
}

TEST_F(BitMapTest, bitwiseXorTest) {
  std::vector<int64_t> bits1 = {1, 4, 6, 323, 211, 56, 99, 8, 32, 323};
  std::vector<int64_t> bits2 = {332, 323, 211, 56, 99, 8, 32, 323, 873, 9887};

  BitMap bitMap1 = BitMap(bits1.size() * 64, bits1);
  BitMap bitMap2 = BitMap(bits2.size() * 64, bits2);

  BitMap bitMap3 = BitMap(bitMap1.length(), bitMap1);
  bitMap1.bitwiseXor(bitMap2);

  for (int i = 0; i < bitMap1.length(); i++) {
    ASSERT_EQ(bitMap1.getBit(i), bitMap3.getBit(i) ^ bitMap2.getBit(i));
  }
}

TEST_F(BitMapTest, serializeTest) {
  // Create a bit map from a bigint integer vector.
  std::vector<int64_t> bits = {100, 4, 6, 323, 211, 56, 99, 8, 32, 323};
  BitMap bitMap = BitMap(bits.size() * 64, bits);

  // Set the last 128 bits to 0.
  for (int i = 0; i < 128; i++) {
    bitMap.setBit(i + bitMap.length() - 128, false);
  }

  // Serialize the bit map.
  auto serialized = bitMap.toIntVector();
  ASSERT_EQ(serialized.size(), 8);

  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(serialized[i], bits[i]);
  }
}

} // namespace facebook::velox::function::aggregate
