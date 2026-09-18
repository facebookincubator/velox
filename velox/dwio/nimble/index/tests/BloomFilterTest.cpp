/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/index/BloomFilter.h"

#include <limits>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble::index {
namespace {

class BloomFilterTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("BloomFilterTest");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(BloomFilterTest, basicInsertAndTest) {
  BloomFilter filter(1'000, 10.0f, leafPool_.get());

  filter.insert("key1");
  filter.insert("key2");
  filter.insert("key3");

  EXPECT_TRUE(filter.testKey("key1"));
  EXPECT_TRUE(filter.testKey("key2"));
  EXPECT_TRUE(filter.testKey("key3"));

  // Keys not inserted should very likely return false (with low FPR at
  // 10 bits/key).
  int falsePositives = 0;
  for (int i = 0; i < 1'000; ++i) {
    if (filter.testKey("nonexistent_" + std::to_string(i))) {
      ++falsePositives;
    }
  }
  // At 10 bits/key, FPR should be ~1%. Allow generous margin.
  EXPECT_LT(falsePositives, 50);
}

TEST_F(BloomFilterTest, serialization) {
  BloomFilter filter(100, 10.0f, leafPool_.get());
  filter.insert("alpha");
  filter.insert("beta");

  // Reconstruct from serialized data.
  BloomFilter restored(
      filter.numBlocks(), filter.data(), filter.dataSize(), leafPool_.get());

  EXPECT_TRUE(restored.testKey("alpha"));
  EXPECT_TRUE(restored.testKey("beta"));
  EXPECT_FALSE(restored.testKey("gamma"));
}

TEST_F(BloomFilterTest, emptyFilter) {
  BloomFilter filter(0, 10.0f, leafPool_.get());

  // Empty filter should have at least one block.
  EXPECT_GE(filter.numBlocks(), 1);
  EXPECT_EQ(
      filter.dataSize(), filter.numBlocks() * BloomFilter::kBlockSizeBytes);

  // No keys inserted — any lookup should return false.
  EXPECT_FALSE(filter.testKey("anything"));
  EXPECT_FALSE(filter.testKey(""));
}

TEST_F(BloomFilterTest, singleEntry) {
  BloomFilter filter(1, 10.0f, leafPool_.get());

  filter.insert("only_key");
  EXPECT_TRUE(filter.testKey("only_key"));
  EXPECT_FALSE(filter.testKey("other_key"));
}

TEST_F(BloomFilterTest, emptyStringKey) {
  BloomFilter filter(100, 10.0f, leafPool_.get());

  filter.insert("");
  EXPECT_TRUE(filter.testKey(""));
  EXPECT_FALSE(filter.testKey("notempty"));
}

TEST_F(BloomFilterTest, duplicateInserts) {
  BloomFilter filter(100, 10.0f, leafPool_.get());

  filter.insert("dup");
  filter.insert("dup");
  filter.insert("dup");

  EXPECT_TRUE(filter.testKey("dup"));
}

TEST_F(BloomFilterTest, numBlocksScaling) {
  // More entries should produce more blocks.
  BloomFilter small(10, 10.0f, leafPool_.get());
  BloomFilter large(10'000, 10.0f, leafPool_.get());

  EXPECT_LT(small.numBlocks(), large.numBlocks());
}

TEST_F(BloomFilterTest, bitsPerKeyAffectsSize) {
  // Higher bits per key should produce more blocks for the same entry count.
  BloomFilter lowBits(1'000, 5.0f, leafPool_.get());
  BloomFilter highBits(1'000, 20.0f, leafPool_.get());

  EXPECT_LE(lowBits.numBlocks(), highBits.numBlocks());
}

TEST_F(BloomFilterTest, bitsPerKeyAccessor) {
  BloomFilter filter(100, 7.5f, leafPool_.get());
  EXPECT_FLOAT_EQ(filter.bitsPerKey(), 7.5f);
}

TEST_F(BloomFilterTest, invalidBitsPerKey) {
  for (const auto bitsPerKey :
       {0.0f,
        -1.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(bitsPerKey);
    EXPECT_THROW(
        BloomFilter(100, bitsPerKey, leafPool_.get()), NimbleUserError);
  }
}

TEST_F(BloomFilterTest, dataSizeMatchesBlocks) {
  BloomFilter filter(500, 10.0f, leafPool_.get());
  EXPECT_EQ(
      filter.dataSize(),
      static_cast<size_t>(filter.numBlocks()) * BloomFilter::kBlockSizeBytes);
}

TEST_F(BloomFilterTest, largeKeySet) {
  const int numKeys = 10'000;
  BloomFilter filter(numKeys, 10.0f, leafPool_.get());

  for (int i = 0; i < numKeys; ++i) {
    filter.insert("key_" + std::to_string(i));
  }

  // All inserted keys must be found (no false negatives).
  for (int i = 0; i < numKeys; ++i) {
    ASSERT_TRUE(filter.testKey("key_" + std::to_string(i)));
  }

  // Check false positive rate on non-inserted keys. The test keys are
  // deterministic for reproducibility — the hash function distributes them
  // uniformly enough that the measured FPR is representative without
  // randomization.
  int falsePositives = 0;
  const int numProbes = 10'000;
  for (int i = 0; i < numProbes; ++i) {
    if (filter.testKey("miss_" + std::to_string(i))) {
      ++falsePositives;
    }
  }
  const double fpr =
      static_cast<double>(falsePositives) / static_cast<double>(numProbes);
  // At 10 bits/key, theoretical FPR is ~0.8%. Allow up to 5%.
  EXPECT_LT(fpr, 0.05);
}

TEST_F(BloomFilterTest, serializationPreservesAllKeys) {
  const int numKeys = 500;
  BloomFilter filter(numKeys, 10.0f, leafPool_.get());

  for (int i = 0; i < numKeys; ++i) {
    filter.insert("ser_key_" + std::to_string(i));
  }

  BloomFilter restored(
      filter.numBlocks(), filter.data(), filter.dataSize(), leafPool_.get());

  // All keys must survive serialization.
  for (int i = 0; i < numKeys; ++i) {
    ASSERT_TRUE(restored.testKey("ser_key_" + std::to_string(i)));
  }
}

TEST_F(BloomFilterTest, dataSizeMismatch) {
  BloomFilter filter(100, 10.0f, leafPool_.get());

  // Wrong data size should throw.
  const size_t wrongSize = filter.dataSize() + 1;
  std::vector<uint8_t> wrongData(wrongSize, 0);
  EXPECT_THROW(
      BloomFilter(
          filter.numBlocks(), wrongData.data(), wrongSize, leafPool_.get()),
      nimble::NimbleInternalError);
}

TEST_F(BloomFilterTest, zeroNumBlocks) {
  // numBlocks=0 with dataSize=0 would cause division by zero in blockIndex().
  NIMBLE_ASSERT_THROW(
      BloomFilter(0, nullptr, 0, leafPool_.get()),
      "numBlocks must be positive");
}

TEST_F(BloomFilterTest, memoryTracking) {
  const auto memoryBefore = leafPool_->usedBytes();

  {
    BloomFilter filter(10'000, 10.0f, leafPool_.get());
    const auto memoryDuring = leafPool_->usedBytes();
    EXPECT_GT(memoryDuring, memoryBefore);
    EXPECT_GE(memoryDuring - memoryBefore, filter.dataSize());
  }

  // Memory should be released after filter is destroyed.
  EXPECT_EQ(leafPool_->usedBytes(), memoryBefore);
}

} // namespace
} // namespace facebook::nimble::index
