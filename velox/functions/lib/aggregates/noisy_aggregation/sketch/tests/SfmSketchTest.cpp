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

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/SfmSketch.h"
#include "gtest/gtest.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {
using facebook::velox::functions::aggregate::SfmSketch;

class SfmSketchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  uint32_t numberOfBuckets_ = 4096;
  uint32_t precision_ = 24;
};

class TestingSeededRandomizationStrategy : public RandomizationStrategy {
 public:
  explicit TestingSeededRandomizationStrategy(int64_t seed)
      : rng_(std::mt19937_64()) {
    rng_.seed(seed);
  }

  bool nextBoolean(double probability) override {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(rng_) < probability;
  }

 private:
  std::mt19937_64 rng_;
};

TEST_F(SfmSketchTest, computeIndexTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    uint32_t index = 5;
    auto hash = static_cast<uint64_t>(index) << (64 - length);
    ASSERT_EQ(SfmSketch::computeIndex(hash, length), index);
  }
}

TEST_F(SfmSketchTest, numOfZerosTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    for (uint32_t zeros = 0; zeros < 63; zeros++) {
      uint64_t hash = 1ULL << zeros;
      ASSERT_EQ(
          SfmSketch::numberOfTrailingZeros(hash, length),
          std::min(zeros, 64 - length));
    }
  }
}

TEST_F(SfmSketchTest, numOfbucketsTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};

  for (auto& length : indexBitLength) {
    ASSERT_EQ(SfmSketch::numberOfBuckets(length), 1U << length);
  }
}

TEST_F(SfmSketchTest, bitMapSizeTest) {
  std::vector<uint32_t> numOfBuckets = {32, 64, 512, 1024, 4096, 32768};
  std::vector<uint32_t> precisions = {1, 2, 3, 8, 24, 32};

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  for (auto& numOfBucket : numOfBuckets) {
    for (auto& precision : precisions) {
      auto sketch = SfmSketch(&allocator);
      sketch.setSketchSize(numOfBucket, precision);
      ASSERT_EQ(sketch.getNumberOfBits(), numOfBucket * precision);
    }
  }
}

} // namespace facebook::velox::functions::aggregate
