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

#include "velox/common/base/SplitBlockBloomFilter.h"
#include "velox/common/testutil/RandomSeed.h"

#include <folly/container/F14Set.h>
#include <gtest/gtest.h>

#include <random>

namespace facebook::velox::test {
namespace {

template <typename Hasher>
SplitBlockBloomFilter makeFilter(
    const folly::F14FastSet<int64_t>& values,
    const Hasher& hasher,
    std::vector<SplitBlockBloomFilter::Block>& blocks) {
  blocks.resize(SplitBlockBloomFilter::numBlocks(values.size(), 0.01));
  bzero(blocks.data(), blocks.size() * sizeof(SplitBlockBloomFilter::Block));
  SplitBlockBloomFilter filter(blocks);
  for (auto& value : values) {
    filter.insert(hasher(value));
  }
  return filter;
}

TEST(SplitBlockBloomFilterTest, numBlocks) {
  ASSERT_EQ(
      SplitBlockBloomFilter::numBlocks(50'000'000, 0.01) *
          sizeof(SplitBlockBloomFilter::Block),
      xsimd::batch<uint32_t>::size == 8 ? 60509568 : 65766912);
  ASSERT_EQ(
      SplitBlockBloomFilter::numBlocks(45'523'964, 0.1) *
          sizeof(SplitBlockBloomFilter::Block),
      xsimd::batch<uint32_t>::size == 8 ? 32848640 : 27546352);
}

TEST(SplitBlockBloomFilterTest, contiguous) {
  constexpr int kSize = 100'000;
  std::default_random_engine gen(common::testutil::getRandomSeed(42));
  std::uniform_int_distribution<> dist(0, 9);
  folly::F14FastSet<int64_t> values;
  values.reserve(kSize / 10);
  for (int i = 0; i < kSize; ++i) {
    if (dist(gen) == 0) {
      values.insert(i);
    }
  }
  std::vector<SplitBlockBloomFilter::Block> blocks;
  auto test = [&](auto hasher) {
    auto filter = makeFilter(values, hasher, blocks);
    int numFalsePositive = 0;
    for (int i = 0; i < kSize; ++i) {
      if (values.contains(i)) {
        ASSERT_TRUE(filter.mayContain(hasher(i)));
      } else {
        numFalsePositive += filter.mayContain(hasher(i));
      }
    }
    ASSERT_LT(1.0 * numFalsePositive / kSize, 0.03);
  };
  {
    SCOPED_TRACE("Folly");
    test(folly::hasher<int64_t>());
  }
  {
    SCOPED_TRACE("Multiplication");
    test([](auto x) { return x * 0xc6a4a7935bd1e995L; });
  }
}

TEST(SplitBlockBloomFilterTest, random) {
  constexpr int kSize = 100'000;
  std::default_random_engine gen(common::testutil::getRandomSeed(42));
  std::uniform_int_distribution<int64_t> dist;
  folly::F14FastSet<int64_t> values;
  values.reserve(kSize);
  for (int i = 0; i < kSize; ++i) {
    values.insert(dist(gen));
  }
  std::vector<SplitBlockBloomFilter::Block> blocks;
  auto test = [&](auto hasher) {
    auto filter = makeFilter(values, hasher, blocks);
    for (auto value : values) {
      ASSERT_TRUE(filter.mayContain(hasher(value)));
    }
    int numFalsePositive = 0;
    for (int i = 0; i < kSize; ++i) {
      auto value = dist(gen);
      if (!values.contains(value)) {
        numFalsePositive += filter.mayContain(hasher(value));
      }
    }
    ASSERT_LT(1.0 * numFalsePositive / kSize, 0.03);
  };
  {
    SCOPED_TRACE("Folly");
    test(folly::hasher<int64_t>());
  }
  {
    SCOPED_TRACE("Multiplication");
    test([](auto x) { return x * 0xc6a4a7935bd1e995L; });
  }
}

} // namespace
} // namespace facebook::velox::test
