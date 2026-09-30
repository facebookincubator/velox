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
#include "velox/dwio/nimble/common/RadixSort.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace facebook::nimble;

namespace {

// The permutation the sort has to reproduce exactly, since callers encode it
// rather than store it.
std::vector<uint32_t> stableSortedOrder(const std::vector<uint64_t>& keys) {
  std::vector<uint32_t> order(keys.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&keys](uint32_t a, uint32_t b) {
    return keys[a] < keys[b];
  });
  return order;
}

std::vector<uint32_t> radixSortedOrder(const std::vector<uint64_t>& keys) {
  std::vector<uint32_t> order(keys.size());
  std::iota(order.begin(), order.end(), 0u);
  RadixSort<uint32_t> sorter;
  sorter.sortStable(
      std::span<uint32_t>(order),
      [&keys](uint32_t row) { return keys[row]; },
      significantBits(std::span<const uint64_t>(keys)));
  return order;
}

TEST(RadixSortTest, MatchesStableSortAcrossKeyWidths) {
  // Widths chosen around the digit boundaries, and counts swept alongside
  // them, because both decide how many passes run: the digit narrows on short
  // inputs, so the same key width takes a different number of passes at
  // different lengths. An odd pass count is the case that has to copy, and it
  // is the one a ping-pong buffer gets wrong, so the sweep has to reach both
  // parities at every width.
  std::mt19937_64 rng{20260907};
  for (const int keyBits : {1, 2, 8, 9, 10, 16, 17, 19, 24, 32, 33, 48, 64}) {
    for (const size_t count :
         {size_t{2}, size_t{17}, size_t{300}, size_t{5000}}) {
      std::vector<uint64_t> keys(count);
      for (auto& key : keys) {
        key = keyBits == 64 ? rng() : (rng() & ((uint64_t{1} << keyBits) - 1));
      }
      EXPECT_EQ(radixSortedOrder(keys), stableSortedOrder(keys))
          << "keyBits " << keyBits << " count " << count;
    }
  }
}

TEST(RadixSortTest, MatchesStableSortWhenKeysRepeatHeavily) {
  // Ties are what stability is about, so a distribution that is nearly all
  // ties exercises it far harder than uniform keys do. It is also the shape
  // the key-derived transform actually sorts, since it only pays where the key
  // has far fewer values than there are rows.
  std::mt19937_64 rng{7};
  for (const uint64_t distinct : {uint64_t{1}, uint64_t{2}, uint64_t{7}}) {
    std::vector<uint64_t> keys(4096);
    for (auto& key : keys) {
      key = rng() % distinct;
    }
    EXPECT_EQ(radixSortedOrder(keys), stableSortedOrder(keys))
        << "distinct " << distinct;
  }
}

TEST(RadixSortTest, LeavesTooShortOrTooNarrowInputUntouched) {
  std::vector<uint32_t> single{9};
  RadixSort<uint32_t> sorter;
  sorter.sortStable(
      std::span<uint32_t>(single), [](uint32_t v) { return v; }, 8);
  EXPECT_THAT(single, testing::ElementsAre(9u));

  // Every key equal means every key is zero bits wide, and a stable sort of
  // equal keys is the identity.
  std::vector<uint32_t> order{0, 1, 2, 3};
  sorter.sortStable(
      std::span<uint32_t>(order), [](uint32_t) { return uint32_t{0}; }, 0);
  EXPECT_THAT(order, testing::ElementsAre(0u, 1u, 2u, 3u));
}

TEST(RadixSortTest, SortsItemsByAnExtractedKey) {
  // The shape a caller sorting records rather than indices needs, which is how
  // the decode path's own radix sort is written.
  struct Row {
    uint32_t source;
    uint32_t row;
  };
  std::vector<Row> rows{{5, 0}, {1, 1}, {5, 2}, {0, 3}, {1, 4}};
  RadixSort<Row> sorter;
  sorter.sortStable(
      std::span<Row>(rows), [](const Row& r) { return r.source; }, 3);

  std::vector<uint32_t> sources;
  std::vector<uint32_t> tiedRows;
  for (const auto& row : rows) {
    sources.push_back(row.source);
    tiedRows.push_back(row.row);
  }
  EXPECT_THAT(sources, testing::ElementsAre(0u, 1u, 1u, 5u, 5u));
  // Ties keep the order they arrived in, which is what makes the sort a
  // substitute for a stable comparison sort.
  EXPECT_THAT(tiedRows, testing::ElementsAre(3u, 1u, 4u, 0u, 2u));
}

TEST(RadixSortTest, SignificantBitsBoundsEveryKey) {
  const std::vector<uint64_t> empty;
  EXPECT_EQ(significantBits(std::span<const uint64_t>(empty)), 0);

  const std::vector<uint64_t> zeros(4, 0);
  EXPECT_EQ(significantBits(std::span<const uint64_t>(zeros)), 0);

  const std::vector<uint64_t> mixed{1, 0, 512, 3};
  EXPECT_EQ(significantBits(std::span<const uint64_t>(mixed)), 10);

  const std::vector<uint64_t> top{uint64_t{1} << 63};
  EXPECT_EQ(significantBits(std::span<const uint64_t>(top)), 64);
}

// A reused sorter must not carry a previous call's bucket offsets or scratch
// size into the next, which is the failure a per-call instance would hide.
TEST(RadixSortTest, ReusedSorterRepeatsItself) {
  std::mt19937_64 rng{99};
  RadixSort<uint32_t> sorter;
  std::vector<std::vector<uint64_t>> inputs;
  for (const size_t count : {size_t{1000}, size_t{4}, size_t{2500}}) {
    std::vector<uint64_t> keys(count);
    for (auto& key : keys) {
      key = rng() & 0x3FFFF;
    }
    inputs.push_back(std::move(keys));
  }
  for (const auto& keys : inputs) {
    std::vector<uint32_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0u);
    sorter.sortStable(
        std::span<uint32_t>(order),
        [&keys](uint32_t row) { return keys[row]; },
        significantBits(std::span<const uint64_t>(keys)));
    EXPECT_EQ(order, stableSortedOrder(keys));
  }
}

} // namespace
