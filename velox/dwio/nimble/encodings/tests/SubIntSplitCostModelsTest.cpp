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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"

using namespace facebook::nimble;
using namespace facebook::nimble::detail::subintsplit;

namespace {

// 1000 values: 990 small values (bit_width <= 4, only 16 distinct) and 10
// "exception" values requiring the full 16-bit segment width. Models a
// column where the vast majority of values fit a narrow baseline and a rare
// few outliers require the full range. The 1% outlier rate keeps PFOR's
// exception side-channels cheap, while the low baseline cardinality (17
// distinct values total) would make Dictionary's per-value index overhead
// (rounded to a byte) more expensive than PFOR's 7-bit base region.
std::vector<uint64_t> makePforFriendlyValues() {
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 990; ++i) {
    values.push_back(static_cast<uint64_t>(i % 16)); // bit_width <= 4
  }
  for (int i = 0; i < 10; ++i) {
    values.push_back(65535); // bit_width == 16
  }
  return values;
}

// 4096 values laid out as 4 blocks of 1024. Each block is locally clustered
// to a narrow 4-bit range, but the blocks' baselines are spread across a
// globally wide ~20-bit range.
std::vector<uint64_t> makeBlockClusteredValues() {
  std::vector<uint64_t> values;
  values.reserve(4096);
  for (int block = 0; block < 4; ++block) {
    const uint64_t base = static_cast<uint64_t>(block) * 250000;
    for (int i = 0; i < 1024; ++i) {
      values.push_back(base + static_cast<uint64_t>(i % 16));
    }
  }
  return values;
}

} // namespace

TEST(SubIntSplitCostModelsTest, PforBeatsFixedBitWidthForBaselinePlusOutliers) {
  const std::vector<uint64_t> values = makePforFriendlyValues();
  const MetricCollector collector;
  const SegmentMetrics m = collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  const double pfor = pforCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth = fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_LT(pfor, fixedBitWidth);
}

TEST(
    SubIntSplitCostModelsTest,
    BlockBitPackingBeatsFixedBitWidthForLocallyClusteredData) {
  const std::vector<uint64_t> values = makeBlockClusteredValues();
  const MetricCollector collector;
  const SegmentMetrics m = collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 20; // bit_width(750015) == 20
  const double blockBitPacking = blockBitPackingCostBits(values, values.size());
  const double fixedBitWidth = fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_LT(blockBitPacking, fixedBitWidth);
}

TEST(SubIntSplitCostModelsTest, SimdForBitpackIsFiniteAndCheaperThanTrivial) {
  // Values span [0, 63] (bit_width <= 6), narrower than the 8-bit segment
  // width below, so SimdForBitpack's tight 6-bit packing beats Trivial's
  // full 8-bit-per-value storage despite group-padding/header overhead.
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<uint64_t>(i % 64)); // bit_width <= 6
  }
  const MetricCollector collector;
  const SegmentMetrics m = collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 8;
  const double simdForBitpack = simdForBitpackCostBits(m, values.size(), kBitWidth);
  const double trivial = trivialCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(simdForBitpack));
  EXPECT_GT(simdForBitpack, 0.0);
  EXPECT_LT(simdForBitpack, trivial);
}

TEST(SubIntSplitCostModelsTest, BestCostBitsSelectsPforForBaselinePlusOutliers) {
  const std::vector<uint64_t> values = makePforFriendlyValues();
  const MetricCollector collector;
  const SegmentMetrics m = collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::PFOR);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsBlockBitPackingForLocallyClusteredData) {
  const std::vector<uint64_t> values = makeBlockClusteredValues();
  const MetricCollector collector;
  const SegmentMetrics m = collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 20;
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::BlockBitPacking);
}

#endif
