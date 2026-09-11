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

// 1000 values: v[i] = i, a strictly monotonically increasing sequence with
// constant step size 1 (bit_width(999) == 10). FOR's small per-frame
// (128-value) local range (~64, 7 bits) beats PFOR/SimdForBitpack/
// BlockBitPacking/FixedBitWidth's full 10-bit packing.
std::vector<uint64_t> makeForFriendlyValues() {
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<uint64_t>(i));
  }
  return values;
}

// 1000 values: v[i] = i * 200'000, a strictly monotonically increasing
// sequence with constant step size 200'000 but a wide overall range
// (~199'800'000, bit_width == 28). Delta's exact-bit deltaBitWidth ==
// bit_width(200'000) == 18 beats FixedBitWidth/PFOR/SimdForBitpack/
// BlockBitPacking's full 28-bit packing, but DeltaBlock still wins overall:
// its per-block residuals collapse to a single repeated value (constant
// step), needing ~0 bits per block regardless of the step's magnitude, while
// Delta's flat per-value cost keeps scaling with deltaBitWidth.
std::vector<uint64_t> makeDeltaFriendlyValues() {
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<uint64_t>(i) * 200'000);
  }
  return values;
}

// 1000 values alternating between 0 and 1000. Half of all consecutive steps
// are decreases, well below Delta's 90% monotonic-non-decreasing threshold.
std::vector<uint64_t> makeAlternatingValues() {
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    values.push_back((i % 2 == 0) ? uint64_t{0} : uint64_t{1000});
  }
  return values;
}

// 1024 values with a Zipfian frequency distribution, interleaved to avoid
// monotonic ordering (keeps Delta cost at infinity). Layout:
//   value 0: 512 occurrences (dominant — wins top-2 tier, topKCoverage[1] high)
//   value 1: 256 occurrences
//   value 2: 128 occurrences
//   value 3: 64 occurrences
//   values 4..67: 1 occurrence each (sparse fallback)
// Interleaving (0,j) pairs ensures monotonicCount / (n-1) ≈ 0.5 < 0.9, so
// Delta returns infinity and FrequencyPartition can win.
std::vector<uint64_t> makeZipfianValues() {
  std::vector<uint64_t> values;
  values.reserve(1024);
  auto push = [&](uint64_t a, uint64_t b, int count) {
    for (int i = 0; i < count; ++i) {
      values.push_back(a);
      values.push_back(b);
    }
  };
  push(0, 1, 256); // 512 values: 256× each of 0 and 1
  push(0, 2, 128); // 256 values: 128× each of 0 and 2
  push(0, 3, 64); // 128 values: 64× each of 0 and 3
  for (uint64_t j = 4; j < 36; ++j) {
    push(0, j, 1); // 64 values: 1× each of 0 and j
  }
  for (uint64_t j = 36; j < 68; ++j) {
    push(0, j, 1); // 64 values: 1× each of 0 and j
  }
  // Total: 512 + 256 + 128 + 64 + 64 = 1024
  return values;
}

// 1000 values: 900 zeros (90% dominant) interleaved with 100 distinct values
// in [128, 227]. Pattern: 9 zeros then one non-zero, repeated 100 times. The
// 90% dominance makes MainlyConstant's exception-list representation (~2760
// bits) cheaper than RLE (~5008 bits, 200 runs), FrequencyPartition (~4139
// bits, 101 uniques), and all others. Non-monotonic enough to keep Delta from
// also being cheap: monotonicFraction ≈ 900/999 ≈ 0.9009 passes Delta's 0.9
// threshold but the large average delta (≈35.5) drives Delta's cost to ~10073
// bits — well above MainlyConstant.
std::vector<uint64_t> makeMainlyConstantValues() {
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 100; ++i) {
    for (int z = 0; z < 9; ++z) {
      values.push_back(0);
    }
    values.push_back(static_cast<uint64_t>(128 + i)); // 128..227
  }
  return values;
}

} // namespace

TEST(SubIntSplitCostModelsTest, PforBeatsFixedBitWidthForBaselinePlusOutliers) {
  const std::vector<uint64_t> values = makePforFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  const double pfor = pforCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_LT(pfor, fixedBitWidth);
}

TEST(
    SubIntSplitCostModelsTest,
    HuffmanIsFiniteAndCheaperThanPforForLowCardinalityBaselinePlusOutliers) {
  // Same data as makePforFriendlyValues: 17 distinct values total (16
  // baseline + 1 outlier). Huffman's exact per-symbol code length beats
  // PFOR's fixed-base-bit-width approximation for this small an alphabet.
  const std::vector<uint64_t> values = makePforFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  const double huffman = huffmanCostBits(values, values.size());
  const double pfor = pforCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(huffman));
  EXPECT_GT(huffman, 0.0);
  EXPECT_LT(huffman, pfor);
}

TEST(SubIntSplitCostModelsTest, HuffmanCostBitsInfiniteWhenAlphabetTooLarge) {
  // Every value distinct and count > HuffmanEncoding<uint64_t>::kMaxSymbols
  // (4096): HuffmanEncoding::estimateSize returns nullopt past this cap.
  std::vector<uint64_t> values;
  values.reserve(HuffmanEncoding<uint64_t>::kMaxSymbols + 1);
  for (uint32_t i = 0; i <= HuffmanEncoding<uint64_t>::kMaxSymbols; ++i) {
    values.push_back(static_cast<uint64_t>(i));
  }

  const double huffman = huffmanCostBits(values, values.size());

  EXPECT_TRUE(std::isinf(huffman));
}

TEST(
    SubIntSplitCostModelsTest,
    DeltaBlockIsFiniteAndCheaperThanDeltaForConstantStepData) {
  const std::vector<uint64_t> values = makeDeltaFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 28; // bit_width(199800000) == 28
  const double deltaBlock = deltaBlockCostBits(values, values.size());
  const double delta = deltaCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(deltaBlock));
  EXPECT_GT(deltaBlock, 0.0);
  EXPECT_LT(deltaBlock, delta);
}

TEST(
    SubIntSplitCostModelsTest,
    DeltaBlockCostBitsInfiniteWhenBlockContainsADecrease) {
  // A single decrease within a deltaBlockSize (default 256) window makes the
  // whole block ineligible for DeltaBlock's non-decreasing-only design.
  std::vector<uint64_t> values;
  values.reserve(300);
  for (uint64_t i = 0; i < 300; ++i) {
    values.push_back(i);
  }
  values[10] = 0; // introduces a decrease within the first 256-value block.

  const double deltaBlock = deltaBlockCostBits(values, values.size());

  EXPECT_TRUE(std::isinf(deltaBlock));
}

TEST(
    SubIntSplitCostModelsTest,
    BlockBitPackingBeatsFixedBitWidthForLocallyClusteredData) {
  const std::vector<uint64_t> values = makeBlockClusteredValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 20; // bit_width(750015) == 20
  const double blockBitPacking = blockBitPackingCostBits(values, values.size());
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

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
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 8;
  const double simdForBitpack =
      simdForBitpackCostBits(m, values.size(), kBitWidth);
  const double trivial = trivialCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(simdForBitpack));
  EXPECT_GT(simdForBitpack, 0.0);
  EXPECT_LT(simdForBitpack, trivial);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsHuffmanForLowCardinalityBaselinePlusOutliers) {
  // Same 17-distinct-value data PFOR was originally tuned for: now that
  // Huffman is a candidate, its exact per-symbol code length beats PFOR's
  // fixed-base-bit-width approximation for this small an alphabet.
  const std::vector<uint64_t> values = makePforFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::Huffman);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsDoesNotUseHuffmanForHighCardinalityBaselinePlusOutliers) {
  // Widen the baseline from 16 to 200 distinct values (bit_width <= 8),
  // still well under HuffmanEncoding's kMaxSymbols cap, with a near-uniform
  // frequency distribution across a wider alphabet -- Huffman's per-symbol
  // savings shrink relative to Dictionary, which packs each of the 210
  // unique values (200 baseline + 10 distinct-looking outliers folded into
  // the same 16-bit alphabet) into an 8-bit index, cheaper than the
  // segment's native 16-bit width (driven up by the rare 65535 outliers).
  // Values are multiplicatively permuted (rather than a plain i % 200
  // sawtooth) to avoid accidentally producing long monotonic runs that would
  // let Delta win instead.
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 990; ++i) {
    values.push_back(static_cast<uint64_t>((i * 37) % 200));
  }
  for (int i = 0; i < 10; ++i) {
    values.push_back(65535);
  }
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 16;
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::Dictionary);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsBlockBitPackingForLocallyClusteredData) {
  const std::vector<uint64_t> values = makeBlockClusteredValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 20;
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::BlockBitPacking);
}

TEST(SubIntSplitCostModelsTest, DeltaCostBitsFiniteForMonotonicData) {
  const std::vector<uint64_t> values = makeDeltaFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 28; // bit_width(199800000) == 28
  const double delta = deltaCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(delta));
  EXPECT_GT(delta, 0.0);
  EXPECT_LT(delta, fixedBitWidth);
}

TEST(SubIntSplitCostModelsTest, DeltaCostBitsInfiniteForNonMonotonicData) {
  const std::vector<uint64_t> values = makeAlternatingValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 10; // bit_width(1000) == 10
  const double delta = deltaCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isinf(delta));
}

TEST(SubIntSplitCostModelsTest, ForCostBitsFiniteAndPositive) {
  const std::vector<uint64_t> values = makeForFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 10; // bit_width(999) == 10
  const double forCost = forCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(forCost));
  EXPECT_GT(forCost, 0.0);
  EXPECT_LT(forCost, fixedBitWidth);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsDeltaBlockForWideRangeConstantStepData) {
  // Strictly increasing, no decreases anywhere in the stream, so every
  // deltaBlockSize block is eligible for DeltaBlock. Its per-block residuals
  // collapse to a single repeated value (constant step) and cost ~0 bits per
  // block regardless of the step's magnitude, while Delta's flat per-value
  // cost keeps scaling with the (now exact, not byte-rounded) deltaBitWidth.
  const std::vector<uint64_t> values = makeDeltaFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 28; // bit_width(199800000) == 28
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::DeltaBlock);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsDeltaBlockForUnitStepMonotonicData) {
  // Strictly increasing with step 1: DeltaBlock's per-block bit-packing of
  // (mostly) zero-width deltas beats FOR's per-frame local-range estimate.
  const std::vector<uint64_t> values = makeForFriendlyValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 10; // bit_width(999) == 10
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::DeltaBlock);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsDeltaForMonotonicDataWithADecrease) {
  // Same step-1 growth as makeForFriendlyValues, but with a single decrease
  // inside the first deltaBlockSize (256) block, which disqualifies that
  // block -- and therefore the whole segment -- from DeltaBlock. Delta pays
  // one restatement plus a single wider (11-bit) delta at the point the
  // sequence resumes; with deltaBitWidth sized exactly (not byte-rounded)
  // and the mostly-false isRestatements stream delegated to
  // SparseBoolEncoding's estimator, that's now cheaper than FOR's per-frame
  // local-range packing.
  std::vector<uint64_t> values;
  values.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    values.push_back(static_cast<uint64_t>(i));
  }
  values[10] = 0;
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 10; // bit_width(999) == 10
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::Delta);
}

TEST(
    SubIntSplitCostModelsTest,
    FrequencyPartitionCostBitsFiniteForZipfianData) {
  const std::vector<uint64_t> values = makeZipfianValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  // Values 0..67 require 7 bits; FPE should be finite and beat fixed-bit-width
  // thanks to the skewed distribution (value 0 covers 50% of entries).
  constexpr int kBitWidth = 7; // bit_width(67) == 7
  const double fpeCost =
      frequencyPartitionCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(fpeCost));
  EXPECT_GT(fpeCost, 0.0);
  EXPECT_LT(fpeCost, fixedBitWidth);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsFrequencyPartitionForZipfianData) {
  const std::vector<uint64_t> values = makeZipfianValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 7; // bit_width(67) == 7
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::FrequencyPartition);
}

TEST(SubIntSplitCostModelsTest, MainlyConstantCostBitsFiniteForDominantData) {
  const std::vector<uint64_t> values = makeMainlyConstantValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  // bit_width(227) == 8; MainlyConstant should be finite and beat FixedBitWidth
  // (~8072 bits) at this dominance level (900/1000 = 90% dominant value).
  constexpr int kBitWidth = 8;
  const double mc = mainlyConstantCostBits(m, values.size(), kBitWidth);
  const double fixedBitWidth =
      fixedBitWidthCostBits(m, values.size(), kBitWidth);

  EXPECT_TRUE(std::isfinite(mc));
  EXPECT_GT(mc, 0.0);
  EXPECT_LT(mc, fixedBitWidth);
}

TEST(
    SubIntSplitCostModelsTest,
    BestCostBitsSelectsMainlyConstantForDominantData) {
  const std::vector<uint64_t> values = makeMainlyConstantValues();
  MetricCollector collector;
  const SegmentMetrics m =
      collector.compute(values, allCostModelRequiredFlags());

  constexpr int kBitWidth = 8; // bit_width(227) == 8
  EncodingType bestEncoding = EncodingType::Trivial;
  const double best =
      bestCostBits(m, values.size(), kBitWidth, values, bestEncoding);

  EXPECT_TRUE(std::isfinite(best));
  EXPECT_EQ(bestEncoding, EncodingType::MainlyConstant);
}

#endif
