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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"

using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

// Compares the two ways of reaching a segment's metrics, over every bit range
// the selector's inner loop visits, stepping the counter exactly as that loop
// does. The flag-checked counting path is the reference: it is what every
// measurement on this encoder was made against.
//
// Both halves of the split are under test here. The three-argument call takes
// the frequency metrics from the counter and the rest from the specialised
// scan; the two-argument call counts and scans in one flag-checked loop. Every
// field has to agree, so the comparison is over all of them rather than over
// the frequency metrics alone.
//
// This is the test that matters, because the way a counter maintained across
// the loop goes wrong is not arithmetic but lockstep. If the counter and the
// extractor ever describe different ranges, the metrics are correct for a range
// nobody asked about, the planner optimises the wrong thing, and every symptom
// is downstream of here. A drift of one column fails this immediately, rather
// than after a full encode and a byte comparison.
void expectPartitionMatchesCounting(
    const std::vector<uint64_t>& samples,
    int bits) {
  const auto flags = allCostModelRequiredFlags();
  MetricCollector counting;
  MetricCollector supplied;
  BitRangeExtractor extractor(samples);
  BitRangeCounter counter(samples);

  for (int l = 0; l < bits; ++l) {
    extractor.reset(l);
    counter.reset(l);
    for (int r = l; r < bits; ++r) {
      extractor.extend(r);
      const std::vector<uint64_t>& segValues = extractor.values();
      const SectionMetrics counted = counting.compute(segValues, flags);
      const RangeCounts rangeCounts = counter.counts(r);
      const SectionMetrics given =
          supplied.compute(segValues, flags, rangeCounts);
      // The frequencies alone, through the path that scans for everything
      // else, so that both of the collector's supplied-counts paths are held
      // to the counting path.
      const SectionMetrics givenFrequencies =
          supplied.compute(segValues, flags, rangeCounts.frequencies);

      const std::string where =
          "bits [" + std::to_string(l) + ", " + std::to_string(r) + "]";
      EXPECT_EQ(given.uniqueCount, counted.uniqueCount) << where;
      EXPECT_EQ(given.uniqueCountCapped, counted.uniqueCountCapped) << where;
      EXPECT_EQ(given.dominantCount, counted.dominantCount) << where;
      EXPECT_EQ(given.dominantCountCapped, counted.dominantCountCapped)
          << where;
      EXPECT_EQ(given.topKCoverage, counted.topKCoverage) << where;

      EXPECT_EQ(given.min, counted.min) << where;
      EXPECT_EQ(given.max, counted.max) << where;
      EXPECT_EQ(given.range, counted.range) << where;
      EXPECT_EQ(given.runCount, counted.runCount) << where;
      EXPECT_EQ(given.avgRunLength, counted.avgRunLength) << where;
      EXPECT_EQ(given.bitWidthBuckets, counted.bitWidthBuckets) << where;
      EXPECT_EQ(given.sumAbsDelta, counted.sumAbsDelta) << where;
      EXPECT_EQ(given.monotonicCount, counted.monotonicCount) << where;
      EXPECT_EQ(given.maxDelta, counted.maxDelta) << where;
      // The singleton and doubleton counts are the stream cardinality
      // estimate's only input, so both supplied paths must forward them. A path
      // that drops them does not fail loudly: the estimate silently falls back
      // to the sample's distinct count and Dictionary is priced on it.
      EXPECT_EQ(given.singletonCount, counted.singletonCount) << where;
      EXPECT_EQ(given.doubletonCount, counted.doubletonCount) << where;
      EXPECT_EQ(given.countedRows, counted.countedRows) << where;
      EXPECT_EQ(givenFrequencies.singletonCount, counted.singletonCount)
          << where;
      EXPECT_EQ(givenFrequencies.doubletonCount, counted.doubletonCount)
          << where;
      EXPECT_EQ(givenFrequencies.countedRows, counted.countedRows) << where;
    }
  }
}

// Repeats are what make the frequency metrics say anything: uniform samples
// turn almost every group into a singleton almost at once, which is the one
// case a counter finds easy. Three fields of different cardinality also give
// the grid ranges that straddle a field boundary, which is where a bit range
// holds part of one field and part of another.
TEST(SubIntSplitSelectorTest, PartitionCountsMatchCountingAtEveryRange) {
  constexpr int kBits = 24;
  constexpr size_t kCount = 500;
  std::mt19937_64 rng(2026);
  std::vector<uint64_t> samples(kCount);
  for (auto& sample : samples) {
    sample = (rng() % 40) | ((rng() % 7) << 8) | ((rng() % 3) << 18);
  }
  expectPartitionMatchesCounting(samples, kBits);
}

// Ranges wider than sixteen bits are the ones the counting path serves with a
// hash map rather than a direct histogram, so this puts the counter against
// that path in particular.
TEST(SubIntSplitSelectorTest, PartitionCountsMatchCountingOnWideRanges) {
  constexpr int kBits = 40;
  constexpr size_t kCount = 400;
  std::mt19937_64 rng(11);
  std::vector<uint64_t> samples(kCount);
  for (auto& sample : samples) {
    sample = rng() & ((uint64_t{1} << kBits) - 1);
  }
  expectPartitionMatchesCounting(samples, kBits);
}

// The full 64-bit grid the encoder walks, on an identifier-shaped sample: a
// slowly rising high field, a small middle field and random low bits, so that
// ranges reaching bit 63 and left edges that merge rather than sort are both
// covered. The sample count is not a multiple of 64, which puts the counter's
// sentinel inside a word rather than at its start.
TEST(SubIntSplitSelectorTest, RangeCountsMatchCountingOverSixtyFourBits) {
  constexpr size_t kCount = 1'000;
  std::mt19937_64 rng(64);
  std::vector<uint64_t> samples(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    samples[i] = ((uint64_t{1} << 62) + (i / 16) * (uint64_t{1} << 22)) |
        ((rng() % 5) << 12) | (rng() & 0xFFF);
  }
  samples[3] = ~uint64_t{0};
  samples[4] = 0;
  expectPartitionMatchesCounting(samples, 64);
}

// Once every group holds one sample, no wider range can split anything, so the
// counter stops working and hands back what it already had. That cache is
// held across cells rather than within one, which makes it the piece most
// likely to be subtly wrong: a cache that survived a reset would report the
// previous left edge's answer for the new one.
//
// The samples are 0..255, so bits 0..7 tell every sample apart and bits 8 and
// above tell none of them apart. Starting at bit 0 therefore reaches all
// singletons and stays there, and starting at bit 8 must report a single group
// of every sample -- which is exactly what a leaked cache could not do.
TEST(SubIntSplitSelectorTest, PartitionRecountsAfterAllGroupsAreSingletons) {
  constexpr size_t kCount = 256;
  std::vector<uint64_t> samples(kCount);
  std::iota(samples.begin(), samples.end(), uint64_t{0});

  BitRangeCounter counter(samples);
  counter.reset(0);
  for (int r = 0; r < 31; ++r) {
    counter.frequencies(r);
  }
  EXPECT_EQ(counter.frequencies(31).uniqueCount, kCount);
  EXPECT_EQ(counter.frequencies(31).dominantCount, 1u);

  // Not the next left edge, so the counter sorts afresh rather than merging.
  counter.reset(8);
  for (int r = 8; r < 31; ++r) {
    counter.frequencies(r);
  }
  EXPECT_EQ(counter.frequencies(31).uniqueCount, 1u);
  EXPECT_EQ(counter.frequencies(31).dominantCount, kCount);

  // And the whole grid on this shape, so the short-circuit is checked against
  // the counting path at every width rather than only at its ends.
  expectPartitionMatchesCounting(samples, 32);
}

// A left edge whose samples are all equal never splits at all, which is the
// opposite corner from all-singletons and reaches the same short-circuit from
// the other side.
TEST(SubIntSplitSelectorTest, PartitionHandlesASegmentOfOneValue) {
  const std::vector<uint64_t> samples(64, uint64_t{0xABCD});
  BitRangeCounter counter(samples);
  counter.reset(0);
  for (int r = 0; r < 23; ++r) {
    counter.frequencies(r);
  }
  EXPECT_EQ(counter.frequencies(23).uniqueCount, 1u);
  EXPECT_EQ(counter.frequencies(23).dominantCount, samples.size());
  expectPartitionMatchesCounting(samples, 24);
}

// The grid prices Dictionary on the stream's estimated cardinality, and the
// grid only ever reaches the metrics through the counter. A near-unique sample
// of a much longer stream must therefore extrapolate there exactly as it does
// through the counting path, rather than reporting the sample's own count.
TEST(SubIntSplitSelectorTest, RangeCountsExtrapolateStreamCardinality) {
  constexpr size_t kCount = 2'048;
  constexpr size_t kStreamRows = 524'288;
  constexpr int kBits = 40;
  std::mt19937_64 rng(5);
  std::vector<uint64_t> samples(kCount);
  for (auto& sample : samples) {
    sample = rng() & ((uint64_t{1} << kBits) - 1);
  }
  samples[1] = samples[0];

  const auto flags = allCostModelRequiredFlags();
  MetricCollector collector;
  const SectionMetrics counted = collector.compute(samples, flags);
  BitRangeCounter counter(samples);
  counter.reset(0);
  const SectionMetrics given =
      collector.compute(samples, flags, counter.counts(kBits - 1));

  const double countedEstimate =
      estimatedStreamUniqueCount(counted, kCount, kBits, kStreamRows);
  EXPECT_GT(countedEstimate, static_cast<double>(counted.uniqueCount));
  EXPECT_EQ(
      estimatedStreamUniqueCount(given, kCount, kBits, kStreamRows),
      countedEstimate);
}

// A column whose low bits repeat heavily and whose high bits run, which is the
// shape that gives the planner a real choice between an encoding that stores a
// section well and one that reads it well.
std::vector<uint64_t> decodeCostSamples() {
  constexpr size_t kCount = 4096;
  std::mt19937_64 rng(7);
  std::vector<uint64_t> samples(kCount);
  uint64_t high = 0;
  for (size_t i = 0; i < kCount; ++i) {
    if (i % 32 == 0) {
      ++high;
    }
    samples[i] = (rng() % 6) | ((rng() % 3) << 8) | (high << 20);
  }
  return samples;
}

// The default must not move a single boundary. This is the property the whole
// change rests on: the decode term is added at every grid cell, so if it were
// not exactly zero by default it would perturb every plan in the encoder.
TEST(SubIntSplitSelectorTest, DecodeWeightZeroReproducesTheSizeOnlyPlan) {
  const auto samples = decodeCostSamples();
  const auto baseline = selectSplits(samples, 32, samples.size());

  SelectorConfig cfg = defaultSelectorConfig();
  cfg.decodeWeighting = DecodeCostWeighting{
      .weight = 0.0, .accessPattern = DecodeAccessPattern::Bulk};
  const auto weighted = selectSplits(samples, 32, samples.size(), cfg);

  ASSERT_EQ(weighted.sections.size(), baseline.sections.size());
  for (size_t i = 0; i < baseline.sections.size(); ++i) {
    EXPECT_EQ(weighted.sections[i].bitStart, baseline.sections[i].bitStart);
    EXPECT_EQ(weighted.sections[i].bitEnd, baseline.sections[i].bitEnd);
    EXPECT_EQ(weighted.sections[i].encoding, baseline.sections[i].encoding);
    EXPECT_DOUBLE_EQ(weighted.sections[i].cost, baseline.sections[i].cost);
  }
  EXPECT_DOUBLE_EQ(weighted.totalCost, baseline.totalCost);
}

// The hybrid planner's shortlist must contain the plan the DP would have
// chosen, or turning it on could only ever lose that plan. Compared on cost
// rather than boundaries, since equal-cost plans may be ranked either way.
TEST(SubIntSplitSelectorTest, KBestSplitsStartsWithTheDpPlanAndRisesInCost) {
  const auto samples = decodeCostSamples();
  const auto cfg = defaultSelectorConfig();
  const AllowedEncodings all;
  const auto dp = selectSplitsRestricted(samples, 32, samples.size(), all, cfg);
  const auto grid = buildSectionCostGrid(
      samples, 32, samples.size(), restrictedSectionCostFn(all, cfg));
  const auto plans = kBestSplits(grid, 32, cfg, 4);

  ASSERT_EQ(plans.size(), 4);
  double previousCost = -std::numeric_limits<double>::infinity();
  for (size_t rank = 0; rank < plans.size(); ++rank) {
    SCOPED_TRACE(rank);
    double cost =
        cfg.splitPenalty * static_cast<double>(plans[rank].size() - 1);
    int nextBit = 0;
    for (const auto& segment : plans[rank]) {
      EXPECT_EQ(segment.bitStart, nextBit);
      nextBit = segment.bitEnd + 1;
      cost += segment.cost;
    }
    EXPECT_EQ(nextBit, 32);
    EXPECT_GE(cost, previousCost);
    previousCost = cost;
    if (rank == 0) {
      EXPECT_DOUBLE_EQ(cost, dp.totalCost);
    }
  }
}

TEST(SubIntSplitSelectorTest, KBestSplitsCutOnlyWhereAllowed) {
  const auto samples = decodeCostSamples();
  const auto cfg = defaultSelectorConfig();
  const AllowedEncodings all;
  const auto grid = buildSectionCostGrid(
      samples, 32, samples.size(), restrictedSectionCostFn(all, cfg));
  std::vector<bool> cuts(33, false);
  cuts[0] = true;
  cuts[16] = true;
  cuts[32] = true;

  // Only two segmentations exist: the whole range, and a cut at bit 16.
  const auto plans = kBestSplits(grid, 32, cfg, 8, cuts);
  EXPECT_EQ(plans.size(), 2);
  for (const auto& plan : plans) {
    for (const auto& segment : plan) {
      EXPECT_TRUE(cuts[segment.bitStart]);
      EXPECT_TRUE(cuts[segment.bitEnd + 1]);
    }
  }
}

// At weight zero a segment's weighted cost and its size are the same number,
// which is what lets a caller read a size-only plan's cost off either field.
//
// The two plan totals still differ, and by exactly the split penalties: the DP
// minimises size plus a charge per boundary, while totalSizeBits is what the
// sections actually store. Asserting they are equal would be asserting the
// penalty away, so the relationship is spelled out instead.
TEST(SubIntSplitSelectorTest, SizeAndWeightedCostAgreeAtWeightZero) {
  const auto samples = decodeCostSamples();
  const auto plan = selectSplits(samples, 32, samples.size());
  ASSERT_FALSE(plan.sections.empty());
  for (const auto& segment : plan.sections) {
    EXPECT_DOUBLE_EQ(segment.sizeCostBits, segment.cost);
  }
  const double penalties = defaultSelectorConfig().splitPenalty *
      static_cast<double>(plan.sections.size() - 1);
  EXPECT_DOUBLE_EQ(plan.totalCost, plan.totalSizeBits + penalties);
}

// A size-only plan still reports what it costs to read, so the trade-off can
// be seen before it is taken. A plan that decoded for free would mean the
// rates never reached the segments.
TEST(SubIntSplitSelectorTest, SizeOnlyPlanStillReportsItsDecodeCost) {
  const auto samples = decodeCostSamples();
  const auto plan = selectSplits(samples, 32, samples.size());
  ASSERT_FALSE(plan.sections.empty());
  EXPECT_GT(plan.totalDecodeNanosPerRow, 0.0);
  // Composition is additive over sections plus the per-section assembly term,
  // so the whole is never cheaper than its parts.
  double sum = 0.0;
  for (const auto& segment : plan.sections) {
    sum += segment.decodeNanosPerRow;
  }
  EXPECT_GE(plan.totalDecodeNanosPerRow, sum);
}

// Turning the weight up buys decode and spends size, in that direction. Which
// is the whole claim: the two axes move against each other and the weight is
// what picks a point on the curve.
TEST(SubIntSplitSelectorTest, DecodeWeightTradesSizeForDecode) {
  const auto samples = decodeCostSamples();
  const auto baseline = selectSplits(samples, 32, samples.size());

  SelectorConfig cfg = defaultSelectorConfig();
  cfg.decodeWeighting = DecodeCostWeighting{
      .weight = 2.0, .accessPattern = DecodeAccessPattern::Bulk};
  const auto weighted = selectSplits(samples, 32, samples.size(), cfg);

  EXPECT_LE(weighted.totalDecodeNanosPerRow, baseline.totalDecodeNanosPerRow);
  EXPECT_GE(weighted.totalSizeBits, baseline.totalSizeBits);
}

// The pattern is not decoration. FrequencyPartition is competitive on bulk and
// dreadful on a probe, so the same weight applied to the two patterns must be
// able to reach different plans.
TEST(SubIntSplitSelectorTest, PointAndBulkRankEncodingsDifferently) {
  EXPECT_GT(
      decodeRate(
          EncodingType::FrequencyPartition,
          DecodeAccessPattern::Point,
          DecodeReadPath::Cursor)
          .baseNanosPerRow,
      decodeRate(
          EncodingType::FixedBitWidth,
          DecodeAccessPattern::Point,
          DecodeReadPath::Cursor)
              .baseNanosPerRow *
          10.0);
  // On bulk the same pair is within a factor of two, which is why bulk alone
  // never declined it.
  EXPECT_LT(
      decodeRate(
          EncodingType::FrequencyPartition,
          DecodeAccessPattern::Bulk,
          DecodeReadPath::Cursor)
          .baseNanosPerRow,
      decodeRate(
          EncodingType::FixedBitWidth,
          DecodeAccessPattern::Bulk,
          DecodeReadPath::Cursor)
              .baseNanosPerRow *
          2.0);
}

// RLE's measured cost is its run count, and the model reaches that through the
// size estimate rather than a second pass: a section that stores more runs is
// priced slower per row even though nothing told the model what a run is.
TEST(SubIntSplitSelectorTest, RleDecodeCostRisesWithEncodedSize) {
  constexpr size_t kRows = 100'000;
  const double fewRuns = decodeNanosPerRow(
      EncodingType::RLE,
      DecodeAccessPattern::Bulk,
      DecodeReadPath::Cursor,
      8.0 * 2'000,
      kRows);
  const double manyRuns = decodeNanosPerRow(
      EncodingType::RLE,
      DecodeAccessPattern::Bulk,
      DecodeReadPath::Cursor,
      8.0 * 300'000,
      kRows);
  EXPECT_GT(manyRuns, fewRuns * 4.0);
  // And a fixed-width section, whose cost is per row rather than per run, is
  // nearly indifferent to the same change.
  const double smallFixed = decodeNanosPerRow(
      EncodingType::FixedBitWidth,
      DecodeAccessPattern::Bulk,
      DecodeReadPath::Cursor,
      8.0 * 2'000,
      kRows);
  const double largeFixed = decodeNanosPerRow(
      EncodingType::FixedBitWidth,
      DecodeAccessPattern::Bulk,
      DecodeReadPath::Cursor,
      8.0 * 300'000,
      kRows);
  EXPECT_LT(largeFixed, smallFixed * 2.0);
}

// Sections add. Measured section times sum to the plan's total rather than
// maxing, which is why a slow section is paid in full, and it is also what
// makes the term admissible in a DP over prefixes of the bit range.
TEST(SubIntSplitSelectorTest, SectionDecodeCostsAddRatherThanMax) {
  const std::vector<double> sections{1.0, 2.0, 4.0};
  const double bulk = combineSectionDecodeNanos(
      DecodeAccessPattern::Bulk,
      DecodeReadPath::Cursor,
      std::span<const double>(sections));
  EXPECT_DOUBLE_EQ(bulk, 7.0 + 3.0 * kAssemblyNanosPerRowPerSection);
  EXPECT_GT(bulk, 4.0);

  const double point = combineSectionDecodeNanos(
      DecodeAccessPattern::Point,
      DecodeReadPath::Cursor,
      std::span<const double>(sections));
  EXPECT_DOUBLE_EQ(point, 7.0 + 3.0 * kProbeNanosPerSection);

  // The view path assembles at its own measured per-section rate.
  EXPECT_DOUBLE_EQ(
      combineSectionDecodeNanos(
          DecodeAccessPattern::Bulk,
          DecodeReadPath::View,
          std::span<const double>(sections)),
      7.0 + 3.0 * kViewAssemblyNanosPerRowPerSection);
}

// A view pays for a section differently from a cursor. Sections whose encoding
// has no usable view are decoded whole when the view is opened, and
// MainlyConstant's real view is slower than its cursor, so both are priced
// dearer on the view path; a section read the same way on both paths is not.
TEST(SubIntSplitSelectorTest, ViewPathPricesSectionsForTheViewItBuilds) {
  const auto bulkRate = [](EncodingType type, DecodeReadPath path) {
    return decodeRate(type, DecodeAccessPattern::Bulk, path).baseNanosPerRow;
  };
  // A materialized fallback costs almost nothing per read once opened and
  // nearly everything when opened, so which of the two a reader pays decides
  // whether the section is cheap or dear.
  EXPECT_LT(
      bulkRate(EncodingType::FrequencyPartition, DecodeReadPath::View), 1.0);
  EXPECT_GT(
      bulkRate(EncodingType::FrequencyPartition, DecodeReadPath::ViewWithOpen),
      20.0);
  EXPECT_GT(
      bulkRate(EncodingType::FOR, DecodeReadPath::ViewWithOpen),
      bulkRate(EncodingType::FOR, DecodeReadPath::Cursor) * 1.5);
  EXPECT_GT(
      bulkRate(EncodingType::MainlyConstant, DecodeReadPath::View),
      bulkRate(EncodingType::MainlyConstant, DecodeReadPath::Cursor) * 1.5);
  EXPECT_LT(
      bulkRate(EncodingType::FixedBitWidth, DecodeReadPath::ViewWithOpen),
      bulkRate(EncodingType::FixedBitWidth, DecodeReadPath::Cursor) * 1.5);

  // The cursor pays FrequencyPartition's construction too, but only when
  // opening is charged; amortised pricing is unchanged.
  EXPECT_GT(
      bulkRate(
          EncodingType::FrequencyPartition, DecodeReadPath::CursorWithOpen),
      bulkRate(EncodingType::FrequencyPartition, DecodeReadPath::Cursor) +
          20.0);
  // A probe's cost does not carry the open.
  EXPECT_DOUBLE_EQ(
      decodeRate(
          EncodingType::FrequencyPartition,
          DecodeAccessPattern::Point,
          DecodeReadPath::CursorWithOpen)
          .baseNanosPerRow,
      decodeRate(
          EncodingType::FrequencyPartition,
          DecodeAccessPattern::Point,
          DecodeReadPath::Cursor)
          .baseNanosPerRow);

  // View point rates are absolute and carry the per-section probe overhead,
  // so no measured section probes for less than that overhead.
  EXPECT_GE(
      decodeRate(
          EncodingType::FixedBitWidth,
          DecodeAccessPattern::Point,
          DecodeReadPath::View)
          .baseNanosPerRow,
      31.0);
}

// The view path must be as inert at weight zero as the cursor path.
TEST(SubIntSplitSelectorTest, DecodeWeightZeroOnTheViewPathReproducesSizeOnly) {
  const auto samples = decodeCostSamples();
  const auto baseline = selectSplits(samples, 32, samples.size());

  SelectorConfig cfg = defaultSelectorConfig();
  cfg.decodeWeighting = DecodeCostWeighting{
      .weight = 0.0,
      .accessPattern = DecodeAccessPattern::Bulk,
      .readPath = DecodeReadPath::View};
  const auto view = selectSplits(samples, 32, samples.size(), cfg);

  ASSERT_EQ(view.sections.size(), baseline.sections.size());
  for (size_t i = 0; i < baseline.sections.size(); ++i) {
    EXPECT_EQ(view.sections[i].bitStart, baseline.sections[i].bitStart);
    EXPECT_EQ(view.sections[i].bitEnd, baseline.sections[i].bitEnd);
    EXPECT_EQ(view.sections[i].encoding, baseline.sections[i].encoding);
  }
  EXPECT_DOUBLE_EQ(view.totalCost, baseline.totalCost);
}

// An unrepresentable candidate must stay unrepresentable. Multiplying an
// infinite size by a zero weight yields a NaN, and a NaN loses every
// comparison silently, so the guard is checked rather than assumed.
TEST(SubIntSplitSelectorTest, InfiniteSizeCandidatesNeverWin) {
  EXPECT_DOUBLE_EQ(
      decodeNanosPerRow(
          EncodingType::Constant,
          DecodeAccessPattern::Bulk,
          DecodeReadPath::Cursor,
          std::numeric_limits<double>::infinity(),
          1'000),
      0.0);
  const auto samples = decodeCostSamples();
  SelectorConfig cfg = defaultSelectorConfig();
  cfg.decodeWeighting = DecodeCostWeighting{
      .weight = 0.5, .accessPattern = DecodeAccessPattern::Point};
  const auto plan = selectSplits(samples, 32, samples.size(), cfg);
  ASSERT_FALSE(plan.sections.empty());
  EXPECT_TRUE(std::isfinite(plan.totalCost));
  EXPECT_TRUE(std::isfinite(plan.totalDecodeNanosPerRow));
}

} // namespace

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
