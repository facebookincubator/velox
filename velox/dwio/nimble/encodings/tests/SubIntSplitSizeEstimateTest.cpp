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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

// The default read factors with SubIntSplit among them. The defaults need not
// list it, and these tests are about how selection treats it once it is a
// candidate.
std::vector<std::pair<EncodingType, float>> readFactorsWithSubIntSplit() {
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  if (std::none_of(
          readFactors.begin(), readFactors.end(), [](const auto& factor) {
            return factor.first == EncodingType::SubIntSplit;
          })) {
    readFactors.emplace_back(EncodingType::SubIntSplit, 0.85f);
  }
  return readFactors;
}

// Fixed so every synthetic stream is reproducible across runs.
constexpr uint64_t kSeed = 20260916;

// 524'288 rows is what the ID-column sweeps measure at, but the estimate is a
// property of the stream's shape rather than its length, and the sample it
// plans over is capped well below either. 64k keeps the test's encodes short.
constexpr size_t kNumRows = 65'536;

// A snowflake-shaped composite key: a slowly advancing timestamp in the high
// bits, a machine id that changes every few hundred rows, and a random
// sequence number in the low bits. The case a split exists for.
std::vector<uint64_t> makeCompositeKeyStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  constexpr uint64_t kMachineIds[] = {1, 3, 0, 2};
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = ((uint64_t{1'700'000'000} + i / 64) << 22) |
        (kMachineIds[(i / 500) % 4] << 12) | (rng() & 0xFFF);
  }
  return values;
}

// Uniform random over the whole 64-bit width: nothing for a split to find, and
// the stream the old range rule was the only thing keeping a split away from.
std::vector<uint64_t> makeUniformRandomStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows);
  for (auto& value : values) {
    value = rng();
  }
  return values;
}

// One value in a hundred differs from a constant, and the rest repeat it:
// MainlyConstant territory, where a split has nothing to add.
std::vector<uint64_t> makeConstantHeavyStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows, uint64_t{0x0123'4567'89AB'CDEF});
  for (size_t i = 0; i < numRows; i += 100) {
    values[i] = rng();
  }
  return values;
}

// A dense monotone counter, where the whole value is one field.
std::vector<uint64_t> makeCounterStream(size_t numRows) {
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = uint64_t{1'000'000'000'000} + i;
  }
  return values;
}

// Forces a split so that what a split writes can be measured, and leaves every
// section to ordinary selection with a split withheld.
template <typename T>
class ForcedSubIntSplitPolicy final : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::SubIntSplit};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /* encodingType */,
      NestedEncodingIdentifier /* identifier */,
      DataType type) override {
    auto readFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == EncodingType::SubIntSplit;
            }),
        readFactors.end());
    ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  }
};

class SubIntSplitSizeEstimateTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  uint64_t estimate(const std::vector<uint64_t>& values) {
    const std::span<const uint64_t> span(values);
    const auto estimated = SubIntSplitEncoding<uint64_t>::estimateSize(
        span.size(),
        span,
        Statistics<uint64_t>::create(span),
        Encoding::Options{});
    EXPECT_TRUE(estimated.has_value());
    return estimated.value_or(0);
  }

  uint64_t actualSubIntSplitBytes(const std::vector<uint64_t>& values) {
    Buffer buffer{*pool_};
    return EncodingFactory::encode<uint64_t>(
               std::make_unique<ForcedSubIntSplitPolicy<uint64_t>>(),
               values,
               buffer)
        .size();
  }

  EncodingType selected(const std::vector<uint64_t>& values) {
    const std::span<const uint64_t> span(values);
    ManualEncodingSelectionPolicy<uint64_t> policy{
        readFactorsWithSubIntSplit(),
        CompressionOptions{},
        std::nullopt,
    };
    return policy
        .select(span, Statistics<uint64_t>::create(span), Encoding::Options{})
        .encodingType;
  }

  // Asserts the estimate lands between 1/`lowerFactor` and `upperFactor` of
  // what a split writes, and reports the ratio either way.
  void expectRatioWithin(
      const std::vector<uint64_t>& values,
      double lowerFactor,
      double upperFactor,
      const std::string& name) {
    const double estimated = static_cast<double>(estimate(values));
    const double actual = static_cast<double>(actualSubIntSplitBytes(values));
    ASSERT_GT(actual, 0.0) << name;
    const double ratio = estimated / actual;
    std::cerr << name << ": estimate " << estimated << " actual " << actual
              << " ratio " << ratio << "\n";
    EXPECT_GE(ratio, 1.0 / lowerFactor) << name << " estimate/actual " << ratio;
    EXPECT_LE(ratio, upperFactor) << name << " estimate/actual " << ratio;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

// The two streams below are where the estimate has to be accurate: what a
// split stores is a material share of the column either way, so a wrong answer
// changes which encoding selection picks. Measured 1.11 and 0.96.
TEST_F(SubIntSplitSizeEstimateTest, compositeKeyEstimateTracksWhatIsWritten) {
  expectRatioWithin(
      makeCompositeKeyStream(kNumRows), 1.25, 1.25, "composite key");
}

TEST_F(SubIntSplitSizeEstimateTest, uniformRandomEstimateTracksWhatIsWritten) {
  expectRatioWithin(
      makeUniformRandomStream(kNumRows), 1.25, 1.25, "uniform random");
}

// On the two below the estimate is loose upward, and deliberately left so.
// Both streams collapse to almost nothing -- 5'672 and 45 bytes out of 512 KiB
// unencoded -- through whole-value routes the DP's sampled cost models do not
// price: MainlyConstant for one, the row frame plus a near-empty residual for
// the other. Selection reaches the same bytes by picking those encodings
// itself, which selectionAvoidsSubIntSplitWhereItIsNot asserts, so the
// looseness costs no bytes. The bounds are regression guards on the direction
// and the order of magnitude: measured 3.46 and 40.7, against the 3.46 and 584
// the same streams gave before the row frame was priced.
TEST_F(SubIntSplitSizeEstimateTest, constantHeavyEstimateIsLooseUpward) {
  expectRatioWithin(
      makeConstantHeavyStream(kNumRows), 1.0, 4.0, "constant heavy");
}

TEST_F(SubIntSplitSizeEstimateTest, counterEstimateIsLooseUpward) {
  expectRatioWithin(makeCounterStream(kNumRows), 1.0, 50.0, "counter");
}

TEST_F(SubIntSplitSizeEstimateTest, splitIsNotPricedUnderSubstreamCompression) {
  // A counter is a stream the estimate does price a split for, through the
  // row frame, and a substream compressor is what makes that pricing wrong:
  // the estimate counts uncompressed bytes and the compressor changes which
  // candidate is smallest.
  const auto values = makeCounterStream(kNumRows);
  const std::span<const uint64_t> span(values);
  const auto statistics = Statistics<uint64_t>::create(span);

  Encoding::Options uncompressed;
  const auto priced = SubIntSplitEncoding<uint64_t>::estimateSize(
      values.size(), span, statistics, uncompressed);
  ASSERT_TRUE(priced.has_value());
  const auto fixedBitWidth = FixedBitWidthEncoding<uint64_t>::estimateSize(
      values.size(), statistics, uncompressed);
  ASSERT_LT(*priced, fixedBitWidth);

  Encoding::Options compressed;
  compressed.substreamCompression = true;
  EXPECT_EQ(
      SubIntSplitEncoding<uint64_t>::estimateSize(
          values.size(), span, statistics, compressed),
      fixedBitWidth);

  // The guard off, the estimate prices the split in both worlds alike.
  compressed.subIntSplitEstimateCompressionGuard = false;
  EXPECT_EQ(
      SubIntSplitEncoding<uint64_t>::estimateSize(
          values.size(), span, statistics, compressed),
      priced);
}

TEST_F(SubIntSplitSizeEstimateTest, lowerBoundRulesOutStreamsWithoutFields) {
  // The screen is off by default, so every case below asks for it.
  // Over every pair. Uniform random is the case where the cap matters: a
  // 1'024-pair profile of it clears the gradient floor on sampling noise, so
  // the screen declines there and the split DP runs. The columns the screen
  // does rule out at the cap -- record counters, a constant-heavy
  // calculation -- have no gradient boundary at all, and no sample size
  // changes that.
  Encoding::Options options;
  options.subIntSplitEstimateBitFlipScreen = true;
  options.subIntSplitAdmissionProfilePairs = 0;
  // 300'000 rows, not kNumRows: a whole-stream flip probability taken from
  // 65'535 pairs still carries 0.002 of standard error, which puts the
  // largest of 63 adjacent differences at the gate's 0.005 floor, and this
  // test is about the bound rather than about that boundary.
  const auto random = makeUniformRandomStream(300'000);
  const std::span<const uint64_t> randomSpan(random);
  const auto randomStatistics = Statistics<uint64_t>::create(randomSpan);
  const auto randomBound =
      SubIntSplitEncoding<uint64_t>::estimateSizeLowerBound(
          randomSpan, randomStatistics, options);
  ASSERT_TRUE(randomBound.has_value());
  EXPECT_EQ(
      *randomBound,
      FixedBitWidthEncoding<uint64_t>::estimateSize(
          random.size(), randomStatistics, options));
  // The bound is the gate's prediction, not a proof, and this stream is where
  // that shows: the split DP prices a uniform-random column about 2% below
  // FixedBitWidth, by taking the bits that never vary out into a constant
  // section, so taking the bound gives up a win that small. It is why the
  // screen is off by default.
  EXPECT_LT(
      *SubIntSplitEncoding<uint64_t>::estimateSize(
          random.size(), randomSpan, randomStatistics, options),
      *randomBound);

  // A composite key has one, so the bound declines to answer and the estimate
  // is what decides.
  const auto composite = makeCompositeKeyStream(kNumRows);
  const std::span<const uint64_t> compositeSpan(composite);
  EXPECT_FALSE(
      SubIntSplitEncoding<uint64_t>::estimateSizeLowerBound(
          compositeSpan, Statistics<uint64_t>::create(compositeSpan), options)
          .has_value());

  // The sampled default declines on this stream, which is the screen's limit
  // and not a bug: it costs a split DP that the estimate then prices out.
  Encoding::Options sampled;
  sampled.subIntSplitEstimateBitFlipScreen = true;
  EXPECT_FALSE(
      SubIntSplitEncoding<uint64_t>::estimateSizeLowerBound(
          randomSpan, randomStatistics, sampled)
          .has_value());

  // Off, the screen never rules anything out.
  Encoding::Options noScreen;
  noScreen.subIntSplitEstimateBitFlipScreen = false;
  noScreen.subIntSplitAdmissionProfilePairs = 0;
  EXPECT_FALSE(
      SubIntSplitEncoding<uint64_t>::estimateSizeLowerBound(
          randomSpan, randomStatistics, noScreen)
          .has_value());
}

TEST_F(SubIntSplitSizeEstimateTest, estimateNeverExceedsFixedBitWidth) {
  // The encoder's whole-value floor stores the values as one FixedBitWidth
  // section rather than let a plan come in above it, so an estimate above
  // FixedBitWidth's would be one selection could never see honoured.
  for (const auto& values :
       {makeCompositeKeyStream(kNumRows),
        makeUniformRandomStream(kNumRows),
        makeConstantHeavyStream(kNumRows),
        makeCounterStream(kNumRows)}) {
    const std::span<const uint64_t> span(values);
    const auto statistics = Statistics<uint64_t>::create(span);
    EXPECT_LE(
        estimate(values),
        FixedBitWidthEncoding<uint64_t>::estimateSize(
            span.size(), statistics, Encoding::Options{}));
  }
}

TEST_F(SubIntSplitSizeEstimateTest, selectionPicksSubIntSplitWhereItIsSmaller) {
  const auto values = makeCompositeKeyStream(kNumRows);
  // The premise the selection assertion rests on: a split really does store
  // this stream in materially fewer bytes than selection's next best.
  const auto splitBytes = actualSubIntSplitBytes(values);
  Buffer buffer{*pool_};
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  readFactors.erase(
      std::remove_if(
          readFactors.begin(),
          readFactors.end(),
          [](const auto& factor) {
            return factor.first == EncodingType::SubIntSplit;
          }),
      readFactors.end());
  const auto withoutSplitBytes =
      EncodingFactory::encode<uint64_t>(
          std::make_unique<ManualEncodingSelectionPolicy<uint64_t>>(
              std::move(readFactors), CompressionOptions{}, std::nullopt),
          values,
          buffer)
          .size();
  ASSERT_LT(splitBytes, withoutSplitBytes * 0.9)
      << splitBytes << " vs " << withoutSplitBytes;

  EXPECT_EQ(selected(values), EncodingType::SubIntSplit);
}

TEST_F(SubIntSplitSizeEstimateTest, selectionAvoidsSubIntSplitWhereItIsNot) {
  // Uniform random has no bit structure to split on, and constant-heavy is
  // stored far better whole. Neither should reach a split now that the range
  // rule no longer withholds the candidate.
  EXPECT_NE(
      selected(makeUniformRandomStream(kNumRows)), EncodingType::SubIntSplit);
  EXPECT_NE(
      selected(makeConstantHeavyStream(kNumRows)), EncodingType::SubIntSplit);
}

} // namespace

#endif
