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

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

TEST(SubIntSplitTuningConfigTest, usesProductionDefaults) {
  const auto& tuning = subintsplit::kDefaultTuningConfig;

  EXPECT_EQ(tuning.sampler.maxSamples, 2'048);
  EXPECT_EQ(tuning.sampler.blockSize, 128);
  EXPECT_DOUBLE_EQ(tuning.selector.boundaryPruneThreshold, 0.001);
  EXPECT_EQ(tuning.selector.maxCandidateBoundaries, 0);
  EXPECT_EQ(tuning.selector.maxSectionWidth, 0);
  EXPECT_EQ(tuning.selector.frequencyMetricsMaxWidth, 0);
  EXPECT_DOUBLE_EQ(tuning.selector.decodeCostBitsPerValue, 0.0);
  EXPECT_EQ(tuning.decodeChunkSize, 4'096);
}

TEST(SubIntSplitCandidateBoundariesTest, zeroThresholdHonorsBoundaryCap) {
  const std::vector<uint64_t> samples{0, 0xff};

  const auto boundaries =
      subintsplit::candidateBoundaries(samples, 0, 7, 0.0, 2);

  EXPECT_EQ(boundaries.size(), 4);
  EXPECT_EQ(boundaries.front(), 0);
  EXPECT_EQ(boundaries.back(), 8);
}

// The planner knobs trade split quality for encode throughput. What has to hold
// for every setting is that the stream still round-trips and the defaults are
// not disturbed; how good the split is, is a tuning question the benchmark
// answers, not a correctness one.
class SubIntSplitPlannerOptionsTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "SubIntSplitPlannerOptionsTest");
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::string encode(
      const std::vector<uint64_t>& values,
      const subintsplit::TuningConfig& tuning) {
    const std::span<const uint64_t> input{values.data(), values.size()};
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<uint64_t> selection{
        EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
        Statistics<uint64_t>::create(input),
        factory.createPolicy(DataType::Uint64)};

    const auto encoded = SubIntSplitEncoding<uint64_t>::encode(
        selection, input, *buffer_, {}, tuning);
    return std::string{encoded};
  }

  std::vector<uint64_t> decode(const std::string& encoded, uint32_t numValues) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    std::vector<uint64_t> output(numValues);
    decoder.materialize(numValues, output.data());
    return output;
  }

  // Snowflake-shaped ids: several distinct bit fields, which is what gives the
  // planner a non-trivial number of candidate boundaries to prune.
  std::vector<uint64_t> makeMultiFieldValues(size_t numValues) {
    std::mt19937_64 rng{42};
    std::vector<uint64_t> values(numValues);
    for (size_t i = 0; i < numValues; ++i) {
      const uint64_t timestamp = 1'700'000'000'000ULL + (i / 64);
      const uint64_t machineId = rng() % 8;
      const uint64_t sequence = i % 4096;
      values[i] = (timestamp << 22) | (machineId << 12) | sequence;
    }
    return values;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

TEST_F(SubIntSplitPlannerOptionsTest, anyBoundaryCapRoundTrips) {
  const auto values = makeMultiFieldValues(10'000);

  for (const uint32_t cap : {1u, 2u, 4u, 16u, 1'000u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.maxCandidateBoundaries = cap;
    EXPECT_EQ(decode(encode(values, tuning), values.size()), values)
        << "boundary cap " << cap;
  }
}

// A width cap below the active range leaves no tiling of narrow cells, which is
// exactly why the full active range is always scored. Without that fallback the
// DP would find no finite plan.
TEST_F(
    SubIntSplitPlannerOptionsTest,
    sectionWidthCapBelowActiveRangeStillEncodes) {
  const auto values = makeMultiFieldValues(10'000);

  for (const uint32_t maxWidth : {1u, 2u, 8u, 24u, 64u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.maxSectionWidth = maxWidth;
    EXPECT_EQ(decode(encode(values, tuning), values.size()), values)
        << "max section width " << maxWidth;
  }
}

// Skipping the frequency pass takes Dictionary and MainlyConstant out of
// contention for wide sections; it must not change what the stream decodes to.
TEST_F(SubIntSplitPlannerOptionsTest, anyFrequencyWidthCapRoundTrips) {
  const auto values = makeMultiFieldValues(10'000);

  for (const uint32_t maxWidth : {1u, 8u, 16u, 32u, 64u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.frequencyMetricsMaxWidth = maxWidth;
    EXPECT_EQ(decode(encode(values, tuning), values.size()), values)
        << "frequency metrics max width " << maxWidth;
  }
}

// The knobs are meant to be combined into a "plan fast" profile, so the most
// aggressive setting of all of them at once still has to produce a valid
// stream.
TEST_F(SubIntSplitPlannerOptionsTest, allHeuristicsAtOnceRoundTrips) {
  const auto values = makeMultiFieldValues(10'000);

  auto fastest = subintsplit::kDefaultTuningConfig;
  fastest.sampler.maxSamples = 64;
  fastest.selector.boundaryPruneThreshold = 0.05;
  fastest.selector.maxCandidateBoundaries = 2;
  fastest.selector.maxSectionWidth = 8;
  fastest.selector.frequencyMetricsMaxWidth = 8;

  EXPECT_EQ(decode(encode(values, fastest), values.size()), values);
}

TEST_F(SubIntSplitPlannerOptionsTest, anySampleCountRoundTrips) {
  const auto values = makeMultiFieldValues(10'000);

  for (const uint32_t maxSamples : {1u, 2u, 64u, 512u, 4096u, 100'000u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.sampler.maxSamples = maxSamples;
    const auto encoded = encode(values, tuning);
    EXPECT_EQ(decode(encoded, values.size()), values)
        << "max samples " << maxSamples;
  }
}

TEST_F(SubIntSplitPlannerOptionsTest, anyPruneThresholdRoundTrips) {
  const auto values = makeMultiFieldValues(10'000);

  // 0.0 considers every bit position; 1.0 is past the maximum possible
  // set-rate change, so only the stream's own edges survive as boundaries.
  for (const double threshold : {0.0, 0.001, 0.02, 0.5, 1.0}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.boundaryPruneThreshold = threshold;
    const auto encoded = encode(values, tuning);
    EXPECT_EQ(decode(encoded, values.size()), values)
        << "threshold " << threshold;
  }
}

// Pruning everything has to leave a usable single-section plan rather than an
// empty one, which is the edge the DP's fallback path exists for.
TEST_F(SubIntSplitPlannerOptionsTest, pruningEveryBoundaryStillEncodes) {
  const auto values = makeMultiFieldValues(10'000);

  auto tuning = subintsplit::kDefaultTuningConfig;
  tuning.selector.boundaryPruneThreshold = 1.0;
  const auto encoded = encode(values, tuning);

  EXPECT_EQ(decode(encoded, values.size()), values);
  EXPECT_GT(encoded.size(), 0u);
}

// A single sample cannot describe the stream, but it must not produce a plan
// that mis-decodes the values it was not shown.
TEST_F(SubIntSplitPlannerOptionsTest, undersampledPlanStillDecodesEveryValue) {
  const auto values = makeMultiFieldValues(10'000);

  auto tuning = subintsplit::kDefaultTuningConfig;
  tuning.sampler.maxSamples = 1;
  const auto encoded = encode(values, tuning);

  EXPECT_EQ(decode(encoded, values.size()), values);
}

} // namespace
