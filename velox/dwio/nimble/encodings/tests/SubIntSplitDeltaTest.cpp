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
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

class SubIntSplitDeltaTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("SubIntSplitDeltaTest");
  }

  // Encodes with the delta pre-transform enabled and decodes the whole stream
  // back through materialize().
  std::vector<uint64_t> roundTrip(const std::vector<uint64_t>& values) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitDeltaPreTransform = true;

    const std::span<const uint64_t> input{values.data(), values.size()};
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<uint64_t> selection{
        EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
        Statistics<uint64_t>::create(input),
        factory.createPolicy(DataType::Uint64)};

    const std::string_view encoded = SubIntSplitEncoding<uint64_t>::encode(
        selection, input, buffer, options);

    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, options};
    std::vector<uint64_t> output(values.size());
    decoder.materialize(static_cast<uint32_t>(values.size()), output.data());
    return output;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(SubIntSplitDeltaTest, denseCounterRoundTrips) {
  std::vector<uint64_t> values(10'000);
  std::iota(values.begin(), values.end(), uint64_t{1'700'000'000'000});
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, decreasingValuesRoundTrip) {
  // Every delta is negative. Without zigzag these wrap to near-2^64 and the
  // split has nothing to work with.
  std::vector<uint64_t> values(10'000);
  uint64_t current = uint64_t{1} << 40;
  for (auto& value : values) {
    current -= 7;
    value = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, alternatingDirectionRoundTrips) {
  std::vector<uint64_t> values(10'000);
  uint64_t current = uint64_t{1} << 32;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 2 == 0) ? 1'000 : -997;
    values[i] = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, nonMonotonicValuesRoundTrip) {
  // Delta should lose to the plain form here; the point is that keep-smaller
  // still produces a correct stream either way.
  std::vector<uint64_t> values(10'000);
  uint64_t state = 88172645463325252ULL;
  for (auto& value : values) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    value = state;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, extremeValuesRoundTrip) {
  const std::vector<uint64_t> values{
      0,
      ~uint64_t{0},
      0,
      1,
      ~uint64_t{0} - 1,
      uint64_t{1} << 63,
      (uint64_t{1} << 63) - 1,
      42};
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, chunkBoundaryRoundTrips) {
  // materialize() works in 4096-row chunks and carries the prefix-sum
  // accumulator across them, so cross a few boundaries.
  std::vector<uint64_t> values(4096 * 3 + 17);
  uint64_t current = 5;
  for (auto& value : values) {
    current += 3;
    value = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(SubIntSplitDeltaTest, splitMaterializeMatchesSingleCall) {
  std::vector<uint64_t> values(10'000);
  uint64_t current = 1'000'000;
  for (auto& value : values) {
    current += 11;
    value = current;
  }

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitDeltaPreTransform = true;
  const std::span<const uint64_t> input{values.data(), values.size()};
  ManualEncodingSelectionPolicyFactory factory;
  EncodingSelection<uint64_t> selection{
      EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
      Statistics<uint64_t>::create(input),
      factory.createPolicy(DataType::Uint64)};
  const std::string_view encoded =
      SubIntSplitEncoding<uint64_t>::encode(selection, input, buffer, options);

  SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, options};
  std::vector<uint64_t> output(values.size());
  decoder.materialize(1'000, output.data());
  decoder.materialize(9'000, output.data() + 1'000);
  EXPECT_EQ(output, values);
}

} // namespace
