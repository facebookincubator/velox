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
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

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

TEST_F(SubIntSplitDeltaTest, resetRestartsSequentialDecode) {
  std::vector<uint64_t> values(1'024);
  std::iota(values.begin(), values.end(), uint64_t{10'000});

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
  std::vector<uint64_t> first(257);
  decoder.materialize(first.size(), first.data());
  decoder.reset();
  std::vector<uint64_t> second(first.size());
  decoder.materialize(second.size(), second.data());

  EXPECT_EQ(first, second);
  EXPECT_EQ(first, std::vector<uint64_t>(values.begin(), values.begin() + 257));
}

TEST_F(SubIntSplitDeltaTest, skipAdvancesDeltaStream) {
  std::vector<uint64_t> values(1'024);
  std::iota(values.begin(), values.end(), uint64_t{10'000});

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
  decoder.skip(17);
  uint64_t actual{0};
  decoder.materialize(1, &actual);
  EXPECT_EQ(actual, values[17]);

  decoder.skip(500);
  decoder.materialize(1, &actual);
  EXPECT_EQ(actual, values[518]);
}

TEST_F(SubIntSplitDeltaTest, interleavedReadsCrossTinyChunkBoundaries) {
  std::vector<uint64_t> values(5'000);
  std::iota(values.begin(), values.end(), uint64_t{1'000'000});

  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitDeltaPreTransform = true;
  options.subIntSplitDecodeChunkSize = 7;
  const std::span<const uint64_t> input{values.data(), values.size()};
  ManualEncodingSelectionPolicyFactory factory;
  EncodingSelection<uint64_t> selection{
      EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
      Statistics<uint64_t>::create(input),
      factory.createPolicy(DataType::Uint64)};
  const std::string_view encoded =
      SubIntSplitEncoding<uint64_t>::encode(selection, input, buffer, options);

  const auto flags =
      static_cast<uint8_t>(encoded[EncodingPrefix::kFixedPrefixSize + 1]);
  ASSERT_NE(flags & subintsplit::kFlagDelta, 0);

  SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, options};
  size_t cursor{0};
  const auto materializeAndExpect = [&](uint32_t count) {
    std::vector<uint64_t> actual(count);
    decoder.materialize(count, actual.data());
    EXPECT_EQ(
        actual,
        std::vector<uint64_t>(
            values.begin() + cursor, values.begin() + cursor + count));
    cursor += count;
  };
  const auto skip = [&](uint32_t count) {
    decoder.skip(count);
    cursor += count;
  };

  materializeAndExpect(5);
  skip(2);
  materializeAndExpect(1);
  skip(248);
  materializeAndExpect(3);
  skip(3'836);
  materializeAndExpect(5);
  skip(17);
  materializeAndExpect(19);

  decoder.reset();
  cursor = 0;
  materializeAndExpect(13);
  skip(4'080);
  materializeAndExpect(20);
}

// A delta stream can only be decoded from row zero, so createEncodingView must
// not serve it positionally: reading the stored residuals as values returns
// wrong results with no error. Every read shape must agree with materialize().
TEST_F(SubIntSplitDeltaTest, encodingViewReadsDeltaStreams) {
  // A random walk with small steps: the absolute bits are random, the deltas
  // narrow, so the encoder keeps the delta form.
  std::mt19937_64 generator{17};
  std::vector<uint64_t> values(10'000);
  uint64_t value = generator();
  for (auto& v : values) {
    value += generator() % 16;
    v = value;
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
  ASSERT_TRUE(
      subintsplit::isDeltaStream(
          encoded,
          EncodingPrefix::prefixSize(encoded, options.useVarintRowCount)));

  const auto view = createEncodingView(encoded, pool_.get(), options);
  ASSERT_EQ(view->rowCount(), values.size());

  std::vector<uint64_t> all(values.size());
  view->read(0, static_cast<uint32_t>(values.size()), all.data());
  EXPECT_EQ(all, values);

  for (const uint32_t row : {0u, 1u, 4'999u, 9'999u}) {
    uint64_t actual{0};
    view->readAt(row, &actual);
    EXPECT_EQ(actual, values[row]) << "row " << row;
  }

  const std::vector<RowRange> ranges{{10, 20}, {500, 501}, {9'990, 10'000}};
  std::vector<uint64_t> ranged(21);
  EXPECT_EQ(
      view->read(
          std::span<const RowRange>(ranges), [](uint32_t) {}, ranged.data()),
      21);
  std::vector<uint64_t> expected(values.begin() + 10, values.begin() + 20);
  expected.push_back(values[500]);
  expected.insert(expected.end(), values.begin() + 9'990, values.end());
  EXPECT_EQ(ranged, expected);
}

// A flag the view does not interpret changes how the stored sections map to
// values, so the view must reject it rather than read the stream as if it
// were absent.
TEST_F(SubIntSplitDeltaTest, encodingViewRejectsUnknownFlags) {
  std::vector<uint64_t> values(1'000);
  std::iota(values.begin(), values.end(), uint64_t{1'700'000'000'000});
  Buffer buffer{*pool_};
  const std::span<const uint64_t> input{values.data(), values.size()};
  ManualEncodingSelectionPolicyFactory factory;
  EncodingSelection<uint64_t> selection{
      EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
      Statistics<uint64_t>::create(input),
      factory.createPolicy(DataType::Uint64)};
  std::string corrupted{SubIntSplitEncoding<uint64_t>::encode(
      selection, input, buffer, Encoding::Options{})};
  const uint32_t flagsOffset = EncodingPrefix::prefixSize(corrupted, false) + 1;
  corrupted[flagsOffset] = static_cast<char>(0x80);
  EXPECT_THROW(
      createEncodingView(corrupted, pool_.get(), Encoding::Options{}),
      NimbleException);
}

} // namespace
