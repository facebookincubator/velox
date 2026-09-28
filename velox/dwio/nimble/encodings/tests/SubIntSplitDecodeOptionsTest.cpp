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

class SubIntSplitDecodeOptionsTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "SubIntSplitDecodeOptionsTest");
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

  std::vector<uint64_t> decode(
      const std::string& encoded,
      uint32_t numValues,
      const subintsplit::TuningConfig& tuning) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}, tuning};
    std::vector<uint64_t> output(numValues);
    decoder.materialize(numValues, output.data());
    return output;
  }

  uint32_t sectionCount(const std::string& encoded) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    const auto debug = decoder.debugString(0);
    const auto marker = std::string("sections=");
    const auto at = debug.find(marker);
    EXPECT_NE(at, std::string::npos);
    return static_cast<uint32_t>(std::stoul(debug.substr(at + marker.size())));
  }

  // Snowflake-shaped ids: a slowly rising timestamp, a low-cardinality machine
  // id and a fast counter. Several distinct bit fields is what makes the
  // planner choose many sections.
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

// Chunk size only governs how decode walks the output, so every setting has to
// reconstruct the same values -- including sizes that do not divide the row
// count, which is where a chunk-boundary bug would show.
TEST_F(SubIntSplitDecodeOptionsTest, decodeChunkSizeDoesNotChangeValues) {
  const auto values = makeMultiFieldValues(10'000);
  const auto encoded = encode(values, {});

  for (const uint32_t chunkSize : {1u, 7u, 512u, 4096u, 65'536u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.decodeChunkSize = chunkSize;
    EXPECT_EQ(decode(encoded, values.size(), tuning), values)
        << "chunk size " << chunkSize;
  }
}

// The decode-cost term is opt-in: zero has to leave the planner's choice, and
// therefore the bytes, exactly as they were.
TEST_F(SubIntSplitDecodeOptionsTest, zeroDecodeCostKeepsTheStorageOnlyPlan) {
  const auto values = makeMultiFieldValues(10'000);

  auto explicitZero = subintsplit::kDefaultTuningConfig;
  explicitZero.selector.decodeCostBitsPerValue = 0.0;

  EXPECT_EQ(encode(values, {}), encode(values, explicitZero));
}

// Charging for sections has to cost sections, monotonically, and the stream
// still has to decode to the original values afterwards.
TEST_F(SubIntSplitDecodeOptionsTest, decodeCostReducesSectionCount) {
  const auto values = makeMultiFieldValues(10'000);

  const auto baselineSections = sectionCount(encode(values, {}));
  ASSERT_GT(baselineSections, 2u)
      << "test data must produce a multi-section plan to be meaningful";

  uint32_t previous = baselineSections;
  for (const double bitsPerValue : {1.0, 4.0, 16.0, 64.0}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.decodeCostBitsPerValue = bitsPerValue;
    const auto encoded = encode(values, tuning);

    const auto sections = sectionCount(encoded);
    EXPECT_LE(sections, previous) << "bits per value " << bitsPerValue;
    previous = sections;

    EXPECT_EQ(decode(encoded, values.size(), {}), values)
        << "bits per value " << bitsPerValue;
  }
  EXPECT_LT(previous, baselineSections);
}

// A term large enough to outweigh any real saving leaves the active bit range
// as a single section. The plan still reports two, because the constant high
// prefix is re-attached after the DP and is not charged for: a constant section
// is folded into a single OR at construction and never decoded per value, so it
// genuinely costs nothing.
TEST_F(SubIntSplitDecodeOptionsTest, hugeDecodeCostCollapsesTheActiveRange) {
  const auto values = makeMultiFieldValues(10'000);

  auto prohibitive = subintsplit::kDefaultTuningConfig;
  prohibitive.selector.decodeCostBitsPerValue = 1'000.0;
  const auto encoded = encode(values, prohibitive);

  EXPECT_EQ(sectionCount(encoded), 2u);
  EXPECT_EQ(decode(encoded, values.size(), {}), values);
}

// The term buys decode time with storage, so the caller should be able to see
// what they paid.
TEST_F(SubIntSplitDecodeOptionsTest, decodeCostCostsStorage) {
  const auto values = makeMultiFieldValues(10'000);

  auto prohibitive = subintsplit::kDefaultTuningConfig;
  prohibitive.selector.decodeCostBitsPerValue = 1'000.0;

  EXPECT_GT(encode(values, prohibitive).size(), encode(values, {}).size());
}

} // namespace
