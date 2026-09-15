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
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

// Config-driven split and per-section encoding: the caller names the bit ranges
// and, optionally, which encoding each range uses. This is what makes
// SubIntSplit usable as a drop-in for an explicitly configured bit-range
// decomposition rather than only as an auto-planner.
class SubIntSplitConfiguredSplitTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "SubIntSplitConfiguredSplitTest");
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::string encodeWithConfig(
      const std::vector<uint64_t>& values,
      const std::unordered_map<std::string, std::string>& config) {
    const std::span<const uint64_t> input{values.data(), values.size()};
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<uint64_t> selection{
        EncodingSelectionResult{
            .encodingType = EncodingType::SubIntSplit,
            .encodingConfig = EncodingLayout::Config{config}},
        Statistics<uint64_t>::create(input),
        factory.createPolicy(DataType::Uint64)};

    const auto encoded = SubIntSplitEncoding<uint64_t>::encode(
        selection, input, *buffer_, /*options=*/{});
    return std::string{encoded};
  }

  std::vector<uint64_t> decode(const std::string& encoded, uint32_t numValues) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    std::vector<uint64_t> output(numValues);
    decoder.materialize(numValues, output.data());
    return output;
  }

  // The bit ranges and the encoding each section actually ended up with, read
  // back off the encoded stream rather than assumed.
  std::vector<std::pair<std::string, std::string>> sectionLayout(
      const std::string& encoded) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    std::vector<std::pair<std::string, std::string>> layout;
    std::istringstream stream{decoder.debugString(0)};
    std::string line;
    std::string pendingRange;
    while (std::getline(stream, line)) {
      const auto open = line.find('[');
      const auto close = line.find(']');
      if (open != std::string::npos && close != std::string::npos &&
          line.find("storageBytes") != std::string::npos) {
        pendingRange = line.substr(open + 1, close - open - 1);
        continue;
      }
      if (!pendingRange.empty()) {
        // The line after a section header is that section's nested encoding.
        auto trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        // debugString renders an encoding as "Name<PhysicalType> ...".
        const auto end = trimmed.find_first_of("< ");
        layout.emplace_back(
            pendingRange,
            end == std::string::npos ? trimmed : trimmed.substr(0, end));
        pendingRange.clear();
      }
    }
    return layout;
  }

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

  // Mirrors BitRangeSplit's canonical three-way configuration.
  std::vector<subintsplit::SectionPlan> pinnedSections() {
    return {
        {.bitStart = 0, .bitEnd = 15},
        {.bitStart = 16, .bitEnd = 58},
        {.bitStart = 59, .bitEnd = 63}};
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

TEST_F(SubIntSplitConfiguredSplitTest, pinnedSplitIsUsedVerbatim) {
  const auto values = makeMultiFieldValues(10'000);
  const auto sections = pinnedSections();

  const auto encoded =
      encodeWithConfig(values, subintsplit::makePreserveSplitConfig(sections));

  const auto layout = sectionLayout(encoded);
  ASSERT_EQ(layout.size(), 3u);
  EXPECT_EQ(layout[0].first, "0..15");
  EXPECT_EQ(layout[1].first, "16..58");
  EXPECT_EQ(layout[2].first, "59..63");
  EXPECT_EQ(decode(encoded, values.size()), values);
}

TEST_F(SubIntSplitConfiguredSplitTest, pinnedSectionEncodingsAreUsedVerbatim) {
  const auto values = makeMultiFieldValues(10'000);
  const auto sections = pinnedSections();

  const std::vector<std::optional<EncodingType>> encodings{
      EncodingType::Trivial,
      EncodingType::FixedBitWidth,
      EncodingType::Constant};

  const auto encoded = encodeWithConfig(
      values, subintsplit::makePreserveSplitConfig(sections, encodings));

  const auto layout = sectionLayout(encoded);
  ASSERT_EQ(layout.size(), 3u);
  EXPECT_EQ(layout[0].second, "Trivial");
  EXPECT_EQ(layout[1].second, "FixedBitWidth");
  EXPECT_EQ(layout[2].second, "Constant");
  EXPECT_EQ(decode(encoded, values.size()), values);
}

// An empty entry is how a caller pins some sections and leaves the rest to
// selection, which is the common case when only one section is interesting.
TEST_F(SubIntSplitConfiguredSplitTest, emptyEntryLeavesSectionToSelection) {
  const auto values = makeMultiFieldValues(10'000);
  const auto sections = pinnedSections();

  const std::vector<std::optional<EncodingType>> encodings{
      EncodingType::Trivial, std::nullopt, EncodingType::Trivial};

  const auto encoded = encodeWithConfig(
      values, subintsplit::makePreserveSplitConfig(sections, encodings));

  const auto layout = sectionLayout(encoded);
  ASSERT_EQ(layout.size(), 3u);
  EXPECT_EQ(layout[0].second, "Trivial");
  EXPECT_EQ(layout[2].second, "Trivial");
  // The middle section holds a wide, high-cardinality counter, which selection
  // would never store as Trivial.
  EXPECT_NE(layout[1].second, "Trivial");
  EXPECT_EQ(decode(encoded, values.size()), values);
}

// The entries index sections positionally, so a list that describes a different
// split has to be rejected rather than silently applied to the wrong ranges.
TEST_F(SubIntSplitConfiguredSplitTest, rejectsWrongNumberOfSectionEncodings) {
  const auto values = makeMultiFieldValues(1'000);
  auto config = subintsplit::makePreserveSplitConfig(pinnedSections());
  config[std::string(subintsplit::kSectionEncodingsConfigKey)] =
      "Trivial;FixedBitWidth";

  NIMBLE_ASSERT_USER_THROW(
      encodeWithConfig(values, config),
      "must name a writable encoding for each of the 3 sections");
}

TEST_F(SubIntSplitConfiguredSplitTest, rejectsUnknownSectionEncoding) {
  const auto values = makeMultiFieldValues(1'000);
  auto config = subintsplit::makePreserveSplitConfig(pinnedSections());
  config[std::string(subintsplit::kSectionEncodingsConfigKey)] =
      "Trivial;NotAnEncoding;Trivial";

  NIMBLE_ASSERT_USER_THROW(
      encodeWithConfig(values, config),
      "must name a writable encoding for each of the 3 sections");
}

TEST_F(SubIntSplitConfiguredSplitTest, sectionEncodingsRoundTripThroughConfig) {
  const std::vector<std::optional<EncodingType>> encodings{
      EncodingType::Trivial, std::nullopt, EncodingType::RLE};

  const auto serialized = subintsplit::serializeSectionEncodings(encodings);
  EXPECT_EQ(serialized, "Trivial;;RLE");

  const auto parsed = subintsplit::parseSectionEncodings(serialized, 3);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(*parsed, encodings);
}

} // namespace
