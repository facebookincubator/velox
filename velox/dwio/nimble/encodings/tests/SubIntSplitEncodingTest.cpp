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
#include <bit>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <folly/String.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"

using namespace facebook;

namespace {

template <typename T>
using PhysicalType = typename nimble::TypeTraits<T>::physicalType;

template <typename T>
using UnsignedPhysicalType = std::make_unsigned_t<PhysicalType<T>>;

template <typename T>
std::vector<T> makeStructuredValues() {
  std::vector<T> values;
  values.reserve(300);

  UnsignedPhysicalType<T> prefix{};
  if constexpr (sizeof(PhysicalType<T>) == 4) {
    prefix = static_cast<UnsignedPhysicalType<T>>(0x12340000u);
  } else {
    prefix = static_cast<UnsignedPhysicalType<T>>(0x1234567890000000ULL);
  }

  for (UnsignedPhysicalType<T> i = 0; i < 300; ++i) {
    const auto bits = static_cast<UnsignedPhysicalType<T>>(prefix + i);
    values.push_back(std::bit_cast<T>(bits));
  }

  return values;
}

// 300 values with a constant high prefix and a low-cardinality, non-monotonic
// low section (cycling through 5 distinct values). Unlike
// makeStructuredValues's unit-step low section -- which plain Delta already
// encodes optimally as a single, unsplit stream -- this low section is not
// monotonic, so encoding the full value directly is comparatively expensive
// and splitting off the constant high prefix (leaving a cheap
// Dictionary/MainlyConstant-friendly low segment) is unambiguously smaller.
// Used to exercise SubIntSplit's multi-segment capture/replay path
// independent of any single candidate's cost-model tuning.
template <typename T>
std::vector<T> makeStructuredValuesWithLowCardinalityNoise() {
  std::vector<T> values;
  values.reserve(300);

  UnsignedPhysicalType<T> prefix{};
  if constexpr (sizeof(PhysicalType<T>) == 4) {
    prefix = static_cast<UnsignedPhysicalType<T>>(0x12340000u);
  } else {
    prefix = static_cast<UnsignedPhysicalType<T>>(0x1234567890000000ULL);
  }

  constexpr UnsignedPhysicalType<T> kLowValues[] = {3, 7, 1, 9, 3, 3, 5, 3};
  for (size_t i = 0; i < 300; ++i) {
    const auto bits =
        static_cast<UnsignedPhysicalType<T>>(prefix + kLowValues[i % 8]);
    values.push_back(std::bit_cast<T>(bits));
  }

  return values;
}

// 300 values whose unsigned physical representation is strictly
// monotonically increasing with a constant, wide step (so consecutive
// values' low-order bit-range segments are also roughly monotonic with a
// constant step). This is the data shape the Delta/FOR cost models
// (SubIntSplitCostModelsTest.cpp's makeDeltaFriendlyValues/
// makeForFriendlyValues) favor, exercised here end-to-end through the real
// SubIntSplit segment-selection pipeline (B.4's extended candidate list).
template <typename T>
std::vector<T> makeWideRangeMonotonicValues() {
  std::vector<T> values;
  values.reserve(300);

  UnsignedPhysicalType<T> base{};
  UnsignedPhysicalType<T> step{};
  if constexpr (sizeof(PhysicalType<T>) == 4) {
    base = static_cast<UnsignedPhysicalType<T>>(0x10000000u);
    step = static_cast<UnsignedPhysicalType<T>>(0x00010000u);
  } else {
    base = static_cast<UnsignedPhysicalType<T>>(0x1000000000000000ULL);
    step = static_cast<UnsignedPhysicalType<T>>(0x0000000100000000ULL);
  }

  for (UnsignedPhysicalType<T> i = 0; i < 300; ++i) {
    const auto bits = static_cast<UnsignedPhysicalType<T>>(base + i * step);
    values.push_back(std::bit_cast<T>(bits));
  }

  return values;
}

// Typed Zipfian generator for SubIntSplitEncodingTest. Values 0..67 are small
// enough to fit in any physicalType; bit_cast is used so the round-trip stays
// bit-exact regardless of the logical type (int32_t, float, etc.).
// Interleaved (0, j) pairs give monotonicCount ≈ 50%, keeping Delta cost
// infinite so FrequencyPartition can win the DP cost model.
template <typename T>
std::vector<T> makeZipfianValues() {
  std::vector<T> values;
  values.reserve(1024);
  auto push = [&](uint64_t a, uint64_t b, int count) {
    for (int i = 0; i < count; ++i) {
      values.push_back(
          std::bit_cast<T>(static_cast<UnsignedPhysicalType<T>>(a)));
      values.push_back(
          std::bit_cast<T>(static_cast<UnsignedPhysicalType<T>>(b)));
    }
  };
  push(0, 1, 256); // 512 values
  push(0, 2, 128); // 256 values
  push(0, 3, 64); // 128 values
  for (uint64_t j = 4; j < 36; ++j) {
    push(0, j, 1); // 64 values
  }
  for (uint64_t j = 36; j < 68; ++j) {
    push(0, j, 1); // 64 values
  }
  // Total: 512 + 256 + 128 + 64 + 64 = 1024
  return values;
}

template <typename T>
std::vector<nimble::subintsplit::SectionPlan> makePreserveSegments() {
  if constexpr (sizeof(PhysicalType<T>) == 4) {
    return {{0, 7}, {8, 15}, {16, 31}};
  } else {
    return {{0, 7}, {8, 15}, {16, 31}, {32, 63}};
  }
}

template <typename T>
std::vector<nimble::subintsplit::SectionPlan> makeFullWidthSegments() {
  return {{0, static_cast<int>(sizeof(PhysicalType<T>) * 8 - 1)}};
}

template <typename T>
std::string_view encodeWithNonRecursiveSubIntSplit(
    const std::vector<T>& values,
    nimble::Buffer& buffer);

nimble::EncodingSelectionPolicyCreator makeLeafPolicyCreator() {
  return [](nimble::DataType type)
             -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == nimble::EncodingType::SubIntSplit;
            }),
        readFactors.end());
    nimble::ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  };
}

template <typename T>
class NonRecursiveSubIntSplitPolicy final
    : public nimble::EncodingSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::SubIntSplit};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* encodingType */,
      nimble::NestedEncodingIdentifier /* identifier */,
      nimble::DataType type) override {
    auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == nimble::EncodingType::SubIntSplit;
            }),
        readFactors.end());
    nimble::ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  }
};

template <typename T>
std::string_view encodeWithNonRecursiveSubIntSplit(
    const std::vector<T>& values,
    nimble::Buffer& buffer) {
  return nimble::EncodingFactory::encode<T>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<T>>(), values, buffer);
}

// Like NonRecursiveSubIntSplitPolicy, but its createImpl() delegates to
// ManualEncodingSelectionPolicy<T>::createImpl() (passing through
// EncodingType::SubIntSplit), exercising the real
// EncodingType::SubIntSplit special case there, which extends the segment's
// candidate list with PFOR / SimdForBitpack / BlockBitPacking.
template <typename T>
class ExtendedSubIntSplitPolicy final
    : public nimble::ManualEncodingSelectionPolicy<T> {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

 public:
  // ManualEncodingSelectionPolicy::createImpl() is protected, so its
  // special-cased SubIntSplit child candidates can only be exercised by
  // inheriting it (rather than delegating to a sibling instance).
  // `config` reaches the SubIntSplit encoder as its selection config, which is
  // how a test pins boundaries while keeping the writer's section candidates.
  explicit ExtendedSubIntSplitPolicy(
      nimble::EncodingLayout::Config config = nimble::EncodingLayout::Config{})
      : nimble::ManualEncodingSelectionPolicy<T>(
            filteredReadFactors(),
            std::nullopt,
            std::nullopt),
        config_{std::move(config)} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {
        .encodingType = nimble::EncodingType::SubIntSplit,
        .encodingConfig = config_};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

 private:
  static std::vector<std::pair<nimble::EncodingType, float>>
  filteredReadFactors() {
    auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
        defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == nimble::EncodingType::SubIntSplit;
            }),
        readFactors.end());
    return readFactors;
  }

  const nimble::EncodingLayout::Config config_;
};

template <typename T>
std::string_view encodeWithExtendedSubIntSplit(
    const std::vector<T>& values,
    nimble::Buffer& buffer) {
  return nimble::EncodingFactory::encode<T>(
      std::make_unique<ExtendedSubIntSplitPolicy<T>>(), values, buffer);
}

template <typename T>
std::string_view encodeWithReplayLayout(
    const nimble::EncodingLayout& layout,
    const std::vector<T>& values,
    nimble::Buffer& buffer) {
  auto leafCreator = makeLeafPolicyCreator();
  return nimble::EncodingFactory::encode<T>(
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<T>>(
          layout, std::nullopt, leafCreator),
      values,
      buffer);
}

template <typename T>
std::unique_ptr<nimble::Encoding> decodeEncoding(
    std::string_view encoded,
    velox::memory::MemoryPool& pool,
    const nimble::Encoding::Options& options = {}) {
  return nimble::EncodingFactory().create(
      pool, encoded, [](uint32_t) { return nullptr; }, options);
}

template <typename T>
std::vector<T> decodeAll(
    std::string_view encoded,
    velox::memory::MemoryPool& pool) {
  auto encoding = decodeEncoding<T>(encoded, pool);
  std::vector<T> result(encoding->rowCount());
  encoding->materialize(encoding->rowCount(), result.data());
  return result;
}

template <typename T>
void expectBitwiseEqual(
    const std::vector<T>& expected,
    const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(
        nimble::EncodingPhysicalType<T>::asEncodingPhysicalType(expected[i]),
        nimble::EncodingPhysicalType<T>::asEncodingPhysicalType(actual[i]))
        << "row " << i;
  }
}

template <typename T>
nimble::EncodingLayout makePreserveLayout(
    const std::vector<nimble::subintsplit::SectionPlan>& segments) {
  std::vector<std::optional<const nimble::EncodingLayout>> children(
      segments.size());
  return nimble::EncodingLayout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{
          nimble::subintsplit::makePreserveSplitConfig(segments)},
      nimble::CompressionType::Uncompressed,
      std::move(children)};
}

void expectSegmentsEqual(
    const std::vector<nimble::subintsplit::SectionPlan>& expected,
    const std::vector<nimble::subintsplit::SectionPlan>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i].bitStart, actual[i].bitStart) << "segment " << i;
    EXPECT_EQ(expected[i].bitEnd, actual[i].bitEnd) << "segment " << i;
  }
}

void expectSameLayout(
    const nimble::EncodingLayout& expected,
    const nimble::EncodingLayout& actual) {
  EXPECT_EQ(expected.encodingType(), actual.encodingType());
  EXPECT_EQ(expected.compressionType(), actual.compressionType());
  EXPECT_EQ(expected.config().values(), actual.config().values());
  ASSERT_EQ(expected.childrenCount(), actual.childrenCount());
  for (nimble::NestedEncodingIdentifier i = 0; i < expected.childrenCount();
       ++i) {
    const auto& expectedChild = expected.child(i);
    const auto& actualChild = actual.child(i);
    ASSERT_EQ(expectedChild.has_value(), actualChild.has_value())
        << "child " << i;
    if (expectedChild.has_value()) {
      expectSameLayout(*expectedChild, *actualChild);
    }
  }
}

// Pre/post-order IDs of a random tree, packed as XMark packs them:
// (depth << 56) | (pre << 28) | post. pre is the row number and post stays
// within a subtree's size of it, so the column follows the line
// (2^28 + 1) * row through its low 56 bits while depth, above them, does not.
std::vector<uint64_t> makeTreeIds(uint32_t numRows, uint32_t seed) {
  std::mt19937 generator{seed};
  std::vector<uint32_t> depths(numRows);
  for (uint32_t row = 1; row < numRows; ++row) {
    // A child, a sibling, or a sibling of an ancestor up to three levels up.
    // Depth wanders rather than settling, as it does in a real document, so
    // the depth field above the counters does not grow with the rows.
    const uint32_t previousDepth = depths[row - 1];
    const uint32_t choice = generator() % 4;
    if (choice == 0 && previousDepth < 12) {
      depths[row] = previousDepth + 1;
    } else if (choice == 1) {
      const uint32_t climb = 1 + generator() % 3;
      depths[row] = previousDepth > climb ? previousDepth - climb : 1;
    } else {
      depths[row] = std::max<uint32_t>(previousDepth, 1);
    }
  }
  // A node's subtree ends where the next node at or above its depth starts.
  std::vector<uint32_t> subtreeSizes(numRows);
  std::vector<uint32_t> open;
  for (uint32_t row = 0; row < numRows; ++row) {
    while (!open.empty() && depths[open.back()] >= depths[row]) {
      subtreeSizes[open.back()] = row - open.back();
      open.pop_back();
    }
    open.push_back(row);
  }
  for (const uint32_t row : open) {
    subtreeSizes[row] = numRows - row;
  }
  std::vector<uint64_t> ids(numRows);
  for (uint32_t row = 0; row < numRows; ++row) {
    const uint64_t post = row + subtreeSizes[row] - 1 - depths[row];
    ids[row] = (uint64_t{depths[row]} << 56) | (uint64_t{row} << 28) | post;
  }
  return ids;
}

nimble::subintsplit::RowFrame parseRowFrame(std::string_view encoded) {
  nimble::subintsplit::RowFrame frame;
  nimble::subintsplit::parseSections(
      encoded, nimble::Encoding::kPrefixSize, nullptr, &frame);
  return frame;
}
} // namespace

TEST(SubIntSplitConfigTests, boundarySerializationAndParsing) {
  const std::vector<nimble::subintsplit::SectionPlan> segments{
      {.bitStart = 0, .bitEnd = 7},
      {.bitStart = 8, .bitEnd = 15},
      {.bitStart = 16, .bitEnd = 31}};

  const auto serialized =
      nimble::subintsplit::serializeSplitBoundaries(segments);
  EXPECT_EQ(serialized, "0-7;8-15;16-31");

  auto parsed = nimble::subintsplit::parseSplitBoundaries(serialized, 32);
  ASSERT_TRUE(parsed.has_value());
  expectSegmentsEqual(segments, *parsed);

  EXPECT_FALSE(nimble::subintsplit::parseSplitBoundaries("", 32).has_value());
  EXPECT_FALSE(
      nimble::subintsplit::parseSplitBoundaries("0-7;9-15", 16).has_value());
  EXPECT_FALSE(
      nimble::subintsplit::parseSplitBoundaries("0-7;8-16", 16).has_value());
  EXPECT_FALSE(
      nimble::subintsplit::parseSplitBoundaries("0-7;8-15", 8).has_value());
}

template <typename T>
class SubIntSplitEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

using SubIntSplitEncodingTypes =
    ::testing::Types<int32_t, uint32_t, int64_t, uint64_t, float, double>;

TYPED_TEST_CASE(SubIntSplitEncodingTest, SubIntSplitEncodingTypes);

TYPED_TEST(SubIntSplitEncodingTest, recomputeRoundTripAndReplay) {
  using T = TypeParam;
  const auto values = makeStructuredValuesWithLowCardinalityNoise<T>();

  const auto encoded =
      encodeWithNonRecursiveSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});

  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  ASSERT_GT(captured.childrenCount(), 1u);

  const auto capturedMode = captured.config().get(
      std::string(nimble::subintsplit::kSplitModeConfigKey));
  ASSERT_TRUE(capturedMode.has_value());
  EXPECT_EQ(*capturedMode, nimble::subintsplit::kSplitModePreserve);

  const auto capturedBoundaries = captured.config().get(
      std::string(nimble::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(capturedBoundaries.has_value());

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);

  const auto replayed =
      encodeWithReplayLayout<T>(captured, values, *this->buffer_);
  const auto replayCaptured = nimble::EncodingLayoutCapture::capture(
      replayed, nimble::Encoding::Options{});
  expectSameLayout(captured, replayCaptured);

  auto encoding = decodeEncoding<T>(replayed, *this->pool_);
  const size_t skipCount = 17;
  ASSERT_GT(values.size(), skipCount);

  encoding->skip(static_cast<uint32_t>(skipCount));
  std::vector<T> suffix(values.size() - skipCount);
  encoding->materialize(static_cast<uint32_t>(suffix.size()), suffix.data());
  expectBitwiseEqual(
      std::vector<T>(values.begin() + skipCount, values.end()), suffix);

  encoding->reset();
  std::vector<T> fullRoundTrip(values.size());
  encoding->materialize(
      static_cast<uint32_t>(values.size()), fullRoundTrip.data());
  expectBitwiseEqual(values, fullRoundTrip);
}

// A column that follows a line through its rows is stored as its distance
// from that line, and every read path adds the line back.
TEST(SubIntSplitEncodingTests, rowFrameRoundTripsTreeIds) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTreeIds(40'000, 7);
  const auto fitted =
      nimble::subintsplit::fitRowFrame(std::span<const uint64_t>(ids));
  EXPECT_EQ(fitted.slope, (uint64_t{1} << 28) + 1);
  // A leaf's post rank sits its depth below its pre rank, so the base is the
  // deepest leaf's distance and lifts every post field back to non-negative.
  int64_t lowestPostMinusPre = 0;
  for (uint32_t row = 0; row < ids.size(); ++row) {
    const auto post =
        static_cast<int64_t>(ids[row] & ((uint64_t{1} << 28) - 1));
    lowestPostMinusPre = std::min(lowestPostMinusPre, post - row);
  }
  EXPECT_LT(lowestPostMinusPre, 0);
  EXPECT_EQ(static_cast<int64_t>(fitted.base), lowestPostMinusPre);

  {
    nimble::Encoding::Options options;
    const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
        ids,
        buffer,
        options);
    const auto frame = parseRowFrame(encoded);
    EXPECT_EQ(frame.slope, fitted.slope);
    EXPECT_EQ(frame.base, fitted.base);
    expectBitwiseEqual(ids, decodeAll<uint64_t>(encoded, *pool));

    // A skip moves the row the frame is evaluated at without decoding.
    auto encoding = decodeEncoding<uint64_t>(encoded, *pool);
    encoding->skip(12'345);
    std::vector<uint64_t> middle(5'000);
    encoding->materialize(5'000, middle.data());
    expectBitwiseEqual(
        std::vector<uint64_t>(ids.begin() + 12'345, ids.begin() + 17'345),
        middle);
    encoding->reset();
    uint64_t first = 0;
    encoding->materialize(1, &first);
    EXPECT_EQ(first, ids[0]);

    // After a reset, alternating skips and reads whose spans straddle the
    // 4'096-row materialize chunk, so each read starts at a row the cursor
    // reached without decoding it.
    encoding->reset();
    uint64_t row = 0;
    for (const auto& [skipRows, readRows] :
         std::vector<std::pair<uint32_t, uint32_t>>{
             {4'095, 3}, {1, 9'000}, {7'777, 1}, {0, 19'123}}) {
      SCOPED_TRACE(fmt::format("row={} read={}", row + skipRows, readRows));
      encoding->skip(skipRows);
      row += skipRows;
      std::vector<uint64_t> chunk(readRows);
      encoding->materialize(readRows, chunk.data());
      expectBitwiseEqual(
          std::vector<uint64_t>(
              ids.begin() + row, ids.begin() + row + readRows),
          chunk);
      row += readRows;
    }
    ASSERT_EQ(row, ids.size());
  }

  // Signed values share the physical bits, and so the frame.
  std::vector<int64_t> signedIds(ids.begin(), ids.end());
  const auto signedEncoded = nimble::EncodingFactory::encode<int64_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<int64_t>>(),
      signedIds,
      buffer);
  EXPECT_TRUE(parseRowFrame(signedEncoded).active());
  expectBitwiseEqual(signedIds, decodeAll<int64_t>(signedEncoded, *pool));
}

// A 32-bit column fits and applies its frame modulo 2^32.
TEST(SubIntSplitEncodingTests, rowFrameRoundTrips32Bit) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  std::mt19937 generator{3};
  std::vector<uint32_t> values(30'000);
  for (uint32_t row = 0; row < values.size(); ++row) {
    values[row] = ((generator() % 16) << 24) | (row * 2 + generator() % 5);
  }
  const auto fitted =
      nimble::subintsplit::fitRowFrame(std::span<const uint32_t>(values));
  EXPECT_EQ(fitted.slope, 2u);
  const auto encoded = nimble::EncodingFactory::encode<uint32_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<uint32_t>>(),
      values,
      buffer);
  expectBitwiseEqual(values, decodeAll<uint32_t>(encoded, *pool));
}

namespace {

// A UUIDv7's high half, as RFC 9562 lays it out with a dedicated counter: a
// millisecond timestamp, the version nibble, and a 12-bit counter that restarts
// at random below 2^11 on each new millisecond and counts up by one within it.
// Arrivals average two per millisecond.
std::vector<uint64_t> makeTimestampCounterIds(uint32_t numRows, uint32_t seed) {
  std::mt19937_64 generator{seed};
  std::exponential_distribution<double> arrivalGap{2.0};
  double time = 1'735'689'600'000.0;
  uint64_t millisecond = 0;
  uint64_t counter = 0;
  std::vector<uint64_t> ids(numRows);
  for (auto& id : ids) {
    time += arrivalGap(generator);
    const auto now = static_cast<uint64_t>(time);
    if (now != millisecond) {
      millisecond = now;
      counter = generator() & 0x7FF;
    } else {
      counter = (counter + 1) & 0xFFF;
    }
    id = (millisecond << 16) | (uint64_t{0x7} << 12) | counter;
  }
  return ids;
}

} // namespace

// Such IDs follow no line through the stream, but most adjacent rows step by
// one, so a step frame turns each millisecond into a run of one residual. The
// residuals and the values are both encoded and the smaller kept, so the
// stream is never larger for the frame, and where the frame is kept every read
// path adds it back.
TEST(SubIntSplitEncodingTests, stepFrameRoundTripsTimestampCounterIds) {
  using nimble::subintsplit::fitRowFrame;
  using nimble::subintsplit::fitStepFrame;
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTimestampCounterIds(65'536, 5);
  EXPECT_FALSE(fitRowFrame(std::span<const uint64_t>(ids)).active());
  const auto fitted = fitStepFrame(std::span<const uint64_t>(ids));
  EXPECT_EQ(fitted.slope, 1u);
  EXPECT_EQ(fitted.base, 0u);

  {
    nimble::Encoding::Options options;
    const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<ExtendedSubIntSplitPolicy<uint64_t>>(),
        ids,
        buffer,
        options);
    nimble::Encoding::Options withoutFrame = options;
    withoutFrame.subIntSplit.rowFrame = false;
    const auto unframed = nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<ExtendedSubIntSplitPolicy<uint64_t>>(),
        ids,
        buffer,
        withoutFrame);
    EXPECT_LE(encoded.size(), unframed.size());
    const auto frame = parseRowFrame(encoded);
    EXPECT_TRUE(!frame.active() || (frame.slope == 1u && frame.base == 0u));
    EXPECT_EQ(frame.active(), encoded.size() < unframed.size());
    expectBitwiseEqual(ids, decodeAll<uint64_t>(encoded, *pool));

    auto encoding = decodeEncoding<uint64_t>(encoded, *pool);
    uint64_t row = 0;
    for (const auto& [skipRows, readRows] :
         std::vector<std::pair<uint32_t, uint32_t>>{
             {4'095, 3}, {1, 9'000}, {7'777, 1}, {0, 44'659}}) {
      SCOPED_TRACE(fmt::format("row={} read={}", row + skipRows, readRows));
      encoding->skip(skipRows);
      row += skipRows;
      std::vector<uint64_t> chunk(readRows);
      encoding->materialize(readRows, chunk.data());
      expectBitwiseEqual(
          std::vector<uint64_t>(
              ids.begin() + row, ids.begin() + row + readRows),
          chunk);
      row += readRows;
    }
    ASSERT_EQ(row, ids.size());
  }

  // No common step: hashes, and sorted values with uneven gaps.
  std::mt19937_64 generator{13};
  std::vector<uint64_t> hashes(40'000);
  std::vector<uint64_t> unevenSorted(40'000);
  uint64_t sorted = 0;
  for (size_t row = 0; row < hashes.size(); ++row) {
    hashes[row] = generator();
    sorted += generator() % 100'000;
    unevenSorted[row] = sorted;
  }
  EXPECT_FALSE(fitStepFrame(std::span<const uint64_t>(hashes)).active());
  EXPECT_FALSE(fitStepFrame(std::span<const uint64_t>(unevenSorted)).active());
}

// The forced ablation keeps a fitted frame without pricing it, so it is never
// smaller than the chosen stream, always carries the frame where one fits, and
// reads back like any framed stream. Where nothing fits it has no frame to
// force and matches the unframed stream.
TEST(SubIntSplitEncodingTests, forcedRowFrameIsKeptWhereItFits) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encode = [&](const std::vector<uint64_t>& values,
                          const nimble::Encoding::Options& options) {
    return nimble::EncodingFactory::encode<uint64_t>(
        std::make_unique<ExtendedSubIntSplitPolicy<uint64_t>>(),
        values,
        buffer,
        options);
  };
  nimble::Encoding::Options forced;
  forced.subIntSplit.rowFrameForceApply = true;

  const auto ids = makeTimestampCounterIds(65'536, 5);
  const auto chosen = encode(ids, nimble::Encoding::Options{});
  const auto kept = encode(ids, forced);
  EXPECT_TRUE(parseRowFrame(kept).active());
  EXPECT_GE(kept.size(), chosen.size());
  expectBitwiseEqual(ids, decodeAll<uint64_t>(kept, *pool));

  std::vector<uint64_t> noise(65'536);
  uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (auto& value : noise) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    value = state >> 20;
  }
  nimble::Encoding::Options unframed;
  unframed.subIntSplit.rowFrame = false;
  const auto forcedNoise = encode(noise, forced);
  EXPECT_FALSE(parseRowFrame(forcedNoise).active());
  EXPECT_EQ(forcedNoise.size(), encode(noise, unframed).size());
  expectBitwiseEqual(noise, decodeAll<uint64_t>(forcedNoise, *pool));
}

// Columns that do not follow a line must not be charged a planner pass, and
// their streams must stay byte-identical to ones written without frames.
TEST(SubIntSplitEncodingTests, rowFrameIsNotFittedWithoutALine) {
  using nimble::subintsplit::fitRowFrame;
  std::mt19937_64 generator{11};
  const uint32_t numRows = 40'000;

  std::vector<uint64_t> hashes(numRows);
  std::vector<uint64_t> unevenSorted(numRows);
  std::vector<uint64_t> wrappingCounters(numRows);
  uint64_t sorted = 0;
  for (uint32_t row = 0; row < numRows; ++row) {
    hashes[row] = generator();
    sorted += generator() % 100'000;
    unevenSorted[row] = sorted;
    // Counters that wrap within a few strides grow by no single multiple.
    wrappingCounters[row] =
        0x1234'5678'9000'0000ULL | (((row / 64) & 0x3F) << 10) | (row & 0x3FF);
  }
  EXPECT_FALSE(fitRowFrame(std::span<const uint64_t>(hashes)).active());
  EXPECT_FALSE(fitRowFrame(std::span<const uint64_t>(unevenSorted)).active());
  EXPECT_FALSE(
      fitRowFrame(std::span<const uint64_t>(wrappingCounters)).active());

  // Too few strides to fit on, however straight the line.
  const auto shortIds = makeTreeIds(10'000, 7);
  EXPECT_FALSE(fitRowFrame(std::span<const uint64_t>(shortIds)).active());

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  nimble::Encoding::Options withoutFrame;
  withoutFrame.subIntSplit.rowFrame = false;
  const auto encoded = nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
      unevenSorted,
      buffer);
  const std::string withDefault{encoded};
  const auto unframed = nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
      unevenSorted,
      buffer,
      withoutFrame);
  EXPECT_EQ(withDefault, std::string{unframed});
}

// The frame is kept on the planner's estimate, so that estimate has to agree
// in direction with what the encoder then stores.
TEST(SubIntSplitEncodingTests, rowFrameEstimateBoundsEncodedSize) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTreeIds(60'000, 19);
  const auto frame =
      nimble::subintsplit::fitRowFrame(std::span<const uint64_t>(ids));
  ASSERT_TRUE(frame.active());
  std::vector<uint64_t> residuals;
  nimble::subintsplit::subtractRowFrame(
      frame, std::span<const uint64_t>(ids), residuals);

  const auto estimateBits = [](const std::vector<uint64_t>& values) {
    std::vector<uint64_t> sample;
    nimble::subintsplit::sampleIntoU64<uint64_t>(
        values, sample, nimble::subintsplit::defaultSamplerConfig());
    return nimble::subintsplit::selectSplitsRestricted(
               sample, 64, values.size(), {})
        .totalSizeBits;
  };
  const double residualBits = estimateBits(residuals);
  const double valueBits = estimateBits(ids);
  EXPECT_LT(residualBits, valueBits);

  nimble::Encoding::Options withoutFrame;
  withoutFrame.subIntSplit.rowFrame = false;
  const size_t framedBytes =
      nimble::EncodingFactory::encode<uint64_t>(
          std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
          ids,
          buffer)
          .size();
  const size_t unframedBytes =
      nimble::EncodingFactory::encode<uint64_t>(
          std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
          ids,
          buffer,
          withoutFrame)
          .size();
  EXPECT_LT(framedBytes, unframedBytes);
  // Each estimate is within a factor of two of the stream it predicts, which
  // is loose on purpose: the check is that the comparison the encoder makes is
  // a comparison of like with like, not that the DP is calibrated.
  EXPECT_LT(residualBits / 8.0, 2.0 * framedBytes);
  EXPECT_GT(residualBits / 8.0, 0.5 * framedBytes);
  EXPECT_LT(valueBits / 8.0, 2.0 * unframedBytes);
  EXPECT_GT(valueBits / 8.0, 0.5 * unframedBytes);
}

// A replay reproduces the captured stream's frame decision rather than making
// its own: the boundaries it replays were planned on either the residuals or
// the values, and taking the other would store the column under a plan nobody
// priced.
TEST(SubIntSplitEncodingTests, rowFrameReplayFollowsCapturedLayout) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTreeIds(40'000, 7);
  const auto frameKey = std::string(nimble::subintsplit::kRowFrameConfigKey);

  const std::string framed{
      encodeWithNonRecursiveSubIntSplit<uint64_t>(ids, buffer)};
  ASSERT_TRUE(parseRowFrame(framed).active());
  const auto framedLayout = nimble::EncodingLayoutCapture::capture(
      framed, nimble::Encoding::Options{});
  ASSERT_TRUE(framedLayout.config().get(frameKey).has_value());

  const std::string replayed{
      encodeWithReplayLayout<uint64_t>(framedLayout, ids, buffer)};
  EXPECT_TRUE(parseRowFrame(replayed).active());
  expectBitwiseEqual(ids, decodeAll<uint64_t>(replayed, *pool));
  expectSameLayout(
      framedLayout,
      nimble::EncodingLayoutCapture::capture(
          replayed, nimble::Encoding::Options{}));

  // Replaying the captured layout stores exactly what the capture did: the
  // section layouts it replays were chosen for the residuals, so a replay
  // that dropped the frame would hand them values they were never chosen for.
  EXPECT_EQ(replayed, framed);

  // A stream written without a frame captures none, so its replay stays
  // without one even though the frame option is on and one would fit.
  nimble::Encoding::Options withoutFrame;
  withoutFrame.subIntSplit.rowFrame = false;
  const std::string unframed{nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
      ids,
      buffer,
      withoutFrame)};
  ASSERT_FALSE(parseRowFrame(unframed).active());
  const auto unframedLayout = nimble::EncodingLayoutCapture::capture(
      unframed, nimble::Encoding::Options{});
  EXPECT_FALSE(unframedLayout.config().get(frameKey).has_value());
  const std::string replayedUnframed{
      encodeWithReplayLayout<uint64_t>(unframedLayout, ids, buffer)};
  EXPECT_FALSE(parseRowFrame(replayedUnframed).active());
  expectBitwiseEqual(ids, decodeAll<uint64_t>(replayedUnframed, *pool));
}

// A reader from before frames must refuse a framed stream rather than return
// residuals. A framed stream keeps the SubIntSplit encoding type, so the
// factory hands it to such a reader, which rejects any flag outside
// kFlagDelta. A reader that skips the flag byte reads the frame's guard as the
// first section's bitStart instead, which is always zero in a stream it can
// read.
TEST(SubIntSplitEncodingTests, rowFrameIsRejectedByPreFrameReaders) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTreeIds(40'000, 7);
  const std::string encoded{
      encodeWithNonRecursiveSubIntSplit<uint64_t>(ids, buffer)};
  ASSERT_TRUE(parseRowFrame(encoded).active());
  ASSERT_EQ(
      static_cast<nimble::EncodingType>(
          encoded[nimble::EncodingPrefix::kEncodingTypeOffset]),
      nimble::EncodingType::SubIntSplit);

  const size_t flagsOffset = nimble::Encoding::kPrefixSize + 1;
  const auto flags = static_cast<uint8_t>(encoded[flagsOffset]);
  EXPECT_EQ(flags, nimble::subintsplit::kFlagRowFrame);
  EXPECT_NE(flags & ~nimble::subintsplit::kFlagDelta, 0);
  EXPECT_EQ(
      static_cast<uint8_t>(encoded[flagsOffset + 1]),
      nimble::subintsplit::kRowFrameGuard);
  EXPECT_NE(nimble::subintsplit::kRowFrameGuard, 0);
}

// The flag byte and the frame's guard come off the wire, so a reader has to
// refuse values it does not know how to reconstruct.
TEST(SubIntSplitEncodingTests, rowFrameHeaderCorruptionThrows) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto ids = makeTreeIds(40'000, 7);
  const std::string encoded{nimble::EncodingFactory::encode<uint64_t>(
      std::make_unique<NonRecursiveSubIntSplitPolicy<uint64_t>>(),
      ids,
      buffer)};
  ASSERT_TRUE(parseRowFrame(encoded).active());
  const size_t flagsOffset = nimble::Encoding::kPrefixSize + 1;

  auto unknownFlag = encoded;
  unknownFlag[flagsOffset] = static_cast<char>(unknownFlag[flagsOffset] | 0x80);
  EXPECT_THROW(
      decodeAll<uint64_t>(unknownFlag, *pool), nimble::NimbleException);

  auto badGuard = encoded;
  badGuard[flagsOffset + 1] = 0;
  EXPECT_THROW(decodeAll<uint64_t>(badGuard, *pool), nimble::NimbleException);
}

TYPED_TEST(SubIntSplitEncodingTest, preserveRoundTripExplicitBoundaries) {
  using T = TypeParam;
  const auto values = makeStructuredValues<T>();
  const auto segments = makePreserveSegments<T>();
  const auto layout = makePreserveLayout<T>(segments);

  const auto encoded =
      encodeWithReplayLayout<T>(layout, values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});

  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  ASSERT_EQ(captured.childrenCount(), segments.size());

  const auto capturedMode = captured.config().get(
      std::string(nimble::subintsplit::kSplitModeConfigKey));
  ASSERT_TRUE(capturedMode.has_value());
  EXPECT_EQ(*capturedMode, nimble::subintsplit::kSplitModePreserve);

  const auto capturedBoundaries = captured.config().get(
      std::string(nimble::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(capturedBoundaries.has_value());
  EXPECT_EQ(
      *capturedBoundaries,
      nimble::subintsplit::serializeSplitBoundaries(segments));

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);
}

TEST(SubIntSplitEncodingTests, preserveModeRequiresBoundaries) {
  const std::vector<int64_t> values{
      0x1234567890000000LL, 0x1234567890000001LL, 0x1234567890000002LL};

  nimble::EncodingLayout layout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{{
          {std::string(nimble::subintsplit::kSplitModeConfigKey),
           std::string(nimble::subintsplit::kSplitModePreserve)},
      }},
      nimble::CompressionType::Uncompressed};

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  auto leafCreator = makeLeafPolicyCreator();
  EXPECT_THROW(
      (nimble::EncodingFactory::encode<int64_t>(
          std::make_unique<nimble::ReplayedEncodingSelectionPolicy<int64_t>>(
              layout, std::nullopt, leafCreator),
          values,
          buffer)),
      nimble::NimbleInternalError);
}

// Every header field is read straight off the wire, so a corrupt or truncated
// stream arrives as arbitrary bytes. Each of these used to walk off the end of
// the buffer, resize a vector by an attacker-chosen count, or shift by a
// negative width, none of which reports anything.
TEST(SubIntSplitEncodingTests, truncatedStreamThrowsRatherThanReadingPastEnd) {
  const std::vector<uint64_t> values{
      0x1234567890000000ULL,
      0x1234567890000001ULL,
      0x1234567890000002ULL,
      0x1234567890000003ULL};
  const auto segments = makeFullWidthSegments<uint64_t>();
  const auto layout = makePreserveLayout<uint64_t>(segments);

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(layout, values, buffer);
  ASSERT_GT(encoded.size(), 1u);

  // Every prefix of a valid stream is invalid, and none of them may be read
  // past. Which check rejects a given prefix is not the point; that one does
  // is.
  for (size_t length = 1; length < encoded.size(); ++length) {
    const std::string_view truncated{encoded.data(), length};
    EXPECT_THROW(decodeAll<uint64_t>(truncated, *pool), nimble::NimbleException)
        << "truncated to " << length << " of " << encoded.size() << " bytes";
  }
}

TEST(SubIntSplitEncodingTests, invalidSectionBitRangeThrows) {
  const std::vector<uint64_t> values{
      0x1234567890000000ULL,
      0x1234567890000001ULL,
      0x1234567890000002ULL,
      0x1234567890000003ULL};
  const auto segments = makeFullWidthSegments<uint64_t>();
  const auto layout = makePreserveLayout<uint64_t>(segments);

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(layout, values, buffer);
  std::string corrupted{encoded};

  // Find the section header by its own contents rather than by hardcoding the
  // prefix size: the first section of a full-width split starts at bit 0 and
  // ends at bit 63, and that pair appears nowhere before it.
  size_t headerPos = std::string::npos;
  for (size_t i = 0; i + 1 < corrupted.size(); ++i) {
    if (static_cast<uint8_t>(corrupted[i]) == 0 &&
        static_cast<uint8_t>(corrupted[i + 1]) == 63) {
      headerPos = i;
      break;
    }
  }
  ASSERT_NE(headerPos, std::string::npos);

  // bitEnd below bitStart gives a negative width, which shifts by a negative
  // amount when the mask is built.
  corrupted[headerPos] = static_cast<char>(40);
  corrupted[headerPos + 1] = static_cast<char>(8);
  EXPECT_THROW(decodeAll<uint64_t>(corrupted, *pool), nimble::NimbleException);

  // A bit index past the width of the value is equally unrepresentable.
  corrupted[headerPos] = static_cast<char>(0);
  corrupted[headerPos + 1] = static_cast<char>(200);
  EXPECT_THROW(decodeAll<uint64_t>(corrupted, *pool), nimble::NimbleException);
}

namespace {

// A valid two-section stream of `values`, split at bit 32 of a 64-bit value,
// with the offset of its SubIntSplit header. The values carry no row frame or
// transform, so the section triples follow the two header bytes directly.
std::pair<std::string, uint32_t> twoSectionStream(
    velox::memory::MemoryPool& pool) {
  std::vector<uint64_t> values(64);
  std::mt19937_64 generator{5};
  for (auto& value : values) {
    value = generator();
  }
  const auto layout = makePreserveLayout<uint64_t>({{0, 31}, {32, 63}});
  nimble::Buffer buffer{pool};
  const std::string encoded{
      encodeWithReplayLayout<uint64_t>(layout, values, buffer)};
  const uint32_t dataOffset =
      decodeEncoding<uint64_t>(encoded, pool)->dataOffset();
  return {encoded, dataOffset};
}

} // namespace

// A section covers at least one bit, so a 32-bit stream can hold at most 32
// sections whatever the 64-bit limit allows.
TEST(SubIntSplitEncodingTests, moreSectionsThanValueBitsThrows) {
  std::vector<uint32_t> values(64);
  std::iota(values.begin(), values.end(), 7u);
  const auto layout =
      makePreserveLayout<uint32_t>(makeFullWidthSegments<uint32_t>());
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  std::string corrupted{
      encodeWithReplayLayout<uint32_t>(layout, values, buffer)};
  const uint32_t dataOffset =
      decodeEncoding<uint32_t>(corrupted, *pool)->dataOffset();
  corrupted[dataOffset] = static_cast<char>(33);
  EXPECT_THROW(decodeAll<uint32_t>(corrupted, *pool), nimble::NimbleException);
}

// Values are reassembled by OR-ing each section in at its bit offset, so a
// gap, an overlap or a top bit no section covers returns wrong values with
// no error unless the header is rejected.
TEST(SubIntSplitEncodingTests, sectionsThatDoNotTileTheValueThrow) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto [encoded, dataOffset] = twoSectionStream(*pool);
  ASSERT_EQ(static_cast<uint8_t>(encoded[dataOffset]), 2);
  ASSERT_EQ(static_cast<uint8_t>(encoded[dataOffset + 1]), 0);
  EXPECT_NO_THROW(decodeAll<uint64_t>(encoded, *pool));

  const size_t first = dataOffset + nimble::subintsplit::kStreamHeaderSize;
  const size_t second = first + nimble::subintsplit::kSectionHeaderSize;
  const auto expectRejected = [&](size_t pos, uint8_t bit) {
    std::string corrupted{encoded};
    corrupted[pos] = static_cast<char>(bit);
    EXPECT_THROW(decodeAll<uint64_t>(corrupted, *pool), nimble::NimbleException)
        << "byte " << pos << " set to " << static_cast<int>(bit);
  };
  expectRejected(second, 33); // gap at bit 32
  expectRejected(second, 31); // overlap at bit 31
  expectRejected(second + 1, 62); // bit 63 uncovered
  expectRejected(first, 1); // bit 0 uncovered
}

// The payloads are read back to back from the section sizes, so bytes the
// sizes do not account for mean the header and the stream disagree.
TEST(SubIntSplitEncodingTests, trailingBytesAfterTheSectionsThrow) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const auto [encoded, dataOffset] = twoSectionStream(*pool);
  std::string padded{encoded};
  padded.push_back('\0');
  EXPECT_THROW(decodeAll<uint64_t>(padded, *pool), nimble::NimbleException);
}

TEST(SubIntSplitEncodingTests, fullWidthSingleSectionRoundTrip) {
  const std::vector<uint64_t> values{
      0x1234567890000000ULL,
      0x1234567890000001ULL,
      0x1234567890000002ULL,
      0x1234567890000003ULL};
  const auto segments = makeFullWidthSegments<uint64_t>();
  const auto layout = makePreserveLayout<uint64_t>(segments);

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(layout, values, buffer);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});

  ASSERT_EQ(captured.childrenCount(), 1u);
  const auto capturedBoundaries = captured.config().get(
      std::string(nimble::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(capturedBoundaries.has_value());
  EXPECT_EQ(
      *capturedBoundaries,
      nimble::subintsplit::serializeSplitBoundaries(segments));

  const auto decoded = decodeAll<uint64_t>(encoded, *pool);
  expectBitwiseEqual(values, decoded);
}

TEST(SubIntSplitEncodingTests, heterogeneousSectionEncodingsRoundTrip) {
  const std::vector<nimble::subintsplit::SectionPlan> segments{
      {0, 7}, {8, 15}, {16, 31}, {32, 63}};
  const nimble::EncodingLayout layout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{
          nimble::subintsplit::makePreserveSplitConfig(segments)},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::FixedBitWidth,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::Dictionary,
              {},
              nimble::CompressionType::Uncompressed,
              {
                  nimble::EncodingLayout{
                      nimble::EncodingType::Trivial,
                      {},
                      nimble::CompressionType::Uncompressed},
                  nimble::EncodingLayout{
                      nimble::EncodingType::FixedBitWidth,
                      {},
                      nimble::CompressionType::Uncompressed},
              }},
          nimble::EncodingLayout{
              nimble::EncodingType::Constant,
              {},
              nimble::CompressionType::Uncompressed},
      }};

  std::vector<uint64_t> values;
  values.reserve(513);
  for (uint64_t i = 0; i < 513; ++i) {
    const uint64_t low = i & 0xff;
    const uint64_t middle = ((i * 7) % 31) << 8;
    const uint64_t dictionary = ((i % 4) * 10'000) << 16;
    values.push_back((uint64_t{0x12345678} << 32) | dictionary | middle | low);
  }

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(layout, values, buffer);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});

  ASSERT_EQ(captured.childrenCount(), segments.size());
  EXPECT_EQ(captured.child(0)->encodingType(), nimble::EncodingType::Trivial);
  EXPECT_EQ(
      captured.child(1)->encodingType(), nimble::EncodingType::FixedBitWidth);
  EXPECT_EQ(
      captured.child(2)->encodingType(), nimble::EncodingType::Dictionary);
  EXPECT_EQ(captured.child(3)->encodingType(), nimble::EncodingType::Constant);

  nimble::Encoding::Options options;
  options.subIntSplitDecodeChunkSize = 7;
  auto decoder = decodeEncoding<uint64_t>(encoded, *pool, options);
  std::vector<uint64_t> decoded(values.size());
  size_t offset{0};
  for (const uint32_t count : {1, 6, 8, 63, 128, 307}) {
    decoder->materialize(count, decoded.data() + offset);
    offset += count;
  }
  EXPECT_EQ(offset, values.size());
  EXPECT_EQ(decoded, values);

  decoder->reset();
  decoder->skip(257);
  std::vector<uint64_t> suffix(values.size() - 257);
  decoder->materialize(suffix.size(), suffix.data());
  EXPECT_EQ(suffix, std::vector<uint64_t>(values.begin() + 257, values.end()));
}

TEST(SubIntSplitEncodingTests, rejectsMalformedStreamHeader) {
  const std::vector<uint64_t> values{
      0x1234567890000000ULL,
      0x1234567890000001ULL,
      0x1234567890000002ULL,
      0x1234567890000003ULL};

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto segments = makePreserveSegments<uint64_t>();
  const auto encoded = encodeWithReplayLayout<uint64_t>(
      makePreserveLayout<uint64_t>(segments), values, buffer);
  const auto headerOffset = nimble::EncodingPrefix::kFixedPrefixSize;
  const auto sectionHeadersOffset =
      headerOffset + nimble::subintsplit::kStreamHeaderSize;
  const auto expectMalformed = [&](std::string malformed,
                                   std::string_view expectedMessage) {
    NIMBLE_ASSERT_FILE_THROW(
        decodeEncoding<uint64_t>(malformed, *pool), expectedMessage);
  };

  {
    std::string malformed{encoded.substr(0, headerOffset + 1)};
    expectMalformed(malformed, "SubIntSplit stream header is truncated.");
  }
  {
    std::string malformed{encoded};
    malformed[headerOffset] = 0;
    expectMalformed(
        malformed, "SubIntSplit stream must contain at least one section.");
  }
  {
    std::string malformed{encoded};
    malformed[headerOffset] = 65;
    expectMalformed(malformed, "SubIntSplit stream has too many sections.");
  }
  {
    std::string malformed{encoded};
    malformed[headerOffset + 1] = static_cast<char>(0x80);
    expectMalformed(malformed, "SubIntSplit stream has unsupported flags.");
  }
  {
    std::string malformed{encoded};
    malformed[headerOffset + 2] = 1;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded.substr(
        0,
        sectionHeadersOffset +
            segments.size() * nimble::subintsplit::kSectionHeaderSize - 1)};
    expectMalformed(malformed, "SubIntSplit section headers are truncated.");
  }
  {
    std::string malformed{encoded};
    malformed[sectionHeadersOffset + nimble::subintsplit::kSectionHeaderSize] =
        9;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded};
    malformed[sectionHeadersOffset + nimble::subintsplit::kSectionHeaderSize] =
        7;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded};
    const auto secondHeader =
        sectionHeadersOffset + nimble::subintsplit::kSectionHeaderSize;
    malformed[secondHeader + 1] = 7;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded};
    const auto lastHeader = sectionHeadersOffset +
        (segments.size() - 1) * nimble::subintsplit::kSectionHeaderSize;
    malformed[lastHeader + 1] = 62;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded};
    const auto lastHeader = sectionHeadersOffset +
        (segments.size() - 1) * nimble::subintsplit::kSectionHeaderSize;
    malformed[lastHeader + 1] = 64;
    expectMalformed(
        malformed,
        "SubIntSplit sections must cover the value bits once in order.");
  }
  {
    std::string malformed{encoded.substr(0, encoded.size() - 1)};
    expectMalformed(
        malformed,
        "SubIntSplit section payload sizes do not match the stream.");
  }
}

TEST(SubIntSplitEncodingTests, largeDecodeChunkSizePreservesCorrectness) {
  std::vector<uint64_t> values;
  values.reserve(100);
  for (uint64_t i = 0; i < 100; ++i) {
    values.push_back(((i + 1) << 56) | (i * 17));
  }
  const std::vector<nimble::subintsplit::SectionPlan> segments{
      {0, 31}, {32, 63}};

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(
      makePreserveLayout<uint64_t>(segments), values, buffer);
  nimble::Encoding::Options options;
  options.subIntSplitDecodeChunkSize = (uint32_t{1} << 29) + 1;
  auto decoder = decodeEncoding<uint64_t>(encoded, *pool, options);

  std::vector<uint64_t> actual(values.size());
  decoder->materialize(actual.size(), actual.data());
  EXPECT_EQ(actual, values);
}

TEST(SubIntSplitEncodingTests, rejectsMismatchedSectionDataType) {
  const std::vector<uint64_t> values{
      0x1234567890000000ULL,
      0x1234567890000001ULL,
      0x1234567890000002ULL,
      0x1234567890000003ULL};
  const auto segments = makePreserveSegments<uint64_t>();

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};
  const auto encoded = encodeWithReplayLayout<uint64_t>(
      makePreserveLayout<uint64_t>(segments), values, buffer);
  std::string malformed{encoded};

  const char* headerPosition = malformed.data() +
      nimble::EncodingPrefix::kFixedPrefixSize +
      nimble::subintsplit::kStreamHeaderSize;
  size_t payloadOffset = nimble::EncodingPrefix::kFixedPrefixSize +
      nimble::subintsplit::specificHeaderSize(segments.size());
  for (size_t i = 0; i + 1 < segments.size(); ++i) {
    payloadOffset +=
        nimble::subintsplit::readSectionHeader(headerPosition).encodedSize;
  }
  malformed[payloadOffset + nimble::EncodingPrefix::kDataTypeOffset] =
      static_cast<char>(nimble::DataType::Uint16);

  NIMBLE_ASSERT_FILE_THROW(
      decodeEncoding<uint64_t>(malformed, *pool),
      "SubIntSplit section data type does not match its bit width.");
}

TEST(
    SubIntSplitEncodingTests,
    CreateImplExtendsCandidatesForSubIntSplitChildren) {
  nimble::ManualEncodingSelectionPolicy<uint64_t> policy{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      std::nullopt,
      std::nullopt};

  auto containsType =
      [](const std::vector<std::pair<nimble::EncodingType, float>>& factors,
         nimble::EncodingType type) {
        return std::any_of(
            factors.begin(), factors.end(), [type](const auto& pair) {
              return pair.first == type;
            });
      };

  // Direct children of a SubIntSplit node get the extended candidate list.
  auto subIntSplitChild =
      policy.create<uint64_t>(nimble::EncodingType::SubIntSplit, 0);
  auto* subIntSplitChildPolicy =
      dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint64_t>*>(
          subIntSplitChild.get());
  ASSERT_NE(subIntSplitChildPolicy, nullptr);
  const auto& extendedFactors =
      subIntSplitChildPolicy->candidateEncodingReadFactors();
  EXPECT_TRUE(containsType(extendedFactors, nimble::EncodingType::PFOR));
  EXPECT_TRUE(
      containsType(extendedFactors, nimble::EncodingType::SimdForBitpack));
  EXPECT_TRUE(
      containsType(extendedFactors, nimble::EncodingType::BlockBitPacking));
  EXPECT_TRUE(containsType(extendedFactors, nimble::EncodingType::Delta));
  EXPECT_TRUE(containsType(extendedFactors, nimble::EncodingType::FOR));
  EXPECT_TRUE(
      containsType(extendedFactors, nimble::EncodingType::FrequencyPartition));

  // Children of a non-SubIntSplit node do not get the extended list.
  auto pforChild = policy.create<uint64_t>(nimble::EncodingType::PFOR, 0);
  auto* pforChildPolicy =
      dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint64_t>*>(
          pforChild.get());
  ASSERT_NE(pforChildPolicy, nullptr);
  const auto& unextendedFactors =
      pforChildPolicy->candidateEncodingReadFactors();
  EXPECT_FALSE(containsType(unextendedFactors, nimble::EncodingType::PFOR));
  EXPECT_FALSE(
      containsType(unextendedFactors, nimble::EncodingType::SimdForBitpack));
  EXPECT_FALSE(
      containsType(unextendedFactors, nimble::EncodingType::BlockBitPacking));
  EXPECT_FALSE(containsType(unextendedFactors, nimble::EncodingType::Delta));
  EXPECT_FALSE(containsType(unextendedFactors, nimble::EncodingType::FOR));
  EXPECT_FALSE(containsType(
      unextendedFactors, nimble::EncodingType::FrequencyPartition));
}

TYPED_TEST(SubIntSplitEncodingTest, ExtendedCandidatesRoundTrip) {
  using T = TypeParam;
  const auto values = makeStructuredValues<T>();

  const auto encoded = encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);
}

// Recursion-sanity test for Delta/FOR as SubIntSplit segment candidates
// (B.4): a wide-range, constant-step monotonic column should round-trip
// bit-exactly and produce a bounded encoding size, regardless of which
// candidate (Delta, FOR, or another) each segment's encodeNested ends up
// selecting.
TYPED_TEST(SubIntSplitEncodingTest, WideRangeMonotonicRoundTrip) {
  using T = TypeParam;
  const auto values = makeWideRangeMonotonicValues<T>();

  const auto encoded = encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);

  // Bounded size: the recursive candidate set must not blow up the encoding
  // beyond a small multiple of the raw data size.
  const size_t rawSize = values.size() * sizeof(T);
  EXPECT_LE(encoded.size(), rawSize * 4 + 1024);
}

// Zipfian-distributed data (FrequencyPartition as SubIntSplit segment
// candidate, Phase C): a column with a dominant value (50% frequency) and a
// long tail should round-trip bit-exactly through SubIntSplit with the
// extended candidate set. The encoding size bound verifies that the
// PerTierBitmaps index override in SubIntSplitEncoding::encode() does not
// inflate the output beyond a reasonable multiple of the raw data.
TYPED_TEST(SubIntSplitEncodingTest, ZipfianRoundTrip) {
  using T = TypeParam;
  const auto values = makeZipfianValues<T>();

  const auto encoded = encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);

  // Bounded size: FrequencyPartition with PerTierBitmaps adds index overhead
  // but should still compress better than storing raw values.
  const size_t rawSize = values.size() * sizeof(T);
  EXPECT_LE(encoded.size(), rawSize * 4 + 1024);
}

#endif
