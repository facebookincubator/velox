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
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

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
  push(0, 3, 64);  // 128 values
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
std::vector<nimble::detail::subintsplit::SegmentPlan> makePreserveSegments() {
  if constexpr (sizeof(PhysicalType<T>) == 4) {
    return {{0, 7}, {8, 15}, {16, 31}};
  } else {
    return {{0, 7}, {8, 15}, {16, 31}, {32, 63}};
  }
}

template <typename T>
std::vector<nimble::detail::subintsplit::SegmentPlan> makeFullWidthSegments() {
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
      nimble::EncodingType encodingType,
      nimble::NestedEncodingIdentifier identifier,
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
    nimble::ManualEncodingSelectionPolicy<T> delegate{
        std::move(readFactors), std::nullopt, std::nullopt};
    return delegate.createImpl(encodingType, identifier, type);
  }
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
    velox::memory::MemoryPool& pool) {
  return nimble::EncodingFactory().create(
      pool, encoded, [](uint32_t) { return nullptr; });
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
    const std::vector<nimble::detail::subintsplit::SegmentPlan>& segments) {
  std::vector<std::optional<const nimble::EncodingLayout>> children(
      segments.size());
  return nimble::EncodingLayout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{
          nimble::detail::subintsplit::makePreserveSplitConfig(segments)},
      nimble::CompressionType::Uncompressed,
      std::move(children)};
}

void expectSegmentsEqual(
    const std::vector<nimble::detail::subintsplit::SegmentPlan>& expected,
    const std::vector<nimble::detail::subintsplit::SegmentPlan>& actual) {
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

} // namespace

TEST(SubIntSplitConfigTests, BoundarySerializationAndParsing) {
  const std::vector<nimble::detail::subintsplit::SegmentPlan> segments{
      {.bitStart = 0, .bitEnd = 7},
      {.bitStart = 8, .bitEnd = 15},
      {.bitStart = 16, .bitEnd = 31}};

  const auto serialized =
      nimble::detail::subintsplit::serializeSplitBoundaries(segments);
  EXPECT_EQ(serialized, "0-7;8-15;16-31");

  auto parsed =
      nimble::detail::subintsplit::parseSplitBoundaries(serialized, 32);
  ASSERT_TRUE(parsed.has_value());
  expectSegmentsEqual(segments, *parsed);

  EXPECT_FALSE(
      nimble::detail::subintsplit::parseSplitBoundaries("", 32).has_value());
  EXPECT_FALSE(
      nimble::detail::subintsplit::parseSplitBoundaries("0-7;9-15", 16)
          .has_value());
  EXPECT_FALSE(
      nimble::detail::subintsplit::parseSplitBoundaries("0-7;8-16", 16)
          .has_value());
  EXPECT_FALSE(
      nimble::detail::subintsplit::parseSplitBoundaries("0-7;8-15", 8)
          .has_value());
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

TYPED_TEST(SubIntSplitEncodingTest, RecomputeRoundTripAndReplay) {
  using T = TypeParam;
  const auto values = makeStructuredValues<T>();

  const auto encoded =
      encodeWithNonRecursiveSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(
      encoded, nimble::Encoding::Options{});

  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);
  ASSERT_GT(captured.childrenCount(), 1u);

  const auto capturedMode = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitModeConfigKey));
  ASSERT_TRUE(capturedMode.has_value());
  EXPECT_EQ(*capturedMode, nimble::detail::subintsplit::kSplitModePreserve);

  const auto capturedBoundaries = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
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

TYPED_TEST(SubIntSplitEncodingTest, PreserveRoundTripExplicitBoundaries) {
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
      std::string(nimble::detail::subintsplit::kSplitModeConfigKey));
  ASSERT_TRUE(capturedMode.has_value());
  EXPECT_EQ(*capturedMode, nimble::detail::subintsplit::kSplitModePreserve);

  const auto capturedBoundaries = captured.config().get(
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(capturedBoundaries.has_value());
  EXPECT_EQ(
      *capturedBoundaries,
      nimble::detail::subintsplit::serializeSplitBoundaries(segments));

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);
}

TEST(SubIntSplitEncodingTests, PreserveModeRequiresBoundaries) {
  const std::vector<int64_t> values{
      0x1234567890000000LL, 0x1234567890000001LL, 0x1234567890000002LL};

  nimble::EncodingLayout layout{
      nimble::EncodingType::SubIntSplit,
      nimble::EncodingLayout::Config{{
          {std::string(nimble::detail::subintsplit::kSplitModeConfigKey),
           std::string(nimble::detail::subintsplit::kSplitModePreserve)},
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

TEST(SubIntSplitEncodingTests, FullWidthSingleSectionRoundTrip) {
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
      std::string(nimble::detail::subintsplit::kSplitBoundariesConfigKey));
  ASSERT_TRUE(capturedBoundaries.has_value());
  EXPECT_EQ(
      *capturedBoundaries,
      nimble::detail::subintsplit::serializeSplitBoundaries(segments));

  const auto decoded = decodeAll<uint64_t>(encoded, *pool);
  expectBitwiseEqual(values, decoded);
}

TEST(SubIntSplitEncodingTests, CreateImplExtendsCandidatesForSubIntSplitChildren) {
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
  auto subIntSplitChild = policy.createImpl(
      nimble::EncodingType::SubIntSplit, 0, nimble::DataType::Uint64);
  auto* subIntSplitChildPolicy =
      dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint64_t>*>(
          subIntSplitChild.get());
  ASSERT_NE(subIntSplitChildPolicy, nullptr);
  const auto& extendedFactors = subIntSplitChildPolicy->readFactors();
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
  auto pforChild = policy.createImpl(
      nimble::EncodingType::PFOR, 0, nimble::DataType::Uint64);
  auto* pforChildPolicy =
      dynamic_cast<nimble::ManualEncodingSelectionPolicy<uint64_t>*>(
          pforChild.get());
  ASSERT_NE(pforChildPolicy, nullptr);
  const auto& unextendedFactors = pforChildPolicy->readFactors();
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

  const auto encoded =
      encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(encoded);
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

  const auto encoded =
      encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(encoded);
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

  const auto encoded =
      encodeWithExtendedSubIntSplit<T>(values, *this->buffer_);
  const auto captured = nimble::EncodingLayoutCapture::capture(encoded);
  ASSERT_EQ(captured.encodingType(), nimble::EncodingType::SubIntSplit);

  const auto decoded = decodeAll<T>(encoded, *this->pool_);
  expectBitwiseEqual(values, decoded);

  // Bounded size: FrequencyPartition with PerTierBitmaps adds index overhead
  // but should still compress better than storing raw values.
  const size_t rawSize = values.size() * sizeof(T);
  EXPECT_LE(encoded.size(), rawSize * 4 + 1024);
}

#endif
