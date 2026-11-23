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
  const auto values = makeStructuredValues<T>();

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

  auto tuning = nimble::subintsplit::kDefaultTuningConfig;
  tuning.decodeChunkSize = 7;
  auto decoder = std::make_unique<nimble::SubIntSplitEncoding<uint64_t>>(
      *pool, encoded, nullptr, nimble::Encoding::Options{}, tuning);
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
  auto tuning = nimble::subintsplit::kDefaultTuningConfig;
  tuning.decodeChunkSize = (uint32_t{1} << 29) + 1;
  auto decoder = std::make_unique<nimble::SubIntSplitEncoding<uint64_t>>(
      *pool, encoded, nullptr, nimble::Encoding::Options{}, tuning);

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

#endif
