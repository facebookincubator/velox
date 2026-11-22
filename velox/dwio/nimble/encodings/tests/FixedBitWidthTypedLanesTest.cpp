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

// FixedBitWidth typed lanes (Encoding::Options::fixedBitWidthTypedLanes) at the
// encoding level. Payloads are compressed with OpenZL, which reads their
// DataType, so chunks take lanes. Its compressed bytes are a function of the
// payload, its DataType and the policy's settings, so chunks can be compared
// byte for byte.

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {
namespace {

using ReadFactors = std::vector<std::pair<EncodingType, float>>;

const ReadFactors kFixedBitWidthOnly{{EncodingType::FixedBitWidth, 1.0}};

velox::memory::MemoryPool& leafPool() {
  static const auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  return *pool;
}

// A compressor that reads the payload's DataType, so chunks can take lanes.
CompressionOptions openzlOptions() {
  return CompressionOptions{
      .compressionType = CompressionType::OpenZL,
      .openzlMinCompressionSize = 0,
  };
}

// Byte-oriented compressors: chunks they compress never take lanes.
CompressionOptions zstdOptions() {
  return CompressionOptions{
      .compressionType = CompressionType::Zstd,
      .zstdMinCompressionSize = 0,
  };
}

CompressionOptions lz4Options() {
  return CompressionOptions{
      .compressionType = CompressionType::Lz4,
      .lz4MinCompressionSize = 0,
  };
}

Encoding::Options lanesOn(uint64_t minValues = 256, bool exactBits = false) {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = exactBits;
  options.fixedBitWidthTypedLanes = true;
  options.fixedBitWidthTypedLanesMinValues = minValues;
  return options;
}

Encoding::Options lanesOff(bool exactBits = false) {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = exactBits;
  return options;
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> manualPolicy(
    ReadFactors readFactors = kFixedBitWidthOnly,
    std::optional<CompressionOptions> compressionOptions = openzlOptions(),
    std::optional<ReadFactors> nestedReadFactors = std::nullopt) {
  return std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::move(readFactors),
      std::move(compressionOptions),
      /*identifier=*/std::nullopt,
      std::move(nestedReadFactors));
}

template <typename T>
std::string encode(
    std::unique_ptr<EncodingSelectionPolicy<T>> policy,
    std::span<const T> values,
    const Encoding::Options& options) {
  Buffer buffer{leafPool()};
  return std::string{
      EncodingFactory::encode<T>(std::move(policy), values, buffer, options)};
}

template <typename T>
std::string encodeNullable(
    std::unique_ptr<EncodingSelectionPolicy<T>> policy,
    std::span<const T> values,
    std::span<const bool> nonNulls,
    const Encoding::Options& options) {
  Buffer buffer{leafPool()};
  return std::string{EncodingFactory::encodeNullable<T>(
      std::move(policy), values, nonNulls, buffer, options)};
}

template <typename T>
std::vector<T> decode(std::string_view encoded) {
  auto encoding =
      EncodingFactory().create(leafPool(), encoded, [](uint32_t) -> void* {
        NIMBLE_UNREACHABLE("No strings.");
      });
  std::vector<T> values(encoding->rowCount());
  encoding->materialize(static_cast<uint32_t>(values.size()), values.data());
  return values;
}

template <typename T>
T fromPhysical(uint64_t physical) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  return std::bit_cast<T>(static_cast<PhysicalType>(physical));
}

template <typename T>
constexpr uint64_t physicalBase() {
  if constexpr (std::is_same_v<T, float>) {
    return 0x3F800000;
  } else if constexpr (std::is_same_v<T, double>) {
    return 0x3FF0000000000000;
  } else {
    return 0;
  }
}

// 'count' values whose physical range needs exactly 'bits' bits: the first two
// pin the minimum and the maximum.
template <typename T>
std::vector<T> spreadValues(uint32_t count, uint32_t bits, uint64_t base = 0) {
  const uint64_t maxOffset =
      bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
  std::vector<T> values;
  values.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t offset = i == 0
        ? 0
        : (i == 1 ? maxOffset
                  : (i * uint64_t{7919}) %
                   (maxOffset == ~uint64_t{0} ? maxOffset : maxOffset + 1));
    values.push_back(fromPhysical<T>(physicalBase<T>() + base + offset));
  }
  return values;
}

// 'count' values from 64 distinct ones over 'bits' bits, so they compress.
template <typename T>
std::vector<T> lowEntropyValues(uint32_t count, uint32_t bits, uint64_t base) {
  const uint64_t range = bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits);
  const uint64_t step = (range - 1) / 63;
  std::vector<T> values(count);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = static_cast<T>(base + ((i * 37) % 64) * step);
  }
  values[0] = static_cast<T>(base);
  values[1] = static_cast<T>(base + 63 * step);
  return values;
}

struct BoolVector {
  explicit BoolVector(size_t size)
      : data{std::make_unique<bool[]>(size)}, size{size} {}

  std::span<const bool> span() const {
    return {data.get(), size};
  }

  std::unique_ptr<bool[]> data;
  size_t size;
};

// Every third row null; 'nonNullCount' non-null rows.
BoolVector everyThirdRowNull(size_t nonNullCount) {
  size_t rowCount = 0;
  for (size_t nonNulls = 0; nonNulls < nonNullCount; ++rowCount) {
    if (rowCount % 3 != 2) {
      ++nonNulls;
    }
  }
  BoolVector bools{rowCount};
  for (size_t row = 0; row < rowCount; ++row) {
    bools.data[row] = row % 3 != 2;
  }
  return bools;
}

// A top-level FixedBitWidth chunk, as stored.
struct Chunk {
  EncodingType encodingType;
  uint32_t rowCount{0};
  CompressionType compressionType{CompressionType::Uncompressed};
  uint64_t baseline{0};
  int bitWidth{0};
  // The payload before compression.
  std::string payload;
};

template <typename T>
Chunk readChunk(std::string_view encoded) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  Chunk chunk{
      .encodingType = EncodingPrefix::encodingType(encoded), .payload = {}};
  chunk.rowCount = EncodingPrefix::readRowCount(encoded, false);
  if (chunk.encodingType != EncodingType::FixedBitWidth) {
    return chunk;
  }
  const char* pos = encoded.data() + EncodingPrefix::kFixedPrefixSize;
  chunk.compressionType = static_cast<CompressionType>(*pos++);
  PhysicalType baseline;
  std::memcpy(&baseline, pos, sizeof(baseline));
  chunk.baseline = baseline;
  pos += sizeof(PhysicalType);
  chunk.bitWidth = static_cast<uint8_t>(*pos++);
  const std::string_view stored{
      pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};
  if (chunk.compressionType == CompressionType::Uncompressed) {
    chunk.payload = std::string{stored};
  } else {
    auto raw = Compression::uncompress(
        leafPool(),
        chunk.compressionType,
        DataType::Undefined,
        stored,
        nullptr);
    chunk.payload = std::string{raw->as<char>(), raw->size()};
  }
  return chunk;
}

enum class Layout { kPacked, kNarrow16, kNarrow32, kFullWidth, kOther };

std::string toString(Layout layout) {
  switch (layout) {
    case Layout::kPacked:
      return "packed";
    case Layout::kNarrow16:
      return "narrow16";
    case Layout::kNarrow32:
      return "narrow32";
    case Layout::kFullWidth:
      return "fullWidth";
    case Layout::kOther:
      return "other";
  }
  return "?";
}

// Classifies a top-level chunk by its bytes.
template <typename T>
Layout layoutOf(std::string_view encoded) {
  constexpr int kTypeBits = 8 * sizeof(typename TypeTraits<T>::physicalType);
  const auto chunk = readChunk<T>(encoded);
  if (chunk.encodingType != EncodingType::FixedBitWidth) {
    return Layout::kOther;
  }
  if (chunk.payload.size() ==
      FixedBitArray::bufferSize(chunk.rowCount, chunk.bitWidth)) {
    return Layout::kPacked;
  }
  EXPECT_EQ(
      chunk.payload.size(),
      FixedBitWidthTypedLanes::payloadSize(chunk.rowCount, chunk.bitWidth));
  if (chunk.bitWidth == kTypeBits) {
    EXPECT_EQ(chunk.baseline, 0);
    return Layout::kFullWidth;
  }
  if (chunk.bitWidth == 16) {
    return Layout::kNarrow16;
  }
  EXPECT_EQ(chunk.bitWidth, 32);
  return Layout::kNarrow32;
}

// The tested lane layout, built independently of FixedBitWidthEncoding:
// 'laneBits'-bit lanes holding value - baseline, 8 zero bytes after them, and
// FixedBitWidth's header. Compressed the way CompressionEncoder does, with the
// lanes' DataType: the stream's own for full-width lanes, Uint16 or Uint32 for
// narrow ones. OpenZL picks its graph by that DataType, so a wrong label gives
// different bytes.
template <typename T>
std::string expectedLanesChunk(
    const std::vector<T>& values,
    int laneBits,
    uint64_t baseline,
    const CompressionPolicy& policy) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  const auto rowCount = static_cast<uint32_t>(values.size());
  std::string payload(rowCount * (laneBits / 8) + 8, '\0');
  for (uint32_t i = 0; i < rowCount; ++i) {
    const uint64_t lane =
        static_cast<uint64_t>(static_cast<PhysicalType>(values[i])) - baseline;
    std::memcpy(payload.data() + i * (laneBits / 8), &lane, laneBits / 8);
  }
  const DataType laneDataType =
      laneBits == 8 * static_cast<int>(sizeof(PhysicalType))
      ? TypeTraits<T>::dataType
      : (laneBits == 16 ? DataType::Uint16 : DataType::Uint32);
  const auto compressed = Compression::compress(
      leafPool(), payload, laneDataType, laneBits, policy);
  const bool accepted =
      compressed.compressionType != CompressionType::Uncompressed;
  std::string stored = accepted
      ? std::string{compressed.buffer->data(), compressed.buffer->size()}
      : payload;

  std::string chunk(
      EncodingPrefix::kFixedPrefixSize + 2 + sizeof(PhysicalType) +
          stored.size(),
      '\0');
  char* pos = chunk.data();
  EncodingPrefix::serialize(
      EncodingType::FixedBitWidth,
      TypeTraits<T>::dataType,
      rowCount,
      /*useVarint=*/false,
      pos);
  encoding::writeChar(static_cast<char>(compressed.compressionType), pos);
  encoding::write(static_cast<PhysicalType>(baseline), pos);
  encoding::writeChar(static_cast<char>(laneBits), pos);
  std::memcpy(pos, stored.data(), stored.size());
  return chunk;
}

// Records each payload handed to compression with its selection: the
// selecting policy's nesting path, the encoding and the value count.
struct Payload {
  std::string path;
  EncodingType encodingType;
  uint64_t values;
  uint64_t uncompressedSize;

  // Lane width when the payload is typed lanes, else 0 (bit-packed or another
  // encoding). Unambiguous for more than 8 values.
  int laneBits() const {
    if (encodingType != EncodingType::FixedBitWidth || values <= 8 ||
        uncompressedSize < 8 || (uncompressedSize - 8) * 8 % values != 0) {
      return 0;
    }
    const auto bits = static_cast<int>((uncompressedSize - 8) * 8 / values);
    return bits == 16 || bits == 32 || bits == 64 ? bits : 0;
  }
};

using PayloadLog = std::vector<Payload>;

class RecordingCompressionPolicy final : public CompressionPolicy {
 public:
  RecordingCompressionPolicy(
      const CompressionOptions& options,
      Payload payload,
      PayloadLog& log)
      : inner_{options, payload.encodingType},
        payload_{std::move(payload)},
        log_{log} {}

  CompressionConfig config() const override {
    return inner_.config();
  }

  bool shouldAccept(
      CompressionType compressionType,
      uint64_t uncompressedSize,
      uint64_t compressedSize) const override {
    auto payload = payload_;
    payload.uncompressedSize = uncompressedSize;
    log_.push_back(std::move(payload));
    return inner_.shouldAccept(
        compressionType, uncompressedSize, compressedSize);
  }

 private:
  const ConfiguredCompressionPolicy inner_;
  const Payload payload_;
  PayloadLog& log_;
};

// Selects 'encoding' (Trivial for bool and string streams), and
// 'nestedEncoding' for its nested streams, FixedBitWidth below those. Reports
// 'alpLike' as ALP candidates, so floating-point nested streams are selected
// with their logical type. Every payload is compressed with OpenZL and
// recorded.
template <typename T>
class ForcedSelectionPolicy final : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  ForcedSelectionPolicy(
      EncodingType encoding,
      EncodingType nestedEncoding,
      bool alpLike,
      std::string path,
      PayloadLog& log)
      : encoding_{encoding},
        nestedEncoding_{nestedEncoding},
        alpLike_{alpLike},
        path_{std::move(path)},
        log_{log} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    const auto encoding =
        std::is_same_v<T, bool> || std::is_same_v<T, std::string_view>
        ? EncodingType::Trivial
        : encoding_;
    return {
        .encodingType = encoding,
        .compressionPolicyFactory =
            [payload =
                 Payload{
                     .path = path_,
                     .encodingType = encoding,
                     .values = values.size(),
                     .uncompressedSize = 0},
             &log = log_]() -> std::unique_ptr<CompressionPolicy> {
          return std::make_unique<RecordingCompressionPolicy>(
              openzlOptions(), payload, log);
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }

  bool hasAlpLikeCandidates() const override {
    return alpLike_;
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ForcedSelectionPolicy,
        nestedEncoding_,
        EncodingType::FixedBitWidth,
        alpLike_,
        fmt::format(
            "{}/{}.{}",
            path_,
            ::facebook::nimble::toString(parentEncodingType),
            nestedEncodingIdentifier),
        log_);
  }

 private:
  const EncodingType encoding_;
  const EncodingType nestedEncoding_;
  const bool alpLike_;
  const std::string path_;
  PayloadLog& log_;
};

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> forcedPolicy(
    EncodingType encoding,
    EncodingType nestedEncoding,
    PayloadLog& log,
    bool alpLike = false) {
  return std::make_unique<ForcedSelectionPolicy<T>>(
      encoding, nestedEncoding, alpLike, "top", log);
}

const Payload& payloadAt(const PayloadLog& log, std::string_view path) {
  for (const auto& payload : log) {
    if (payload.path == path) {
      return payload;
    }
  }
  ADD_FAILURE() << "No payload at " << path;
  static const Payload kNone{};
  return kNone;
}

template <typename T>
class TypedLanesIntegerTest : public ::testing::Test {};

using IntegerTypes = ::testing::Types<
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t>;

TYPED_TEST_SUITE(TypedLanesIntegerTest, IntegerTypes);

// Every integer type at 17 bits (8-bit types: their full 8): 8-bit types stay
// bit-packed (below 16 stored bits), 16-bit and 32-bit types take full-width
// lanes (more than half their width), 64-bit types narrow 32-bit lanes.
TYPED_TEST(TypedLanesIntegerTest, laneLayoutPerType) {
  using T = TypeParam;
  const uint32_t bits = std::min<uint32_t>(17, 8 * sizeof(T));
  const auto values = spreadValues<T>(1'000, bits);
  const auto expected = sizeof(T) == 1 ? Layout::kPacked
      : sizeof(T) == 8                 ? Layout::kNarrow32
                                       : Layout::kFullWidth;

  const auto encoded = encode<T>(manualPolicy<T>(), values, lanesOn());

  EXPECT_EQ(toString(layoutOf<T>(encoded)), toString(expected));
  EXPECT_EQ(decode<T>(encoded), values);
  // Off: the bit-packed chunk.
  EXPECT_EQ(
      toString(layoutOf<T>(encode<T>(manualPolicy<T>(), values, lanesOff()))),
      toString(Layout::kPacked));
}

// A typed-lane payload is FixedBitArray::bufferSize() rounded up to whole
// lanes: n * L / 8 + 8 bytes for 16-, 32- and 64-bit lanes.
TEST(FixedBitWidthTypedLanesTest, payloadSizeIsBufferSizeWithLanePadding) {
  for (const int laneBits : {16, 32, 64}) {
    for (const uint64_t rows :
         {uint64_t{0}, uint64_t{1}, uint64_t{255}, uint64_t{4'099}}) {
      SCOPED_TRACE(fmt::format("{} rows of {} bits", rows, laneBits));
      const auto size = FixedBitWidthTypedLanes::payloadSize(rows, laneBits);
      EXPECT_EQ(size % (laneBits / 8), uint64_t{0});
      EXPECT_GE(size, FixedBitArray::bufferSize(rows, laneBits));
      EXPECT_EQ(size, rows * laneBits / 8 + 8);
    }
  }
}

// Port of the tested rule's threshold cases, through the encoder.
TEST(FixedBitWidthTypedLanesTest, thresholdBoundaries) {
  struct BoundaryCase {
    uint32_t count;
    uint32_t bits;
    bool exactBits;
    Layout int64Layout;
    Layout uint32Layout;
  };
  const std::vector<BoundaryCase> cases{
      // Value count, at the default floor.
      {255, 17, false, Layout::kPacked, Layout::kPacked},
      {256, 17, false, Layout::kNarrow32, Layout::kFullWidth},
      // Byte-rounded width: 8 bits stays 8, 9 rounds up to 16.
      {1'000, 8, false, Layout::kPacked, Layout::kPacked},
      {1'000, 9, false, Layout::kNarrow16, Layout::kNarrow16},
      {1'000, 16, false, Layout::kNarrow16, Layout::kNarrow16},
      {1'000, 17, false, Layout::kNarrow32, Layout::kFullWidth},
      // Exact width.
      {1'000, 9, true, Layout::kPacked, Layout::kPacked},
      {1'000, 15, true, Layout::kPacked, Layout::kPacked},
      {1'000, 16, true, Layout::kNarrow16, Layout::kNarrow16},
      {1'000, 17, true, Layout::kNarrow32, Layout::kFullWidth},
  };
  for (const auto& [count, bits, exactBits, int64Layout, uint32Layout] :
       cases) {
    SCOPED_TRACE(
        fmt::format("count {}, bits {}, exact {}", count, bits, exactBits));
    const auto options = lanesOn(256, exactBits);
    const auto int64Values = spreadValues<int64_t>(count, bits);
    const auto int64Encoded =
        encode<int64_t>(manualPolicy<int64_t>(), int64Values, options);
    EXPECT_EQ(toString(layoutOf<int64_t>(int64Encoded)), toString(int64Layout));
    EXPECT_EQ(decode<int64_t>(int64Encoded), int64Values);

    const auto uint32Values = spreadValues<uint32_t>(count, bits);
    const auto uint32Encoded =
        encode<uint32_t>(manualPolicy<uint32_t>(), uint32Values, options);
    EXPECT_EQ(
        toString(layoutOf<uint32_t>(uint32Encoded)), toString(uint32Layout));
    EXPECT_EQ(decode<uint32_t>(uint32Encoded), uint32Values);
  }
}

// The floor: the default 256, and an explicit 1024.
TEST(FixedBitWidthTypedLanesTest, floor) {
  EXPECT_EQ(Encoding::Options{}.fixedBitWidthTypedLanesMinValues, 256);
  for (const auto& [floor, count, expected] :
       std::vector<std::tuple<uint64_t, uint32_t, Layout>>{
           {256, 255, Layout::kPacked},
           {256, 256, Layout::kNarrow32},
           {256, 257, Layout::kNarrow32},
           {1'024, 1'023, Layout::kPacked},
           {1'024, 1'024, Layout::kNarrow32},
           {1'024, 1'025, Layout::kNarrow32},
       }) {
    SCOPED_TRACE(fmt::format("floor {}, count {}", floor, count));
    auto options = Encoding::Options{};
    options.fixedBitWidthTypedLanes = true;
    if (floor != 256) {
      options.fixedBitWidthTypedLanesMinValues = floor;
    }
    const auto values = spreadValues<int64_t>(count, 24);
    EXPECT_EQ(
        toString(
            layoutOf<int64_t>(
                encode<int64_t>(manualPolicy<int64_t>(), values, options))),
        toString(expected));
  }
}

// Narrow lanes are byte for byte the tested layout: 16- and 32-bit lanes of
// value - min, n * w / 8 + 8 bytes, FixedBitWidth's header.
TEST(FixedBitWidthTypedLanesTest, narrowLanesMatchTestedLayout) {
  const ConfiguredCompressionPolicy policy{
      openzlOptions(), EncodingType::FixedBitWidth};
  const auto expectNarrow = [&](const auto& values, int laneBits) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    SCOPED_TRACE(fmt::format("sizeof {}, lane {}", sizeof(T), laneBits));
    using PhysicalType = typename TypeTraits<T>::physicalType;
    PhysicalType minimum = static_cast<PhysicalType>(values[0]);
    for (const auto value : values) {
      minimum = std::min(minimum, static_cast<PhysicalType>(value));
    }
    const auto encoded = encode<T>(manualPolicy<T>(), values, lanesOn());
    EXPECT_EQ(
        encoded, expectedLanesChunk<T>(values, laneBits, minimum, policy));
    EXPECT_EQ(decode<T>(encoded), values);
  };
  expectNarrow(lowEntropyValues<int64_t>(8'192, 16, 1'000), 16);
  expectNarrow(lowEntropyValues<uint64_t>(8'192, 24, 7), 32);
  expectNarrow(lowEntropyValues<int64_t>(8'192, 32, 1'000'000), 32);
  expectNarrow(lowEntropyValues<uint32_t>(4'099, 16, 5), 16);
  expectNarrow(lowEntropyValues<int32_t>(4'099, 12, 5), 16);
  // Incompressible lanes, random over 32 bits: the compressor's output is not
  // small enough, so they are stored uncompressed, still padded.
  std::vector<uint64_t> incompressible(4'099);
  std::mt19937_64 generator{incompressible.size()};
  for (auto& value : incompressible) {
    value = generator() >> 32;
  }
  incompressible[0] = 0;
  incompressible[1] = 0xFFFF'FFFF;
  expectNarrow(incompressible, 32);
  EXPECT_EQ(
      readChunk<uint64_t>(
          encode<uint64_t>(manualPolicy<uint64_t>(), incompressible, lanesOn()))
          .compressionType,
      CompressionType::Uncompressed);
}

// Full-width lanes hold the raw values, as TrivialEncoding does, plus 8 zero
// bytes, at baseline 0.
TEST(FixedBitWidthTypedLanesTest, fullWidthPayloadIsTrivialPayloadPlusPad) {
  const auto expectFullWidth = [](const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    SCOPED_TRACE(fmt::format("sizeof {}", sizeof(T)));
    const auto encoded = encode<T>(manualPolicy<T>(), values, lanesOn());
    const auto chunk = readChunk<T>(encoded);
    ASSERT_EQ(chunk.encodingType, EncodingType::FixedBitWidth);
    EXPECT_EQ(chunk.bitWidth, 8 * sizeof(T));
    EXPECT_EQ(chunk.baseline, 0);

    const auto trivial = encode<T>(
        manualPolicy<T>({{EncodingType::Trivial, 1.0}}), values, lanesOff());
    ASSERT_EQ(EncodingPrefix::encodingType(trivial), EncodingType::Trivial);
    const char* pos = trivial.data() + EncodingPrefix::kFixedPrefixSize;
    const auto compressionType = static_cast<CompressionType>(*pos++);
    std::string_view stored{
        pos, static_cast<size_t>(trivial.data() + trivial.size() - pos)};
    std::string trivialPayload{stored};
    if (compressionType != CompressionType::Uncompressed) {
      auto raw = Compression::uncompress(
          leafPool(), compressionType, DataType::Undefined, stored, nullptr);
      trivialPayload = std::string{raw->as<char>(), raw->size()};
    }
    EXPECT_EQ(chunk.payload, trivialPayload + std::string(8, '\0'));
    EXPECT_EQ(decode<T>(encoded), values);
  };
  expectFullWidth(lowEntropyValues<uint16_t>(4'096, 16, 0));
  expectFullWidth(lowEntropyValues<int16_t>(4'096, 12, 100));
  expectFullWidth(lowEntropyValues<uint32_t>(4'096, 24, 3));
  expectFullWidth(lowEntropyValues<int32_t>(4'096, 32, 0));
  expectFullWidth(lowEntropyValues<uint64_t>(4'096, 40, 1'000));
  // 41 to 48 bits: 64-bit chunks over 44 and 48 bits are more than half the
  // type width, so they take full-width lanes too.
  expectFullWidth(lowEntropyValues<int64_t>(4'096, 44, 1'000));
  expectFullWidth(lowEntropyValues<uint64_t>(4'096, 48, 7));
  expectFullWidth(lowEntropyValues<int64_t>(4'096, 64, 0));
  expectFullWidth(spreadValues<uint64_t>(4'099, 64));
  // Signed values whose two's complement spans the whole type.
  std::vector<int64_t> signedMix(4'096);
  for (size_t i = 0; i < signedMix.size(); ++i) {
    signedMix[i] = (i % 2 == 0 ? -1 : 1) * static_cast<int64_t>(i * 977);
  }
  expectFullWidth(signedMix);
}

// A full-width chunk is priced at its packed width, the byte-rounded bit width,
// not at the width of its lanes: the estimate is the same with lanes on or off,
// so selection makes the same choice either way. The chunk is still stored in
// lanes of the type's width plus 8 zero bytes.
TEST(FixedBitWidthTypedLanesTest, fullWidthChunksArePricedAtPackedWidth) {
  const auto check = [](auto tag, uint32_t bits) {
    using T = decltype(tag);
    using PhysicalType = typename TypeTraits<T>::physicalType;
    SCOPED_TRACE(fmt::format("sizeof {}, {} bits", sizeof(T), bits));
    constexpr uint32_t kRows{4'096};
    const auto values = spreadValues<T>(kRows, bits);
    const std::span<const PhysicalType> physical{
        reinterpret_cast<const PhysicalType*>(values.data()), values.size()};
    const auto statistics = Statistics<PhysicalType>::create(physical);
    // Data layout: prefix, compression type, baseline, bit width, payload.
    const uint64_t header =
        EncodingPrefix::kFixedPrefixSize + 2 + sizeof(PhysicalType);
    const uint64_t packedBits = (bits + 7) / 8 * 8;
    const uint64_t packed = header + kRows * packedBits / 8;
    EXPECT_EQ(
        FixedBitWidthEncoding<T>::estimateSize(kRows, statistics, lanesOn()),
        packed);
    EXPECT_EQ(
        FixedBitWidthEncoding<T>::estimateSize(kRows, statistics, lanesOff()),
        packed);

    const auto encoded = encode<T>(manualPolicy<T>(), values, lanesOn());
    const auto chunk = readChunk<T>(encoded);
    ASSERT_EQ(chunk.encodingType, EncodingType::FixedBitWidth);
    EXPECT_EQ(chunk.bitWidth, 8 * sizeof(T));
    EXPECT_EQ(chunk.payload.size(), kRows * sizeof(T) + 8);
    EXPECT_EQ(decode<T>(encoded), values);
  };
  check(uint32_t{}, 24);
  check(int32_t{}, 20);
  check(int64_t{}, 40);
  check(uint64_t{}, 48);
  check(int64_t{}, 56);
}

// Lanes need compression: an Uncompressed policy, or no compression options,
// keep the bit-packed bytes.
TEST(FixedBitWidthTypedLanesTest, compressionOffKeepsBitPackedBytes) {
  const auto values = spreadValues<int64_t>(1'000, 17);
  auto uncompressed = zstdOptions();
  uncompressed.compressionType = CompressionType::Uncompressed;
  for (const auto& [name, compressionOptions] :
       std::vector<std::pair<std::string, std::optional<CompressionOptions>>>{
           {"Uncompressed", uncompressed},
           {"NoCompressionOptions", std::nullopt},
       }) {
    SCOPED_TRACE(name);
    const auto withLanes = encode<int64_t>(
        manualPolicy<int64_t>(kFixedBitWidthOnly, compressionOptions),
        values,
        lanesOn());
    EXPECT_EQ(
        toString(layoutOf<int64_t>(withLanes)), toString(Layout::kPacked));
    EXPECT_EQ(
        withLanes,
        encode<int64_t>(
            manualPolicy<int64_t>(kFixedBitWidthOnly, compressionOptions),
            values,
            lanesOff()));
  }
}

// Lanes need a compressor that reads the payload's DataType. Zstd and Lz4
// compress bytes, so with lanes on they write exactly the bytes they write
// with lanes off, for chunks that take each lane layout under OpenZL.
TEST(FixedBitWidthTypedLanesTest, byteCompressorsKeepBitPackedBytes) {
  const auto expectBitPacked = [](const auto& values, Layout openzlLayout) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    SCOPED_TRACE(
        fmt::format("sizeof {}, {}", sizeof(T), toString(openzlLayout)));
    EXPECT_EQ(
        toString(layoutOf<T>(encode<T>(manualPolicy<T>(), values, lanesOn()))),
        toString(openzlLayout));
    for (const auto& [name, compressionOptions] :
         std::vector<std::pair<std::string, CompressionOptions>>{
             {"Zstd", zstdOptions()},
             {"Lz4", lz4Options()},
         }) {
      SCOPED_TRACE(name);
      const auto withLanes = encode<T>(
          manualPolicy<T>(kFixedBitWidthOnly, compressionOptions),
          values,
          lanesOn());
      EXPECT_EQ(toString(layoutOf<T>(withLanes)), toString(Layout::kPacked));
      EXPECT_EQ(
          withLanes,
          encode<T>(
              manualPolicy<T>(kFixedBitWidthOnly, compressionOptions),
              values,
              lanesOff()));
      EXPECT_EQ(decode<T>(withLanes), values);
    }
  };
  expectBitPacked(
      lowEntropyValues<int64_t>(4'096, 12, 1'000), Layout::kNarrow16);
  expectBitPacked(lowEntropyValues<uint64_t>(4'096, 24, 7), Layout::kNarrow32);
  expectBitPacked(lowEntropyValues<uint16_t>(4'096, 16, 0), Layout::kFullWidth);
  expectBitPacked(lowEntropyValues<uint32_t>(4'096, 24, 3), Layout::kFullWidth);
  expectBitPacked(
      lowEntropyValues<int64_t>(4'096, 48, 1'000), Layout::kFullWidth);
}

// Float and double streams are never lanes, though their physical types are
// integers; bool streams are untouched.
TEST(FixedBitWidthTypedLanesTest, floatDoubleAndBoolKeepBytes) {
  const auto expectSameBytes = [](const auto& values) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    SCOPED_TRACE(fmt::format("sizeof {}", sizeof(T)));
    const auto withLanes = encode<T>(manualPolicy<T>(), values, lanesOn(0));
    ASSERT_EQ(
        EncodingPrefix::encodingType(withLanes), EncodingType::FixedBitWidth);
    EXPECT_EQ(withLanes, encode<T>(manualPolicy<T>(), values, lanesOff()));
  };
  expectSameBytes(spreadValues<float>(1'000, 24));
  expectSameBytes(spreadValues<double>(1'000, 24));
  expectSameBytes(spreadValues<double>(1'000, 40));

  BoolVector bools{4'000};
  for (size_t i = 0; i < bools.size; ++i) {
    bools.data[i] = i % 3 == 0;
  }
  const ReadFactors leafOnly{
      {EncodingType::Trivial, 1.0}, {EncodingType::FixedBitWidth, 1.0}};
  EXPECT_EQ(
      encode<bool>(manualPolicy<bool>(leafOnly), bools.span(), lanesOn(0)),
      encode<bool>(manualPolicy<bool>(leafOnly), bools.span(), lanesOff()));
}

// Lanes apply to nested integer streams: the non-null values of Nullable
// integer streams and of Nullable float and double streams (as bit patterns),
// Dictionary indices, RLE run values, and the lengths of Trivial strings.
TEST(FixedBitWidthTypedLanesTest, nestedIntegerStreamsGetLanes) {
  {
    SCOPED_TRACE("Nullable int64");
    const auto values = spreadValues<int64_t>(1'000, 17);
    const auto nonNulls = everyThirdRowNull(values.size());
    const auto encoded = encodeNullable<int64_t>(
        manualPolicy<int64_t>(), values, nonNulls.span(), lanesOn());
    ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Nullable);
    EXPECT_NE(
        encoded,
        encodeNullable<int64_t>(
            manualPolicy<int64_t>(), values, nonNulls.span(), lanesOff()));
    PayloadLog log;
    encodeNullable<int64_t>(
        forcedPolicy<int64_t>(
            EncodingType::FixedBitWidth, EncodingType::FixedBitWidth, log),
        values,
        nonNulls.span(),
        lanesOn());
    EXPECT_EQ(payloadAt(log, "top/Nullable.0").laneBits(), 32);
  }
  for (const bool isDouble : {false, true}) {
    SCOPED_TRACE(isDouble ? "Nullable double" : "Nullable float");
    PayloadLog log;
    if (isDouble) {
      const auto values = spreadValues<double>(1'000, 24);
      const auto nonNulls = everyThirdRowNull(values.size());
      encodeNullable<double>(
          forcedPolicy<double>(
              EncodingType::FixedBitWidth, EncodingType::FixedBitWidth, log),
          values,
          nonNulls.span(),
          lanesOn());
      // 24 of 64 bits: narrow 32-bit lanes of the bit patterns.
      EXPECT_EQ(payloadAt(log, "top/Nullable.0").laneBits(), 32);
    } else {
      const auto values = spreadValues<float>(1'000, 24);
      const auto nonNulls = everyThirdRowNull(values.size());
      encodeNullable<float>(
          forcedPolicy<float>(
              EncodingType::FixedBitWidth, EncodingType::FixedBitWidth, log),
          values,
          nonNulls.span(),
          lanesOn());
      // 24 of 32 bits: full-width lanes of the bit patterns.
      EXPECT_EQ(payloadAt(log, "top/Nullable.0").laneBits(), 32);
    }
  }
  {
    SCOPED_TRACE("Dictionary");
    // 1,000 distinct values: the indices need 10 bits, stored in 16.
    std::vector<int64_t> values(4'000);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<int64_t>((i * 7) % 1'000) << 20;
    }
    PayloadLog log;
    const auto encoded = encode<int64_t>(
        forcedPolicy<int64_t>(
            EncodingType::Dictionary, EncodingType::FixedBitWidth, log),
        values,
        lanesOn());
    EXPECT_EQ(decode<int64_t>(encoded), values);
    EXPECT_EQ(
        payloadAt(
            log,
            fmt::format(
                "top/Dictionary.{}", EncodingIdentifiers::Dictionary::Indices))
            .laneBits(),
        16);
    // The alphabet: 1,000 values over 30 bits, stored in 32: narrow lanes.
    EXPECT_EQ(
        payloadAt(
            log,
            fmt::format(
                "top/Dictionary.{}", EncodingIdentifiers::Dictionary::Alphabet))
            .laneBits(),
        32);
  }
  {
    SCOPED_TRACE("RLE");
    std::vector<int64_t> values;
    for (const auto value : spreadValues<int64_t>(400, 17)) {
      values.insert(values.end(), 10, value);
    }
    PayloadLog log;
    const auto encoded = encode<int64_t>(
        forcedPolicy<int64_t>(
            EncodingType::RLE, EncodingType::FixedBitWidth, log),
        values,
        lanesOn());
    EXPECT_EQ(decode<int64_t>(encoded), values);
    EXPECT_EQ(
        payloadAt(
            log,
            fmt::format(
                "top/RLE.{}", EncodingIdentifiers::RunLength::RunValues))
            .laneBits(),
        32);
    // Run lengths are all 10: 0 bits, bit-packed.
    for (const auto& payload : log) {
      if (payload.path ==
          fmt::format(
              "top/RLE.{}", EncodingIdentifiers::RunLength::RunLengths)) {
        EXPECT_EQ(payload.laneBits(), 0);
      }
    }
  }
  {
    SCOPED_TRACE("Trivial string lengths");
    // Lengths from 0 to 70,000: 17 bits, stored in 24 of the 32-bit lengths,
    // so full-width lanes.
    std::vector<std::string> storage;
    storage.reserve(300);
    for (uint32_t i = 0; i < 300; ++i) {
      storage.emplace_back(i == 1 ? 70'000 : i % 50, 'x');
    }
    std::vector<std::string_view> values{storage.begin(), storage.end()};
    PayloadLog log;
    Buffer buffer{leafPool()};
    EncodingFactory::encode<std::string_view>(
        forcedPolicy<std::string_view>(
            EncodingType::Trivial, EncodingType::FixedBitWidth, log),
        values,
        buffer,
        lanesOn());
    EXPECT_EQ(
        payloadAt(
            log,
            fmt::format(
                "top/Trivial.{}", EncodingIdentifiers::Trivial::Lengths))
            .laneBits(),
        32);
  }
}

// The non-null values of a Nullable float or double stream are stored as
// integers whenever FixedBitWidth is selected for them, so they take lanes by
// the same rule as any integer chunk: also when the policy has ALP candidates
// and selection runs with the floating-point type. Selected either way, the
// stream is stored byte for byte the same, as a replayed layout (selected as
// integers) stores it.
TEST(FixedBitWidthTypedLanesTest, floatValuesStoredAsIntegersTakeLanes) {
  // 10,000 float bit patterns: 9,500 within 8 bits of the minimum and 500
  // spread over 24 bits, so PFOR has about 500 exceptions.
  std::vector<float> values(10'000);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint64_t offset =
        i % 20 == 0 ? 256 + (i * 7919) % ((1 << 24) - 256) : i % 200;
    values[i] = fromPhysical<float>(physicalBase<float>() + offset);
  }
  values[20] = fromPhysical<float>(physicalBase<float>() + (1 << 24) - 1);
  const auto nonNulls = everyThirdRowNull(values.size());
  const auto nonNullCount = values.size();

  const auto encodeFixedBitWidth =
      [&](bool alpLike, const Encoding::Options& options, PayloadLog& log) {
        return encodeNullable<float>(
            forcedPolicy<float>(
                EncodingType::FixedBitWidth,
                EncodingType::FixedBitWidth,
                log,
                alpLike),
            values,
            nonNulls.span(),
            options);
      };
  PayloadLog withAlpLog;
  const auto withAlp =
      encodeFixedBitWidth(/*alpLike=*/true, lanesOn(), withAlpLog);
  PayloadLog withoutAlpLog;
  const auto withoutAlp =
      encodeFixedBitWidth(/*alpLike=*/false, lanesOn(), withoutAlpLog);
  for (const auto* log : {&withAlpLog, &withoutAlpLog}) {
    // 24 bits of 32: full-width lanes of the bit patterns.
    const auto& data = payloadAt(*log, "top/Nullable.0");
    EXPECT_EQ(data.laneBits(), 32);
    EXPECT_EQ(
        data.uncompressedSize,
        FixedBitWidthTypedLanes::payloadSize(nonNullCount, 32));
  }
  EXPECT_EQ(withAlp, withoutAlp);
  PayloadLog offLog;
  EXPECT_NE(withAlp, encodeFixedBitWidth(/*alpLike=*/true, lanesOff(), offLog));

  {
    SCOPED_TRACE("PFOR, ALP candidates");
    PayloadLog log;
    encodeNullable<float>(
        forcedPolicy<float>(
            EncodingType::PFOR,
            EncodingType::PFOR,
            log,
            /*alpLike=*/true),
        values,
        nonNulls.span(),
        lanesOn());
    // Exception positions span 10,000 rows (14 bits, stored in 16): narrow
    // lanes. Exception values span 24 bits of 32: full-width lanes.
    const auto& positions = payloadAt(
        log,
        fmt::format(
            "top/Nullable.0/PFOR.{}",
            EncodingIdentifiers::Pfor::ExceptionPositions));
    const auto& exceptions = payloadAt(
        log,
        fmt::format(
            "top/Nullable.0/PFOR.{}",
            EncodingIdentifiers::Pfor::ExceptionValues));
    ASSERT_GE(positions.values, 256);
    EXPECT_EQ(positions.laneBits(), 16);
    EXPECT_EQ(exceptions.laneBits(), 32);
  }
}

// Logical-type floating-point streams nested in Dictionary, RLE and
// MainlyConstant (alphabets, run values, uncommon values) are encoded with
// their floating-point type, so they keep the bit-packed layout, while the
// integer streams next to them (indices) take lanes.
TEST(FixedBitWidthTypedLanesTest, floatNestedValuesKeepBits) {
  std::vector<float> values(4'000);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] =
        fromPhysical<float>(physicalBase<float>() + ((i * 7) % 1'000) * 4'099);
  }
  for (const auto encoding :
       {EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::MainlyConstant}) {
    SCOPED_TRACE(::facebook::nimble::toString(encoding));
    PayloadLog log;
    std::vector<float> input = values;
    if (encoding == EncodingType::RLE) {
      input.clear();
      for (size_t i = 0; i < 400; ++i) {
        input.insert(input.end(), 10, values[i]);
      }
    } else if (encoding == EncodingType::MainlyConstant) {
      input.assign(4'000, values[0]);
      for (size_t i = 0; i < input.size(); i += 10) {
        input[i] = values[i];
      }
    }
    const auto encoded = encode<float>(
        forcedPolicy<float>(encoding, EncodingType::FixedBitWidth, log),
        input,
        lanesOn(0));
    EXPECT_EQ(decode<float>(encoded), input);
    for (const auto& payload : log) {
      const bool valueStream =
          (encoding == EncodingType::Dictionary &&
           payload.path ==
               fmt::format(
                   "top/Dictionary.{}",
                   EncodingIdentifiers::Dictionary::Alphabet)) ||
          (encoding == EncodingType::RLE &&
           payload.path ==
               fmt::format(
                   "top/RLE.{}", EncodingIdentifiers::RunLength::RunValues)) ||
          (encoding == EncodingType::MainlyConstant &&
           payload.path ==
               fmt::format(
                   "top/MainlyConstant.{}",
                   EncodingIdentifiers::MainlyConstant::OtherValues));
      if (valueStream) {
        SCOPED_TRACE(payload.path);
        EXPECT_EQ(payload.laneBits(), 0);
      }
    }
  }
}

// With exact bits and lanes on, float and double chunks keep the bit-packed
// layout at the exact width, byte for byte as with lanes off, and
// FixedBitWidthEncoding of the floating-point type prices them as without
// lanes.
TEST(FixedBitWidthTypedLanesTest, floatingPointKeepsExactWidthWithLanes) {
  const auto check = [](auto tag, uint32_t bits) {
    using T = decltype(tag);
    SCOPED_TRACE(fmt::format("sizeof {}, bits {}", sizeof(T), bits));
    const auto values = spreadValues<T>(1'000, bits, 3);
    const auto withLanes =
        encode<T>(manualPolicy<T>(), values, lanesOn(256, true));
    EXPECT_EQ(withLanes, encode<T>(manualPolicy<T>(), values, lanesOff(true)));
    const auto chunk = readChunk<T>(withLanes);
    ASSERT_EQ(chunk.encodingType, EncodingType::FixedBitWidth);
    EXPECT_EQ(chunk.bitWidth, static_cast<int>(bits));
    EXPECT_EQ(decode<T>(withLanes), values);
    const uint64_t maxValue = (uint64_t{1} << bits) - 1;
    EXPECT_EQ(
        FixedBitWidthEncoding<T>::estimateSize(
            1'000, 0, maxValue, lanesOn(256, true)),
        FixedBitWidthEncoding<T>::estimateSize(
            1'000, 0, maxValue, lanesOff(true)));
  };
  for (const uint32_t bits : {9u, 12u, 15u, 17u, 20u, 24u}) {
    check(float{}, bits);
    check(double{}, bits);
  }
}

// slice() of a lane chunk re-packs its rows at the stored width, uncompressed.
TEST(FixedBitWidthTypedLanesTest, sliceOfLanes) {
  for (const int bits : {16, 24, 64}) {
    SCOPED_TRACE(bits);
    const auto values = lowEntropyValues<uint64_t>(4'096, bits, 0);
    const auto encoded =
        encode<uint64_t>(manualPolicy<uint64_t>(), values, lanesOn());
    ASSERT_NE(toString(layoutOf<uint64_t>(encoded)), toString(Layout::kPacked));
    Buffer buffer{leafPool()};
    const auto sliced =
        FixedBitWidthEncoding<uint64_t>::slice(encoded, 100, 1'000, buffer);
    EXPECT_EQ(
        decode<uint64_t>(sliced),
        std::vector<uint64_t>(values.begin() + 100, values.begin() + 1'100));
  }
}

// SubIntSplit packs its sections at their exact width, never as lanes.
TEST(FixedBitWidthTypedLanesTest, subIntSplitSectionsStayBitPacked) {
  // Two fields: a 3-bit tag at bit 40 and a 16-bit counter.
  std::vector<uint64_t> values(4'096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = (static_cast<uint64_t>(i % 7) << 40) | ((i * 7919) % 65'521);
  }
  PayloadLog log;
  const auto withLanes = encode<uint64_t>(
      forcedPolicy<uint64_t>(
          EncodingType::SubIntSplit, EncodingType::FixedBitWidth, log),
      values,
      lanesOn(0));
  ASSERT_EQ(EncodingPrefix::encodingType(withLanes), EncodingType::SubIntSplit);
  EXPECT_EQ(decode<uint64_t>(withLanes), values);
  bool sawSection = false;
  for (const auto& payload : log) {
    if (payload.path != "top") {
      SCOPED_TRACE(payload.path);
      sawSection = true;
      EXPECT_EQ(payload.laneBits(), 0);
    }
  }
  EXPECT_TRUE(sawSection);
  PayloadLog offLog;
  EXPECT_EQ(
      withLanes,
      encode<uint64_t>(
          forcedPolicy<uint64_t>(
              EncodingType::SubIntSplit, EncodingType::FixedBitWidth, offLog),
          values,
          lanesOff()));
}

// Off is the bit-packed layout for every width and type: the payload is
// FixedBitArray::bufferSize() bytes at the stored width, baseline the minimum.
TEST(FixedBitWidthTypedLanesTest, offIsBitPacked) {
  EXPECT_FALSE(Encoding::Options{}.fixedBitWidthTypedLanes);
  const auto expectPacked = [](auto tag, int maxBits) {
    using T = decltype(tag);
    for (int bits = 1; bits <= maxBits; ++bits) {
      for (const bool exactBits : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "sizeof {}, bits {}, exact {}", sizeof(T), bits, exactBits));
        const auto values = spreadValues<T>(300, bits, 3);
        const auto encoded =
            encode<T>(manualPolicy<T>(), values, lanesOff(exactBits));
        const auto chunk = readChunk<T>(encoded);
        ASSERT_EQ(chunk.encodingType, EncodingType::FixedBitWidth);
        const int stored = exactBits ? bits : (bits + 7) / 8 * 8;
        EXPECT_EQ(chunk.bitWidth, stored);
        EXPECT_EQ(chunk.payload.size(), FixedBitArray::bufferSize(300, stored));
        EXPECT_EQ(decode<T>(encoded), values);
      }
    }
  };
  expectPacked(uint16_t{}, 15);
  expectPacked(uint32_t{}, 31);
  expectPacked(int64_t{}, 63);
}

} // namespace
} // namespace facebook::nimble
