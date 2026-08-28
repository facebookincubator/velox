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

#include "velox/dwio/nimble/encodings/benchmarks/NimbleEncodingRunner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/String.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <folly/ssl/OpenSSLHash.h>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BlockBitPackingBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/DeltaBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/FixedBitWidthBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/FsstBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/PFOREncodingBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/PrefixBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/SimdForBitpackBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/SparseBoolBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/VarintBenchmarkData.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "xplat/secure_lib/secure_string.h"

namespace facebook::nimble::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;

enum class RunnerEncoding {
  RLE,
  Dictionary,
  FixedBitWidth,
  Delta,
  SparseBool,
  PFOR,
  SimdForBitpack,
  BlockBitPacking,
  Varint,
  Prefix,
  Fsst,
  Nullable,
  ALP,
  DeltaBlock,
};

enum class RunnerLane {
  Encode,
  DecodeConstruct,
  DecodeDense,
  DecodeRange50,
  DecodeScatter10,
  DecodeScatter1,
  SkipSeek,
  ViewRandom,
  Slice,
  SelectionE2E,
};

constexpr uint16_t laneBit(RunnerLane lane) {
  return 1U << static_cast<uint16_t>(lane);
}

constexpr uint16_t kAllLaneMask = (1U << 10) - 1;
constexpr uint16_t kNoViewLaneMask =
    kAllLaneMask & ~laneBit(RunnerLane::ViewRandom);
constexpr uint16_t kFixedBitWidthLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::DecodeScatter10) |
    laneBit(RunnerLane::DecodeScatter1) | laneBit(RunnerLane::SkipSeek) |
    laneBit(RunnerLane::SelectionE2E);
constexpr uint16_t kDeltaLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kSparseBoolLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kPforLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kSimdForBitpackLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kBlockBitPackingLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kVarintLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kPrefixLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);
constexpr uint16_t kFsstLaneMask = laneBit(RunnerLane::Encode) |
    laneBit(RunnerLane::DecodeDense) | laneBit(RunnerLane::SkipSeek);

struct TaskSpec {
  RunnerEncoding encoding;
  RunnerLane lane;
  std::string_view encodingSlug;
  std::string_view laneName;
  EncodingType encodingType;
  std::string_view dataType;
};

constexpr std::array<std::pair<std::string_view, RunnerLane>, 10> kLanes{{
    {"encode", RunnerLane::Encode},
    {"decode_construct", RunnerLane::DecodeConstruct},
    {"decode_dense", RunnerLane::DecodeDense},
    {"decode_range50", RunnerLane::DecodeRange50},
    {"decode_scatter10", RunnerLane::DecodeScatter10},
    {"decode_scatter1", RunnerLane::DecodeScatter1},
    {"skip_seek", RunnerLane::SkipSeek},
    {"view_random", RunnerLane::ViewRandom},
    {"slice", RunnerLane::Slice},
    {"selection_e2e", RunnerLane::SelectionE2E},
}};

struct EncodingSpec {
  std::string_view slug;
  RunnerEncoding encoding;
  EncodingType encodingType;
  std::string_view dataType;
  uint16_t supportedLanes;
};

constexpr std::array<EncodingSpec, 14> kEncodings{{
    {"rle", RunnerEncoding::RLE, EncodingType::RLE, "int64", kAllLaneMask},
    {"dictionary",
     RunnerEncoding::Dictionary,
     EncodingType::Dictionary,
     "int64",
     kAllLaneMask},
    {"fixed_bit_width",
     RunnerEncoding::FixedBitWidth,
     EncodingType::FixedBitWidth,
     "uint64",
     kFixedBitWidthLaneMask},
    {"delta",
     RunnerEncoding::Delta,
     EncodingType::Delta,
     "uint32",
     kDeltaLaneMask},
    {"sparse_bool",
     RunnerEncoding::SparseBool,
     EncodingType::SparseBool,
     "bool",
     kSparseBoolLaneMask},
    {"pfor", RunnerEncoding::PFOR, EncodingType::PFOR, "uint32", kPforLaneMask},
    {"simd_for_bitpack",
     RunnerEncoding::SimdForBitpack,
     EncodingType::SimdForBitpack,
     "uint32",
     kSimdForBitpackLaneMask},
    {"block_bit_packing",
     RunnerEncoding::BlockBitPacking,
     EncodingType::BlockBitPacking,
     "uint32",
     kBlockBitPackingLaneMask},
    {"varint",
     RunnerEncoding::Varint,
     EncodingType::Varint,
     "uint32",
     kVarintLaneMask},
    {"prefix",
     RunnerEncoding::Prefix,
     EncodingType::Prefix,
     "string",
     kPrefixLaneMask},
    {"fsst", RunnerEncoding::Fsst, EncodingType::Fsst, "string", kFsstLaneMask},
    {"nullable",
     RunnerEncoding::Nullable,
     EncodingType::Nullable,
     "int64",
     kNoViewLaneMask},
    {"alp", RunnerEncoding::ALP, EncodingType::ALP, "double", kAllLaneMask},
    {"delta_block",
     RunnerEncoding::DeltaBlock,
     EncodingType::DeltaBlock,
     "int64",
     kAllLaneMask},
}};

constexpr uint32_t kMaxRowCount = 1'048'576;
constexpr uint32_t kMaxWarmups = 100;
constexpr uint32_t kMaxSamples = 100;
constexpr uint32_t kMaxMinSampleTimeMicros = 10'000'000;
constexpr uint32_t kMaxCalibratedIterations = 1'000'000'000;
constexpr uint64_t kMaxStringDecodedBytes = 256ULL * 1024 * 1024;
constexpr uint32_t kMaxStringDecodedValueBytes = 16 * 1024 * 1024;
constexpr size_t kMaxStringPages = 2048;
constexpr uint64_t kFsstMaxExpansion = 8;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error{message};
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

// Fixed-size header shared by every Nimble encoding artifact. Decoding these
// three fields up front lets the preflight checks reject a mismatched or
// truncated artifact before any child stream is interpreted.
struct FixedArtifactPrefix {
  // Encoding kind stored at the artifact root (e.g. RLE, Trivial).
  EncodingType encodingType;
  // Logical value type carried by the encoding.
  DataType dataType;
  // Number of logical rows the artifact claims to hold.
  uint32_t rowCount;
};

// Copy a trivially-copyable value out of the artifact at a byte offset, but
// only after bounds-checking the read so a truncated or lying artifact cannot
// drive an out-of-bounds access.
template <typename T>
T readBoundedValue(
    std::string_view data,
    size_t offset,
    std::string_view field) {
  require(
      offset <= data.size() && sizeof(T) <= data.size() - offset,
      fmt::format("{} is truncated", field));
  T value;
  require(
      try_checked_memcpy_robust(
          &value,
          sizeof(value),
          data.data() + offset,
          data.size() - offset,
          sizeof(value)) == 0,
      fmt::format("{} could not be copied safely", field));
  return value;
}

uint32_t readBoundedUint32(
    std::string_view data,
    size_t offset,
    std::string_view field) {
  return readBoundedValue<uint32_t>(data, offset, field);
}

uint32_t readBoundedVarint32(
    std::string_view data,
    size_t& offset,
    std::string_view field) {
  uint32_t value{0};
  for (uint32_t byteIndex = 0; byteIndex < 5; ++byteIndex) {
    require(offset < data.size(), fmt::format("{} is truncated", field));
    const auto byte = static_cast<uint8_t>(data[offset++]);
    if (byteIndex == 4) {
      require(
          (byte & 0xF0) == 0,
          fmt::format("{} overflows a 32-bit varint", field));
    }
    const auto payload = static_cast<uint32_t>(byte & 0x7F);
    value |= payload << (byteIndex * 7);
    if ((byte & 0x80) == 0) {
      require(
          byteIndex == 0 || payload != 0,
          fmt::format("{} is not minimally encoded", field));
      return value;
    }
  }
  fail(fmt::format("{} exceeds five bytes", field));
}

void validateFixedBitArrayPadding(
    std::string_view packed,
    uint32_t rowCount,
    uint8_t bitWidth,
    std::string_view field) {
  const uint64_t logicalBits = static_cast<uint64_t>(rowCount) * bitWidth;
  const size_t logicalBytes = static_cast<size_t>((logicalBits + 7) / 8);
  require(
      logicalBytes <= packed.size(),
      fmt::format("{} payload is truncated", field));
  const uint32_t usedBits = logicalBits % 8;
  if (usedBits != 0) {
    const auto paddingMask = static_cast<uint8_t>(0xFFU << usedBits);
    require(
        (static_cast<uint8_t>(packed[logicalBytes - 1]) & paddingMask) == 0,
        fmt::format("{} has non-zero padding bits", field));
  }
  for (size_t offset = logicalBytes; offset < packed.size(); ++offset) {
    require(
        packed[offset] == 0, fmt::format("{} has non-zero slop bytes", field));
  }
}

// Decode the fixed prefix from the front of an artifact, refusing to read past
// a prefix that is shorter than the on-disk layout guarantees.
FixedArtifactPrefix readFixedArtifactPrefix(
    std::string_view data,
    std::string_view field) {
  require(
      data.size() >= EncodingPrefix::kFixedPrefixSize,
      fmt::format("{} prefix is truncated", field));
  return FixedArtifactPrefix{
      .encodingType = static_cast<EncodingType>(data[0]),
      .dataType = static_cast<DataType>(data[1]),
      .rowCount =
          readBoundedUint32(data, EncodingPrefix::kRowCountOffset, field),
  };
}

// Confirm the artifact root matches the encoding, data type, and row count the
// benchmark task expects, so a corpus/artifact mismatch fails fast instead of
// silently benchmarking the wrong payload.
void validateArtifactPrefix(
    std::string_view artifact,
    EncodingType encodingType,
    DataType dataType,
    uint32_t rowCount) {
  const auto prefix = readFixedArtifactPrefix(artifact, "artifact");
  require(
      prefix.encodingType == encodingType,
      "artifact root encoding does not match task");
  require(
      prefix.dataType == dataType, "artifact data type does not match task");
  require(
      prefix.rowCount == rowCount, "artifact row count does not match corpus");
}

// Verify a Trivial-encoded child stream is uncompressed and its payload is
// exactly rowCount fixed-size values. Checking the size against the declared
// row count keeps the decoder from reading beyond a short buffer.
template <typename T>
void validateUncompressedTrivialChild(
    std::string_view child,
    DataType dataType,
    uint32_t rowCount,
    std::string_view field) {
  const auto prefix = readFixedArtifactPrefix(child, field);
  require(
      prefix.encodingType == EncodingType::Trivial,
      fmt::format("{} must use Trivial encoding", field));
  require(
      prefix.dataType == dataType,
      fmt::format("{} has an unexpected data type", field));
  require(
      prefix.rowCount == rowCount,
      fmt::format("{} row count does not match expected rows", field));
  constexpr size_t kPayloadOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  require(
      child.size() >= kPayloadOffset,
      fmt::format("{} payload is truncated", field));
  require(
      static_cast<CompressionType>(child[EncodingPrefix::kFixedPrefixSize]) ==
          CompressionType::Uncompressed,
      fmt::format("{} must be uncompressed", field));
  const uint64_t expectedSize =
      kPayloadOffset + static_cast<uint64_t>(rowCount) * sizeof(T);
  require(
      child.size() == expectedSize,
      fmt::format("{} payload size is invalid", field));
}

// Verify a FixedBitWidth-encoded child stream is uncompressed, carries a bit
// width no wider than the value type, and is sized for exactly rowCount packed
// values. Callers explicitly allow width zero only for an empty Delta residual
// stream; rejecting it for non-empty streams keeps the FixedBitArray decoder
// off its unsupported zero-width path.
template <typename T>
void validateUncompressedFixedBitWidthChild(
    std::string_view child,
    DataType dataType,
    uint32_t rowCount,
    std::string_view field,
    bool allowEmptyZeroBitWidth) {
  const auto prefix = readFixedArtifactPrefix(child, field);
  require(
      prefix.encodingType == EncodingType::FixedBitWidth,
      fmt::format("{} must use FixedBitWidth encoding", field));
  require(
      prefix.dataType == dataType,
      fmt::format("{} has an unexpected data type", field));
  require(
      prefix.rowCount == rowCount,
      fmt::format("{} row count does not match expected rows", field));
  constexpr size_t kBitWidthOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(T);
  constexpr size_t kPayloadOffset = kBitWidthOffset + sizeof(uint8_t);
  require(
      child.size() >= kPayloadOffset,
      fmt::format("{} payload is truncated", field));
  require(
      static_cast<CompressionType>(child[EncodingPrefix::kFixedPrefixSize]) ==
          CompressionType::Uncompressed,
      fmt::format("{} must be uncompressed", field));
  const auto bitWidth = static_cast<uint8_t>(child[kBitWidthOffset]);
  require(
      (bitWidth > 0 || (allowEmptyZeroBitWidth && rowCount == 0)) &&
          bitWidth <= std::numeric_limits<T>::digits,
      fmt::format("{} bit width is invalid", field));
  const uint64_t expectedSize = kPayloadOffset +
      FixedBitArray::bufferSize(rowCount, static_cast<int>(bitWidth));
  require(
      child.size() == expectedSize,
      fmt::format("{} payload size is invalid", field));
}

// Verify an uncompressed Trivial bool child has exactly one bit per row and
// only zero padding/slop bits. This prevents a malformed Delta flag stream
// from exposing FixedBitArray to bytes outside the declared logical bitmap.
void validateUncompressedTrivialBoolChild(
    std::string_view child,
    uint32_t rowCount,
    std::string_view field) {
  const auto prefix = readFixedArtifactPrefix(child, field);
  require(
      prefix.encodingType == EncodingType::Trivial,
      fmt::format("{} must use Trivial encoding", field));
  require(
      prefix.dataType == DataType::Bool,
      fmt::format("{} has an unexpected data type", field));
  require(
      prefix.rowCount == rowCount,
      fmt::format("{} row count does not match expected rows", field));
  constexpr size_t kPayloadOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  require(
      child.size() >= kPayloadOffset,
      fmt::format("{} payload is truncated", field));
  require(
      static_cast<CompressionType>(child[EncodingPrefix::kFixedPrefixSize]) ==
          CompressionType::Uncompressed,
      fmt::format("{} must be uncompressed", field));
  const uint64_t expectedSize =
      kPayloadOffset + FixedBitArray::bufferSize(rowCount, 1);
  require(
      child.size() == expectedSize,
      fmt::format("{} payload size is invalid", field));
  const uint64_t bitmapBytes = (static_cast<uint64_t>(rowCount) + 7) / 8;
  const uint32_t usedBits = rowCount % 8;
  if (usedBits != 0) {
    const auto paddingMask = static_cast<uint8_t>(0xFFU << usedBits);
    require(
        (static_cast<uint8_t>(child[kPayloadOffset + bitmapBytes - 1]) &
         paddingMask) == 0,
        fmt::format("{} has non-zero padding bits", field));
  }
  for (size_t offset = kPayloadOffset + bitmapBytes; offset < child.size();
       ++offset) {
    require(
        child[offset] == 0, fmt::format("{} has non-zero slop bytes", field));
  }
}

// Dispatch a numeric child to the matching validator, restricting RLE children
// to the only two uncompressed encodings the replay path can safely decode.
template <typename T>
void validateUncompressedNumericChild(
    std::string_view child,
    DataType dataType,
    uint32_t rowCount,
    std::string_view field) {
  const auto prefix = readFixedArtifactPrefix(child, field);
  if (prefix.encodingType == EncodingType::Trivial) {
    validateUncompressedTrivialChild<T>(child, dataType, rowCount, field);
    return;
  }
  if (prefix.encodingType == EncodingType::FixedBitWidth) {
    validateUncompressedFixedBitWidthChild<T>(
        child, dataType, rowCount, field, false);
    return;
  }
  fail(fmt::format("{} must use Trivial or FixedBitWidth encoding", field));
}

// Read a single decoded value out of an already-validated numeric child,
// reconstructing FixedBitWidth values as baseline + delta. The index and the
// delta-overflow checks guard against a child whose declared layout does not
// match its contents.
template <typename T>
T readUncompressedNumericChildValue(
    std::string_view child,
    uint32_t index,
    std::string_view field) {
  const auto prefix = readFixedArtifactPrefix(child, field);
  require(index < prefix.rowCount, fmt::format("{} index is invalid", field));
  constexpr size_t kPayloadHeaderOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  if (prefix.encodingType == EncodingType::Trivial) {
    return readBoundedValue<T>(
        child,
        kPayloadHeaderOffset + static_cast<size_t>(index) * sizeof(T),
        field);
  }

  require(
      prefix.encodingType == EncodingType::FixedBitWidth,
      fmt::format("{} has an unsupported encoding", field));
  const auto baseline = readBoundedValue<T>(child, kPayloadHeaderOffset, field);
  const size_t bitWidthOffset = kPayloadHeaderOffset + sizeof(T);
  const auto bitWidth = static_cast<uint8_t>(child[bitWidthOffset]);
  const auto packed = child.substr(bitWidthOffset + sizeof(uint8_t));
  const FixedBitArray values{packed, static_cast<int>(bitWidth)};
  const uint64_t delta = values.get(index);
  require(
      delta <= std::numeric_limits<T>::max() - baseline,
      fmt::format("{} value overflows its data type", field));
  return static_cast<T>(baseline + delta);
}

// Preflight a full RLE artifact before a decoder is ever constructed. The
// run-length and run-value children are bounds-checked, required to share a
// run count, and every run length is summed and required to reproduce exactly
// rowCount rows. Enforcing these invariants here is what lets the consumer
// lanes replay the artifact without exposing the decoder to malformed input.
void validateRleArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  constexpr size_t kRunLengthsSizeOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kRunLengthsOffset = kRunLengthsSizeOffset + sizeof(uint32_t);
  const auto runLengthsSize =
      readBoundedUint32(artifact, kRunLengthsSizeOffset, "RLE run-length size");
  require(
      runLengthsSize <= artifact.size() - kRunLengthsOffset,
      "RLE run-length child exceeds artifact bounds");
  const auto runLengths = artifact.substr(kRunLengthsOffset, runLengthsSize);
  const auto runValues = artifact.substr(kRunLengthsOffset + runLengthsSize);
  const auto runLengthsPrefix =
      readFixedArtifactPrefix(runLengths, "RLE run-length child");
  require(
      runLengthsPrefix.rowCount > 0 && runLengthsPrefix.rowCount <= rowCount,
      "RLE run count is invalid");
  validateUncompressedNumericChild<uint32_t>(
      runLengths,
      DataType::Uint32,
      runLengthsPrefix.rowCount,
      "RLE run-length child");

  const auto runValuesPrefix =
      readFixedArtifactPrefix(runValues, "RLE run-value child");
  require(
      runValuesPrefix.rowCount == runLengthsPrefix.rowCount,
      "RLE child row counts do not match");
  validateUncompressedNumericChild<uint64_t>(
      runValues,
      DataType::Uint64,
      runLengthsPrefix.rowCount,
      "RLE run-value child");

  uint64_t decodedRows{0};
  for (uint32_t run = 0; run < runLengthsPrefix.rowCount; ++run) {
    const auto runLength = readUncompressedNumericChildValue<uint32_t>(
        runLengths, run, "RLE run-length child");
    require(runLength > 0, "RLE run lengths must be positive");
    decodedRows += runLength;
    require(decodedRows <= rowCount, "RLE run lengths exceed row count");
  }
  require(decodedRows == rowCount, "RLE run lengths do not match row count");
}

// Parse the Delta root's value, restatement, and restatement-flag children
// before decoder construction. Bound every child, validate its uncompressed
// wire layout and row relationship, then require the flag bitmap to identify
// the first row and exactly partition all rows between the two value streams.
void validateDeltaArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  constexpr size_t kDeltasSizeOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kRestatementsSizeOffset =
      kDeltasSizeOffset + sizeof(uint32_t);
  constexpr size_t kDeltasOffset = kRestatementsSizeOffset + sizeof(uint32_t);
  const auto deltasSize =
      readBoundedUint32(artifact, kDeltasSizeOffset, "Delta child size");
  const auto restatementsSize = readBoundedUint32(
      artifact, kRestatementsSizeOffset, "Delta restatement child size");
  require(
      deltasSize <= artifact.size() - kDeltasOffset,
      "Delta child exceeds artifact bounds");
  const size_t restatementsOffset = kDeltasOffset + deltasSize;
  require(
      restatementsSize <= artifact.size() - restatementsOffset,
      "Delta restatement child exceeds artifact bounds");
  const size_t flagsOffset = restatementsOffset + restatementsSize;
  const auto deltas = artifact.substr(kDeltasOffset, deltasSize);
  const auto restatements =
      artifact.substr(restatementsOffset, restatementsSize);
  const auto flags = artifact.substr(flagsOffset);

  const auto deltasPrefix =
      readFixedArtifactPrefix(deltas, "Delta value child");
  const auto restatementsPrefix =
      readFixedArtifactPrefix(restatements, "Delta restatement child");
  const auto flagsPrefix =
      readFixedArtifactPrefix(flags, "Delta restatement flag child");
  require(
      deltasPrefix.rowCount <= rowCount,
      "Delta value child row count exceeds root rows");
  require(
      restatementsPrefix.rowCount <= rowCount,
      "Delta restatement child row count exceeds root rows");
  require(
      flagsPrefix.rowCount == rowCount,
      "Delta restatement flag row count does not match root rows");
  validateUncompressedFixedBitWidthChild<uint32_t>(
      deltas,
      DataType::Uint32,
      deltasPrefix.rowCount,
      "Delta value child",
      true);
  validateUncompressedTrivialChild<uint32_t>(
      restatements,
      DataType::Uint32,
      restatementsPrefix.rowCount,
      "Delta restatement child");
  validateUncompressedTrivialBoolChild(
      flags, rowCount, "Delta restatement flag child");

  constexpr size_t kFlagPayloadOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  const FixedBitArray flagValues{flags.substr(kFlagPayloadOffset), 1};
  uint32_t numRestatements{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    numRestatements += flagValues.get(row);
  }
  require(flagValues.get(0) == 1, "Delta first row must be a restatement");
  require(
      numRestatements == restatementsPrefix.rowCount,
      "Delta restatement flags do not match restatement child rows");
  require(
      rowCount - numRestatements == deltasPrefix.rowCount,
      "Delta restatement flags do not match value child rows");
}

void validateSparseBoolArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  constexpr size_t kSparseValueOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kIndicesOffset = kSparseValueOffset + sizeof(uint8_t);
  require(
      artifact.size() >= kIndicesOffset + EncodingPrefix::kFixedPrefixSize,
      "SparseBool indices child is missing");
  require(
      static_cast<uint8_t>(artifact[kSparseValueOffset]) <= 1,
      "SparseBool sparse value must be zero or one");

  const auto indices = artifact.substr(kIndicesOffset);
  const auto indicesPrefix =
      readFixedArtifactPrefix(indices, "SparseBool indices child");
  require(
      indicesPrefix.rowCount > 0 && indicesPrefix.rowCount <= rowCount + 1,
      "SparseBool index count is invalid");
  validateUncompressedFixedBitWidthChild<uint32_t>(
      indices,
      DataType::Uint32,
      indicesPrefix.rowCount,
      "SparseBool indices child",
      false);

  uint32_t previousIndex{0};
  for (uint32_t index = 0; index < indicesPrefix.rowCount; ++index) {
    const auto sparseIndex = readUncompressedNumericChildValue<uint32_t>(
        indices, index, "SparseBool indices child");
    if (index + 1 == indicesPrefix.rowCount) {
      require(
          sparseIndex == rowCount,
          "SparseBool indices must end with the row-count sentinel");
    } else {
      require(sparseIndex < rowCount, "SparseBool index exceeds root rows");
      require(
          index == 0 || sparseIndex > previousIndex,
          "SparseBool indices must be strictly increasing");
    }
    previousIndex = sparseIndex;
  }
}

void validatePforArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  size_t offset = EncodingPrefix::kFixedPrefixSize;
  const auto baseline = readBoundedUint32(artifact, offset, "PFOR baseline");
  offset += sizeof(uint32_t);
  require(offset < artifact.size(), "PFOR base bit width is truncated");
  const auto baseBitWidth = static_cast<uint8_t>(artifact[offset++]);
  require(baseBitWidth <= 32, "PFOR base bit width exceeds uint32");

  const auto numExceptions =
      readBoundedVarint32(artifact, offset, "PFOR exception count");
  require(numExceptions <= rowCount, "PFOR exception count exceeds root rows");

  const auto positionsSize =
      readBoundedVarint32(artifact, offset, "PFOR positions size");
  require(
      positionsSize <= artifact.size() - offset,
      "PFOR positions child exceeds artifact bounds");
  const auto positions = artifact.substr(offset, positionsSize);
  offset += positionsSize;

  const auto valuesSize =
      readBoundedVarint32(artifact, offset, "PFOR values size");
  require(
      valuesSize <= artifact.size() - offset,
      "PFOR values child exceeds artifact bounds");
  const auto values = artifact.substr(offset, valuesSize);
  offset += valuesSize;

  require(
      (numExceptions == 0) == (positionsSize == 0 && valuesSize == 0),
      "PFOR child presence does not match exception count");
  if (numExceptions > 0) {
    require(
        positionsSize > 0 && valuesSize > 0,
        "PFOR exception children must both be present");
  }
  require(
      baseBitWidth < 32 || numExceptions == 0,
      "PFOR full-width payload cannot have exceptions");

  const uint64_t packedSize =
      baseBitWidth == 0 ? 0 : FixedBitArray::bufferSize(rowCount, baseBitWidth);
  require(
      packedSize == artifact.size() - offset,
      "PFOR packed payload size is invalid");
  const auto packed = artifact.substr(offset);
  if (baseBitWidth > 0) {
    validateFixedBitArrayPadding(
        packed, rowCount, baseBitWidth, "PFOR packed payload");
  }
  const std::optional<FixedBitArray> baseResiduals = baseBitWidth == 0
      ? std::nullopt
      : std::optional<FixedBitArray>{FixedBitArray{packed, baseBitWidth}};
  if (baseResiduals.has_value()) {
    const uint32_t maxResidual =
        std::numeric_limits<uint32_t>::max() - baseline;
    for (uint32_t row = 0; row < rowCount; ++row) {
      require(
          baseResiduals->get(row) <= maxResidual,
          "PFOR packed residual overflows uint32");
    }
  }

  if (numExceptions == 0) {
    return;
  }

  validateUncompressedFixedBitWidthChild<uint32_t>(
      positions, DataType::Uint32, numExceptions, "PFOR positions child", true);
  validateUncompressedFixedBitWidthChild<uint32_t>(
      values, DataType::Uint32, numExceptions, "PFOR values child", true);

  constexpr size_t kChildBitWidthOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(uint32_t);
  constexpr size_t kChildPayloadOffset = kChildBitWidthOffset + sizeof(uint8_t);
  const auto positionsBitWidth =
      static_cast<uint8_t>(positions[kChildBitWidthOffset]);
  const auto valuesBitWidth =
      static_cast<uint8_t>(values[kChildBitWidthOffset]);
  validateFixedBitArrayPadding(
      positions.substr(kChildPayloadOffset),
      numExceptions,
      positionsBitWidth,
      "PFOR positions child");
  validateFixedBitArrayPadding(
      values.substr(kChildPayloadOffset),
      numExceptions,
      valuesBitWidth,
      "PFOR values child");

  const uint32_t baseMask = baseBitWidth == 32
      ? std::numeric_limits<uint32_t>::max()
      : baseBitWidth == 0 ? 0
                          : (uint32_t{1} << baseBitWidth) - 1;
  uint32_t previousPosition{0};
  for (uint32_t index = 0; index < numExceptions; ++index) {
    const auto position = readUncompressedNumericChildValue<uint32_t>(
        positions, index, "PFOR positions child");
    require(position < rowCount, "PFOR exception position exceeds root rows");
    require(
        index == 0 || position > previousPosition,
        "PFOR exception positions must be strictly increasing");
    previousPosition = position;

    const auto residual = readUncompressedNumericChildValue<uint32_t>(
        values, index, "PFOR values child");
    require(
        residual > baseMask,
        "PFOR exception residual fits in the base payload");
    require(
        residual <= std::numeric_limits<uint32_t>::max() - baseline,
        "PFOR exception value overflows uint32");
    if (baseResiduals.has_value()) {
      require(
          baseResiduals->get(position) == 0,
          "PFOR exception slot must be zero in the base payload");
    }
  }
}

void validateSimdForBitpackArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  size_t offset = EncodingPrefix::kFixedPrefixSize;
  readBoundedUint32(artifact, offset, "SIMD_FOR baseline");
  offset += sizeof(uint32_t);
  require(offset < artifact.size(), "SIMD_FOR bit width is truncated");
  const auto bitWidth = static_cast<uint8_t>(artifact[offset++]);
  require(
      bitWidth <= std::numeric_limits<uint32_t>::digits,
      "SIMD_FOR bit width exceeds uint32");

  constexpr uint32_t kGroupSize = SimdForBitpackEncoding<uint32_t>::kGroupSize;
  const auto firstGroupRows =
      readBoundedVarint32(artifact, offset, "SIMD_FOR first group rows");
  require(
      firstGroupRows > 0 && firstGroupRows <= std::min(rowCount, kGroupSize),
      "SIMD_FOR first group row count is invalid");
  const uint64_t remainingRows = rowCount - firstGroupRows;
  const uint64_t numGroups = 1 + (remainingRows + kGroupSize - 1) / kGroupSize;
  const uint64_t expectedPackedSize = numGroups * bitWidth * sizeof(uint32_t);
  require(
      expectedPackedSize == artifact.size() - offset,
      "SIMD_FOR packed payload size is invalid");
}

void validateBlockBitPackingArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  size_t offset = EncodingPrefix::kFixedPrefixSize;
  require(offset < artifact.size(), "BlockBitPacking compression is truncated");
  const auto compression = static_cast<CompressionType>(artifact[offset++]);
  require(
      compression == CompressionType::Uncompressed,
      "BlockBitPacking payload must be uncompressed");

  const auto blockSize =
      readBoundedVarint32(artifact, offset, "BlockBitPacking block size");
  require(
      blockSize == kBlockBitPackingBlockSize,
      "BlockBitPacking block size does not match the benchmark profile");
  const auto blockCount =
      readBoundedVarint32(artifact, offset, "BlockBitPacking block count");
  require(
      blockCount > 0 &&
          blockCount <= BlockBitPackingEncoding<uint32_t>::kMaxBlockCount,
      "BlockBitPacking block count is invalid");

  const auto readChild = [&](std::string_view field) {
    const auto childSize = readBoundedVarint32(artifact, offset, field);
    require(
        childSize <= artifact.size() - offset,
        fmt::format("{} exceeds artifact bounds", field));
    const auto child = artifact.substr(offset, childSize);
    offset += childSize;
    return child;
  };
  const auto baselines = readChild("BlockBitPacking baselines size");
  const auto bitWidths = readChild("BlockBitPacking bit widths size");
  const auto blockOffsets = readChild("BlockBitPacking offsets size");
  const auto firstBlockRows =
      readBoundedVarint32(artifact, offset, "BlockBitPacking first block rows");
  require(
      firstBlockRows == std::min(rowCount, blockSize),
      "BlockBitPacking first block row count is invalid");
  const uint64_t remainingRows = rowCount - firstBlockRows;
  const uint64_t expectedBlockCount =
      1 + (remainingRows + blockSize - 1) / blockSize;
  require(
      blockCount == expectedBlockCount,
      "BlockBitPacking block count does not match root rows");

  validateUncompressedTrivialChild<uint32_t>(
      baselines,
      DataType::Uint32,
      blockCount,
      "BlockBitPacking baselines child");
  validateUncompressedTrivialChild<uint8_t>(
      bitWidths,
      DataType::Uint8,
      blockCount,
      "BlockBitPacking bit widths child");
  validateUncompressedTrivialChild<uint32_t>(
      blockOffsets,
      DataType::Uint32,
      blockCount,
      "BlockBitPacking offsets child");

  const auto packed = artifact.substr(offset);
  uint64_t expectedOffset{0};
  for (uint32_t block = 0; block < blockCount; ++block) {
    const auto baseline = readUncompressedNumericChildValue<uint32_t>(
        baselines, block, "BlockBitPacking baselines child");
    const auto bitWidth = readUncompressedNumericChildValue<uint8_t>(
        bitWidths, block, "BlockBitPacking bit widths child");
    const auto actualOffset = readUncompressedNumericChildValue<uint32_t>(
        blockOffsets, block, "BlockBitPacking offsets child");
    require(
        actualOffset == expectedOffset,
        "BlockBitPacking offsets are not contiguous");

    const uint32_t blockRows = block == 0 ? firstBlockRows
        : block + 1 == blockCount
        ? rowCount - firstBlockRows - (blockCount - 2) * blockSize
        : blockSize;
    uint64_t packedSize{0};
    if (bitWidth == BlockBitPackingEncoding<uint32_t>::kRawBlockBitWidth) {
      packedSize = static_cast<uint64_t>(blockRows) * sizeof(uint32_t);
    } else {
      require(
          bitWidth <= std::numeric_limits<uint32_t>::digits,
          "BlockBitPacking bit width exceeds uint32");
      packedSize =
          bitWidth == 0 ? 0 : FixedBitArray::bufferSize(blockRows, bitWidth);
    }
    require(
        packedSize <= packed.size() - expectedOffset,
        "BlockBitPacking block payload exceeds artifact bounds");

    if (bitWidth > 0 &&
        bitWidth != BlockBitPackingEncoding<uint32_t>::kRawBlockBitWidth) {
      const auto blockPayload = packed.substr(expectedOffset, packedSize);
      validateFixedBitArrayPadding(
          blockPayload, blockRows, bitWidth, "BlockBitPacking block payload");
      const FixedBitArray residuals{blockPayload, bitWidth};
      const uint32_t maxResidual =
          std::numeric_limits<uint32_t>::max() - baseline;
      for (uint32_t row = 0; row < blockRows; ++row) {
        require(
            residuals.get(row) <= maxResidual,
            "BlockBitPacking residual overflows uint32");
      }
    }
    expectedOffset += packedSize;
  }
  require(
      expectedOffset == packed.size(),
      "BlockBitPacking packed payload size is invalid");
}

void validateVarintArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  size_t offset = EncodingPrefix::kFixedPrefixSize;
  const auto baseline = readBoundedUint32(artifact, offset, "Varint baseline");
  offset += sizeof(uint32_t);

  bool sawZeroResidual{false};
  for (uint32_t row = 0; row < rowCount; ++row) {
    const auto residual =
        readBoundedVarint32(artifact, offset, "Varint residual");
    require(
        residual <= std::numeric_limits<uint32_t>::max() - baseline,
        "Varint baseline plus residual overflows uint32");
    sawZeroResidual |= residual == 0;
  }
  require(sawZeroResidual, "Varint baseline is not the minimum value");
  require(offset == artifact.size(), "Varint payload has trailing bytes");
}

void validatePrefixArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  validateArtifactPrefix(
      artifact, EncodingType::Prefix, DataType::String, rowCount);
  const size_t intervalOffset = EncodingPrefix::kFixedPrefixSize;
  const uint32_t restartInterval =
      readBoundedUint32(artifact, intervalOffset, "Prefix restart interval");
  require(restartInterval > 0, "Prefix restart interval must be positive");

  const uint32_t numRestarts =
      rowCount == 0 ? 0 : 1 + (rowCount - 1) / restartInterval;
  const size_t restartOffsets = intervalOffset + sizeof(uint32_t);
  const uint64_t restartTableBytes =
      static_cast<uint64_t>(numRestarts) * sizeof(uint32_t);
  require(
      restartTableBytes <= artifact.size() - restartOffsets,
      "Prefix restart offset table is truncated");
  const size_t dataStart =
      restartOffsets + static_cast<size_t>(restartTableBytes);

  size_t cursor = dataStart;
  uint32_t previousLength{0};
  uint64_t totalDecodedBytes{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    if (row % restartInterval == 0) {
      const uint32_t restartIndex = row / restartInterval;
      const uint32_t storedOffset = readBoundedUint32(
          artifact,
          restartOffsets + static_cast<size_t>(restartIndex) * sizeof(uint32_t),
          "Prefix restart offset");
      require(
          storedOffset == cursor - dataStart,
          "Prefix restart offset does not match its entry");
    }

    const uint32_t shared =
        readBoundedUint32(artifact, cursor, "Prefix shared length");
    cursor += sizeof(uint32_t);
    const uint32_t suffix =
        readBoundedUint32(artifact, cursor, "Prefix suffix length");
    cursor += sizeof(uint32_t);
    if (row % restartInterval == 0) {
      require(shared == 0, "Prefix restart entry must have zero shared bytes");
    }
    require(
        shared <= previousLength,
        "Prefix shared length exceeds the previous decoded value");
    require(
        shared <= kMaxStringDecodedValueBytes,
        "Prefix shared length exceeds the value-size limit");
    require(
        suffix <= kMaxStringDecodedValueBytes - shared,
        "Prefix decoded value exceeds the value-size limit");
    const uint32_t decodedLength = shared + suffix;
    require(
        decodedLength <= kMaxStringDecodedValueBytes,
        "Prefix decoded value exceeds the value-size limit");
    require(
        decodedLength <= kMaxStringDecodedBytes - totalDecodedBytes,
        "Prefix decoded values exceed the total-size limit");
    totalDecodedBytes += decodedLength;
    require(
        suffix <= artifact.size() - cursor,
        "Prefix suffix payload is truncated");
    cursor += suffix;
    previousLength = decodedLength;
  }
  require(cursor == artifact.size(), "Prefix payload has trailing bytes");
}

void validateFsstArtifactStructure(
    std::string_view artifact,
    uint32_t rowCount) {
  validateArtifactPrefix(
      artifact, EncodingType::Fsst, DataType::String, rowCount);

  size_t offset = EncodingPrefix::kFixedPrefixSize;
  const uint32_t symbolTableSize =
      readBoundedVarint32(artifact, offset, "FSST symbol table size");
  require(symbolTableSize > 0, "FSST symbol table must not be empty");
  require(
      symbolTableSize <= FSST_MAXHEADER,
      "FSST symbol table exceeds the canonical header limit");
  require(
      symbolTableSize <= artifact.size() - offset,
      "FSST symbol table exceeds artifact bounds");
  offset += symbolTableSize;

  const uint32_t lengthsSize =
      readBoundedVarint32(artifact, offset, "FSST lengths child size");
  require(lengthsSize > 0, "FSST lengths child must not be empty");
  require(
      lengthsSize <= artifact.size() - offset,
      "FSST lengths child exceeds artifact bounds");
  const auto lengths = artifact.substr(offset, lengthsSize);
  offset += lengthsSize;

  validateUncompressedFixedBitWidthChild<uint32_t>(
      lengths, DataType::Uint32, rowCount, "FSST lengths child", false);
  constexpr size_t kLengthsBitWidthOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(uint32_t);
  constexpr size_t kLengthsPayloadOffset =
      kLengthsBitWidthOffset + sizeof(uint8_t);
  const auto bitWidth = static_cast<uint8_t>(lengths[kLengthsBitWidthOffset]);
  validateFixedBitArrayPadding(
      lengths.substr(kLengthsPayloadOffset),
      rowCount,
      bitWidth,
      "FSST lengths child");

  uint64_t totalExpandedBytes{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    const uint32_t length = readUncompressedNumericChildValue<uint32_t>(
        lengths, row, "FSST lengths child");
    require(
        length <= kMaxStringDecodedValueBytes / kFsstMaxExpansion,
        "FSST per-row expansion exceeds the value-size limit");
    const uint64_t expandedBytes =
        static_cast<uint64_t>(length) * kFsstMaxExpansion;
    require(
        expandedBytes <= kMaxStringDecodedBytes - totalExpandedBytes,
        "FSST total expansion exceeds the decoded-byte limit");
    totalExpandedBytes += expandedBytes;
  }

  const auto blob = artifact.substr(offset);
  size_t blobOffset{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    const uint32_t length = readUncompressedNumericChildValue<uint32_t>(
        lengths, row, "FSST lengths child");
    require(
        length <= blob.size() - blobOffset,
        "FSST compressed length exceeds the remaining blob");
    const auto compressed = blob.substr(blobOffset, length);
    size_t codeOffset{0};
    while (codeOffset < compressed.size()) {
      if (static_cast<uint8_t>(compressed[codeOffset]) == FSST_ESC) {
        require(
            compressed.size() - codeOffset >= 2,
            "FSST escape code has no literal byte in its row");
        codeOffset += 2;
      } else {
        ++codeOffset;
      }
    }
    blobOffset += length;
  }
  require(blobOffset == blob.size(), "FSST lengths do not match the blob");
}

TaskSpec parseTaskId(std::string_view taskId) {
  constexpr std::string_view kPrefix{"nimble."};
  constexpr std::string_view kSuffix{".v1"};
  if (!taskId.starts_with(kPrefix) || !taskId.ends_with(kSuffix)) {
    throw std::invalid_argument(
        "task_id must match nimble.<encoding>.<lane>.v1");
  }

  const auto body = taskId.substr(
      kPrefix.size(), taskId.size() - kPrefix.size() - kSuffix.size());
  for (const auto& encoding : kEncodings) {
    const auto encodingPrefix = fmt::format("{}.", encoding.slug);
    if (!body.starts_with(encodingPrefix)) {
      continue;
    }
    const auto laneName = body.substr(encodingPrefix.size());
    for (const auto& [candidateName, lane] : kLanes) {
      if (laneName != candidateName) {
        continue;
      }
      if ((encoding.supportedLanes & laneBit(lane)) == 0) {
        throw std::invalid_argument(
            fmt::format(
                "{} does not support {} in the executable runner",
                encoding.slug,
                laneName));
      }
      return TaskSpec{
          .encoding = encoding.encoding,
          .lane = lane,
          .encodingSlug = encoding.slug,
          .laneName = candidateName,
          .encodingType = encoding.encodingType,
          .dataType = encoding.dataType,
      };
    }
    throw std::invalid_argument(
        fmt::format("unsupported executable lane: {}", laneName));
  }
  throw std::invalid_argument(
      "encoding runner currently supports RLE, Dictionary, FixedBitWidth, "
      "Delta, SparseBool, PFOR, SimdForBitpack, BlockBitPacking, Varint, "
      "Prefix, Fsst, Nullable, ALP, and DeltaBlock");
}

TaskSpec validateConfig(const EncodingRunnerConfig& config) {
  if (config.rowCount < 100) {
    throw std::invalid_argument("row_count must be at least 100");
  }
  if (config.rowCount > kMaxRowCount) {
    throw std::invalid_argument("row_count exceeds the runner limit");
  }
  if (config.samples == 0) {
    throw std::invalid_argument("samples must be positive");
  }
  if (config.samples > kMaxSamples) {
    throw std::invalid_argument("samples exceeds the runner limit");
  }
  if (config.warmups > kMaxWarmups) {
    throw std::invalid_argument("warmups exceeds the runner limit");
  }
  if (config.minSampleTimeMicros > kMaxMinSampleTimeMicros) {
    throw std::invalid_argument(
        "min_sample_time_micros exceeds the runner limit");
  }
  if (config.innerIterations == 0) {
    throw std::invalid_argument("inner_iterations must be positive");
  }
  if (config.innerIterations > kMaxCalibratedIterations) {
    throw std::invalid_argument("inner_iterations exceeds the runner limit");
  }
  if (config.seed >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "seed must fit in a signed 64-bit JSON integer");
  }
  return parseTaskId(config.taskId);
}

std::string digest(std::span<const std::byte> bytes) {
  std::array<uint8_t, 32> result{};
  folly::ssl::OpenSSLHash::sha256(
      folly::MutableByteRange{result.data(), result.size()},
      folly::ByteRange{
          reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()});
  return folly::hexlify(folly::ByteRange{result.data(), result.size()});
}

std::string digest(std::string_view value) {
  return digest(std::as_bytes(std::span{value.data(), value.size()}));
}

void appendUint64LittleEndian(std::string& output, uint64_t value) {
  for (uint32_t byte = 0; byte < sizeof(value); ++byte) {
    output.push_back(
        static_cast<char>(static_cast<uint8_t>(value >> (byte * 8))));
  }
}

std::string semanticStringDigest(
    std::span<const std::string_view> values,
    std::span<const bool> nonNulls,
    std::string_view dataType) {
  require(
      values.size() == nonNulls.size(),
      "semantic string digest nullable metadata size mismatch");
  std::string canonical{"nimble-semantic-digest-v1"};
  appendUint64LittleEndian(canonical, dataType.size());
  canonical.append(dataType);
  appendUint64LittleEndian(canonical, values.size());
  appendUint64LittleEndian(canonical, nonNulls.size());
  for (size_t index = 0; index < values.size(); ++index) {
    canonical.push_back(nonNulls[index] ? '\1' : '\0');
    const std::string_view value =
        nonNulls[index] ? values[index] : std::string_view{};
    appendUint64LittleEndian(canonical, value.size());
    canonical.append(value);
  }
  return digest(canonical);
}

std::string semanticDigest(
    std::span<const std::string_view> values,
    std::span<const bool> nonNulls,
    std::string_view dataType) {
  return semanticStringDigest(values, nonNulls, dataType);
}

template <typename T>
std::string semanticDigest(
    std::span<const T> values,
    std::span<const bool> nonNulls,
    std::string_view dataType) {
  const auto valueBytes =
      std::as_bytes(std::span{values.data(), values.size()});
  std::string canonical;
  canonical.reserve(dataType.size() + 1 + nonNulls.size() + valueBytes.size());
  canonical.append(dataType);
  canonical.push_back('\0');
  for (const bool nonNull : nonNulls) {
    canonical.push_back(nonNull ? '\1' : '\0');
  }
  canonical.append(
      reinterpret_cast<const char*>(valueBytes.data()), valueBytes.size());
  return digest(canonical);
}

template <typename T>
typename TypeTraits<T>::physicalType toPhysical(T value) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  static_assert(sizeof(T) == sizeof(PhysicalType));
  return std::bit_cast<PhysicalType>(value);
}

template <typename T>
struct Corpus {
  using PhysicalType = typename TypeTraits<T>::physicalType;

  explicit Corpus(velox::memory::MemoryPool& pool, uint32_t rowCount)
      : logicalValues{&pool},
        expectedPhysical{&pool},
        nonNulls{&pool, rowCount, true} {
    logicalValues.reserve(rowCount);
    expectedPhysical.reserve(rowCount);
  }

  Vector<T> logicalValues;
  Vector<PhysicalType> expectedPhysical;
  Vector<bool> nonNulls;
};

Corpus<int64_t> makeIntegerCorpus(
    const TaskSpec& spec,
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<int64_t> corpus{pool, config.rowCount};
  std::mt19937_64 rng{config.seed};

  if (spec.encoding == RunnerEncoding::DeltaBlock) {
    int64_t value{-1'000'000};
    for (uint32_t row = 0; row < config.rowCount; ++row) {
      value += static_cast<int64_t>(rng() % 17);
      corpus.logicalValues.push_back(value);
      corpus.expectedPhysical.push_back(toPhysical(value));
    }
    return corpus;
  }

  if (spec.encoding == RunnerEncoding::RLE) {
    while (corpus.logicalValues.size() < config.rowCount) {
      int64_t value = static_cast<int64_t>(rng() % 64) - 32;
      if (!corpus.logicalValues.empty() &&
          value == corpus.logicalValues.back()) {
        value = value == 31 ? -32 : value + 1;
      }
      const uint32_t remaining =
          config.rowCount - static_cast<uint32_t>(corpus.logicalValues.size());
      const uint32_t runLength = std::min<uint32_t>(1 + rng() % 31, remaining);
      for (uint32_t index = 0; index < runLength; ++index) {
        corpus.logicalValues.push_back(value);
        corpus.expectedPhysical.push_back(toPhysical(value));
      }
    }
    return corpus;
  }

  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const bool isNonNull = spec.encoding != RunnerEncoding::Nullable ||
        (row != 0 && rng() % 5 != 0);
    corpus.nonNulls[row] = isNonNull;
    const int64_t value = static_cast<int64_t>(rng() % 64) - 32;
    if (isNonNull) {
      corpus.logicalValues.push_back(value);
      corpus.expectedPhysical.push_back(toPhysical(value));
    } else {
      corpus.expectedPhysical.push_back(0);
    }
  }
  return corpus;
}

Corpus<double> makeDoubleCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<double> corpus{pool, config.rowCount};
  std::mt19937_64 rng{config.seed};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    double value =
        static_cast<double>(static_cast<int64_t>(rng() % 100'001) - 50'000) /
        100.0;
    if (row % 257 == 0) {
      value = -0.0;
    } else if (row % 1021 == 0) {
      value = std::numeric_limits<double>::infinity();
    }
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint64_t> makeFixedBitWidthCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  constexpr int kBitWidth = 40;
  Corpus<uint64_t> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    uint64_t value = fixedBitWidthBenchmarkValue(config.seed + row, kBitWidth);
    if (row == 0) {
      value = fixedBitWidthBenchmarkBaseline(kBitWidth);
    } else if (row == 1) {
      value = fixedBitWidthBenchmarkBaseline(kBitWidth) +
          fixedBitWidthBenchmarkMask(kBitWidth);
    }
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint32_t> makeDeltaCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<uint32_t> corpus{pool, config.rowCount};
  uint32_t value = deltaBenchmarkInitialValue(config.seed);
  corpus.logicalValues.push_back(value);
  corpus.expectedPhysical.push_back(toPhysical(value));
  for (uint32_t row = 1; row < config.rowCount; ++row) {
    value += deltaBenchmarkIncrement(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint32_t> makePforCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<uint32_t> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const uint32_t value = pforBenchmarkValue(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint32_t> makeSimdForBitpackCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<uint32_t> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const uint32_t value = simdForBitpackBenchmarkValue(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint32_t> makeBlockBitPackingCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<uint32_t> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const uint32_t value = blockBitPackingBenchmarkValue(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<uint32_t> makeVarintCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<uint32_t> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const uint32_t value = varintBenchmarkValue(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

Corpus<bool> makeSparseBoolCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<bool> corpus{pool, config.rowCount};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const bool value = sparseBoolBenchmarkValue(row, config.seed);
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(value);
  }
  return corpus;
}

EncodingLayout trivialLayout() {
  return EncodingLayout{
      EncodingType::Trivial, {}, CompressionType::Uncompressed};
}

EncodingLayout fixedBitWidthLayout() {
  return EncodingLayout{
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
}

EncodingLayout varintLayout() {
  return EncodingLayout{
      EncodingType::Varint, {}, CompressionType::Uncompressed};
}

EncodingLayout replayLayout(RunnerEncoding encoding) {
  switch (encoding) {
    case RunnerEncoding::RLE:
      return EncodingLayout{
          EncodingType::RLE,
          {},
          CompressionType::Uncompressed,
          {trivialLayout(), fixedBitWidthLayout()}};
    case RunnerEncoding::Dictionary:
      return EncodingLayout{
          EncodingType::Dictionary,
          {},
          CompressionType::Uncompressed,
          {trivialLayout(), fixedBitWidthLayout()}};
    case RunnerEncoding::FixedBitWidth:
      return fixedBitWidthLayout();
    case RunnerEncoding::Delta:
      return EncodingLayout{
          EncodingType::Delta,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout(), trivialLayout(), trivialLayout()}};
    case RunnerEncoding::SparseBool:
      return EncodingLayout{
          EncodingType::SparseBool,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout()}};
    case RunnerEncoding::PFOR:
      return EncodingLayout{
          EncodingType::PFOR,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout(), fixedBitWidthLayout()}};
    case RunnerEncoding::SimdForBitpack:
      return EncodingLayout{
          EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
    case RunnerEncoding::BlockBitPacking:
      return EncodingLayout{
          EncodingType::BlockBitPacking,
          {},
          CompressionType::Uncompressed,
          {trivialLayout(), trivialLayout(), trivialLayout()}};
    case RunnerEncoding::Varint:
      return varintLayout();
    case RunnerEncoding::Prefix:
      return EncodingLayout{
          EncodingType::Prefix, {}, CompressionType::Uncompressed};
    case RunnerEncoding::Fsst:
      return EncodingLayout{
          EncodingType::Fsst,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout()}};
    case RunnerEncoding::Nullable:
      return trivialLayout();
    case RunnerEncoding::ALP:
      return EncodingLayout{
          EncodingType::ALP,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout(), varintLayout(), trivialLayout()}};
    case RunnerEncoding::DeltaBlock:
      return EncodingLayout{
          EncodingType::DeltaBlock, {}, CompressionType::Uncompressed};
  }
  fail("unknown runner encoding");
}

EncodingSelectionPolicyCreator fallbackPolicyCreator() {
  return [](DataType dataType) {
    return ManualEncodingSelectionPolicyFactory{
        {{EncodingType::Trivial, 1.0}},
        /*compressionOptions=*/std::nullopt}
        .createPolicy(dataType);
  };
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> replayPolicy(
    RunnerEncoding encoding) {
  return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      replayLayout(encoding),
      /*compressionOptions=*/std::nullopt,
      fallbackPolicyCreator());
}

Encoding::Options fsstOptions() {
  Encoding::Options options;
  options.fsstCompressionTargetRatio = std::numeric_limits<double>::max();
  return options;
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> selectionPolicy(
    EncodingType encodingType) {
  ManualEncodingSelectionPolicyFactory factory{
      {{encodingType, 1.0}},
      /*compressionOptions=*/std::nullopt};
  auto policy = factory.createPolicy(TypeTraits<T>::dataType);
  return std::unique_ptr<EncodingSelectionPolicy<T>>{
      static_cast<EncodingSelectionPolicy<T>*>(policy.release())};
}

std::vector<uint32_t>
scatterPositions(uint32_t rowCount, uint32_t percentage, uint64_t seed) {
  const uint32_t count = std::max<uint32_t>(
      1,
      static_cast<uint32_t>(
          static_cast<uint64_t>(rowCount) * percentage / 100));
  const uint32_t rotation = static_cast<uint32_t>(seed % rowCount);
  std::vector<uint32_t> positions;
  positions.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    positions.push_back(
        (static_cast<uint64_t>(index) * rowCount / count + rotation) %
        rowCount);
  }
  std::sort(positions.begin(), positions.end());
  require(
      std::adjacent_find(positions.begin(), positions.end()) == positions.end(),
      "scatter position generation produced a duplicate");
  return positions;
}

std::vector<uint32_t> randomViewPositions(uint32_t rowCount, uint64_t seed) {
  auto positions = scatterPositions(rowCount, 10, seed);
  std::mt19937_64 rng{seed ^ 0x9E3779B97F4A7C15ULL};
  for (size_t remaining = positions.size(); remaining > 1; --remaining) {
    std::swap(positions[remaining - 1], positions[rng() % remaining]);
  }
  if (std::is_sorted(positions.begin(), positions.end())) {
    std::rotate(positions.begin(), positions.begin() + 1, positions.end());
  }
  return positions;
}

std::string makeDecoderBacking(
    RunnerEncoding encoding,
    std::string_view artifact) {
  if (encoding != RunnerEncoding::Varint) {
    return {};
  }
  std::string backing{artifact};
  backing.append(kVarintBulkDecodePaddingBytes, '\0');
  return backing;
}

class StringPageArena {
 public:
  explicit StringPageArena(velox::memory::MemoryPool& pool) : pool_{pool} {}

  void* allocate(uint32_t bytes) {
    require(bytes > 0, "string page must not be empty");
    require(
        bytes <= kMaxStringDecodedValueBytes,
        "string page exceeds the value-size limit");
    require(
        pages_.size() < kMaxStringPages,
        "string page count exceeds the runner limit");
    require(
        bytes <= kMaxStringDecodedBytes - allocatedBytes_,
        "string pages exceed the runner byte limit");
    auto& page = pages_.emplace_back(
        velox::AlignedBuffer::allocate<char>(bytes, &pool_));
    allocatedBytes_ += bytes;
    return page->asMutable<void>();
  }

  size_t pageCount() const {
    return pages_.size();
  }

  uint64_t allocatedBytes() const {
    return allocatedBytes_;
  }

 private:
  velox::memory::MemoryPool& pool_;
  std::vector<velox::BufferPtr> pages_;
  uint64_t allocatedBytes_{0};
};

class PrefixRunner {
 public:
  PrefixRunner(
      const EncodingRunnerConfig& config,
      TaskSpec spec,
      velox::memory::MemoryPool& pool,
      std::optional<std::string_view> artifact = std::nullopt)
      : config_{config},
        spec_{spec},
        pool_{pool},
        corpus_{makePrefixBenchmarkCorpus(config.rowCount, config.seed)},
        nonNulls_{&pool_, config.rowCount, true},
        encodedArtifact_{
            artifact.has_value() ? std::string{*artifact} : encodeArtifact()},
        timingBuffer_{pool_},
        stringPages_{pool_},
        output_{&pool_, config.rowCount},
        skipSeekOutput_{&pool_, config.rowCount} {
    require(!encodedArtifact_.empty(), "encoded artifact must not be empty");
    require(
        encodedArtifact_.size() <= kMaxEncodingArtifactBytes,
        "encoded artifact exceeds the runner size limit");
    validatePrefixArtifactStructure(encodedArtifact_, config_.rowCount);
    decoder_ = createEncoding(encodedArtifact_);
    validateGold();
    stablePageCount_ = stringPages_.pageCount();
    stablePageBytes_ = stringPages_.allocatedBytes();
  }

  EncodingRunnerMeasurement run() {
    const uint32_t iterations = calibrateIterations();
    runIterations(iterations);
    validateTimedResult();
    for (uint32_t warmup = 0; warmup < config_.warmups; ++warmup) {
      runIterations(iterations);
      validateTimedResult();
    }

    std::vector<double> samples;
    samples.reserve(config_.samples);
    for (uint32_t sample = 0; sample < config_.samples; ++sample) {
      const double elapsed = runIterations(iterations) / iterations;
      require(
          std::isfinite(elapsed) && elapsed > 0.0,
          "timing sample must be finite and positive");
      samples.push_back(elapsed);
      validateTimedResult();
    }

    return EncodingRunnerMeasurement{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .lane = std::string{spec_.laneName},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
        .samplesSeconds = std::move(samples),
        .encodedArtifact = encodedArtifact_,
    };
  }

  EncodingArtifactVerification verification() const {
    return EncodingArtifactVerification{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
    };
  }

 private:
  std::span<const bool> nonNulls() const {
    return {nonNulls_.data(), nonNulls_.size()};
  }

  std::string inputDigest() const {
    return semanticStringDigest(corpus_.values, nonNulls(), spec_.dataType);
  }

  std::string_view encodeToBuffer(Buffer& buffer) const {
    return EncodingFactory::encode<std::string_view>(
        replayPolicy<std::string_view>(RunnerEncoding::Prefix),
        corpus_.values,
        buffer,
        options_);
  }

  std::string encodeArtifact() const {
    Buffer buffer{pool_};
    return std::string{encodeToBuffer(buffer)};
  }

  std::unique_ptr<Encoding> createEncoding(std::string_view artifact) {
    return EncodingFactory{options_}.create(
        pool_, artifact, [this](uint32_t bytes) -> void* {
          return stringPages_.allocate(bytes);
        });
  }

  void validateGold() {
    require(
        decoder_->encodingType() == EncodingType::Prefix,
        "artifact root encoding does not match task");
    require(
        decoder_->rowCount() == config_.rowCount,
        "artifact row count does not match corpus");

    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    require(
        std::equal(output_.begin(), output_.end(), corpus_.values.begin()),
        "dense round trip failed");
    outputDigest_ = semanticStringDigest(
        std::span<const std::string_view>{output_.data(), output_.size()},
        nonNulls(),
        spec_.dataType);

    runSkipSeek();
    validateSkipSeek();
    decoder_->reset();
  }

  void runEncode() {
    timingBuffer_.reset();
    timedArtifact_ = encodeToBuffer(timingBuffer_);
    folly::doNotOptimizeAway(timedArtifact_.size());
  }

  void runDense() {
    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    folly::doNotOptimizeAway(output_.back());
  }

  void runSkipSeek() {
    decoder_->reset();
    uint32_t cursor{0};
    skipSeekReadCount_ = 0;
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      decoder_->skip(skip);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      decoder_->materialize(read, skipSeekOutput_.data() + skipSeekReadCount_);
      skipSeekReadCount_ += read;
      cursor += read;
    }
    require(skipSeekReadCount_ > 0, "skip_seek did not read a value");
    folly::doNotOptimizeAway(skipSeekOutput_[skipSeekReadCount_ - 1]);
  }

  void validateSkipSeek() const {
    uint32_t cursor{0};
    uint32_t outputIndex{0};
    while (cursor < config_.rowCount) {
      cursor += std::min<uint32_t>(31, config_.rowCount - cursor);
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      require(
          std::equal(
              skipSeekOutput_.begin() + outputIndex,
              skipSeekOutput_.begin() + outputIndex + read,
              corpus_.values.begin() + cursor),
          "timed skip_seek decode failed");
      outputIndex += read;
      cursor += read;
    }
    require(
        outputIndex == skipSeekReadCount_,
        "timed skip_seek returned the wrong value count");
  }

  void validateAllocationHighWater() const {
    require(
        stringPages_.pageCount() == stablePageCount_ &&
            stringPages_.allocatedBytes() == stablePageBytes_,
        "Prefix decode allocated string pages during timing");
  }

  void validateTimedResult() const {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        require(
            timedArtifact_ == encodedArtifact_,
            "timed encode did not reproduce the validated artifact");
        return;
      case RunnerLane::DecodeDense:
        require(
            std::equal(output_.begin(), output_.end(), corpus_.values.begin()),
            "timed dense decode failed");
        validateAllocationHighWater();
        return;
      case RunnerLane::SkipSeek:
        validateSkipSeek();
        validateAllocationHighWater();
        return;
      default:
        fail("Prefix runner received an unsupported lane");
    }
  }

  void runOnce() {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        runEncode();
        return;
      case RunnerLane::DecodeDense:
        runDense();
        return;
      case RunnerLane::SkipSeek:
        runSkipSeek();
        return;
      default:
        fail("Prefix runner received an unsupported lane");
    }
  }

  double runIterations(uint32_t iterations) {
    const auto start = Clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      runOnce();
    }
    return std::chrono::duration<double>{Clock::now() - start}.count();
  }

  uint32_t calibrateIterations() {
    uint32_t iterations = config_.innerIterations;
    if (config_.minSampleTimeMicros == 0) {
      return iterations;
    }
    const double targetSeconds =
        static_cast<double>(config_.minSampleTimeMicros) / 1'000'000.0;
    for (uint32_t attempt = 0; attempt < 16; ++attempt) {
      const double elapsed = runIterations(iterations);
      if (elapsed >= targetSeconds) {
        return iterations;
      }
      const double safeElapsed = std::max(elapsed, 1e-9);
      const auto multiplier = static_cast<uint32_t>(
          std::clamp(std::ceil(targetSeconds / safeElapsed), 2.0, 16.0));
      if (iterations > kMaxCalibratedIterations / multiplier) {
        fail("timing iteration calibration overflowed");
      }
      iterations *= multiplier;
    }
    fail("timing iteration calibration did not reach the requested duration");
  }

  const EncodingRunnerConfig& config_;
  TaskSpec spec_;
  velox::memory::MemoryPool& pool_;
  StringBenchmarkCorpus corpus_;
  Vector<bool> nonNulls_;
  Encoding::Options options_{};
  std::string encodedArtifact_;
  Buffer timingBuffer_;
  StringPageArena stringPages_;
  std::unique_ptr<Encoding> decoder_;
  Vector<std::string_view> output_;
  Vector<std::string_view> skipSeekOutput_;
  std::string outputDigest_;
  std::string_view timedArtifact_;
  size_t stablePageCount_{0};
  uint64_t stablePageBytes_{0};
  uint32_t skipSeekReadCount_{0};
};

class FsstRunner {
 public:
  FsstRunner(
      const EncodingRunnerConfig& config,
      TaskSpec spec,
      velox::memory::MemoryPool& pool,
      std::optional<std::string_view> artifact = std::nullopt)
      : config_{config},
        spec_{spec},
        pool_{pool},
        corpus_{makeFsstBenchmarkCorpus(config.rowCount, config.seed)},
        nonNulls_{&pool_, config.rowCount, true},
        options_{fsstOptions()},
        encodedArtifact_{
            artifact.has_value() ? std::string{*artifact} : encodeArtifact()},
        timingBuffer_{pool_},
        stringPages_{pool_},
        output_{&pool_, config.rowCount},
        skipSeekOutput_{&pool_, config.rowCount} {
    require(!encodedArtifact_.empty(), "encoded artifact must not be empty");
    require(
        encodedArtifact_.size() <= kMaxEncodingArtifactBytes,
        "encoded artifact exceeds the runner size limit");
    validateFsstArtifactStructure(encodedArtifact_, config_.rowCount);
    require(
        encodedArtifact_.size() < corpus_.rawBytes,
        "FSST full artifact did not compress the benchmark corpus");
    pristineArtifact_ = encodedArtifact_;
    inputDigest_ = inputDigest();
    artifactDigest_ = digest(encodedArtifact_);
    decoder_ = createEncoding(encodedArtifact_);
    validateGold();
    require(
        inputDigest_ == outputDigest_,
        "FSST output digest does not match its stable input digest");
    stablePageCount_ = stringPages_.pageCount();
    stablePageBytes_ = stringPages_.allocatedBytes();
  }

  EncodingRunnerMeasurement run() {
    const uint32_t iterations = calibrateIterations();
    runIterations(iterations);
    validateTimedResult();
    for (uint32_t warmup = 0; warmup < config_.warmups; ++warmup) {
      runIterations(iterations);
      validateTimedResult();
    }

    std::vector<double> samples;
    samples.reserve(config_.samples);
    for (uint32_t sample = 0; sample < config_.samples; ++sample) {
      const double elapsed = runIterations(iterations) / iterations;
      require(
          std::isfinite(elapsed) && elapsed > 0.0,
          "timing sample must be finite and positive");
      samples.push_back(elapsed);
      validateTimedResult();
    }

    return EncodingRunnerMeasurement{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .lane = std::string{spec_.laneName},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest_,
        .outputDigest = outputDigest_,
        .artifactDigest = artifactDigest_,
        .samplesSeconds = std::move(samples),
        .encodedArtifact = encodedArtifact_,
    };
  }

  EncodingArtifactVerification verification() const {
    return EncodingArtifactVerification{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest_,
        .outputDigest = outputDigest_,
        .artifactDigest = artifactDigest_,
    };
  }

 private:
  std::span<const bool> nonNulls() const {
    return {nonNulls_.data(), nonNulls_.size()};
  }

  std::string inputDigest() const {
    return semanticStringDigest(corpus_.values, nonNulls(), spec_.dataType);
  }

  std::string_view encodeToBuffer(Buffer& buffer) const {
    return EncodingFactory::encode<std::string_view>(
        replayPolicy<std::string_view>(RunnerEncoding::Fsst),
        corpus_.values,
        buffer,
        options_);
  }

  std::string encodeArtifact() const {
    Buffer buffer{pool_};
    return std::string{encodeToBuffer(buffer)};
  }

  std::unique_ptr<Encoding> createEncoding(std::string_view artifact) {
    return EncodingFactory{options_}.create(
        pool_, artifact, [this](uint32_t bytes) -> void* {
          return stringPages_.allocate(bytes);
        });
  }

  void validateGold() {
    require(
        decoder_->encodingType() == EncodingType::Fsst,
        "artifact root encoding does not match task");
    require(
        decoder_->dataType() == DataType::String,
        "artifact data type does not match task");
    require(
        decoder_->rowCount() == config_.rowCount,
        "artifact row count does not match corpus");

    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    require(
        std::equal(output_.begin(), output_.end(), corpus_.values.begin()),
        "dense round trip failed");
    outputDigest_ = semanticStringDigest(
        std::span<const std::string_view>{output_.data(), output_.size()},
        nonNulls(),
        spec_.dataType);

    runSkipSeek();
    validateSkipSeek();
    decoder_->reset();
  }

  void runEncode() {
    timingBuffer_.reset();
    timedArtifact_ = encodeToBuffer(timingBuffer_);
    folly::doNotOptimizeAway(timedArtifact_.size());
  }

  void runDense() {
    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    folly::doNotOptimizeAway(output_.back());
  }

  void runSkipSeek() {
    decoder_->reset();
    uint32_t cursor{0};
    skipSeekReadCount_ = 0;
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      decoder_->skip(skip);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      decoder_->materialize(read, skipSeekOutput_.data() + skipSeekReadCount_);
      skipSeekReadCount_ += read;
      cursor += read;
    }
    require(skipSeekReadCount_ > 0, "skip_seek did not read a value");
    folly::doNotOptimizeAway(skipSeekOutput_[skipSeekReadCount_ - 1]);
  }

  void validateSkipSeek() const {
    uint32_t cursor{0};
    uint32_t outputIndex{0};
    while (cursor < config_.rowCount) {
      cursor += std::min<uint32_t>(31, config_.rowCount - cursor);
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      require(
          std::equal(
              skipSeekOutput_.begin() + outputIndex,
              skipSeekOutput_.begin() + outputIndex + read,
              corpus_.values.begin() + cursor),
          "timed skip_seek decode failed");
      outputIndex += read;
      cursor += read;
    }
    require(
        outputIndex == skipSeekReadCount_,
        "timed skip_seek returned the wrong value count");
  }

  void validateAllocationHighWater() const {
    require(
        stringPages_.pageCount() == stablePageCount_ &&
            stringPages_.allocatedBytes() == stablePageBytes_,
        "FSST decode allocated string pages during timing");
  }

  void validateArtifactStability() const {
    require(
        encodedArtifact_ == pristineArtifact_,
        "FSST canonical artifact changed during execution");
    require(
        digest(encodedArtifact_) == artifactDigest_,
        "FSST artifact digest changed during execution");
    require(
        inputDigest() == inputDigest_,
        "FSST input content digest changed during execution");
  }

  void validateTimedResult() const {
    validateArtifactStability();
    switch (spec_.lane) {
      case RunnerLane::Encode:
        require(
            timedArtifact_ == encodedArtifact_,
            "timed encode did not reproduce the validated artifact");
        return;
      case RunnerLane::DecodeDense:
        require(
            std::equal(output_.begin(), output_.end(), corpus_.values.begin()),
            "timed dense decode failed");
        validateAllocationHighWater();
        return;
      case RunnerLane::SkipSeek:
        validateSkipSeek();
        validateAllocationHighWater();
        return;
      default:
        fail("FSST runner received an unsupported lane");
    }
  }

  void runOnce() {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        runEncode();
        return;
      case RunnerLane::DecodeDense:
        runDense();
        return;
      case RunnerLane::SkipSeek:
        runSkipSeek();
        return;
      default:
        fail("FSST runner received an unsupported lane");
    }
  }

  double runIterations(uint32_t iterations) {
    const auto start = Clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      runOnce();
    }
    return std::chrono::duration<double>{Clock::now() - start}.count();
  }

  uint32_t calibrateIterations() {
    uint32_t iterations = config_.innerIterations;
    if (config_.minSampleTimeMicros == 0) {
      return iterations;
    }
    const double targetSeconds =
        static_cast<double>(config_.minSampleTimeMicros) / 1'000'000.0;
    for (uint32_t attempt = 0; attempt < 16; ++attempt) {
      const double elapsed = runIterations(iterations);
      if (elapsed >= targetSeconds) {
        return iterations;
      }
      const double safeElapsed = std::max(elapsed, 1e-9);
      const auto multiplier = static_cast<uint32_t>(
          std::clamp(std::ceil(targetSeconds / safeElapsed), 2.0, 16.0));
      if (iterations > kMaxCalibratedIterations / multiplier) {
        fail("timing iteration calibration overflowed");
      }
      iterations *= multiplier;
    }
    fail("timing iteration calibration did not reach the requested duration");
  }

  const EncodingRunnerConfig& config_;
  TaskSpec spec_;
  velox::memory::MemoryPool& pool_;
  StringBenchmarkCorpus corpus_;
  Vector<bool> nonNulls_;
  Encoding::Options options_;
  std::string encodedArtifact_;
  std::string pristineArtifact_;
  std::string inputDigest_;
  std::string outputDigest_;
  std::string artifactDigest_;
  Buffer timingBuffer_;
  StringPageArena stringPages_;
  std::unique_ptr<Encoding> decoder_;
  Vector<std::string_view> output_;
  Vector<std::string_view> skipSeekOutput_;
  std::string_view timedArtifact_;
  size_t stablePageCount_{0};
  uint64_t stablePageBytes_{0};
  uint32_t skipSeekReadCount_{0};
};

template <typename T>
class TypedRunner {
 public:
  using PhysicalType = typename TypeTraits<T>::physicalType;

  TypedRunner(
      const EncodingRunnerConfig& config,
      TaskSpec spec,
      velox::memory::MemoryPool& pool,
      std::optional<std::string_view> artifact = std::nullopt)
      : config_{config},
        spec_{spec},
        pool_{pool},
        corpus_{makeCorpus()},
        encodedArtifact_{
            artifact.has_value() ? std::string{*artifact} : encodeArtifact()},
        decoderBacking_{makeDecoderBacking(spec_.encoding, encodedArtifact_)},
        timingBuffer_{pool_},
        output_{&pool_, config.rowCount},
        rangeOutput_{&pool_, config.rowCount / 2},
        scatter10_{scatterPositions(config.rowCount, 10, config.seed)},
        scatter1_{scatterPositions(config.rowCount, 1, config.seed)},
        viewPositions_{randomViewPositions(config.rowCount, config.seed)},
        scatterOutput_{&pool_, scatter10_.size()},
        skipSeekOutput_{&pool_, config.rowCount},
        viewOutput_{&pool_, viewPositions_.size()} {
    validateArtifact();
    decoder_ = createEncoding(decoderArtifact());
    if (spec_.lane == RunnerLane::ViewRandom) {
      view_ = createEncodingView(decoderArtifact(), &pool_, options_);
    }
  }

  EncodingRunnerMeasurement run() {
    if (spec_.lane == RunnerLane::SelectionE2E) {
      validateSelection();
    }
    const uint32_t iterations = calibrateIterations();
    for (uint32_t warmup = 0; warmup < config_.warmups; ++warmup) {
      runIterations(iterations);
      validateTimedResult();
    }

    std::vector<double> samples;
    samples.reserve(config_.samples);
    for (uint32_t sample = 0; sample < config_.samples; ++sample) {
      const double elapsed = runIterations(iterations) / iterations;
      require(
          std::isfinite(elapsed) && elapsed > 0.0,
          "timing sample must be finite and positive");
      samples.push_back(elapsed);
      validateTimedResult();
    }

    return EncodingRunnerMeasurement{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .lane = std::string{spec_.laneName},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
        .samplesSeconds = std::move(samples),
        .encodedArtifact = encodedArtifact_,
    };
  }

  EncodingArtifactVerification verification() const {
    return EncodingArtifactVerification{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
    };
  }

 private:
  std::span<const bool> nonNulls() const {
    return {corpus_.nonNulls.data(), corpus_.nonNulls.size()};
  }

  std::string inputDigest() const {
    return semanticDigest(physicalValues(), nonNulls(), spec_.dataType);
  }

  std::span<const PhysicalType> physicalValues() const {
    return {corpus_.expectedPhysical.data(), corpus_.expectedPhysical.size()};
  }

  Corpus<T> makeCorpus() {
    if constexpr (std::is_same_v<T, double>) {
      return makeDoubleCorpus(config_, pool_);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return makeFixedBitWidthCorpus(config_, pool_);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      if (spec_.encoding == RunnerEncoding::PFOR) {
        return makePforCorpus(config_, pool_);
      }
      if (spec_.encoding == RunnerEncoding::SimdForBitpack) {
        return makeSimdForBitpackCorpus(config_, pool_);
      }
      if (spec_.encoding == RunnerEncoding::BlockBitPacking) {
        return makeBlockBitPackingCorpus(config_, pool_);
      }
      if (spec_.encoding == RunnerEncoding::Varint) {
        return makeVarintCorpus(config_, pool_);
      }
      return makeDeltaCorpus(config_, pool_);
    } else if constexpr (std::is_same_v<T, bool>) {
      return makeSparseBoolCorpus(config_, pool_);
    } else {
      return makeIntegerCorpus(spec_, config_, pool_);
    }
  }

  std::string_view encodeToBuffer(Buffer& buffer, bool runSelection) const {
    auto policy = runSelection ? selectionPolicy<T>(spec_.encodingType)
                               : replayPolicy<T>(spec_.encoding);
    std::string_view encoded;
    if (spec_.encoding == RunnerEncoding::Nullable) {
      encoded = EncodingFactory::encodeNullable<T>(
          std::move(policy),
          std::span<const T>{
              corpus_.logicalValues.data(), corpus_.logicalValues.size()},
          std::span<const bool>{
              corpus_.nonNulls.data(), corpus_.nonNulls.size()},
          buffer,
          options_);
    } else {
      encoded = EncodingFactory::encode<T>(
          std::move(policy),
          std::span<const T>{
              corpus_.logicalValues.data(), corpus_.logicalValues.size()},
          buffer,
          options_);
    }
    return encoded;
  }

  std::string encodeArtifact() {
    Buffer buffer{pool_};
    return std::string{
        encodeToBuffer(buffer, spec_.lane == RunnerLane::SelectionE2E)};
  }

  std::unique_ptr<Encoding> createEncoding(std::string_view artifact) const {
    return EncodingFactory{options_}.create(
        pool_, artifact, [](uint32_t) -> void* { return nullptr; });
  }

  std::string_view decoderArtifact() const {
    if (spec_.encoding != RunnerEncoding::Varint) {
      return encodedArtifact_;
    }
    return {decoderBacking_.data(), encodedArtifact_.size()};
  }

  void validateArtifact() {
    require(!encodedArtifact_.empty(), "encoded artifact must not be empty");
    require(
        encodedArtifact_.size() <= kMaxEncodingArtifactBytes,
        "encoded artifact exceeds the runner size limit");
    if (spec_.encoding == RunnerEncoding::RLE) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateRleArtifactStructure(encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::FixedBitWidth) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateUncompressedFixedBitWidthChild<PhysicalType>(
          encodedArtifact_,
          TypeTraits<T>::dataType,
          config_.rowCount,
          "artifact",
          false);
    } else if (spec_.encoding == RunnerEncoding::Delta) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateDeltaArtifactStructure(encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::SparseBool) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateSparseBoolArtifactStructure(encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::PFOR) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validatePforArtifactStructure(encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::SimdForBitpack) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateSimdForBitpackArtifactStructure(
          encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::BlockBitPacking) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateBlockBitPackingArtifactStructure(
          encodedArtifact_, config_.rowCount);
    } else if (spec_.encoding == RunnerEncoding::Varint) {
      validateArtifactPrefix(
          encodedArtifact_,
          spec_.encodingType,
          TypeTraits<T>::dataType,
          config_.rowCount);
      validateVarintArtifactStructure(encodedArtifact_, config_.rowCount);
    }
    auto encoding = createEncoding(decoderArtifact());
    require(
        encoding->encodingType() == spec_.encodingType,
        "artifact root encoding does not match task");
    require(
        encoding->rowCount() == config_.rowCount,
        "artifact row count does not match corpus");

    validateDense(*encoding);
    validateFragmented(*encoding);
    validateRange(*encoding);
    validateScatter(*encoding, scatter10_);
    validateScatter(*encoding, scatter1_);
    validateNullable(*encoding);
    validateSlice();
    validateView();
  }

  void validateDense(Encoding& encoding) {
    Vector<PhysicalType> actual{&pool_, config_.rowCount};
    encoding.reset();
    encoding.materialize(config_.rowCount, actual.data());
    require(
        std::equal(actual.begin(), actual.end(), physicalValues().begin()),
        "dense round trip failed");
    outputDigest_ = semanticDigest(
        std::span<const PhysicalType>{actual.data(), actual.size()},
        nonNulls(),
        spec_.dataType);
  }

  void validateFragmented(Encoding& encoding) const {
    Vector<PhysicalType> actual{&pool_, config_.rowCount};
    encoding.reset();
    uint32_t offset{0};
    while (offset < config_.rowCount) {
      const uint32_t count =
          std::min<uint32_t>(1 + offset % 31, config_.rowCount - offset);
      encoding.materialize(count, actual.data() + offset);
      offset += count;
    }
    require(
        std::equal(actual.begin(), actual.end(), physicalValues().begin()),
        "fragmented materialization failed");
  }

  void validateRange(Encoding& encoding) const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    Vector<PhysicalType> actual{&pool_, count};
    encoding.reset();
    encoding.skip(offset);
    encoding.materialize(count, actual.data());
    require(
        std::equal(
            actual.begin(),
            actual.end(),
            corpus_.expectedPhysical.begin() + offset),
        "range materialization failed");
  }

  void validateScatter(
      Encoding& encoding,
      const std::vector<uint32_t>& positions) const {
    Vector<PhysicalType> actual{&pool_, positions.size()};
    encoding.reset();
    uint32_t cursor{0};
    for (uint32_t index = 0; index < positions.size(); ++index) {
      const uint32_t position = positions[index];
      encoding.skip(position - cursor);
      encoding.materialize(1, actual.data() + index);
      cursor = position + 1;
    }
    for (uint32_t index = 0; index < positions.size(); ++index) {
      require(
          actual[index] == corpus_.expectedPhysical[positions[index]],
          "scatter materialization failed");
    }
  }

  void validateNullable(Encoding& encoding) const {
    if (spec_.encoding != RunnerEncoding::Nullable) {
      return;
    }
    Vector<PhysicalType> actual{&pool_, config_.rowCount};
    std::vector<uint64_t> nonNullBitmap((config_.rowCount + 63) / 64, 0);
    encoding.reset();
    const auto actualNonNullCount = encoding.materializeNullable(
        config_.rowCount, actual.data(), [&nonNullBitmap]() -> void* {
          return nonNullBitmap.data();
        });
    require(
        actualNonNullCount == corpus_.logicalValues.size(),
        "nullable non-null count mismatch");
    for (uint32_t row = 0; row < config_.rowCount; ++row) {
      const bool actualNonNull =
          (nonNullBitmap[row / 64] & (uint64_t{1} << (row % 64))) != 0;
      require(
          actualNonNull == corpus_.nonNulls[row], "nullable bitmap mismatch");
      if (actualNonNull) {
        require(
            actual[row] == corpus_.expectedPhysical[row],
            "nullable value mismatch");
      }
    }
  }

  void validateSlice() const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    Buffer buffer{pool_};
    const auto sliced = EncodingFactory::slice(
        decoderArtifact(), offset, count, buffer, options_);
    validateSliceArtifact(sliced);
  }

  void validateSliceArtifact(std::string_view sliced) const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    if (spec_.encoding == RunnerEncoding::Varint) {
      validateArtifactPrefix(
          sliced, spec_.encodingType, TypeTraits<T>::dataType, count);
      validateVarintArtifactStructure(sliced, count);
    }
    const auto backing = makeDecoderBacking(spec_.encoding, sliced);
    const std::string_view decoderInput =
        spec_.encoding == RunnerEncoding::Varint
        ? std::string_view{backing.data(), sliced.size()}
        : sliced;
    auto encoding = createEncoding(decoderInput);
    require(
        encoding->rowCount() == count, "sliced artifact row count mismatch");
    Vector<PhysicalType> actual{&pool_, count};
    encoding->materialize(count, actual.data());
    require(
        std::equal(
            actual.begin(),
            actual.end(),
            corpus_.expectedPhysical.begin() + offset),
        "sliced artifact materialization failed");
  }

  void validateView() const {
    if (!supportsEncodingView(spec_.encodingType)) {
      return;
    }
    auto view = createEncodingView(decoderArtifact(), &pool_, options_);
    for (const auto position : viewPositions_) {
      PhysicalType actual{};
      view->readAt(position, &actual);
      require(
          actual == corpus_.expectedPhysical[position],
          "encoding view read failed");
    }
  }

  void validateSelection() const {
    Buffer buffer{pool_};
    const auto selected = encodeToBuffer(buffer, true);
    require(
        selected == encodedArtifact_,
        "selection_e2e did not reproduce the exported artifact");
    auto encoding = createEncoding(selected);
    require(
        encoding->encodingType() == spec_.encodingType,
        "selection_e2e selected a different root encoding");
    Vector<PhysicalType> actual{&pool_, config_.rowCount};
    encoding->materialize(config_.rowCount, actual.data());
    require(
        std::equal(actual.begin(), actual.end(), physicalValues().begin()),
        "selection_e2e round trip failed");
  }

  void validateTimedEncode(bool runSelection) const {
    require(
        timedArtifact_ == encodedArtifact_,
        "timed encode did not reproduce the validated artifact");
    if (runSelection) {
      require(
          std::equal(output_.begin(), output_.end(), physicalValues().begin()),
          "timed selection_e2e decode failed");
    }
  }

  void validateTimedConstruct() const {
    require(
        constructedRowCount_ == config_.rowCount,
        "timed construction returned the wrong row count");
    require(
        constructedEncodingType_ == spec_.encodingType,
        "timed construction returned the wrong encoding type");
  }

  void validateTimedScatter(const std::vector<uint32_t>& positions) const {
    for (uint32_t index = 0; index < positions.size(); ++index) {
      require(
          scatterOutput_[index] == corpus_.expectedPhysical[positions[index]],
          "timed scatter decode failed");
    }
  }

  void validateTimedSkipSeek() const {
    require(skipSeekReadCount_ > 0, "timed skip_seek did not read a value");
    uint32_t cursor{0};
    uint32_t outputIndex{0};
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      require(
          std::equal(
              skipSeekOutput_.begin() + outputIndex,
              skipSeekOutput_.begin() + outputIndex + read,
              corpus_.expectedPhysical.begin() + cursor),
          "timed skip_seek decode failed");
      outputIndex += read;
      cursor += read;
    }
    require(
        outputIndex == skipSeekReadCount_,
        "timed skip_seek returned the wrong value count");
  }

  void validateTimedResult() const {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        validateTimedEncode(false);
        return;
      case RunnerLane::DecodeConstruct:
        validateTimedConstruct();
        return;
      case RunnerLane::DecodeDense:
        require(
            std::equal(
                output_.begin(), output_.end(), physicalValues().begin()),
            "timed dense decode failed");
        return;
      case RunnerLane::DecodeRange50:
        require(
            std::equal(
                rangeOutput_.begin(),
                rangeOutput_.end(),
                corpus_.expectedPhysical.begin() + config_.rowCount / 4),
            "timed range decode failed");
        return;
      case RunnerLane::DecodeScatter10:
        validateTimedScatter(scatter10_);
        return;
      case RunnerLane::DecodeScatter1:
        validateTimedScatter(scatter1_);
        return;
      case RunnerLane::SkipSeek:
        validateTimedSkipSeek();
        return;
      case RunnerLane::ViewRandom:
        for (uint32_t index = 0; index < viewPositions_.size(); ++index) {
          require(
              viewOutput_[index] ==
                  corpus_.expectedPhysical[viewPositions_[index]],
              "timed view read failed");
        }
        return;
      case RunnerLane::Slice:
        validateSliceArtifact(timedArtifact_);
        return;
      case RunnerLane::SelectionE2E:
        validateTimedEncode(true);
        return;
    }
    fail("unknown runner lane");
  }

  void runOnce() {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        runEncode(false);
        return;
      case RunnerLane::DecodeConstruct:
        runConstruct();
        return;
      case RunnerLane::DecodeDense:
        runDense();
        return;
      case RunnerLane::DecodeRange50:
        runRange();
        return;
      case RunnerLane::DecodeScatter10:
        runScatter(scatter10_);
        return;
      case RunnerLane::DecodeScatter1:
        runScatter(scatter1_);
        return;
      case RunnerLane::SkipSeek:
        runSkipSeek();
        return;
      case RunnerLane::ViewRandom:
        runView();
        return;
      case RunnerLane::Slice:
        runSlice();
        return;
      case RunnerLane::SelectionE2E:
        runEncode(true);
        return;
    }
    fail("unknown runner lane");
  }

  void runEncode(bool runSelection) {
    timingBuffer_.reset();
    const auto encoded = encodeToBuffer(timingBuffer_, runSelection);
    timedArtifact_ = encoded;
    if (runSelection) {
      auto encoding = createEncoding(encoded);
      require(
          encoding->encodingType() == spec_.encodingType,
          "selection_e2e selected a different root encoding");
      encoding->materialize(config_.rowCount, output_.data());
      folly::doNotOptimizeAway(output_.back());
    } else {
      folly::doNotOptimizeAway(encoded.size());
    }
  }

  void runConstruct() {
    auto encoding = createEncoding(decoderArtifact());
    constructedRowCount_ = encoding->rowCount();
    constructedEncodingType_ = encoding->encodingType();
    folly::doNotOptimizeAway(*constructedRowCount_);
  }

  void runDense() {
    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    folly::doNotOptimizeAway(output_.back());
  }

  void runRange() {
    decoder_->reset();
    decoder_->skip(config_.rowCount / 4);
    decoder_->materialize(config_.rowCount / 2, rangeOutput_.data());
    folly::doNotOptimizeAway(rangeOutput_.back());
  }

  void runScatter(const std::vector<uint32_t>& positions) {
    decoder_->reset();
    uint32_t cursor{0};
    for (uint32_t index = 0; index < positions.size(); ++index) {
      decoder_->skip(positions[index] - cursor);
      decoder_->materialize(1, scatterOutput_.data() + index);
      cursor = positions[index] + 1;
    }
    folly::doNotOptimizeAway(scatterOutput_[positions.size() - 1]);
  }

  void runSkipSeek() {
    decoder_->reset();
    uint32_t cursor{0};
    skipSeekReadCount_ = 0;
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      decoder_->skip(skip);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      decoder_->materialize(read, skipSeekOutput_.data() + skipSeekReadCount_);
      skipSeekReadCount_ += read;
      cursor += read;
    }
    folly::doNotOptimizeAway(skipSeekOutput_[skipSeekReadCount_ - 1]);
  }

  void runView() {
    require(view_ != nullptr, "view_random requires a supported encoding view");
    for (uint32_t index = 0; index < viewPositions_.size(); ++index) {
      view_->readAt(viewPositions_[index], viewOutput_.data() + index);
    }
    folly::doNotOptimizeAway(viewOutput_.back());
  }

  void runSlice() {
    timingBuffer_.reset();
    timedArtifact_ = EncodingFactory::slice(
        decoderArtifact(),
        config_.rowCount / 4,
        config_.rowCount / 2,
        timingBuffer_,
        options_);
    folly::doNotOptimizeAway(timedArtifact_.size());
  }

  double runIterations(uint32_t iterations) {
    const auto start = Clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      runOnce();
    }
    const auto elapsed = Clock::now() - start;
    return std::chrono::duration<double>{elapsed}.count();
  }

  uint32_t calibrateIterations() {
    uint32_t iterations = config_.innerIterations;
    if (config_.minSampleTimeMicros == 0) {
      return iterations;
    }
    const double targetSeconds =
        static_cast<double>(config_.minSampleTimeMicros) / 1'000'000.0;
    for (uint32_t attempt = 0; attempt < 16; ++attempt) {
      const double elapsed = runIterations(iterations);
      if (elapsed >= targetSeconds) {
        return iterations;
      }
      const double safeElapsed = std::max(elapsed, 1e-9);
      const auto multiplier = static_cast<uint32_t>(
          std::clamp(std::ceil(targetSeconds / safeElapsed), 2.0, 16.0));
      if (iterations > kMaxCalibratedIterations / multiplier) {
        fail("timing iteration calibration overflowed");
      }
      iterations *= multiplier;
    }
    fail("timing iteration calibration did not reach the requested duration");
  }

  const EncodingRunnerConfig& config_;
  TaskSpec spec_;
  velox::memory::MemoryPool& pool_;
  Corpus<T> corpus_;
  Encoding::Options options_{};
  std::string encodedArtifact_;
  std::string decoderBacking_;
  std::string outputDigest_;
  Buffer timingBuffer_;
  std::unique_ptr<Encoding> decoder_;
  std::unique_ptr<EncodingView> view_;
  std::optional<uint32_t> constructedRowCount_;
  std::optional<EncodingType> constructedEncodingType_;
  std::string_view timedArtifact_;
  Vector<PhysicalType> output_;
  Vector<PhysicalType> rangeOutput_;
  std::vector<uint32_t> scatter10_;
  std::vector<uint32_t> scatter1_;
  std::vector<uint32_t> viewPositions_;
  Vector<PhysicalType> scatterOutput_;
  Vector<PhysicalType> skipSeekOutput_;
  Vector<PhysicalType> viewOutput_;
  uint32_t skipSeekReadCount_{0};
};

template <typename Result>
folly::dynamic commonJson(const Result& result, std::string_view kind) {
  return folly::dynamic::object("schema_version", 1)("kind", kind)(
      "task_id", result.taskId)("encoding", result.encoding)(
      "data_type", result.dataType)("seed", static_cast<int64_t>(result.seed))(
      "row_count", result.rowCount)("encoded_bytes", result.encodedBytes)(
      "input_digest", result.inputDigest)("output_digest", result.outputDigest)(
      "artifact_digest", result.artifactDigest)("correctness", true);
}

folly::dynamic samplesJson(const std::vector<double>& samples) {
  folly::dynamic result = folly::dynamic::array;
  for (const auto sample : samples) {
    result.push_back(sample);
  }
  return result;
}

bool isSha256(std::string_view value) {
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  return value.size() == 64 &&
      value.find_first_not_of(kHexDigits) == std::string_view::npos;
}

template <typename Result>
void validateCommonResult(const Result& result) {
  const auto spec = parseTaskId(result.taskId);
  require(result.encoding == spec.encodingSlug, "result encoding mismatch");
  require(result.dataType == spec.dataType, "result data type mismatch");
  require(
      result.seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "result seed exceeds the JSON integer limit");
  require(
      result.rowCount >= 100 && result.rowCount <= kMaxRowCount,
      "result row count is outside runner limits");
  require(
      result.encodedBytes > 0 &&
          result.encodedBytes <= kMaxEncodingArtifactBytes,
      "result encoded size is outside runner limits");
  require(
      result.inputDigest == result.outputDigest,
      "result output digest does not match its input digest");
  require(isSha256(result.inputDigest), "result input digest is not SHA-256");
  require(
      isSha256(result.artifactDigest), "result artifact digest is not SHA-256");
}

void validateMeasurementResult(const EncodingRunnerMeasurement& measurement) {
  validateCommonResult(measurement);
  const auto spec = parseTaskId(measurement.taskId);
  require(measurement.lane == spec.laneName, "measurement lane mismatch");
  require(
      measurement.encodedBytes == measurement.encodedArtifact.size(),
      "measurement artifact size mismatch");
  require(
      !measurement.samplesSeconds.empty() &&
          measurement.samplesSeconds.size() <= kMaxSamples,
      "measurement sample count is outside runner limits");
  for (const double sample : measurement.samplesSeconds) {
    require(
        std::isfinite(sample) && sample > 0.0,
        "measurement samples must be finite and positive");
  }
}

bool consumesCanonicalArtifact(RunnerLane lane) {
  return lane != RunnerLane::Encode && lane != RunnerLane::SelectionE2E;
}

} // namespace

std::string detail::semanticStringDigestForTesting(
    std::span<const std::string_view> values,
    std::span<const bool> nonNulls,
    std::string_view dataType) {
  return semanticStringDigest(values, nonNulls, dataType);
}

EncodingRunnerMeasurement runEncodingBenchmark(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool,
    std::optional<std::string_view> benchmarkArtifact) {
  const auto spec = validateConfig(config);
  if (benchmarkArtifact.has_value() && !consumesCanonicalArtifact(spec.lane)) {
    throw std::invalid_argument(
        "encode and selection_e2e cannot consume a benchmark artifact");
  }
  if (benchmarkArtifact.has_value() &&
      benchmarkArtifact->size() > kMaxEncodingArtifactBytes) {
    throw std::invalid_argument("benchmark artifact exceeds the runner limit");
  }
  if (spec.encoding == RunnerEncoding::Prefix) {
    return PrefixRunner{config, spec, pool, benchmarkArtifact}.run();
  }
  if (spec.encoding == RunnerEncoding::Fsst) {
    return FsstRunner{config, spec, pool, benchmarkArtifact}.run();
  }
  if (spec.encoding == RunnerEncoding::ALP) {
    return TypedRunner<double>{config, spec, pool, benchmarkArtifact}.run();
  }
  if (spec.encoding == RunnerEncoding::FixedBitWidth) {
    return TypedRunner<uint64_t>{config, spec, pool, benchmarkArtifact}.run();
  }
  if (spec.encoding == RunnerEncoding::Delta ||
      spec.encoding == RunnerEncoding::PFOR ||
      spec.encoding == RunnerEncoding::SimdForBitpack ||
      spec.encoding == RunnerEncoding::BlockBitPacking ||
      spec.encoding == RunnerEncoding::Varint) {
    return TypedRunner<uint32_t>{config, spec, pool, benchmarkArtifact}.run();
  }
  if (spec.encoding == RunnerEncoding::SparseBool) {
    return TypedRunner<bool>{config, spec, pool, benchmarkArtifact}.run();
  }
  return TypedRunner<int64_t>{config, spec, pool, benchmarkArtifact}.run();
}

EncodingArtifactVerification verifyEncodingArtifact(
    const EncodingRunnerConfig& config,
    std::string_view encodedArtifact,
    velox::memory::MemoryPool& pool) {
  const auto spec = validateConfig(config);
  if (encodedArtifact.size() > kMaxEncodingArtifactBytes) {
    throw std::invalid_argument(
        "verification artifact exceeds the runner limit");
  }
  if (spec.encoding == RunnerEncoding::Prefix) {
    return PrefixRunner{config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  if (spec.encoding == RunnerEncoding::Fsst) {
    return FsstRunner{config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  if (spec.encoding == RunnerEncoding::ALP) {
    return TypedRunner<double>{
        config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  if (spec.encoding == RunnerEncoding::FixedBitWidth) {
    return TypedRunner<uint64_t>{
        config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  if (spec.encoding == RunnerEncoding::Delta ||
      spec.encoding == RunnerEncoding::PFOR ||
      spec.encoding == RunnerEncoding::SimdForBitpack ||
      spec.encoding == RunnerEncoding::BlockBitPacking ||
      spec.encoding == RunnerEncoding::Varint) {
    return TypedRunner<uint32_t>{
        config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  if (spec.encoding == RunnerEncoding::SparseBool) {
    return TypedRunner<bool>{config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  return TypedRunner<int64_t>{
      config, spec, pool, std::optional{encodedArtifact}}
      .verification();
}

std::string measurementToJson(const EncodingRunnerMeasurement& measurement) {
  validateMeasurementResult(measurement);
  auto result = commonJson(measurement, "nimble_encoding_measurement");
  result["lane"] = measurement.lane;
  result["samples_seconds"] = samplesJson(measurement.samplesSeconds);
  return folly::toJson(result);
}

std::string verificationToJson(
    const EncodingArtifactVerification& verification) {
  validateCommonResult(verification);
  return folly::toJson(
      commonJson(verification, "nimble_encoding_verification"));
}

} // namespace facebook::nimble::benchmarks
