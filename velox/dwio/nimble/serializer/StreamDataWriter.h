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

#pragma once

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "folly/io/Cursor.h"
#include "folly/io/IOBuf.h"
#include "velox/common/Casts.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Zigzag.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/velox/RowRange.h"
#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble::serde {
namespace detail {

/// Get total size of a string field
uint32_t getStringsTotalSize(std::string_view input);

/// Encode a string field with supplied total size
void encodeStrings(std::string_view input, uint32_t size, char* output);

/// Encode non-string field
uint32_t
encode(const SerializerOptions& options, std::string_view input, char* output);

/// Write zeros for missing streams in kLegacy format.
/// Each missing stream is a zero-length stream (size=0, u32 = 4 bytes).
template <typename T>
void writeMissingStreams(T& buffer, uint32_t lastStream, uint32_t nextStream) {
  NIMBLE_CHECK_LE(lastStream + 1, nextStream, "unexpected stream offset");
  const auto missingStreamCount = nextStream - lastStream - 1;
  if (missingStreamCount > 0) {
    const auto oldByteSize = buffer.size();
    buffer.resize(oldByteSize + missingStreamCount * sizeof(uint32_t));
    auto begin = reinterpret_cast<uint32_t*>(buffer.data() + oldByteSize);
    std::fill(begin, begin + missingStreamCount, 0);
  }
}

/// Encode typed values using a given encoding selection policy factory.
/// When encodingLayout is provided, replays the captured encoding with
/// compressionOptions. policyFactory is used as fallback for nested encodings
/// not captured in the layout tree. When encodingLayout is not provided, uses
/// policyFactory directly and compressionOptions is ignored.
template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> makeEncodingPolicy(
    const EncodingSelectionPolicyCreator& policyFactory,
    const EncodingLayout* encodingLayout,
    const std::optional<CompressionOptions>& compressionOptions) {
  if (encodingLayout != nullptr) {
    return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        *encodingLayout, compressionOptions, policyFactory);
  }
  return velox::checkedPointerCast<EncodingSelectionPolicy<T>>(
      policyFactory(TypeTraits<T>::dataType));
}

template <typename T>
std::string_view encodeTyped(
    std::span<const T> values,
    nimble::Buffer& encodingBuffer,
    const EncodingSelectionPolicyCreator& policyFactory,
    const Encoding::Options& encodingOptions = {.useVarintRowCount = true},
    const EncodingLayout* encodingLayout = nullptr,
    std::optional<CompressionOptions> compressionOptions = std::nullopt) {
  return EncodingFactory::encode<T>(
      makeEncodingPolicy<T>(policyFactory, encodingLayout, compressionOptions),
      values,
      encodingBuffer,
      encodingOptions);
}

template <typename T>
std::string_view encodeNullableTyped(
    const SerializerOptions& options,
    std::string_view data,
    std::span<const bool> nonNulls,
    nimble::Buffer& encodingBuffer,
    const EncodingLayout* encodingLayout,
    const Encoding::Options& encodingOptions) {
  const auto count = data.size() / sizeof(T);
  std::span<const T> values{reinterpret_cast<const T*>(data.data()), count};

  return EncodingFactory::encodeNullable<T>(
      makeEncodingPolicy<T>(
          options.encodingSelectionPolicyCreator,
          encodingLayout,
          options.compressionOptions),
      values,
      nonNulls,
      encodingBuffer,
      encodingOptions);
}

inline std::string_view boolsAsStringView(std::span<const bool> values) {
  return {
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(bool)};
}

/// Returns an upper-bound estimate of the trailer size for the two-array
/// sparse layout. Conservatively assumes every stream slot is non-zero
/// (worst case for sparseness) and that sizes can be up to UINT32_MAX.
size_t estimateTrailerSize(
    size_t numStreams,
    EncodingType indicesEncodingType,
    EncodingType sizesEncodingType);

/// Returns an upper-bound estimate for the kTablet dedup trailer layout:
/// present stream IDs, per-present-stream size indices, and unique stream
/// sizes. `numPresentStreams` is the number of present streams;
/// `numUniqueStreams` is the number of streams with unique content.
size_t estimateTrailerSize(
    size_t numPresentStreams,
    size_t numUniqueStreams,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType);

/// Overload that accepts SerializationVersion for API compatibility.
inline size_t estimateTrailerSize(
    SerializationVersion /* outputVersion */,
    size_t numStreams,
    std::optional<EncodingType> indicesEncodingType = std::nullopt,
    std::optional<EncodingType> sizesEncodingType = std::nullopt) {
  return estimateTrailerSize(
      numStreams,
      indicesEncodingType.value_or(EncodingType::FixedBitWidth),
      sizesEncodingType.value_or(EncodingType::FixedBitWidth));
}

/// Writes the Trivial section payload: count varint + raw u32 array.
/// Wire: [count:varint][v_0:u32]...[v_N:u32]
template <typename T>
void writeTrivialSection(const std::vector<uint32_t>& values, T& buffer) {
  const auto count = static_cast<uint32_t>(values.size());
  const auto countVarintSize = varint::varintSize(count);
  const uint32_t payloadBytes = count * sizeof(uint32_t);
  auto* pos = extend(buffer, countVarintSize + payloadBytes);
  varint::writeVarint(count, &pos);
  if (payloadBytes > 0) {
    std::memcpy(pos, values.data(), payloadBytes);
  }
}

/// Writes the Varint section payload: count varint + each value as varint.
/// Wire: [count:varint][v_0:varint]...[v_N:varint]
template <typename T>
void writeVarintSection(const std::vector<uint32_t>& values, T& buffer) {
  const auto count = static_cast<uint32_t>(values.size());
  const auto countVarintSize = varint::varintSize(count);
  const auto dataVarintSize =
      static_cast<uint32_t>(varint::bulkVarintSize32(values));
  auto* pos = extend(buffer, countVarintSize + dataVarintSize);
  varint::writeVarint(count, &pos);
  for (const auto v : values) {
    varint::writeVarint(v, &pos);
  }
}

/// Writes the Delta section payload: count varint + first value + per-element
/// deltas. Wire: [count:varint][first:varint][delta_1:varint]...
template <typename T>
void writeDeltaSection(const std::vector<uint32_t>& values, T& buffer) {
  const auto count = static_cast<uint32_t>(values.size());
  const auto countVarintSize = varint::varintSize(count);
  uint32_t dataVarintSize = 0;
  if (count > 0) {
    dataVarintSize = varint::varintSize(values[0]);
    for (uint32_t i = 1; i < count; ++i) {
      const auto delta = values[i] - values[i - 1];
      dataVarintSize += varint::varintSize(delta);
    }
  }
  auto* pos = extend(buffer, countVarintSize + dataVarintSize);
  varint::writeVarint(count, &pos);
  if (count > 0) {
    varint::writeVarint(values[0], &pos);
    for (uint32_t i = 1; i < count; ++i) {
      const auto delta = values[i] - values[i - 1];
      varint::writeVarint(delta, &pos);
    }
  }
}

/// Writes the FixedBitWidth section payload: bitWidth + count + bit-packed.
/// Wire: [bitWidth:1B][count:varint][bit-packed data]
template <typename T>
void writeFixedBitWidthSection(const std::vector<uint32_t>& values, T& buffer) {
  const auto count = static_cast<uint32_t>(values.size());
  uint32_t maxVal = 0;
  for (const auto v : values) {
    maxVal = std::max(maxVal, v);
  }
  const uint8_t bitWidth =
      maxVal == 0 ? 0 : static_cast<uint8_t>(std::bit_width(maxVal));
  const uint32_t packedBytes = (bitWidth > 0 && count > 0)
      ? static_cast<uint32_t>(FixedBitArray::bufferSize(count, bitWidth))
      : 0;
  const auto countVarintSize = varint::varintSize(count);
  auto* pos = extend(buffer, sizeof(uint8_t) + countVarintSize + packedBytes);
  *pos++ = static_cast<char>(bitWidth);
  varint::writeVarint(count, &pos);
  if (bitWidth > 0 && count > 0) {
    std::memset(pos, 0, packedBytes);
    FixedBitArray arr{pos, static_cast<int>(bitWidth)};
    arr.bulkSet32(0, count, values.data());
  }
}

/// Dispatches a section write to the encoding-specific writer.
template <typename T>
void writeSection(
    EncodingType encodingType,
    const std::vector<uint32_t>& values,
    T& buffer) {
  switch (getTrailerEncodingType(encodingType)) {
    case EncodingType::Trivial:
      writeTrivialSection(values, buffer);
      break;
    case EncodingType::Varint:
      writeVarintSection(values, buffer);
      break;
    case EncodingType::Delta:
      writeDeltaSection(values, buffer);
      break;
    case EncodingType::FixedBitWidth:
      writeFixedBitWidthSection(values, buffer);
      break;
    default:
      NIMBLE_FAIL(
          "Unsupported EncodingType for stream sizes trailer section: {}",
          encodingType);
  }
}

/// Writes the two-array sparse stream-sizes trailer. Walks
/// `denseStreamSizes` once to collect the stream IDs of non-zero entries and
/// their sizes, then encodes each section with the caller-specified encoding
/// type.
/// Wire: [indicesEncType:1B][indicesPayload]
///       [sizesEncType:1B][sizesPayload][trailer_size:u32]
template <typename T>
void writeTrailer(
    const std::vector<uint32_t>& denseStreamSizes,
    EncodingType indicesEncodingType,
    EncodingType sizesEncodingType,
    T& buffer) {
  const auto streamCount = denseStreamSizes.size();
  std::vector<uint32_t> streamIds;
  std::vector<uint32_t> streamSizes;
  streamIds.reserve(streamCount);
  streamSizes.reserve(streamCount);
  for (uint32_t i = 0; i < streamCount; ++i) {
    if (denseStreamSizes[i] != 0) {
      streamIds.emplace_back(i);
      streamSizes.emplace_back(denseStreamSizes[i]);
    }
  }

  const auto trailerStartOffset = buffer.size();
  auto* indicesTypePos = extend(buffer, sizeof(uint8_t));
  *indicesTypePos = static_cast<char>(indicesEncodingType);
  writeSection(indicesEncodingType, streamIds, buffer);

  auto* sizesTypePos = extend(buffer, sizeof(uint8_t));
  *sizesTypePos = static_cast<char>(sizesEncodingType);
  writeSection(sizesEncodingType, streamSizes, buffer);

  const uint32_t trailerSize =
      static_cast<uint32_t>(buffer.size() - trailerStartOffset);
  auto* sizePos = extend(buffer, sizeof(uint32_t));
  encoding::writeUint32(trailerSize, sizePos);
}

/// Writes the kTablet dedup trailer layout used when projected streams may be
/// duplicated. The caller supplies the three trailer sections directly:
/// `streamIds` (present slot ids),
/// `streamSizeIndices` (per present slot, an index into `uniqueStreamSizes`),
/// and `uniqueStreamSizes` (the distinct stream sizes in body order). Neither
/// offsets nor duplicated sizes are stored: offsets are implied by
/// prefix-summing `uniqueStreamSizes`, and duplicate slots reuse a unique-size
/// entry through `streamSizeIndices`. The caller chooses the encoding for each
/// section. Wire:
/// [streamIdsEncType:1B][streamIdsPayload]
///       [sizeIndicesEncType:1B][sizeIndicesPayload]
///       [uniqueSizesEncType:1B][uniqueSizesPayload][trailer_size:u32]
template <typename T>
void writeTrailer(
    const std::vector<uint32_t>& streamIds,
    const std::vector<uint32_t>& streamSizeIndices,
    const std::vector<uint32_t>& uniqueStreamSizes,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType,
    T& buffer) {
  NIMBLE_CHECK_EQ(
      streamSizeIndices.size(),
      streamIds.size(),
      "Stream ids and size indices must have the same length");

  const auto trailerStartOffset = buffer.size();
  auto* streamIdsTypePos = extend(buffer, sizeof(uint8_t));
  *streamIdsTypePos = static_cast<char>(streamIdsEncodingType);
  writeSection(streamIdsEncodingType, streamIds, buffer);

  auto* sizeIndicesTypePos = extend(buffer, sizeof(uint8_t));
  *sizeIndicesTypePos = static_cast<char>(sizeIndicesEncodingType);
  writeSection(sizeIndicesEncodingType, streamSizeIndices, buffer);

  auto* uniqueSizesTypePos = extend(buffer, sizeof(uint8_t));
  *uniqueSizesTypePos = static_cast<char>(uniqueSizesEncodingType);
  writeSection(uniqueSizesEncodingType, uniqueStreamSizes, buffer);

  const uint32_t trailerSize =
      static_cast<uint32_t>(buffer.size() - trailerStartOffset);
  auto* sizePos = extend(buffer, sizeof(uint32_t));
  encoding::writeUint32(trailerSize, sizePos);
}

/// Encode scalar data using nimble encoding framework with serializer options.
template <typename T, typename Buffer>
std::string_view encodeTyped(
    const SerializerOptions& options,
    std::string_view data,
    velox::memory::MemoryPool& pool,
    nimble::Buffer& encodingBuffer,
    const EncodingLayout* encodingLayout,
    const Encoding::Options& encodingOptions) {
  const auto count = data.size() / sizeof(T);
  std::span<const T> values{reinterpret_cast<const T*>(data.data()), count};
  return encodeTyped<T>(
      values,
      encodingBuffer,
      options.encodingSelectionPolicyCreator,
      encodingOptions,
      encodingLayout,
      options.compressionOptions);
}

/// Dispatch to typed nimble encoding based on ScalarKind.
template <typename Buffer>
std::string_view encodeScalar(
    const SerializerOptions& options,
    ScalarKind scalarKind,
    std::string_view data,
    velox::memory::MemoryPool& pool,
    nimble::Buffer& encodingBuffer,
    const EncodingLayout* encodingLayout,
    const Encoding::Options& encodingOptions) {
  switch (scalarKind) {
    case ScalarKind::Bool:
      return encodeTyped<bool, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Int8:
      return encodeTyped<int8_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::UInt8:
      return encodeTyped<uint8_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Int16:
      return encodeTyped<int16_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::UInt16:
      return encodeTyped<uint16_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Int32:
      return encodeTyped<int32_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::UInt32:
      return encodeTyped<uint32_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Int64:
      return encodeTyped<int64_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::UInt64:
      return encodeTyped<uint64_t, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Float:
      return encodeTyped<float, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::Double:
      return encodeTyped<double, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    case ScalarKind::String:
      [[fallthrough]];
    case ScalarKind::Binary:
      return encodeTyped<std::string_view, Buffer>(
          options, data, pool, encodingBuffer, encodingLayout, encodingOptions);
    default:
      NIMBLE_UNSUPPORTED(
          "Unsupported scalar kind for nimble encoding: {}", scalarKind);
  }
}

template <typename Buffer>
std::string_view encodeNullableScalar(
    const SerializerOptions& options,
    ScalarKind scalarKind,
    std::string_view data,
    std::span<const bool> nonNulls,
    nimble::Buffer& encodingBuffer,
    const EncodingLayout* encodingLayout,
    const Encoding::Options& encodingOptions) {
  switch (scalarKind) {
    case ScalarKind::Bool:
      return encodeNullableTyped<bool>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Int8:
      return encodeNullableTyped<int8_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::UInt8:
      return encodeNullableTyped<uint8_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Int16:
      return encodeNullableTyped<int16_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::UInt16:
      return encodeNullableTyped<uint16_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Int32:
      return encodeNullableTyped<int32_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::UInt32:
      return encodeNullableTyped<uint32_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Int64:
      return encodeNullableTyped<int64_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::UInt64:
      return encodeNullableTyped<uint64_t>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Float:
      return encodeNullableTyped<float>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Double:
      return encodeNullableTyped<double>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::String:
      [[fallthrough]];
    case ScalarKind::Binary:
      return encodeNullableTyped<std::string_view>(
          options,
          data,
          nonNulls,
          encodingBuffer,
          encodingLayout,
          encodingOptions);
    case ScalarKind::Undefined:
      NIMBLE_UNSUPPORTED(
          "Unsupported scalar kind for nullable nimble encoding: {}",
          toString(scalarKind));
    default:
      NIMBLE_UNSUPPORTED(
          "Unsupported scalar kind for nullable nimble encoding: {}",
          toString(scalarKind));
  }
}
} // namespace detail

// Writes the serialized stream-data payload that StreamData and
// StreamDataParser read back.
//
// NOTE: `nimble::StreamData` below is the writer-side buffer from
// velox/dwio/nimble/velox/StreamData.h, NOT `serde::StreamData` from
// velox/dwio/nimble/serializer/StreamData.h. Keep every reference explicitly
// qualified.
template <typename T>
class StreamDataWriter {
 public:
  /// Constructor. For kLegacy, writes the header immediately. For
  /// kSerialization, writes the compact header prefix (version, row count,
  /// flags).
  ///
  /// @param pool Memory pool for encoding buffer allocation.
  /// @param streamEncodingLayouts Optional encoding layouts for replaying
  ///        captured encodings. When provided, looks up EncodingLayout by
  ///        stream offset and uses ReplayedEncodingSelectionPolicy.
  StreamDataWriter(
      const SerializerOptions& options,
      T& buffer,
      uint32_t rowCount,
      velox::memory::MemoryPool* pool,
      const std::unordered_map<uint32_t, const EncodingLayout*>*
          streamEncodingLayouts);

  /// Write data for a single stream.
  void writeData(const nimble::StreamData& streamData);

  /// Close the writer. For kLegacy, fills trailing zeros up to nodeCount. For
  /// kSerialization, writes the stream-sizes trailer.
  void close(uint32_t nodeCount = 0);

 private:
  void encodeStream(
      ScalarKind scalarKind,
      uint32_t streamOffset,
      std::string_view data,
      std::span<const bool> nonNulls = {});

  // --- Const members ---
  const SerializerOptions& options_;
  // Memory pool for encoding scratch buffers.
  velox::memory::MemoryPool* const pool_;
  // Scratch buffer holding one encoded stream before it is copied to output.
  const std::unique_ptr<nimble::Buffer> streamEncodingBuffer_;
  // Optional map from stream offset to encoding layout for replaying captured
  // encodings. Only set when options_.encodingLayoutTree is specified.
  const std::unordered_map<uint32_t, const EncodingLayout*>* const
      streamEncodingLayouts_;

  // --- Mutable members ---
  // Final serialized output. Encoded stream bytes and stream metadata are
  // appended here after each stream encode completes.
  T& outputBuffer_;
  // Track last stream offset for kLegacy format zero-filling.
  uint32_t lastStream_{0xffffffff};
  // Dense stream sizes. streamSizes_[i] = byte size of stream i (0 for
  // missing/empty).
  std::vector<uint32_t> streamSizes_;
  // Byte offset of the serialization header flags byte, patched in close().
  size_t headerFlagsOffset_{0};
  bool writesHeaderFlags_{false};
  bool requiresNullBarrier_{false};
};

template <typename T>
StreamDataWriter<T>::StreamDataWriter(
    const SerializerOptions& options,
    T& buffer,
    uint32_t rowCount,
    velox::memory::MemoryPool* pool,
    const std::unordered_map<uint32_t, const EncodingLayout*>*
        streamEncodingLayouts)
    : options_{options},
      pool_{pool},
      streamEncodingBuffer_{
          options.enableEncoding() ? std::make_unique<nimble::Buffer>(*pool)
                                   : nullptr},
      streamEncodingLayouts_{streamEncodingLayouts},
      outputBuffer_{buffer} {
  NIMBLE_CHECK(
      streamEncodingLayouts_ == nullptr || options_.enableEncoding(),
      "streamEncodingLayouts can only be set when encoding is enabled");
  NIMBLE_CHECK_NOT_NULL(pool, "Memory pool cannot be null");

  std::optional<SerializationVersion> version;
  if (options_.hasVersionHeader()) {
    version = options_.serializationVersion();
  }

  writesHeaderFlags_ = usesCompactHeaderFlags(version);
  if (writesHeaderFlags_) {
    headerFlagsOffset_ =
        writeSerializationHeader(outputBuffer_, version.value(), rowCount);
    NIMBLE_CHECK_LT(
        headerFlagsOffset_,
        outputBuffer_.size(),
        "Invalid null barrier flag offset");
    NIMBLE_CHECK_EQ(
        static_cast<uint8_t>(outputBuffer_.data()[headerFlagsOffset_]),
        SerializationHeader::kStreamVarintRowCountFlag,
        "Non-tablet header should initialize the varint row-count flag");
  } else {
    writeLegacySerializationHeader(outputBuffer_, version, rowCount);
  }
}

template <typename T>
void StreamDataWriter<T>::writeData(const nimble::StreamData& streamData) {
  const auto scalarKind = streamData.descriptor().scalarKind();
  const auto streamOffset = streamData.descriptor().offset();
  const auto nonNulls = streamData.nonNulls();
  const auto data = streamData.data();

  // Streams with no physical payload are omitted. All-true Row/FlatMap null
  // streams normally remain unmaterialized and are reconstructed on read.
  if (data.empty() && nonNulls.empty()) {
    return;
  }

  if (!options_.enableEncoding()) {
    NIMBLE_CHECK(
        nonNulls.empty() ||
            std::all_of(
                nonNulls.begin(),
                nonNulls.end(),
                [](bool notNull) { return notNull; }),
        "Null values are not supported in legacy serialization formats. "
        "Use kSerialization for nullable support.");

    NIMBLE_CHECK_LE(lastStream_ + 1, streamOffset, "unexpected stream offset");
    detail::writeMissingStreams(outputBuffer_, lastStream_, streamOffset);
    lastStream_ = streamOffset;
    encodeStream(scalarKind, streamOffset, data);
    return;
  }

  if (streamData.isNullStream()) {
    requiresNullBarrier_ |= streamData.hasNullValues();
    NIMBLE_CHECK(
        data.empty(), "null streams should not carry a separate data payload");
    const auto streamPayload = detail::boolsAsStringView(nonNulls);
    NIMBLE_CHECK(!streamPayload.empty(), "Expected null stream payload");
    encodeStream(scalarKind, streamOffset, streamPayload);
    return;
  }

  // Nullable content streams store only non-null payload values; an all-null
  // chunk has an empty value payload but still needs its non-null bits encoded.
  NIMBLE_CHECK(
      !data.empty() || streamData.hasNullValues(),
      "Expected content stream payload");
  encodeStream(scalarKind, streamOffset, data, nonNulls);
}

template <typename T>
void StreamDataWriter<T>::encodeStream(
    ScalarKind scalarKind,
    uint32_t streamOffset,
    std::string_view data,
    std::span<const bool> nonNulls) {
  if (!options_.enableEncoding()) {
    if (scalarKind == ScalarKind::String || scalarKind == ScalarKind::Binary) {
      // Legacy string encoding: [total_size:u32][len_0:u32][data_0]...
      const auto size = detail::getStringsTotalSize(data);
      auto* pos = detail::extend(outputBuffer_, size + sizeof(uint32_t));
      detail::encodeStrings(data, size, pos);
    } else {
      // Legacy scalar encoding:
      //   Zstd: [size:u32][compType:i8][data...]
      //   LZ4:  [size:u32][compType:i8][origSize:u32][data...]
      const auto bufferStart = outputBuffer_.size();
      const uint32_t maxSize = data.size() + 2 * sizeof(uint32_t) + 1;
      auto* pos = detail::extend(outputBuffer_, maxSize);
      const auto encodedSize = detail::encode(options_, data, pos);
      if (encodedSize < maxSize) {
        outputBuffer_.resize(bufferStart + encodedSize);
      }
    }
    return;
  }

  const EncodingLayout* encodingLayout = nullptr;
  if (streamEncodingLayouts_ != nullptr) {
    auto it = streamEncodingLayouts_->find(streamOffset);
    if (it != streamEncodingLayouts_->end()) {
      encodingLayout = it->second;
    }
  }

  std::string_view encoded;
  if (!facebook::nimble::StreamData::hasNullValues(nonNulls)) {
    encoded = detail::encodeScalar<T>(
        options_,
        scalarKind,
        data,
        *pool_,
        *streamEncodingBuffer_,
        encodingLayout,
        options_.encodingOptions);
  } else {
    encoded = detail::encodeNullableScalar<T>(
        options_,
        scalarKind,
        data,
        nonNulls,
        *streamEncodingBuffer_,
        encodingLayout,
        options_.encodingOptions);
  }

  // Track size for trailer.
  if (streamOffset >= streamSizes_.size()) {
    streamSizes_.resize(streamOffset + 1, 0);
  }
  streamSizes_[streamOffset] = static_cast<uint32_t>(encoded.size());
  auto* pos =
      detail::extend(outputBuffer_, static_cast<uint32_t>(encoded.size()));
  std::memcpy(pos, encoded.data(), encoded.size());
  streamEncodingBuffer_->reset();
}

template <typename T>
void StreamDataWriter<T>::close(uint32_t nodeCount) {
  if (!options_.enableEncoding()) {
    detail::writeMissingStreams(outputBuffer_, lastStream_, nodeCount);
    return;
  }

  if (writesHeaderFlags_) {
    NIMBLE_CHECK_LT(
        headerFlagsOffset_,
        outputBuffer_.size(),
        "Invalid null barrier flag offset");
    NIMBLE_CHECK(
        options_.encodingOptions.useVarintRowCount,
        "Non-tablet writers must use varint stream row counts");
    outputBuffer_.data()[headerFlagsOffset_] =
        static_cast<char>(detail::makeFlagsByte(requiresNullBarrier_));
  }

  detail::writeTrailer(
      streamSizes_,
      options_.streamIndicesEncodingType,
      options_.streamSizesEncodingType,
      outputBuffer_);
}

} // namespace facebook::nimble::serde
