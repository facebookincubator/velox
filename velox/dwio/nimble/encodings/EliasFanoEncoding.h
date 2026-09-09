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
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>
#include <folly/Range.h>
#include <folly/compression/elias_fano/EliasFanoCoding.h>
#include <folly/lang/Bits.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Encodes non-decreasing integers for compact positional and lower-bound
/// access. This encoding is most useful for sparse sorted identifiers and
/// offsets where random lookup matters as much as sequential decoding.
template <typename T>
class EliasFanoEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Opens an Elias-Fano stream without copying its encoded payload.
  EliasFanoEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  /// Rewinds sequential decoding to the start of the stream.
  void reset() final;

  /// Advances the sequential decoder by 'rowCount' rows.
  void skip(uint32_t rowCount) final;

  /// Materializes rows from the current sequential decoder position.
  void materialize(uint32_t rowCount, void* buffer) final;

  /// Materializes an absolute row range without changing sequential state.
  /// This supports concurrent range reads through EncodingView.
  void materializeAt(uint32_t offset, uint32_t rowCount, physicalType* output)
      const;

  /// Returns the value at an absolute row without changing sequential state.
  /// This supports random index-key reconstruction and EncodingView reads.
  physicalType valueAt(uint32_t row) const;

  /// Returns the first row whose value is at least 'value'. This supports
  /// value-to-row lookup in sorted index streams.
  uint32_t lowerBound(physicalType value) const;

  /// Decodes selected rows through the generic visitor interface.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Encodes a non-empty, non-decreasing integer span.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Encodes rows [offset, offset + length) as a standalone Elias-Fano stream.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates encoded size, or returns nullopt for empty, unsorted, or
  /// oversized input. Requires 'statistics' derived from 'values'.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {});

  /// Returns a human-readable description of the encoded layout.
  std::string debugString(int offset) const final;

 private:
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<T, bool>,
      "EliasFanoEncoding only supports non-bool integer types.");

  // The wire layout is [prefix][base][lower bit count][upper byte count]
  // [alignment padding][skip pointers][forward pointers][lower bits]
  // [upper bits][trailing padding]. The constants below define this layout and
  // must remain stable unless the Nimble wire format is versioned.
  static constexpr size_t kSkipQuantum{128};
  static constexpr size_t kForwardQuantum{256};
  static constexpr bool kUpperFirst{false};
  using Encoder = folly::compression::EliasFanoEncoder<
      uint64_t,
      uint64_t,
      kSkipQuantum,
      kForwardQuantum,
      kUpperFirst>;
  using Reader = folly::compression::EliasFanoReader<Encoder>;

  // Number of bytes between the common prefix and the Folly payload.
  static constexpr uint8_t kHeaderSize{/*base=*/sizeof(uint64_t) +
                                       /*lowerBits=*/sizeof(uint8_t) +
                                       /*upperBytes=*/sizeof(uint32_t)};

  // Makes Folly's word-at-a-time payload reads safe at the end of the stream.
  static constexpr size_t kPayloadPaddingBytes{
      kUpperFirst ? folly::compression::kLowerTrailingBytes
                  : folly::compression::kUpperTrailingBytes};

  // Aligns Folly's uint64 pointer tables while encoding into the arena.
  static constexpr size_t kPayloadAlignment{alignof(uint64_t)};

  // Folly's encoder writes lower values with writeBits56().
  static constexpr uint8_t kMaxLowerBits{56};

  // Moves signed values into monotonically ordered unsigned space.
  static constexpr uint64_t kSignMask{
      uint64_t{1} << (sizeof(physicalType) * 8 - 1)};

  // Signed logical types use an unsigned physical type of the same width, so
  // toggling its sign bit maps two's-complement order to unsigned order.
  static uint64_t toOrdered(physicalType value) {
    if constexpr (std::is_signed_v<T>) {
      using UnsignedPhysicalType = std::make_unsigned_t<physicalType>;
      return static_cast<uint64_t>(static_cast<UnsignedPhysicalType>(value)) ^
          kSignMask;
    } else {
      return static_cast<uint64_t>(value);
    }
  }

  // Restores a physical value from its unsigned sortable representation.
  static physicalType fromOrdered(uint64_t value) {
    if constexpr (std::is_signed_v<T>) {
      using UnsignedPhysicalType = std::make_unsigned_t<physicalType>;
      return static_cast<physicalType>(
          static_cast<UnsignedPhysicalType>(value ^ kSignMask));
    } else {
      return static_cast<physicalType>(value);
    }
  }

  // Holds the normalized base and the corresponding Folly payload layout.
  struct ValueLayout {
    // Absolute value represented by relative value zero in the payload.
    uint64_t base{0};

    // Describes the byte layout of the base-relative values.
    Encoder::Layout payload;
  };

  // Returns the normalized base and payload layout for valid input.
  static std::optional<ValueLayout> layoutForValues(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics);

  // Returns the format-defined payload offset for the selected prefix size.
  static constexpr size_t payloadOffset(uint32_t prefixSize) {
    return (prefixSize + kHeaderSize + kPayloadAlignment - 1) &
        ~(kPayloadAlignment - 1);
  }

  // Serializes values supplied relative to 'base' into a complete stream.
  // Invokes 'relativeValueAt' exactly once per row in ascending row order.
  template <typename RelativeValueAt>
  static std::string_view encode(
      uint32_t rowCount,
      uint64_t base,
      const Encoder::Layout& layout,
      RelativeValueAt&& relativeValueAt,
      Buffer& buffer,
      const Encoding::Options& options);

  // Reads an absolute row with an independent, allocation-free cursor.
  physicalType readValue(uint32_t row) const;

  struct RandomReadCache {
    uint64_t encodingId{0};
    const uint8_t* upper{nullptr};
    std::optional<Reader> reader;
  };

  // Returns this thread's reusable cursor for random reads on this encoding.
  Reader& randomReader() const;

  // Assigns a never-reused key so thread-local cursors survive address reuse.
  static uint64_t nextEncodingId() {
    static /* library-local */ std::atomic_uint64_t counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
  }

  // Reads and advances the stateful sequential cursor.
  physicalType readNextValue();

  // Creates the stateful cursor on its first sequential operation.
  Reader& ensureReader();

  // Absolute value represented by relative value zero in the payload.
  uint64_t base_{0};

  // Bit width of each lower-value component in the Folly layout.
  uint8_t lowerBits_{0};

  // Byte length of the unary-coded upper-value component.
  uint32_t upperBytes_{0};

  // Non-owning parsed view over the serialized Folly payload.
  Encoder::CompressedList list_;

  // Identifies this parsed stream in the per-thread random-read cache.
  const uint64_t encodingId_{nextEncodingId()};

  // Cursor created on demand for the stateful Encoding interface.
  std::optional<Reader> reader_;

  // Next logical row consumed by the stateful Encoding interface.
  uint32_t row_{0};
};

namespace detail::elias_fano {

// Returns nullopt so encoding selection can skip oversized streams. encode()
// turns nullopt into an exception if the selected encoding exceeds the limit.
constexpr std::optional<uint32_t> trySerializedSize(
    uint64_t payloadOffset,
    uint64_t payloadBytes,
    uint64_t paddingBytes) noexcept {
  constexpr uint64_t kMaxSerializedSize{std::numeric_limits<uint32_t>::max()};
  if (payloadOffset > kMaxSerializedSize ||
      paddingBytes > kMaxSerializedSize - payloadOffset ||
      payloadBytes > kMaxSerializedSize - payloadOffset - paddingBytes) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(payloadOffset + payloadBytes + paddingBytes);
}

} // namespace detail::elias_fano

template <typename T>
EliasFanoEncoding<T>::EliasFanoEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& /*stringBufferFactory*/,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options} {
  NIMBLE_CHECK_FILE_GT(
      this->rowCount(), 0, "EliasFano encoding requires at least one row.");
  const auto dataOffset = this->dataOffset();
  NIMBLE_CHECK_FILE_LE(dataOffset, data.size(), "Truncated EliasFano header.");
  NIMBLE_CHECK_FILE_GE(
      data.size() - dataOffset, kHeaderSize, "Truncated EliasFano header.");
  const char* position = data.data() + dataOffset;
  base_ = encoding::read<uint64_t>(position);
  lowerBits_ = encoding::read<uint8_t>(position);
  upperBytes_ = encoding::readUint32(position);
  NIMBLE_CHECK_FILE_LE(
      lowerBits_, kMaxLowerBits, "Invalid EliasFano lower bit count.");
  NIMBLE_CHECK_FILE_GE(
      static_cast<uint64_t>(upperBytes_) * 8,
      this->rowCount(),
      "Invalid EliasFano upper bit count.");

  // Reconstructing the layout performs arithmetic only; it does not scan or
  // decode values.
  const auto layout = Encoder::Layout::fromInternalSizes(
      lowerBits_, upperBytes_, this->rowCount());
  const auto encodedPayloadOffset = payloadOffset(dataOffset);
  NIMBLE_CHECK_FILE_LE(
      encodedPayloadOffset, data.size(), "Invalid EliasFano payload size.");
  position = data.data() + encodedPayloadOffset;
  const auto remainingBytes = data.size() - encodedPayloadOffset;
  NIMBLE_CHECK_FILE_GE(
      remainingBytes, kPayloadPaddingBytes, "Invalid EliasFano payload size.");
  NIMBLE_CHECK_FILE_LE(
      layout.bytes(),
      remainingBytes - kPayloadPaddingBytes,
      "Invalid EliasFano payload size.");
  folly::ByteRange range{
      reinterpret_cast<const uint8_t*>(position), layout.bytes()};
  // openList creates non-owning pointers into 'range' without allocation.
  list_ = layout.openList(range);
  uint64_t upperValueCount{0};
  size_t upperByteOffset{0};
  for (; upperByteOffset + sizeof(uint64_t) <= upperBytes_;
       upperByteOffset += sizeof(uint64_t)) {
    upperValueCount += std::popcount(
        folly::loadUnaligned<uint64_t>(list_.upper + upperByteOffset));
  }
  if (upperByteOffset < upperBytes_) {
    const auto remainingUpperBytes = upperBytes_ - upperByteOffset;
    // The persisted trailing padding makes the final word readable.
    const auto upperTail =
        folly::loadUnaligned<uint64_t>(list_.upper + upperByteOffset) &
        ((uint64_t{1} << (remainingUpperBytes * 8)) - 1);
    upperValueCount += std::popcount(upperTail);
  }
  NIMBLE_CHECK_FILE_EQ(
      upperValueCount, this->rowCount(), "Invalid EliasFano upper bits.");
}

template <typename T>
void EliasFanoEncoding<T>::reset() {
  if (reader_.has_value()) {
    reader_->reset();
  }
  row_ = 0;
}

template <typename T>
typename EliasFanoEncoding<T>::Reader& EliasFanoEncoding<T>::ensureReader() {
  if (!reader_.has_value()) {
    reader_.emplace(list_);
  }
  return *reader_;
}

template <typename T>
typename EliasFanoEncoding<T>::Reader& EliasFanoEncoding<T>::randomReader()
    const {
  static thread_local RandomReadCache cache;
  if (cache.encodingId != encodingId_ || cache.upper != list_.upper) {
    cache.encodingId = encodingId_;
    cache.upper = list_.upper;
    cache.reader.emplace(list_);
  }
  return *cache.reader;
}

template <typename T>
void EliasFanoEncoding<T>::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(row_, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Skipping past end of encoding.");
  if (rowCount == 0) {
    // Folly reports its initial before-first cursor as invalid for skip(0).
    // The generic Encoding interface treats a zero-row advance as a no-op.
    return;
  }
  const bool skipped = ensureReader().skip(rowCount);
  NIMBLE_CHECK(skipped, "Unexpected end of EliasFano stream.");
  row_ += rowCount;
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType EliasFanoEncoding<T>::readValue(
    uint32_t row) const {
  NIMBLE_CHECK_LT(row, this->rowCount(), "Reading past end of encoding.");
  auto& reader = randomReader();
  const bool found = reader.jump(row);
  NIMBLE_CHECK(found, "Failed to seek EliasFano row.");
  return fromOrdered(base_ + reader.value());
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType
EliasFanoEncoding<T>::readNextValue() {
  NIMBLE_CHECK_LT(row_, this->rowCount(), "Reading past end of encoding.");
  auto& reader = ensureReader();
  const bool advanced = reader.next();
  NIMBLE_CHECK(advanced, "Unexpected end of EliasFano stream.");
  ++row_;
  return fromOrdered(base_ + reader.value());
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType EliasFanoEncoding<T>::valueAt(
    uint32_t row) const {
  return readValue(row);
}

template <typename T>
uint32_t EliasFanoEncoding<T>::lowerBound(physicalType value) const {
  const auto ordered = toOrdered(value);
  if (ordered <= base_) {
    return 0;
  }
  Reader reader{list_};
  if (!reader.jumpTo(ordered - base_)) {
    return this->rowCount();
  }
  return reader.position();
}

template <typename T>
void EliasFanoEncoding<T>::materializeAt(
    uint32_t offset,
    uint32_t rowCount,
    physicalType* output) const {
  NIMBLE_CHECK_LE(offset, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - offset, "Reading past end of encoding.");
  if (rowCount == 0) {
    // Zero-length reads are valid at any in-range offset.
    return;
  }

  Reader reader{list_};
  const bool found = reader.jump(offset);
  NIMBLE_CHECK(found, "Failed to seek EliasFano row.");
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = fromOrdered(base_ + reader.value());
    if (i + 1 < rowCount) {
      const bool advanced = reader.next();
      NIMBLE_CHECK(advanced, "Unexpected end of EliasFano stream.");
    }
  }
}

template <typename T>
void EliasFanoEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(row_, this->rowCount(), "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Reading past end of encoding.");
  auto* output = static_cast<physicalType*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = readNextValue();
  }
}

template <typename T>
template <typename DecoderVisitor>
void EliasFanoEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return readNextValue(); });
}

template <typename T>
std::optional<typename EliasFanoEncoding<T>::ValueLayout>
EliasFanoEncoding<T>::layoutForValues(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics) {
  if (values.empty() || !statistics.template isNonDecreasing<T>()) {
    return std::nullopt;
  }

  const auto base = toOrdered(values.front());
  return ValueLayout{
      .base = base,
      .payload = Encoder::Layout::fromUpperBoundAndSize(
          toOrdered(values.back()) - base, values.size()),
  };
}

template <typename T>
std::optional<uint64_t> EliasFanoEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (values.size() > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  const auto layout = layoutForValues(values, statistics);
  if (!layout.has_value()) {
    return std::nullopt;
  }
  const auto rowCount = static_cast<uint32_t>(values.size());
  const auto prefixSize =
      EncodingPrefix::serializedSize(rowCount, options.useVarintRowCount);
  const auto encodingSize = detail::elias_fano::trySerializedSize(
      payloadOffset(prefixSize), layout->payload.bytes(), kPayloadPaddingBytes);
  return encodingSize;
}

template <typename T>
template <typename RelativeValueAt>
std::string_view EliasFanoEncoding<T>::encode(
    uint32_t rowCount,
    uint64_t base,
    const Encoder::Layout& layout,
    RelativeValueAt&& relativeValueAt,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto payloadBytes = layout.bytes();
  const auto prefixSize =
      Encoding::serializePrefixSize(rowCount, options.useVarintRowCount);
  const auto encodedPayloadOffset = payloadOffset(prefixSize);
  const auto encodingSize = detail::elias_fano::trySerializedSize(
      encodedPayloadOffset, payloadBytes, kPayloadPaddingBytes);
  NIMBLE_CHECK(
      encodingSize.has_value(), "EliasFano encoding exceeds uint32 size.");

  const auto allocationSize = *encodingSize + kPayloadAlignment - 1;
  void* reserved = buffer.reserve(allocationSize);
  auto availableBytes = static_cast<size_t>(allocationSize);
  NIMBLE_CHECK_NOT_NULL(
      std::align(
          kPayloadAlignment,
          static_cast<size_t>(*encodingSize),
          reserved,
          availableBytes));
  auto* const encodedBegin = static_cast<char*>(reserved);
  char* position = encodedBegin;
  Encoding::serializePrefix(
      EncodingType::EliasFano,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      position);
  // The normalized base can require ten varint bytes for signed and unsigned
  // 64-bit values. A fixed-width field is smaller in that case and keeps the
  // payload offset constant.
  encoding::write<uint64_t>(base, position);
  encoding::write<uint8_t>(layout.numLowerBits, position);
  encoding::writeUint32(static_cast<uint32_t>(layout.upper), position);
  const auto payloadAlignmentPadding =
      encodedPayloadOffset - static_cast<size_t>(position - encodedBegin);
  std::memset(position, 0, payloadAlignmentPadding);
  position += payloadAlignmentPadding;
  NIMBLE_DCHECK_EQ(
      reinterpret_cast<uintptr_t>(position) % kPayloadAlignment, 0);
  folly::MutableByteRange payload{
      reinterpret_cast<uint8_t*>(position), payloadBytes};
  Encoder encoder{layout.openList(payload)};
  for (uint32_t row{0}; row < rowCount; ++row) {
    encoder.add(relativeValueAt(row));
  }
  encoder.finish();
  position += payloadBytes;
  std::memset(position, 0, kPayloadPaddingBytes);
  position += kPayloadPaddingBytes;
  NIMBLE_CHECK_EQ(
      position - encodedBegin, *encodingSize, "Encoding size mismatch.");
  return {encodedBegin, static_cast<size_t>(*encodingSize)};
}

template <typename T>
std::string_view EliasFanoEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(
      values.size(),
      std::numeric_limits<uint32_t>::max(),
      "EliasFano row count exceeds uint32.");
  const auto layout = layoutForValues(values, selection.statistics());
  if (!layout.has_value()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "EliasFano requires non-empty non-decreasing values.");
  }

  return encode(
      static_cast<uint32_t>(values.size()),
      layout->base,
      layout->payload,
      [&](uint32_t row) { return toOrdered(values[row]) - layout->base; },
      buffer,
      options);
}

template <typename T>
std::string_view EliasFanoEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const EliasFanoEncoding<T> source{
      buffer.getMemoryPool(), encoded, {}, options};
  Reader reader{source.list_};
  const bool foundFirst = reader.jump(offset);
  NIMBLE_CHECK(foundFirst, "Failed to seek first EliasFano slice row.");
  const auto firstRelative = reader.value();
  const bool foundLast = reader.jump(offset + length - 1);
  NIMBLE_CHECK(foundLast, "Failed to seek last EliasFano slice row.");
  const auto lastRelative = reader.value();
  const auto sliceBase = source.base_ + firstRelative;
  const auto layout = Encoder::Layout::fromUpperBoundAndSize(
      lastRelative - firstRelative, length);

  const bool rewound = reader.jump(offset);
  NIMBLE_CHECK(rewound, "Failed to rewind to first EliasFano slice row.");
  // Encode directly from the compressed cursor. Generic slicing would first
  // materialize the selected values into a temporary vector. Folly exposes
  // scalar Reader::next() and Encoder::add() operations, and the new base and
  // layout prevent copying the source payload as an encoded byte range.
  return encode(
      length,
      sliceBase,
      layout,
      [&](uint32_t row) {
        const auto relative = reader.value() - firstRelative;
        if (row + 1 < length) {
          const bool advanced = reader.next();
          NIMBLE_CHECK(advanced, "Unexpected end of EliasFano stream.");
        }
        return relative;
      },
      buffer,
      options);
}

template <typename T>
std::string EliasFanoEncoding<T>::debugString(int offset) const {
  return Encoding::debugString(offset) +
      fmt::format(
             "\n{}lowerBits={}, upperBytes={}",
             std::string(offset, ' '),
             lowerBits_,
             upperBytes_);
}

} // namespace facebook::nimble
