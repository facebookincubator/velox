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
#include "velox/dwio/nimble/common/OrderedInteger.h"
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

  /// Returns the first row in [begin, end) whose value is at least 'value', or
  /// 'end'. Stores the value of a returned row before 'end' in 'found', which
  /// spares a caller that compares it a second seek.
  uint32_t lowerBound(
      uint32_t begin,
      uint32_t end,
      physicalType value,
      physicalType& found) const;

  /// Returns the first row whose value equals 'value', or std::nullopt when no
  /// row holds it.
  std::optional<uint32_t> find(physicalType value) const;

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

  /// Encodes a non-empty, non-decreasing span with a layout derived from its
  /// first and last values. Lets another encoding store an ascending sequence
  /// it computes from its own input, whose statistics it does not have.
  static std::string_view encodeNonDecreasing(
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options);

  /// Estimates the encoded size of 'rowCount' non-decreasing values whose last
  /// value exceeds the first by 'range'.
  static std::optional<uint64_t> estimateNonDecreasingSize(
      uint32_t rowCount,
      uint64_t range,
      const Encoding::Options& options);

  /// Returns a human-readable description of the encoded layout.
  std::string debugString(int offset) const final;

  /// Reads a stream without the encoding object. Holds the parsed header and a
  /// pointer to the encoded payload, which must outlive it. It is small, so an
  /// encoding made of several lists can keep each one inline where a read
  /// reaches it.
  class List;

  /// Returns the parsed header.
  const List& list() const {
    return list_;
  }

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
    const uint8_t* payload{nullptr};
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

  // Non-owning parsed view over the serialized Folly payload.
  List list_;

  // Identifies this parsed stream in the per-thread random-read cache.
  const uint64_t encodingId_{nextEncodingId()};

  // Cursor created on demand for the stateful Encoding interface.
  std::optional<Reader> reader_;

  // Next logical row consumed by the stateful Encoding interface.
  uint32_t row_{0};
};

/// A stateless reader over one Elias-Fano stream. It holds the header that
/// EliasFanoEncoding validated and a pointer to the serialized Folly payload,
/// and opens a fresh cursor per read, so concurrent reads need no locking.
/// The payload must outlive the list; the encoding object need not, so an
/// encoding made of several lists can keep each one inline.
template <typename T>
class EliasFanoEncoding<T>::List {
 public:
  List() = default;

  /// Wraps a header that EliasFanoEncoding has validated against 'payload',
  /// the start of the serialized Folly payload.
  List(
      uint64_t base,
      uint8_t lowerBits,
      uint32_t upperBytes,
      uint32_t rowCount,
      const uint8_t* payload)
      : base_{base},
        payload_{payload},
        rowCount_{rowCount},
        upperBytes_{upperBytes},
        lowerBits_{lowerBits} {}

  /// Returns the value at 'row'.
  physicalType valueAt(uint32_t row) const;

  /// Returns the first row whose value is at least 'value', or the row count,
  /// storing the value of a returned row in 'found'.
  uint32_t lowerBound(physicalType value, physicalType& found) const;

  /// Materializes rows [offset, offset + rowCount). A zero-length read is
  /// valid at any offset up to the row count.
  void materializeAt(uint32_t offset, uint32_t rowCount, physicalType* output)
      const;

  /// Opens the Folly list. Rebuilding its layout is arithmetic only, so the
  /// header alone is enough to read.
  Encoder::CompressedList open() const;

  uint64_t base() const {
    return base_;
  }

  const uint8_t* payload() const {
    return payload_;
  }

  uint32_t upperBytes() const {
    return upperBytes_;
  }

  uint8_t lowerBits() const {
    return lowerBits_;
  }

 private:
  // Absolute value represented by relative value zero in the payload.
  uint64_t base_{0};

  // Start of the serialized Folly payload.
  const uint8_t* payload_{nullptr};

  // Number of values in the list.
  uint32_t rowCount_{0};

  // Byte length of the unary-coded upper-value component.
  uint32_t upperBytes_{0};

  // Bit width of each lower-value component in the Folly layout.
  uint8_t lowerBits_{0};
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
  const auto base = encoding::read<uint64_t>(position);
  const auto lowerBits = encoding::read<uint8_t>(position);
  const auto upperBytes = encoding::readUint32(position);
  NIMBLE_CHECK_FILE_LE(
      lowerBits, kMaxLowerBits, "Invalid EliasFano lower bit count.");
  NIMBLE_CHECK_FILE_GE(
      static_cast<uint64_t>(upperBytes) * 8,
      this->rowCount(),
      "Invalid EliasFano upper bit count.");

  // Reconstructing the layout performs arithmetic only; it does not scan or
  // decode values.
  const auto layout = Encoder::Layout::fromInternalSizes(
      lowerBits, upperBytes, this->rowCount());
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
  list_ = List{
      base,
      lowerBits,
      upperBytes,
      this->rowCount(),
      reinterpret_cast<const uint8_t*>(position)};
  const auto* const upper = list_.open().upper;
  uint64_t upperValueCount{0};
  size_t upperByteOffset{0};
  for (; upperByteOffset + sizeof(uint64_t) <= upperBytes;
       upperByteOffset += sizeof(uint64_t)) {
    upperValueCount +=
        std::popcount(folly::loadUnaligned<uint64_t>(upper + upperByteOffset));
  }
  if (upperByteOffset < upperBytes) {
    const auto remainingUpperBytes = upperBytes - upperByteOffset;
    // The persisted trailing padding makes the final word readable.
    const auto upperTail =
        folly::loadUnaligned<uint64_t>(upper + upperByteOffset) &
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
    reader_.emplace(list_.open());
  }
  return *reader_;
}

template <typename T>
typename EliasFanoEncoding<T>::Reader& EliasFanoEncoding<T>::randomReader()
    const {
  static thread_local RandomReadCache cache;
  if (cache.encodingId != encodingId_ || cache.payload != list_.payload()) {
    cache.encodingId = encodingId_;
    cache.payload = list_.payload();
    cache.reader.emplace(list_.open());
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
  return fromOrdered<T>(list_.base() + reader.value());
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType
EliasFanoEncoding<T>::readNextValue() {
  NIMBLE_CHECK_LT(row_, this->rowCount(), "Reading past end of encoding.");
  auto& reader = ensureReader();
  const bool advanced = reader.next();
  NIMBLE_CHECK(advanced, "Unexpected end of EliasFano stream.");
  ++row_;
  return fromOrdered<T>(list_.base() + reader.value());
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType EliasFanoEncoding<T>::valueAt(
    uint32_t row) const {
  return readValue(row);
}

template <typename T>
uint32_t EliasFanoEncoding<T>::lowerBound(physicalType value) const {
  physicalType found;
  return lowerBound(0, this->rowCount(), value, found);
}

template <typename T>
uint32_t EliasFanoEncoding<T>::lowerBound(
    uint32_t begin,
    uint32_t end,
    physicalType value,
    physicalType& found) const {
  NIMBLE_CHECK_LE(begin, end, "Invalid lowerBound range.");
  NIMBLE_CHECK_LE(end, this->rowCount(), "Invalid lowerBound range.");
  if (begin == end) {
    return end;
  }
  // The stream ascends as a whole, so its lower bound clamped to the range is
  // the lower bound within [begin, end).
  physicalType listFound;
  const auto row = list_.lowerBound(value, listFound);
  if (row < begin) {
    found = valueAt(begin);
    return begin;
  }
  if (row >= end) {
    return end;
  }
  found = listFound;
  return row;
}

template <typename T>
typename EliasFanoEncoding<T>::Encoder::CompressedList
EliasFanoEncoding<T>::List::open() const {
  const auto layout =
      Encoder::Layout::fromInternalSizes(lowerBits_, upperBytes_, rowCount_);
  folly::ByteRange range{payload_, layout.bytes()};
  // openList creates non-owning pointers into 'range' without allocation.
  return layout.openList(range);
}

template <typename T>
typename EliasFanoEncoding<T>::physicalType EliasFanoEncoding<T>::List::valueAt(
    uint32_t row) const {
  Reader reader{open()};
  const bool found = reader.jump(row);
  NIMBLE_CHECK(found, "Failed to seek EliasFano row.");
  return fromOrdered<T>(base_ + reader.value());
}

template <typename T>
uint32_t EliasFanoEncoding<T>::List::lowerBound(
    physicalType value,
    physicalType& found) const {
  const auto ordered = toOrdered<T>(value);
  if (ordered <= base_) {
    // The first row stores relative value zero.
    found = fromOrdered<T>(base_);
    return 0;
  }
  Reader reader{open()};
  if (!reader.jumpTo(ordered - base_)) {
    return rowCount_;
  }
  found = fromOrdered<T>(base_ + reader.value());
  return static_cast<uint32_t>(reader.position());
}

template <typename T>
std::optional<uint32_t> EliasFanoEncoding<T>::find(physicalType value) const {
  physicalType found;
  const auto row = lowerBound(0, this->rowCount(), value, found);
  if (row == this->rowCount() || found != value) {
    return std::nullopt;
  }
  return row;
}

template <typename T>
void EliasFanoEncoding<T>::materializeAt(
    uint32_t offset,
    uint32_t rowCount,
    physicalType* output) const {
  list_.materializeAt(offset, rowCount, output);
}

template <typename T>
void EliasFanoEncoding<T>::List::materializeAt(
    uint32_t offset,
    uint32_t rowCount,
    physicalType* output) const {
  NIMBLE_CHECK_LE(offset, rowCount_, "Invalid encoding position.");
  NIMBLE_CHECK_LE(
      rowCount, rowCount_ - offset, "Reading past end of encoding.");
  if (rowCount == 0) {
    return;
  }

  Reader reader{open()};
  const bool found = reader.jump(offset);
  NIMBLE_CHECK(found, "Failed to seek EliasFano row.");
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = fromOrdered<T>(base_ + reader.value());
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
std::optional<uint64_t> EliasFanoEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (values.empty() || values.size() > std::numeric_limits<uint32_t>::max() ||
      !statistics.template isNonDecreasing<T>()) {
    return std::nullopt;
  }
  return estimateNonDecreasingSize(
      static_cast<uint32_t>(values.size()),
      toOrdered<T>(values.back()) - toOrdered<T>(values.front()),
      options);
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
  if (values.empty() || !selection.statistics().template isNonDecreasing<T>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "EliasFano requires non-empty non-decreasing values.");
  }
  return encodeNonDecreasing(values, buffer, options);
}

template <typename T>
std::string_view EliasFanoEncoding<T>::encodeNonDecreasing(
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK(!values.empty(), "EliasFano encoding requires values.");
  NIMBLE_CHECK_LE(
      values.size(),
      std::numeric_limits<uint32_t>::max(),
      "EliasFano row count exceeds uint32.");
  const auto base = toOrdered<T>(values.front());
  const auto layout = Encoder::Layout::fromUpperBoundAndSize(
      toOrdered<T>(values.back()) - base, values.size());
  return encode(
      static_cast<uint32_t>(values.size()),
      base,
      layout,
      [&](uint32_t row) { return toOrdered<T>(values[row]) - base; },
      buffer,
      options);
}

template <typename T>
std::optional<uint64_t> EliasFanoEncoding<T>::estimateNonDecreasingSize(
    uint32_t rowCount,
    uint64_t range,
    const Encoding::Options& options) {
  const auto layout = Encoder::Layout::fromUpperBoundAndSize(range, rowCount);
  const auto prefixSize =
      EncodingPrefix::serializedSize(rowCount, options.useVarintRowCount);
  return detail::elias_fano::trySerializedSize(
      payloadOffset(prefixSize), layout.bytes(), kPayloadPaddingBytes);
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
  Reader reader{source.list_.open()};
  const bool foundFirst = reader.jump(offset);
  NIMBLE_CHECK(foundFirst, "Failed to seek first EliasFano slice row.");
  const auto firstRelative = reader.value();
  const bool foundLast = reader.jump(offset + length - 1);
  NIMBLE_CHECK(foundLast, "Failed to seek last EliasFano slice row.");
  const auto lastRelative = reader.value();
  const auto sliceBase = source.list_.base() + firstRelative;
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
             list_.lowerBits(),
             list_.upperBytes());
}

} // namespace facebook::nimble
