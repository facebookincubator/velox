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
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/OrderedInteger.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/PartitionedEliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace facebook::nimble {

/// Complements the order-preserving form of each integer value, so descending
/// values ascend, and stores the result in a nested encoding. Encodings that
/// require non-decreasing values, such as EliasFano, PartitionedEliasFano and
/// DeltaBlock, then apply to descending data without each supporting a
/// direction.
///
/// Each value is mapped to its toOrdered() form and complemented, which is one
/// XOR with a constant mask. A non-increasing stream becomes non-decreasing in
/// the unsigned order of the nested stream, including signed values that cross
/// zero. The mapping is its own inverse and keeps the distance between any two
/// values. For int8_t:
///
///   value:    5     1    -1    -3
///   stored:   0x7A  0x7E  0x80  0x82      stored = value ^ 0x7F
///
/// Reversed is a candidate only for streams whose values mostly descend. Its
/// size estimate is that of EliasFano, PartitionedEliasFano or DeltaBlock over
/// the stored values, so estimating it runs no nested selection. Encoding then
/// selects the nested encoding among every candidate, as other encodings with
/// nested streams do. Reversed decodes its nested stream plus one XOR per
/// value, so give it the read factor of those encodings.
///
/// Data layout is:
///   EncodingPrefix bytes: standard Encoding prefix
///   X bytes: the nested encoding of the stored values
template <typename T>
class ReversedEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Opens a stream and its nested stream without copying them.
  ReversedEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  /// Returns the buffer of rows decoded ahead to the pool.
  ~ReversedEncoding() override {
    this->releaseVectorBuffer(decoded_);
  }

  /// Keeps ownership of pooled buffers in the decoder that acquired them.
  ReversedEncoding(const ReversedEncoding&) = delete;
  ReversedEncoding& operator=(const ReversedEncoding&) = delete;
  ReversedEncoding(ReversedEncoding&&) = delete;
  ReversedEncoding& operator=(ReversedEncoding&&) = delete;

  /// Rewinds sequential decoding to the first row.
  void reset() final;

  /// Advances sequential decoding by 'rowCount' rows.
  void skip(uint32_t rowCount) final;

  /// Materializes 'rowCount' rows from the sequential position.
  void materialize(uint32_t rowCount, void* buffer) final;

  /// Decodes selected rows through the generic visitor interface, decoding
  /// the nested stream kDecodeBatchRows rows at a time.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Encodes the stored values with the nested encoding that 'selection'
  /// selects among every candidate.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates the stream size from the nested encoding with the lowest size
  /// times read factor in 'nestedEncodingReadFactors' among EliasFano,
  /// PartitionedEliasFano and DeltaBlock. The nested selection in encode()
  /// offers every candidate, so it picks an encoding of no higher estimated
  /// cost. Returns nullopt for empty input, for values that descend no more
  /// often than they ascend, and when none of those encodings is offered or
  /// applies to the stored values.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options,
      std::span<const std::pair<EncodingType, float>>
          nestedEncodingReadFactors);

  /// Keeps rows [offset, offset + length) by slicing the nested encoding.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Returns a human-readable description of the stream and its nested
  /// stream.
  std::string debugString(int offset) const final;

  /// Maps 'numValues' values to their stored form in place, or back, since
  /// the mapping is its own inverse.
  static void flip(physicalType* values, size_t numValues) {
    for (size_t i = 0; i < numValues; ++i) {
      values[i] ^= kMask;
    }
  }

  /// Maps a value to its stored form and back.
  static constexpr physicalType kMask =
      static_cast<physicalType>(~toOrdered<T>(physicalType{0}));

 private:
  static_assert(
      isIntegralType<T>(),
      "ReversedEncoding only supports non-bool integer types.");

  // Serializes 'rowCount' rows around 'child', the serialized nested stream of
  // the stored values.
  static std::string_view serialize(
      uint32_t rowCount,
      std::string_view child,
      Buffer& buffer,
      const Encoding::Options& options);

  // Returns the next row for readWithVisitor(), refilling 'decoded_' from the
  // nested encoding when it is used up.
  physicalType decodeNext();

  // Returns the rows of 'decoded_' not read yet.
  uint32_t numRemainingDecoded() const {
    return static_cast<uint32_t>(decoded_.size()) - decodedOffset_;
  }

  // Rows that readWithVisitor() decodes from the nested encoding at once.
  static constexpr uint32_t kDecodeBatchRows{1'024};

  // Nested encoding of the stored values.
  const std::unique_ptr<Encoding> valuesEncoding_;

  // Rows read from 'valuesEncoding_', including the rows in 'decoded_'.
  uint32_t numReadValues_{0};

  // Rows decoded ahead by readWithVisitor(), with the mask applied.
  Vector<physicalType> decoded_;

  // Rows of 'decoded_' already read.
  uint32_t decodedOffset_{0};
};

//
// End of public API. Implementations follow.
//

template <typename T>
ReversedEncoding<T>::ReversedEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>(pool, data, options),
      valuesEncoding_{EncodingFactory().create(
          *this->pool_,
          data.substr(this->dataOffset()),
          std::move(stringBufferFactory),
          options)},
      decoded_{this->template getVectorBuffer<physicalType>()} {
  NIMBLE_CHECK_FILE_EQ(
      valuesEncoding_->dataType(),
      TypeTraits<physicalType>::dataType,
      "Reversed nested stream has the wrong data type.");
  NIMBLE_CHECK_FILE_EQ(
      valuesEncoding_->rowCount(),
      this->rowCount(),
      "Reversed nested stream has the wrong row count.");
}

template <typename T>
void ReversedEncoding<T>::reset() {
  valuesEncoding_->reset();
  numReadValues_ = 0;
  decoded_.clear();
  decodedOffset_ = 0;
}

template <typename T>
void ReversedEncoding<T>::skip(uint32_t rowCount) {
  // Rows that readWithVisitor() decoded ahead come first.
  const auto numFromDecoded = std::min(rowCount, numRemainingDecoded());
  decodedOffset_ += numFromDecoded;
  const auto numFromValues = rowCount - numFromDecoded;
  if (numFromValues > 0) {
    valuesEncoding_->skip(numFromValues);
    numReadValues_ += numFromValues;
  }
}

template <typename T>
void ReversedEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* values = static_cast<physicalType*>(buffer);
  // Rows that readWithVisitor() decoded ahead come first.
  const auto numFromDecoded = std::min(rowCount, numRemainingDecoded());
  std::copy_n(decoded_.data() + decodedOffset_, numFromDecoded, values);
  decodedOffset_ += numFromDecoded;
  const auto numFromValues = rowCount - numFromDecoded;
  if (numFromValues > 0) {
    valuesEncoding_->materialize(numFromValues, values + numFromDecoded);
    numReadValues_ += numFromValues;
    flip(values + numFromDecoded, numFromValues);
  }
}

template <typename T>
typename ReversedEncoding<T>::physicalType ReversedEncoding<T>::decodeNext() {
  if (numRemainingDecoded() == 0) {
    const auto numRows =
        std::min(kDecodeBatchRows, this->rowCount() - numReadValues_);
    NIMBLE_CHECK_GT(numRows, 0, "Read past the last Reversed row.");
    decoded_.resize(numRows);
    valuesEncoding_->materialize(numRows, decoded_.data());
    numReadValues_ += numRows;
    flip(decoded_.data(), numRows);
    decodedOffset_ = 0;
  }
  return decoded_[decodedOffset_++];
}

template <typename T>
template <typename DecoderVisitor>
void ReversedEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return decodeNext(); });
}

template <typename T>
std::string_view ReversedEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  auto* pool = &buffer.getMemoryPool();
  ScopedVector<physicalType> stored{values.size(), pool, options.bufferPool};
  std::copy(values.begin(), values.end(), stored.data());
  flip(stored.data(), stored.size());
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto child = selection.template encodeNested<physicalType>(
      EncodingIdentifiers::Reversed::Values,
      {stored.data(), stored.size()},
      scopedBuffer.get(),
      options);
  return serialize(
      static_cast<uint32_t>(values.size()), child, buffer, options);
}

template <typename T>
std::optional<uint64_t> ReversedEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options,
    std::span<const std::pair<EncodingType, float>> nestedEncodingReadFactors) {
  if (values.empty()) {
    return std::nullopt;
  }
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  // Values that never descend have nothing to reverse. isNonDecreasing() stops
  // at the first descent, so this skips counting the pairs of sorted streams.
  if (statistics.template isNonDecreasing<T>()) {
    return std::nullopt;
  }
  // EliasFano and DeltaBlock need every pair of values to descend, and
  // PartitionedEliasFano starts a partition where a stored value drops, which
  // is where a value rises. Reversed thus stores fewer partitions than
  // PartitionedEliasFano over the values themselves only when the values
  // descend more often than they ascend. Skips copying the values unless they
  // descend more than 'reversedMinDescendingRatio' times as often.
  const auto order = statistics.template adjacentOrder<T>();
  if (static_cast<double>(order.numDescending) <=
      options.reversedMinDescendingRatio *
          static_cast<double>(order.numAscending)) {
    return std::nullopt;
  }
  // TODO: Allocate 'stored' from a memory pool, as encode() does.
  std::vector<physicalType> stored{values.begin(), values.end()};
  flip(stored.data(), stored.size());
  const std::span<const physicalType> storedValues{stored};
  const auto storedStatistics = Statistics<physicalType>::create(storedValues);
  std::optional<uint64_t> nestedSize;
  double minCost{std::numeric_limits<double>::max()};
  for (const auto& [encodingType, readFactor] : nestedEncodingReadFactors) {
    std::optional<uint64_t> size;
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (encodingType) {
      case EncodingType::EliasFano:
        size = EliasFanoEncoding<physicalType>::estimateSize(
            storedValues, storedStatistics, options);
        break;
      case EncodingType::PartitionedEliasFano:
        size = PartitionedEliasFanoEncoding<physicalType>::estimateSize(
            storedValues, options);
        break;
      case EncodingType::DeltaBlock:
        size = DeltaBlockEncoding<physicalType>::estimateSize(
            storedValues, options);
        break;
      default:
        // Other encodings store the reversed values no smaller than the
        // values themselves.
        break;
    }
    if (!size.has_value()) {
      continue;
    }
    const auto cost = static_cast<double>(*size) * readFactor;
    if (cost < minCost) {
      minCost = cost;
      nestedSize = size;
    }
  }
  if (!nestedSize.has_value()) {
    return std::nullopt;
  }
  return EncodingPrefix::serializedSize(
             static_cast<uint32_t>(values.size()), options.useVarintRowCount) +
      *nestedSize;
}

template <typename T>
std::string_view ReversedEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  EncodingPrefix::checkSliceRange(
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount),
      offset,
      length);
  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  // Slices the nested stream of the stored values.
  const auto child = EncodingFactory::slice(
      encoded.substr(
          EncodingPrefix::prefixSize(encoded, options.useVarintRowCount)),
      offset,
      length,
      scopedBuffer.get(),
      options);
  return serialize(length, child, buffer, options);
}

template <typename T>
std::string_view ReversedEncoding<T>::serialize(
    uint32_t rowCount,
    std::string_view child,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const size_t encodingSize =
      EncodingPrefix::serializedSize(rowCount, useVarint) + child.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Reversed,
      TypeTraits<T>::dataType,
      rowCount,
      useVarint,
      pos);
  encoding::writeBytes(child, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string ReversedEncoding<T>::debugString(int offset) const {
  return fmt::format(
      "{}\n{}values child:\n{}",
      Encoding::debugString(offset),
      std::string(offset + 2, ' '),
      valuesEncoding_->debugString(offset + 4));
}

} // namespace facebook::nimble
