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

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/OrderedInteger.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

namespace detail::reverse_elias_fano {

// Returns what the child stream stores for 'value' of logical type T: the
// complement of its order-preserving form within the type's width. A
// non-increasing sequence becomes non-decreasing, and the gaps between values
// stay the same.
template <typename T>
typename TypeTraits<T>::physicalType toStored(
    typename TypeTraits<T>::physicalType value) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  return static_cast<PhysicalType>(~toOrdered<T>(value));
}

// Restores a value of logical type T from what the child stream stores.
template <typename T>
typename TypeTraits<T>::physicalType fromStored(
    typename TypeTraits<T>::physicalType stored) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  return fromOrdered<T>(static_cast<PhysicalType>(~stored));
}

// Returns the child stream that follows the 'prefixSize' bytes of a stream's
// prefix, after checking that it is an EliasFano stream of 'rowCount' values
// of T's unsigned physical type.
template <typename T>
std::string_view childStream(
    std::string_view data,
    uint32_t prefixSize,
    uint32_t rowCount,
    bool useVarintRowCount) {
  NIMBLE_CHECK_FILE_LE(
      prefixSize, data.size(), "Truncated ReverseEliasFano stream.");
  const auto child = data.substr(prefixSize);

  // EncodingPrefix::consume() reports truncation as an internal error.
  NIMBLE_CHECK_FILE_GE(
      child.size(),
      EncodingPrefix::serializedSize(rowCount, useVarintRowCount),
      "Truncated ReverseEliasFano child prefix.");
  auto cursor = child;
  const auto childPrefix = EncodingPrefix::consume(cursor, useVarintRowCount);
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::encodingType(childPrefix),
      EncodingType::EliasFano,
      "ReverseEliasFano child must be an EliasFano stream.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(childPrefix),
      TypeTraits<typename TypeTraits<T>::physicalType>::dataType,
      "ReverseEliasFano child must hold the unsigned physical type.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::readRowCount(childPrefix, useVarintRowCount),
      rowCount,
      "ReverseEliasFano child row count must match the stream.");
  return child;
}

// Writes a stream of 'rowCount' rows: its prefix, followed by 'child'.
template <typename T>
std::string_view serialize(
    uint32_t rowCount,
    std::string_view child,
    Buffer& buffer,
    const Encoding::Options& options) {
  const size_t encodingSize =
      EncodingPrefix::serializedSize(rowCount, options.useVarintRowCount) +
      child.size();
  char* reserved = buffer.reserve(encodingSize);
  char* position = reserved;
  EncodingPrefix::serialize(
      EncodingType::ReverseEliasFano,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      position);
  encoding::writeBytes(child, position);
  NIMBLE_DCHECK_EQ(
      position - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

} // namespace detail::reverse_elias_fano

/// Encodes non-increasing integers with Elias-Fano, which only stores
/// non-decreasing ones. Each value is stored as the complement of its
/// order-preserving form within the type's width, so a descending stream
/// ascends and rows keep their order. For uint8_t:
///
///   values:   9    7    4    2
///   stored: 246  248  251  253      (255 - value)
///
/// Complementing keeps the gaps between values, so the stored stream costs what
/// an ascending stream with the same gaps costs.
///
/// Layout: [prefix][complete EliasFano stream of the stored values]. The child
/// stream holds T's unsigned physical type, so its values need no further
/// mapping.
template <typename T>
class ReverseEliasFanoEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Opens a stream and its child stream without copying them.
  ReverseEliasFanoEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  /// Rewinds sequential decoding to the first row.
  void reset() final;

  /// Advances sequential decoding by 'rowCount' rows.
  void skip(uint32_t rowCount) final;

  /// Materializes 'rowCount' rows from the sequential position.
  void materialize(uint32_t rowCount, void* buffer) final;

  /// Materializes an absolute row range without changing sequential state.
  /// This supports concurrent range reads through EncodingView.
  void materializeAt(uint32_t offset, uint32_t rowCount, physicalType* output)
      const;

  /// Returns the value at an absolute row without changing sequential state.
  physicalType valueAt(uint32_t row) const;

  /// Decodes selected rows through the generic visitor interface.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Encodes a non-empty, non-increasing integer span.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Encodes rows [offset, offset + length) as a standalone stream.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates encoded size, or returns nullopt for empty or oversized input,
  /// or for input where a value exceeds the one before it. Requires
  /// 'statistics' derived from 'values'.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {});

  /// Returns a human-readable description of the stream and its child.
  std::string debugString(int offset) const final;

 private:
  static_assert(
      std::is_integral_v<T> && !std::is_same_v<T, bool>,
      "ReverseEliasFanoEncoding only supports non-bool integer types.");

  // Stores the values over T's unsigned physical type.
  using ChildEncoding = EliasFanoEncoding<physicalType>;

  // Holds the stored values in row order.
  ChildEncoding child_;
};

template <typename T>
ReverseEliasFanoEncoding<T>::ReverseEliasFanoEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      child_{
          pool,
          detail::reverse_elias_fano::childStream<T>(
              data,
              this->dataOffset(),
              this->rowCount(),
              options.useVarintRowCount),
          stringBufferFactory,
          options} {}

template <typename T>
void ReverseEliasFanoEncoding<T>::reset() {
  child_.reset();
}

template <typename T>
void ReverseEliasFanoEncoding<T>::skip(uint32_t rowCount) {
  child_.skip(rowCount);
}

template <typename T>
void ReverseEliasFanoEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  child_.materialize(rowCount, buffer);
  auto* output = static_cast<physicalType*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = detail::reverse_elias_fano::fromStored<T>(output[i]);
  }
}

template <typename T>
void ReverseEliasFanoEncoding<T>::materializeAt(
    uint32_t offset,
    uint32_t rowCount,
    physicalType* output) const {
  child_.materializeAt(offset, rowCount, output);
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = detail::reverse_elias_fano::fromStored<T>(output[i]);
  }
}

template <typename T>
typename ReverseEliasFanoEncoding<T>::physicalType
ReverseEliasFanoEncoding<T>::valueAt(uint32_t row) const {
  return detail::reverse_elias_fano::fromStored<T>(child_.valueAt(row));
}

template <typename T>
template <typename DecoderVisitor>
void ReverseEliasFanoEncoding<T>::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        physicalType value;
        materialize(1, &value);
        return value;
      });
}

template <typename T>
std::string_view ReverseEliasFanoEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if (values.empty() || !selection.statistics().template isNonIncreasing<T>()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "ReverseEliasFano requires non-empty non-increasing values.");
  }
  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  const auto child = ChildEncoding::encodeNonDecreasing(
      values, scopedBuffer.get(), options, [](physicalType value) {
        return detail::reverse_elias_fano::toStored<T>(value);
      });
  return detail::reverse_elias_fano::serialize<T>(
      static_cast<uint32_t>(values.size()), child, buffer, options);
}

template <typename T>
std::string_view ReverseEliasFanoEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  const auto child = detail::reverse_elias_fano::childStream<T>(
      encoded,
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount),
      rowCount,
      options.useVarintRowCount);
  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  const auto childSlice =
      ChildEncoding::slice(child, offset, length, scopedBuffer.get(), options);
  return detail::reverse_elias_fano::serialize<T>(
      length, childSlice, buffer, options);
}

template <typename T>
std::optional<uint64_t> ReverseEliasFanoEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (values.empty() || values.size() > std::numeric_limits<uint32_t>::max() ||
      !statistics.template isNonIncreasing<T>()) {
    return std::nullopt;
  }
  const auto rowCount = static_cast<uint32_t>(values.size());

  // The stored values ascend by as much as the values descend.
  const auto childSize = ChildEncoding::estimateNonDecreasingSize(
      rowCount,
      toOrdered<T>(values.front()) - toOrdered<T>(values.back()),
      options);
  if (!childSize.has_value()) {
    return std::nullopt;
  }
  return EncodingPrefix::serializedSize(rowCount, options.useVarintRowCount) +
      *childSize;
}

template <typename T>
std::string ReverseEliasFanoEncoding<T>::debugString(int offset) const {
  return Encoding::debugString(offset) +
      fmt::format(
             "\n{}child:\n{}",
             std::string(offset + 2, ' '),
             child_.debugString(offset + 4));
}

} // namespace facebook::nimble
