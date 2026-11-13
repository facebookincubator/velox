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
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Stores an integer stream whose values form an arithmetic progression, such
/// as the row ids 0, 1, 2, ... of a table, as the value of row 0 and the step
/// between adjacent rows. Row i holds base + i * step. The arithmetic is done
/// in the unsigned physical type, modulo 2^bits, so decreasing progressions
/// and progressions that wrap around round-trip too. Reads compute values
/// instead of decoding them, so skipping and reading any row costs O(1).
///
/// encode() rejects any other stream with NIMBLE_INCOMPATIBLE_ENCODING, so a
/// replayed layout falls back to encoding selection.
///
/// Data layout after the standard prefix:
///   sizeof(physicalType) bytes: base, the value of row 0.
///   sizeof(physicalType) bytes: step, the difference between adjacent rows.
///
/// For example, the int64 stream 7, 10, 13, 16 is stored as base 7 and step 3,
/// and the uint8 stream 254, 255, 0, 1 as base 254 and step 1.
template <typename T>
class SequenceEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  SequenceEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  /// Returns the value of row 'row' of the progression that starts at 'base'
  /// and advances by 'step'. Computes in 64-bit unsigned arithmetic, which
  /// wraps like the physical type and avoids the promotion of narrow types to
  /// int.
  static physicalType
  valueAt(physicalType base, physicalType step, uint64_t row) {
    return static_cast<physicalType>(
        static_cast<uint64_t>(base) + row * static_cast<uint64_t>(step));
  }

  /// Returns the value of row 0 of the serialized sequence 'data', whose
  /// standard prefix takes 'dataOffset' bytes.
  static physicalType readBase(std::string_view data, uint32_t dataOffset);

  /// Returns the step of the serialized sequence 'data', whose standard prefix
  /// takes 'dataOffset' bytes.
  static physicalType readStep(std::string_view data, uint32_t dataOffset);

  void reset() final;

  void skip(uint32_t rowCount) final;

  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Computes the selected rows of a chunk into the visitor's values and
  /// applies its filter or hook. Invoked through 'detail::readWithVisitorFast'.
  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  /// Returns the encoded size of 'values', or std::nullopt when they do not
  /// form an arithmetic progression. 'statistics' describe 'values'.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Writes the sequence of rows [offset, offset + length) of 'encoded' to
  /// 'buffer'. The slice keeps the step and starts at the value of row
  /// 'offset'.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

 private:
  static_assert(
      isIntegralType<T>(),
      "SequenceEncoding only supports non-bool integer types.");

  // Bytes after the standard prefix: the base and the step.
  static constexpr uint32_t kDataSize{2 * sizeof(physicalType)};

  // Checks that 'data' holds exactly the base and the step after its prefix.
  static void checkDataSize(std::string_view data, uint32_t dataOffset);

  const physicalType base_;
  const physicalType step_;
  // Next row materialize(), skip() and readWithVisitor() read.
  uint32_t row_{0};
};

//
// End of public API. Implementation follows.
//

template <typename T>
SequenceEncoding<T>::SequenceEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& /*stringBufferFactory*/,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      base_{readBase(data, this->dataOffset())},
      step_{readStep(data, this->dataOffset())} {}

template <typename T>
void SequenceEncoding<T>::checkDataSize(
    std::string_view data,
    uint32_t dataOffset) {
  NIMBLE_CHECK_FILE_EQ(
      data.size(),
      static_cast<size_t>(dataOffset) + kDataSize,
      "Unexpected sequence encoding size.");
}

template <typename T>
typename SequenceEncoding<T>::physicalType SequenceEncoding<T>::readBase(
    std::string_view data,
    uint32_t dataOffset) {
  checkDataSize(data, dataOffset);
  return encoding::peek<physicalType>(data.data() + dataOffset);
}

template <typename T>
typename SequenceEncoding<T>::physicalType SequenceEncoding<T>::readStep(
    std::string_view data,
    uint32_t dataOffset) {
  checkDataSize(data, dataOffset);
  return encoding::peek<physicalType>(
      data.data() + dataOffset + sizeof(physicalType));
}

template <typename T>
void SequenceEncoding<T>::reset() {
  row_ = 0;
}

template <typename T>
void SequenceEncoding<T>::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Skipping past end of encoding.");
  row_ += rowCount;
}

template <typename T>
void SequenceEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(
      rowCount, this->rowCount() - row_, "Reading past end of encoding.");
  auto* output = static_cast<physicalType*>(buffer);
  const uint64_t first = valueAt(base_, step_, row_);
  const uint64_t step = step_;
  for (uint32_t i = 0; i < rowCount; ++i) {
    output[i] = static_cast<physicalType>(first + i * step);
  }
  row_ += rowCount;
}

template <typename T>
template <typename V>
void SequenceEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  using OutputType = detail::ValueType<typename V::DataType>;
  // The bulk path filters 4- and 8-byte values, and writes them to an integral
  // output at least as wide as the physical type.
  constexpr bool kBulkTypes = (isFourByteIntegralType<physicalType>() ||
                               isEightByteIntegralType<physicalType>()) &&
      std::is_integral_v<OutputType> &&
      sizeof(OutputType) >= sizeof(physicalType);
  if constexpr (kBulkTypes) {
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if ((detail::bulkFastPathByDefault<V>() || params.scanFastPath) &&
        velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return valueAt(base_, step_, row_++); });
}

template <typename T>
template <bool kScatter, typename V>
void SequenceEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using OutputType = detail::ValueType<typename V::DataType>;
  if (numSelected == 0) {
    return;
  }
  const auto numRows = visitor.numRows() - visitor.rowIndex();
  // 'selectedRows' count from 'currentRow', which is row 'row_' here.
  const auto offset = static_cast<int64_t>(row_) - currentRow;
  auto* values = detail::mutableValues<OutputType>(visitor, numRows);
  if constexpr (V::dense) {
    const uint64_t first =
        valueAt(base_, step_, static_cast<uint64_t>(selectedRows[0] + offset));
    const uint64_t step = step_;
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = static_cast<OutputType>(
          static_cast<physicalType>(first + static_cast<uint64_t>(i) * step));
    }
  } else {
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = static_cast<OutputType>(valueAt(
          base_, step_, static_cast<uint64_t>(selectedRows[i] + offset)));
    }
  }
  row_ += selectedRows[numSelected - 1] - currentRow + 1;

  // Without scatter, filter or hook, the values are already the output.
  if constexpr (!kScatter && !V::kHasFilter && !V::kHasHook) {
    visitor.addNumValues(numRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }
  detail::applyFixedWidthRun<kScatter>(
      visitor, selectedRows, numSelected, scatterRows, values, numRows);
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
std::optional<uint64_t> SequenceEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (values.empty() || !statistics.sequenceStep().has_value()) {
    return std::nullopt;
  }
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return Encoding::serializePrefixSize(
             static_cast<uint32_t>(values.size()), options.useVarintRowCount) +
      kDataSize;
}

template <typename T>
std::string_view SequenceEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("Sequence encoding cannot be empty.");
  }
  const auto step = selection.statistics().sequenceStep();
  if (!step.has_value()) {
    NIMBLE_INCOMPATIBLE_ENCODING(
        "Sequence encoding requires an arithmetic progression.");
  }
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  const auto rowCount = static_cast<uint32_t>(values.size());
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, options.useVarintRowCount) +
      kDataSize;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Sequence,
      TypeTraits<T>::dataType,
      rowCount,
      options.useVarintRowCount,
      pos);
  encoding::write<physicalType>(values.front(), pos);
  encoding::write<physicalType>(step.value(), pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string_view SequenceEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  EncodingPrefix::checkSliceRange(
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount),
      offset,
      length);
  const auto dataOffset =
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const auto step = readStep(encoded, dataOffset);
  const auto base = valueAt(readBase(encoded, dataOffset), step, offset);
  const uint32_t encodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      kDataSize;
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Sequence,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      pos);
  encoding::write<physicalType>(base, pos);
  encoding::write<physicalType>(step, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string SequenceEncoding<T>::debugString(int offset) const {
  return Encoding::debugString(offset) +
      fmt::format(
             " base={} step={}", static_cast<T>(base_), static_cast<T>(step_));
}

} // namespace facebook::nimble
