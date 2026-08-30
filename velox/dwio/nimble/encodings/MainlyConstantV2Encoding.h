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
#include <limits>
#include <optional>
#include <span>

#include <fmt/core.h>

#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Encodes streams with one common value and stores exception row positions
/// directly as an outer child stream.
///
/// Decode fills the output with the common value, then scatters exception
/// values at the stored positions. Supports non-boolean numeric types.
///
/// Wire format:
///   prefix
///   common value
///   uint32_t exception positions stream byte size
///   exception positions stream bytes
///   uint32_t exception values stream byte size
///   exception values stream bytes
template <typename T>
class MainlyConstantV2Encoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  /// Logical C++ value type exposed by this encoding.
  using cppDataType = T;

  /// Physical value type stored in the encoded stream.
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Deserializes an encoded stream and validates its child streams.
  MainlyConstantV2Encoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options),
        exceptionPositions_(&pool),
        exceptionValuesBuffer_(this->template getVectorBuffer<physicalType>()) {
    const char* cursor = data.data() + this->dataOffset();
    const char* const end = data.end();
    const auto requireBytes = [&](size_t size, std::string_view field) {
      NIMBLE_CHECK_FILE(
          cursor <= end && static_cast<size_t>(end - cursor) >= size,
          "Truncated MainlyConstantV2 {}.",
          field);
    };

    requireBytes(sizeof(physicalType), "common value");
    commonValue_ = encoding::read<physicalType>(cursor);

    requireBytes(sizeof(uint32_t), "exception positions size");
    const uint32_t exceptionPositionsBytes = encoding::readUint32(cursor);
    requireBytes(
        static_cast<size_t>(exceptionPositionsBytes) + sizeof(uint32_t),
        "exception positions stream");
    if (exceptionPositionsBytes != 0) {
      const std::string_view encodedPositions{cursor, exceptionPositionsBytes};
      auto positionsEncoding = EncodingFactory{options}.create(
          *this->pool_, encodedPositions, stringBufferFactory);
      NIMBLE_CHECK_FILE(
          positionsEncoding->dataType() == DataType::Uint32,
          "MainlyConstantV2 positions child must use Uint32.");
      exceptionPositions_.resize(positionsEncoding->rowCount());
      positionsEncoding->materialize(
          exceptionPositions_.size(), exceptionPositions_.data());
    }
    cursor += exceptionPositionsBytes;

    requireBytes(sizeof(uint32_t), "exception values size");
    const uint32_t exceptionValuesBytes = encoding::readUint32(cursor);
    requireBytes(exceptionValuesBytes, "exception values stream");
    NIMBLE_CHECK_FILE(
        (exceptionPositionsBytes == 0) == (exceptionValuesBytes == 0),
        "MainlyConstantV2 child streams must both be empty or non-empty.");
    if (exceptionValuesBytes != 0) {
      exceptionValues_ = EncodingFactory{options}.create(
          *this->pool_, {cursor, exceptionValuesBytes}, stringBufferFactory);
      NIMBLE_CHECK_FILE(
          exceptionValues_->dataType() == TypeTraits<physicalType>::dataType,
          "MainlyConstantV2 values child has an unexpected data type.");
      NIMBLE_CHECK_FILE(
          exceptionValues_->rowCount() == exceptionPositions_.size(),
          "MainlyConstantV2 child row counts do not match.");
    }
    cursor += exceptionValuesBytes;

    NIMBLE_CHECK_FILE(cursor == end, "Unexpected MainlyConstantV2 end.");
    for (uint32_t i = 0; i < exceptionPositions_.size(); ++i) {
      NIMBLE_CHECK_FILE(
          exceptionPositions_[i] < this->rowCount(),
          "MainlyConstantV2 exception position is out of range.");
      NIMBLE_CHECK_FILE(
          i == 0 || exceptionPositions_[i - 1] < exceptionPositions_[i],
          "MainlyConstantV2 exception positions must be strictly increasing.");
    }
  }

  /// Releases the reusable exception-values buffer.
  ~MainlyConstantV2Encoding() override {
    this->releaseVectorBuffer(exceptionValuesBuffer_);
  }

  /// Resets sequential decoding to the first row.
  void reset() final {
    row_ = 0;
    nextExceptionPositionIndex_ = 0;
    if (exceptionValues_ != nullptr) {
      exceptionValues_->reset();
    }
  }

  /// Advances sequential decoding by `rowCount` rows.
  void skip(uint32_t rowCount) final {
    NIMBLE_CHECK_LE(
        rowCount,
        this->rowCount() - row_,
        "Cannot skip beyond MainlyConstantV2 row count.");
    const auto rowEnd = row_ + rowCount;
    const auto beginIndex = nextExceptionPositionIndex_;
    advanceExceptionPositionIndex(rowEnd);
    const auto numExceptions = nextExceptionPositionIndex_ - beginIndex;
    if (numExceptions != 0) {
      NIMBLE_CHECK_NOT_NULL(exceptionValues_);
      exceptionValues_->skip(numExceptions);
    }
    row_ = rowEnd;
  }

  /// Decodes `rowCount` sequential rows into `buffer` and advances state.
  void materialize(uint32_t rowCount, void* buffer) final {
    NIMBLE_CHECK_LE(
        rowCount,
        this->rowCount() - row_,
        "Cannot materialize beyond MainlyConstantV2 row count.");
    const auto rowBegin = row_;
    const auto rowEnd = row_ + rowCount;
    auto* output = static_cast<physicalType*>(buffer);
    velox::simd::simdFill(output, commonValue_, rowCount);

    const auto beginIndex = nextExceptionPositionIndex_;
    advanceExceptionPositionIndex(rowEnd);
    const auto numExceptions = nextExceptionPositionIndex_ - beginIndex;
    if (numExceptions != 0) {
      NIMBLE_CHECK_NOT_NULL(exceptionValues_);
      exceptionValuesBuffer_.resize(numExceptions);
      exceptionValues_->materialize(
          numExceptions, exceptionValuesBuffer_.data());
      for (uint32_t i = 0; i < numExceptions; ++i) {
        output[exceptionPositions_[beginIndex + i] - rowBegin] =
            exceptionValuesBuffer_[i];
      }
    }

    row_ = rowEnd;
  }

  /// Reads selected rows through the standard decoder visitor interface.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params) {
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

  /// Encodes values using one common value and two exception child streams.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    if constexpr (
        !isNumericType<physicalType>() || isBoolType<physicalType>()) {
      NIMBLE_INCOMPATIBLE_ENCODING(
          "MainlyConstantV2 encoding only supports non-bool numeric data.");
    }
    if (values.empty()) {
      NIMBLE_INCOMPATIBLE_ENCODING("MainlyConstantV2 cannot be empty.");
    }
    NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());

    const auto commonElement =
        findCommonValue(selection.statistics().uniqueCounts().value());
    const auto commonValue = commonElement->first;
    const auto numExceptions = values.size() - commonElement->second;

    auto* pool = &buffer.getMemoryPool();
    Vector<uint32_t> exceptionPositions{pool};
    exceptionPositions.reserve(numExceptions);
    Vector<physicalType> exceptionValues{pool};
    exceptionValues.reserve(numExceptions);
    for (uint32_t i = 0; i < values.size(); ++i) {
      if (values[i] != commonValue) {
        exceptionPositions.push_back(i);
        exceptionValues.push_back(values[i]);
      }
    }

    ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
    const auto serializedExceptionPositions = exceptionPositions.empty()
        ? std::string_view{}
        : selection.template encodeNested<uint32_t>(
              EncodingIdentifiers::MainlyConstantV2::ExceptionPositions,
              exceptionPositions,
              scopedBuffer.get(),
              options);
    const auto serializedExceptionValues = exceptionValues.empty()
        ? std::string_view{}
        : selection.template encodeNested<physicalType>(
              EncodingIdentifiers::MainlyConstantV2::ExceptionValues,
              exceptionValues,
              scopedBuffer.get(),
              options);

    NIMBLE_CHECK_LE(
        serializedExceptionPositions.size(),
        std::numeric_limits<uint32_t>::max(),
        "MainlyConstantV2 positions child is too large.");
    NIMBLE_CHECK_LE(
        serializedExceptionValues.size(),
        std::numeric_limits<uint32_t>::max(),
        "MainlyConstantV2 values child is too large.");
    const uint64_t encodingSize =
        Encoding::serializePrefixSize(
            values.size(), options.useVarintRowCount) +
        sizeof(physicalType) + sizeof(uint32_t) +
        serializedExceptionPositions.size() + sizeof(uint32_t) +
        serializedExceptionValues.size();
    NIMBLE_CHECK_LE(
        encodingSize,
        std::numeric_limits<uint32_t>::max(),
        "MainlyConstantV2 stream is too large.");

    char* reserved = buffer.reserve(encodingSize);
    char* cursor = reserved;
    Encoding::serializePrefix(
        EncodingType::MainlyConstantV2,
        TypeTraits<T>::dataType,
        values.size(),
        options.useVarintRowCount,
        cursor);
    encoding::write<physicalType>(commonValue, cursor);
    encoding::writeString(serializedExceptionPositions, cursor);
    encoding::writeString(serializedExceptionValues, cursor);
    NIMBLE_DCHECK_EQ(
        cursor - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  /// Estimates the encoded size for the supplied value statistics.
  static std::optional<uint64_t> estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {}) {
    if (rowCount == 0 || !statistics.uniqueCounts().has_value()) {
      return std::nullopt;
    }
    const auto commonElement =
        findCommonValue(statistics.uniqueCounts().value());
    const auto numExceptions = rowCount - commonElement->second;

    return estimatePositionLayoutSize(
        rowCount, numExceptions, statistics, options);
  }

  /// Returns a human-readable summary of the decoded stream state.
  std::string debugString(int offset) const final {
    return fmt::format(
        "{}{}<{}> rowCount={} commonValue={} exceptionPositions={}",
        std::string(offset, ' '),
        toString(this->encodingType()),
        toString(this->dataType()),
        this->rowCount(),
        commonValue_,
        exceptionPositions_.size());
  }

 private:
  // Estimates the direct-position layout using fixed-bit-width child streams.
  static std::optional<uint64_t> estimatePositionLayoutSize(
      uint64_t rowCount,
      uint64_t numExceptions,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    const auto positionsSize = numExceptions == 0
        ? 0
        : FixedBitWidthEncoding<uint32_t>::estimateSize(
              numExceptions, 0, rowCount - 1, options);
    const auto exceptionValuesSize = numExceptions == 0
        ? 0
        : FixedBitWidthEncoding<physicalType>::estimateSize(
              numExceptions, statistics.min(), statistics.max(), options);

    return EncodingPrefix::serializedSize(
               static_cast<uint32_t>(rowCount), options.useVarintRowCount) +
        sizeof(physicalType) + 2 * sizeof(uint32_t) + positionsSize +
        exceptionValuesSize;
  }

  // Selects the most frequent value and breaks ties by the smaller value.
  template <typename UniqueCounts>
  static auto findCommonValue(const UniqueCounts& uniqueCounts) {
    return std::max_element(
        uniqueCounts.cbegin(),
        uniqueCounts.cend(),
        [](const auto& left, const auto& right) {
          if (left.second != right.second) {
            return left.second < right.second;
          }
          return left.first > right.first;
        });
  }

  // Advances to the first exception position at or after `rowEnd`.
  void advanceExceptionPositionIndex(uint32_t rowEnd) {
    while (nextExceptionPositionIndex_ < exceptionPositions_.size() &&
           exceptionPositions_[nextExceptionPositionIndex_] < rowEnd) {
      ++nextExceptionPositionIndex_;
    }
  }

  // Value used for rows without an exception.
  physicalType commonValue_;

  // Strictly increasing row positions for exception values.
  Vector<uint32_t> exceptionPositions_;

  // Index of the next exception position in sequential decoding state.
  uint32_t nextExceptionPositionIndex_{0};

  // Nested encoding for exception values, absent when all rows are common.
  std::unique_ptr<Encoding> exceptionValues_;

  // Reusable decode buffer for exception values.
  Vector<physicalType> exceptionValuesBuffer_;

  // Next row in sequential decoding state.
  uint32_t row_{0};
};

} // namespace facebook::nimble
