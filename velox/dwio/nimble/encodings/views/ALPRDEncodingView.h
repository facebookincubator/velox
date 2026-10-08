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
#include <array>
#include <bit>

#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// Reads ALPRD values without advancing a decoder or re-encoding a slice.
/// Keeps validated exception arrays and delegates main-stream access to views.
template <typename T>
class ALPRDEncodingView final : public TypedEncodingView<T>,
                                private ALPRDEncodingBase {
  static_assert(isFloatingPointType<T>());

 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  /// Validates metadata before the base view reads the encoding prefix.
  ALPRDEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : ALPRDEncodingView(data, pool, options, readMetadata(data, options)) {}

 private:
  // Requires views for the main children. Exception children use decoders so
  // their ordering, positions and high bits are fully checked at construction.
  ALPRDEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options,
      const Metadata& metadata)
      : TypedEncodingView<T>{data, pool, options},
        parameters_{metadata.parameters},
        exceptionPositions_{0, pool, options.bufferPool},
        exceptionHighParts_{0, pool, options.bufferPool} {
    NIMBLE_CHECK_EQ(metadata.dataType, TypeTraits<T>::dataType);
    codes_ = detail::createTypedEncodingView<uint16_t>(
        metadata.children[0], pool, options);
    rightParts_ = detail::createTypedEncodingView<physicalType>(
        metadata.children[1], pool, options);
    loadExceptions(
        *pool, metadata, options, exceptionPositions_, exceptionHighParts_);
  }

  T readTypedAt(uint32_t index) const final {
    return std::bit_cast<T>(readPhysicalAt(index));
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    // Validate the main values even when an exception replaces the high bits.
    const auto value =
        decode(codes_->readAt(index), rightParts_->readAt(index));
    if (exceptionPositions_.empty()) {
      return value;
    }
    const auto* begin = exceptionPositions_.data();
    const auto* end = begin + exceptionPositions_.size();
    const auto* position = std::lower_bound(begin, end, index);
    if (position == end || *position != index) {
      return value;
    }
    return patch(value, position - begin);
  }

  // Batches main-stream reads and locates the first exception only once.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    uint32_t exceptionIndex{0};
    if (!exceptionPositions_.empty()) {
      const auto* begin = exceptionPositions_.data();
      exceptionIndex =
          std::lower_bound(begin, begin + exceptionPositions_.size(), offset) -
          begin;
    }
    // Child reads overwrite the used prefix before it is inspected.
    std::array<uint16_t, 256> codes;
    while (length != 0) {
      const auto count = std::min<uint32_t>(length, codes.size());
      codes_->read(offset, count, codes.data());
      rightParts_->read(offset, count, output);
      for (uint32_t i = 0; i < count; ++i) {
        output[i] = decode(codes[i], output[i]);
      }
      while (exceptionIndex < exceptionPositions_.size() &&
             exceptionPositions_[exceptionIndex] < offset + count) {
        const auto index = exceptionPositions_[exceptionIndex] - offset;
        output[index] = patch(output[index], exceptionIndex);
        ++exceptionIndex;
      }
      offset += count;
      length -= count;
      output += count;
    }
  }

  // Reconstructs dictionary values with the same checks as the decoder.
  physicalType decode(uint16_t code, physicalType right) const {
    NIMBLE_CHECK_LT(
        code, parameters_.dictionarySize, "Invalid ALPRD dictionary code.");
    NIMBLE_CHECK_LE(right, rightMask(), "Invalid ALPRD low part.");
    return right |
        (static_cast<physicalType>(parameters_.dictionary[code])
         << parameters_.rightBitWidth);
  }

  // Replaces only the high bits; the original low bits remain unchanged.
  physicalType patch(physicalType value, uint32_t exceptionIndex) const {
    return (value & rightMask()) |
        (static_cast<physicalType>(exceptionHighParts_[exceptionIndex])
         << parameters_.rightBitWidth);
  }

  physicalType rightMask() const {
    return (physicalType{1} << parameters_.rightBitWidth) - 1;
  }

  // Holds the dictionary and split shared by every row.
  Parameters parameters_;
  // References one dictionary code per non-null value.
  std::unique_ptr<TypedEncodingView<uint16_t>> codes_;
  // References the low bits of each non-null value.
  std::unique_ptr<TypedEncodingView<physicalType>> rightParts_;
  // Holds sorted, validated positions for binary searches and range patching.
  ScopedVector<uint32_t> exceptionPositions_;
  // Preserves construction-time validation and avoids repeated high-bit reads.
  ScopedVector<uint16_t> exceptionHighParts_;
};

} // namespace facebook::nimble
