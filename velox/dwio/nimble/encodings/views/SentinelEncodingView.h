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
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::detail {

/// Represents nulls with the reserved physical value of a child encoding.
/// Single-value reads require a non-null row. Uses null-aware reads for
/// batches.
template <typename T>
class SentinelEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;
  using TypedEncodingView<T>::read;

  /// Creates a random-access view over a sentinel-encoded stream.
  SentinelEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Sentinel);
    NIMBLE_CHECK_LE(
        this->dataOffset_,
        data.size(),
        "Sentinel encoding prefix exceeds payload bounds");
    const auto payloadSize = data.size() - this->dataOffset_;
    NIMBLE_CHECK_LE(
        2 * sizeof(uint32_t),
        payloadSize,
        "Sentinel encoding is missing its null count or value-stream length");
    const char* position = data.data() + this->dataOffset_;
    const auto numNulls = encoding::readUint32(position);
    NIMBLE_CHECK_LE(
        numNulls, this->rowCount_, "Sentinel null count exceeds row count");
    const auto valuesSize = encoding::readUint32(position);
    const auto remainingSize = static_cast<size_t>(data.end() - position);
    constexpr auto kSentinelSize =
        std::is_same_v<physicalType, std::string_view> ? sizeof(uint32_t)
                                                       : sizeof(physicalType);
    NIMBLE_CHECK_LE(
        kSentinelSize,
        remainingSize,
        "Sentinel encoding is missing its sentinel value");
    NIMBLE_CHECK_LE(
        valuesSize,
        remainingSize - kSentinelSize,
        "Sentinel value encoding exceeds payload bounds");
    values_ = createTypedEncodingView<T>({position, valuesSize}, pool, options);
    position += valuesSize;
    if constexpr (std::is_same_v<physicalType, std::string_view>) {
      const auto sentinelSize = encoding::readUint32(position);
      NIMBLE_CHECK_LE(
          sentinelSize,
          static_cast<size_t>(data.end() - position),
          "Sentinel string exceeds payload bounds");
      sentinel_ = {position, sentinelSize};
      position += sentinelSize;
    } else {
      sentinel_ = encoding::read<physicalType>(position);
    }
    NIMBLE_CHECK_EQ(
        position, data.end(), "Sentinel encoding has trailing data");
    NIMBLE_CHECK_EQ(
        values_->rowCount(),
        this->rowCount_,
        "Sentinel child row count must match its parent");
  }

  uint32_t read(
      std::span<const uint32_t> indices,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final {
    NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
    auto* physicalOutput = static_cast<physicalType*>(output);
    values_->readAt(indices, physicalOutput);
    return normalizeNulls(
        physicalOutput, static_cast<uint32_t>(indices.size()), &setNull);
  }

  uint32_t read(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final {
    const auto numRows = this->checkReadRanges(ranges);
    auto* physicalOutput = static_cast<physicalType*>(output);
    uint32_t outputOffset{0};
    for (const auto& range : ranges) {
      values_->read(
          range.startRow, range.numRows(), physicalOutput + outputOffset);
      outputOffset += range.numRows();
    }
    return normalizeNulls(physicalOutput, numRows, &setNull);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    return detail::castFromPhysicalType<T>(readPhysicalAt(index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    values_->read(offset, length, output);
    normalizeNulls(output, length, nullptr);
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value;
    values_->readAt(index, &value);
    NIMBLE_CHECK_NE(
        value,
        sentinel_,
        "Sentinel EncodingView physical reads require a non-null row");
    return value;
  }

  // Clears reserved values and optionally reports null output positions.
  uint32_t normalizeNulls(
      physicalType* output,
      uint32_t numRows,
      const std::function<void(uint32_t)>* setNull) const {
    uint32_t numNulls{0};
    for (uint32_t row{0}; row < numRows; ++row) {
      if (output[row] == sentinel_) {
        ++numNulls;
        if (setNull != nullptr) {
          (*setNull)(static_cast<uint32_t>(row));
        }
        output[row] = physicalType{};
      }
    }
    return numRows - numNulls;
  }

  // Owns random access to the physical values, including null sentinels.
  std::unique_ptr<TypedEncodingView<T>> values_;
  // Identifies the physical value reserved for null rows.
  physicalType sentinel_{};
};

} // namespace facebook::nimble::detail
