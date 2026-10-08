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

#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class SimdForBitpackEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  SimdForBitpackEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::SimdForBitpack);
    header_ = SimdForBitpackEncoding<T>::parseHeader(data, options);
    NIMBLE_CHECK_EQ(header_.rowCount, this->rowCount_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (header_.bitWidth == 0) {
      return detail::castFromPhysicalType<T>(header_.baseline);
    }

    const auto position = SimdForBitpackEncoding<T>::groupInfo(header_, index);
    std::array<physicalType, kGroupSize> residuals{};
    unpackGroup(position.groupIndex, residuals.data());
    return detail::castFromPhysicalType<T>(static_cast<physicalType>(
        residuals[position.rowOffset] + header_.baseline));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (header_.bitWidth == 0) {
      std::fill(output, output + length, header_.baseline);
      return;
    }

    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto position =
          SimdForBitpackEncoding<T>::groupInfo(header_, offset);
      const auto count = std::min<uint32_t>(
          length - outputOffset, position.rowCount - position.rowOffset);
      std::array<physicalType, kGroupSize> residuals{};
      unpackGroup(position.groupIndex, residuals.data());
      for (uint32_t i = 0; i < count; ++i) {
        output[outputOffset + i] = static_cast<physicalType>(
            residuals[position.rowOffset + i] + header_.baseline);
      }
      outputOffset += count;
      offset += count;
    }
  }

  static constexpr uint32_t kGroupSize = SimdForBitpackEncoding<T>::kGroupSize;

  void unpackGroup(uint32_t groupIndex, physicalType* output) const {
    static_assert(
        isOneByteIntegralType<physicalType>() ||
            isTwoByteIntegralType<physicalType>() ||
            isFourByteIntegralType<physicalType>() ||
            isEightByteIntegralType<physicalType>(),
        "Unexpected SimdForBitpack physical type width.");
    const auto* groupStart =
        reinterpret_cast<const uint32_t*>(header_.packedData) +
        static_cast<uint64_t>(groupIndex) * header_.bitWidth;
    if constexpr (isEightByteIntegralType<physicalType>()) {
      facebook::velox::fastpforlib::fastunpack(
          groupStart, reinterpret_cast<uint64_t*>(output), header_.bitWidth);
    } else if constexpr (isFourByteIntegralType<physicalType>()) {
      facebook::velox::fastpforlib::fastunpack(
          groupStart, reinterpret_cast<uint32_t*>(output), header_.bitWidth);
    } else {
      std::array<uint32_t, kGroupSize> temp{};
      facebook::velox::fastpforlib::fastunpack(
          groupStart, temp.data(), header_.bitWidth);
      for (uint32_t i = 0; i < kGroupSize; ++i) {
        output[i] = static_cast<physicalType>(temp[i]);
      }
    }
  }

  typename SimdForBitpackEncoding<T>::Header header_;
};

} // namespace facebook::nimble
