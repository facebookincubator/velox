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

#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class PFOREncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  PFOREncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        exceptionPositions_{this->template getVectorBuffer<uint32_t>()},
        exceptionValues_{this->template getVectorBuffer<physicalType>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::PFOR);
    const char* pos = data.data() + this->dataOffset_;
    baseline_ = encoding::read<physicalType>(pos);
    baseBitWidth_ = static_cast<uint8_t>(encoding::readChar(pos));
    NIMBLE_CHECK_LE(baseBitWidth_, sizeof(physicalType) * 8);
    numExceptions_ = varint::readVarint32(&pos);
    NIMBLE_CHECK_LE(numExceptions_, this->rowCount_);

    const auto exceptionPositionsSize = varint::readVarint32(&pos);
    if (numExceptions_ == 0) {
      NIMBLE_CHECK_EQ(
          exceptionPositionsSize,
          0,
          "Empty Pfor exception positions stream has data.");
    } else {
      auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
      auto exceptionPositions = EncodingFactory().create(
          *this->pool_,
          {pos, exceptionPositionsSize},
          noStringBufferFactory,
          options);
      NIMBLE_CHECK_NOT_NULL(exceptionPositions);
      exceptionPositions_.resize(numExceptions_);
      exceptionPositions->materialize(
          numExceptions_, exceptionPositions_.data());
    }
    pos += exceptionPositionsSize;

    const auto exceptionValuesSize = varint::readVarint32(&pos);
    if (numExceptions_ == 0) {
      NIMBLE_CHECK_EQ(
          exceptionValuesSize,
          0,
          "Empty Pfor exception values stream has data.");
    } else {
      auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
      auto exceptionValues = EncodingFactory().create(
          *this->pool_,
          {pos, exceptionValuesSize},
          noStringBufferFactory,
          options);
      NIMBLE_CHECK_NOT_NULL(exceptionValues);
      exceptionValues_.resize(numExceptions_);
      exceptionValues->materialize(numExceptions_, exceptionValues_.data());
    }
    pos += exceptionValuesSize;

    if (baseBitWidth_ > 0) {
      fixedBitArray_ = FixedBitArray{
          {pos, FixedBitArray::bufferSize(this->rowCount_, baseBitWidth_)},
          static_cast<int>(baseBitWidth_)};
    }
  }

  ~PFOREncodingView() override {
    this->releaseVectorBuffer(exceptionValues_);
    this->releaseVectorBuffer(exceptionPositions_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    auto residual = baseBitWidth_ == 0
        ? physicalType{0}
        : static_cast<physicalType>(fixedBitArray_.get(index));
    const auto exceptionIt = std::lower_bound(
        exceptionPositions_.begin(), exceptionPositions_.end(), index);
    if (exceptionIt != exceptionPositions_.end() && *exceptionIt == index) {
      residual = exceptionValues_[exceptionIt - exceptionPositions_.begin()];
    }
    return detail::castFromPhysicalType<T>(
        static_cast<physicalType>(baseline_ + residual));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (baseBitWidth_ == 0) {
      std::fill(output, output + length, baseline_);
    } else {
      fixedBitArray_.bulkGetWithBaseline(offset, length, output, baseline_);
    }

    auto exceptionIt = std::lower_bound(
        exceptionPositions_.begin(), exceptionPositions_.end(), offset);
    const auto end = offset + length;
    while (exceptionIt != exceptionPositions_.end() && *exceptionIt < end) {
      output[*exceptionIt - offset] = static_cast<physicalType>(
          baseline_ +
          exceptionValues_[exceptionIt - exceptionPositions_.begin()]);
      ++exceptionIt;
    }
  }

  physicalType baseline_{};
  uint8_t baseBitWidth_{0};
  uint32_t numExceptions_{0};
  Vector<uint32_t> exceptionPositions_;
  Vector<physicalType> exceptionValues_;
  FixedBitArray fixedBitArray_{};
};

} // namespace facebook::nimble
