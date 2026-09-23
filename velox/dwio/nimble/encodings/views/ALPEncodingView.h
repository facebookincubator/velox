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
#include <optional>

#include "velox/common/encode/Coding.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class ALPEncodingView final : public TypedEncodingView<T> {
  static_assert(
      isFloatingPointType<T>(),
      "ALPEncodingView only supports float and double types.");

 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  ALPEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::ALP);
    const char* pos = data.data() + this->dataOffset_;
    const auto header = detail::alp::readHeader(pos);
    exponent_ = header.exponent;
    factor_ = header.factor;
    exceptionCount_ = header.hasExceptions ? varint::readVarint32(&pos) : 0;
    const auto encodedValuesSize = varint::readVarint32(&pos);

    encodedValues_ = detail::createTypedEncodingView<uint64_t>(
        {pos, encodedValuesSize}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(encodedValues_);
    pos += encodedValuesSize;

    if (exceptionCount_ == 0) {
      return;
    }

    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    const EncodingFactory encodingFactory{options};

    const auto exceptionPositionsSize = varint::readVarint32(&pos);
    auto exceptionPositionsEncoding = encodingFactory.create(
        *this->pool_, {pos, exceptionPositionsSize}, noStringBufferFactory);
    pos += exceptionPositionsSize;

    const auto exceptionValuesSize = varint::readVarint32(&pos);
    exceptionValues_ = detail::createTypedEncodingView<physicalType>(
        {pos, exceptionValuesSize}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(exceptionValues_);

    exceptionPositionsBuffer_.emplace(
        exceptionCount_, this->pool_, options.bufferPool);
    exceptionPositionsEncoding->materialize(
        exceptionCount_, exceptionPositionsBuffer_->data());
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (const auto exception = findException(index)) {
      return detail::castFromPhysicalType<T>(*exception);
    }

    const auto encoded = velox::ZigZag::decode(encodedValues_->readAt(index));
    return decodeValue(encoded);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (exceptionCount_ == 0) {
      for (uint32_t i = 0; i < length; ++i) {
        output[i] = detail::alp::toPhysical<T>(decodeValue(
            velox::ZigZag::decode(encodedValues_->readAt(offset + i))));
      }
      return;
    }

    const auto* exceptionPositionsBegin = exceptionPositionsBuffer_->data();
    const auto* exceptionPositionsEnd =
        exceptionPositionsBegin + exceptionCount_;
    const auto* exceptionPosition = std::lower_bound(
        exceptionPositionsBegin, exceptionPositionsEnd, offset);
    for (uint32_t i = 0; i < length; ++i) {
      const auto row = offset + i;
      if (exceptionPosition != exceptionPositionsEnd &&
          *exceptionPosition == row) {
        const auto exceptionIndex = exceptionPosition - exceptionPositionsBegin;
        output[i] = exceptionValues_->readAt(exceptionIndex);
        ++exceptionPosition;
      } else {
        output[i] = detail::alp::toPhysical<T>(
            decodeValue(velox::ZigZag::decode(encodedValues_->readAt(row))));
      }
    }
  }

  std::optional<physicalType> findException(uint32_t index) const {
    if (exceptionCount_ == 0) {
      return std::nullopt;
    }

    const auto* begin = exceptionPositionsBuffer_->data();
    const auto* end = begin + exceptionCount_;
    const auto* it = std::lower_bound(begin, end, index);
    if (it == end || *it != index) {
      return std::nullopt;
    }
    return exceptionValues_->readAt(it - begin);
  }

  T decodeValue(int64_t encoded) const {
    return static_cast<T>(
        static_cast<double>(encoded) * ALPEncoding<T>::kPow10Double[factor_] /
        ALPEncoding<T>::kPow10Double[exponent_]);
  }

  uint8_t exponent_{0};
  uint8_t factor_{0};
  uint32_t exceptionCount_{0};
  std::unique_ptr<TypedEncodingView<uint64_t>> encodedValues_;
  std::unique_ptr<TypedEncodingView<physicalType>> exceptionValues_;
  std::optional<ScopedVector<uint32_t>> exceptionPositionsBuffer_;
};

} // namespace facebook::nimble
