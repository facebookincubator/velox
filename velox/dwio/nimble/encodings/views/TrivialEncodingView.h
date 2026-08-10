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
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class TrivialEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const auto compressionType =
        static_cast<CompressionType>(data[this->dataOffset_]);
    NIMBLE_CHECK_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed Trivial streams.");
    values_ = reinterpret_cast<const physicalType*>(
        data.data() + this->dataOffset_ + sizeof(uint8_t));
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return detail::castFromPhysicalType<T>(values_[index]);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    std::copy(values_ + offset, values_ + offset + length, output);
  }

  const physicalType* values_;
};

template <>
class TrivialEncodingView<bool> final : public TypedEncodingView<bool> {
 public:
  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<bool>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const char* pos = data.data() + this->dataOffset_;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed Trivial streams.");
    bitmap_ = pos;
  }

 private:
  bool readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return velox::bits::isBitSet(
        reinterpret_cast<const uint8_t*>(bitmap_), index);
  }

  void readPhysical(uint32_t offset, uint32_t length, bool* output)
      const final {
    this->checkReadRange(offset, length);
    const auto* bitmap = reinterpret_cast<const uint8_t*>(bitmap_);
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = velox::bits::isBitSet(bitmap, offset + i);
    }
  }

  const char* bitmap_{nullptr};
};

template <>
class TrivialEncodingView<std::string_view> final
    : public TypedEncodingView<std::string_view> {
 public:
  TrivialEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<std::string_view>{data, pool, options},
        offsets_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Trivial);
    const char* pos = data.data() + this->dataOffset_;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed Trivial streams.");
    const auto lengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto lengths = EncodingFactory(options).create(
        *this->pool_, {pos, lengthsSize}, noStringBufferFactory);
    NIMBLE_CHECK_NOT_NULL(lengths);

    offsets_.resize(this->rowCount_ + 1);
    offsets_[0] = 0;
    lengths->materialize(this->rowCount_, offsets_.data() + 1);
    for (uint32_t i = 0; i < this->rowCount_; ++i) {
      offsets_[i + 1] += offsets_[i];
    }
    blob_ = pos + lengthsSize;
  }

  ~TrivialEncodingView() override {
    this->releaseVectorBuffer(offsets_);
  }

 private:
  std::string_view readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return {blob_ + offsets_[index], offsets_[index + 1] - offsets_[index]};
  }

  void readPhysical(uint32_t offset, uint32_t length, std::string_view* output)
      const final {
    this->checkReadRange(offset, length);
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = {
          blob_ + offsets_[offset + i],
          offsets_[offset + i + 1] - offsets_[offset + i]};
    }
  }

  const char* blob_{nullptr};
  Vector<uint32_t> offsets_;
};

} // namespace facebook::nimble
