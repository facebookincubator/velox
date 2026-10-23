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

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

/// Reads a Varint stream through the default Varint decoder, which only reads
/// forward. With Encoding::Options::cacheEncodingView, the view decodes every
/// value at construction. Otherwise each read decodes up to its row with a new
/// decoder, which skips the preceding values without decoding them.
template <typename T>
class VarintEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  VarintEncodingView(const VarintEncodingView&) = delete;
  VarintEncodingView& operator=(const VarintEncodingView&) = delete;
  VarintEncodingView(VarintEncodingView&&) = delete;
  VarintEncodingView& operator=(VarintEncodingView&&) = delete;

  VarintEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        values_{this->template getVectorBuffer<physicalType>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Varint);
    if (!this->options_.cacheEncodingView) {
      return;
    }
    auto encoding = createDecoder();
    values_.resize(this->rowCount_);
    encoding->materialize(this->rowCount_, values_.data());
  }

  ~VarintEncodingView() override {
    this->releaseVectorBuffer(values_);
  }

 private:
  std::unique_ptr<Encoding> createDecoder() const {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto encoding = EncodingFactory().create(
        *this->pool_, this->data_, noStringBufferFactory, this->options_);
    NIMBLE_CHECK_NOT_NULL(encoding);
    return encoding;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (!this->options_.cacheEncodingView) {
      physicalType value;
      readPhysical(index, 1, &value);
      return detail::castFromPhysicalType<T>(value);
    }
    return detail::castFromPhysicalType<T>(values_[index]);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (!this->options_.cacheEncodingView) {
      auto encoding = createDecoder();
      encoding->skip(offset);
      encoding->materialize(length, output);
      return;
    }
    std::copy(
        values_.data() + offset, values_.data() + offset + length, output);
  }

  // Holds every decoded value. Empty without cacheEncodingView.
  Vector<physicalType> values_;
};

} // namespace facebook::nimble
