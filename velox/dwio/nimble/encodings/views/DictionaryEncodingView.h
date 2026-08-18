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

#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class DictionaryEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  DictionaryEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Dictionary);
    const char* pos = data.data() + this->dataOffset_;
    const auto alphabetSize = encoding::readUint32(pos);
    alphabet_ = detail::createTypedEncodingView<T>(
        {pos, alphabetSize}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(alphabet_);
    pos += alphabetSize;
    indices_ = detail::createTypedEncodingView<uint32_t>(
        {pos, static_cast<size_t>(data.end() - pos)}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(indices_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return alphabet_->readAt(indices_->readAt(index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    auto indices = this->template getVectorBuffer<uint32_t>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(indices);
    };
    indices.resize(length);
    indices_->read(offset, length, indices.data());
    for (uint32_t i = 0; i < length; ++i) {
      alphabet_->readAt(indices[i], output + i);
    }
  }

  std::unique_ptr<TypedEncodingView<T>> alphabet_;
  std::unique_ptr<TypedEncodingView<uint32_t>> indices_;
};

} // namespace facebook::nimble
