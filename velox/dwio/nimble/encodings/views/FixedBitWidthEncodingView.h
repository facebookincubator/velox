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
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class FixedBitWidthEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  FixedBitWidthEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::FixedBitWidth);
    const char* pos = data.data() + this->dataOffset_;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed FixedBitWidth streams.");
    baseline_ = encoding::read<physicalType>(pos);
    bitWidth_ = static_cast<uint32_t>(encoding::readChar(pos));
    fixedBitArray_ = FixedBitArray{
        {pos, static_cast<size_t>(data.end() - pos)},
        static_cast<int>(bitWidth_)};
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto value =
        static_cast<physicalType>(fixedBitArray_.get(index) + baseline_);
    return detail::castFromPhysicalType<T>(value);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (bitWidth_ == 0) {
      std::fill(output, output + length, baseline_);
      return;
    }
    fixedBitArray_.bulkGetWithBaseline(offset, length, output, baseline_);
  }

  physicalType baseline_;
  uint32_t bitWidth_;
  FixedBitArray fixedBitArray_{};
};

} // namespace facebook::nimble
