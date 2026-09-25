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

#include "velox/common/memory/RawVector.h"
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
    baseline_ = encoding::read<physicalType>(pos);
    bitWidth_ = static_cast<uint32_t>(encoding::readChar(pos));
    // Match FixedBitWidthEncoding: the packed payload has no scalar data type.
    const auto payload = this->decompressPayload(
        compressionType,
        DataType::Undefined,
        {pos, static_cast<size_t>(data.data() + data.size() - pos)});
    fixedBitArray_ = FixedBitArray{payload, static_cast<int>(bitWidth_)};
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

  // Raw values already sit in [0, 2^bitWidth_), so a direct-mapped table can
  // translate value to id without hashing. Declined past
  // kMaxDirectTableBitWidth, since a section needing that many bits was not
  // chosen for having few distinct values.
  bool denseRunIds(
      uint32_t offset,
      uint32_t length,
      std::vector<uint32_t>& ids,
      std::vector<uint64_t>& table) const final {
    static constexpr uint32_t kMaxDirectTableBitWidth = 20; // 1M entries.
    if (bitWidth_ > kMaxDirectTableBitWidth) {
      return false;
    }
    this->checkReadRange(offset, length);

    // Raw, pre-baseline values; every element is overwritten below before
    // being read, so the uninitialised allocation is safe.
    velox::raw_vector<physicalType> raw(length);
    if (bitWidth_ == 0) {
      std::fill(raw.begin(), raw.end(), physicalType{0});
    } else {
      fixedBitArray_.bulkGetWithBaseline(
          offset, length, raw.data(), physicalType{0});
    }

    constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
    const uint32_t alphabetSize = uint32_t{1} << bitWidth_;
    // kUnassigned is a sentinel tested for below, not just initial padding.
    std::vector<uint32_t> valueToId(alphabetSize, kUnassigned);
    ids.clear();
    ids.reserve(length);
    table.clear();
    for (uint32_t i = 0; i < length; ++i) {
      const auto rawValue = static_cast<uint32_t>(raw[i]);
      auto& id = valueToId[rawValue];
      if (id == kUnassigned) {
        id = static_cast<uint32_t>(table.size());
        uint64_t bits = 0;
        const physicalType value =
            static_cast<physicalType>(rawValue) + baseline_;
        __builtin_memcpy(&bits, &value, sizeof(physicalType));
        table.push_back(bits);
      }
      ids.push_back(id);
    }
    return true;
  }

  physicalType baseline_;
  uint32_t bitWidth_;
  FixedBitArray fixedBitArray_{};
};

} // namespace facebook::nimble
