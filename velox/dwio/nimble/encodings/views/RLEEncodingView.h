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
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class RLEEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  RLEEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        runEnds_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::RLE);
    const char* pos = data.data() + this->dataOffset_;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto runLengths = EncodingFactory().create(
        *this->pool_, {pos, runLengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);
    runEnds_.resize(runLengths->rowCount());
    runLengths->materialize(runLengths->rowCount(), runEnds_.data());
    uint32_t end = 0;
    for (auto& runEnd : runEnds_) {
      end += runEnd;
      runEnd = end;
    }
    NIMBLE_CHECK_EQ(end, this->rowCount_);

    pos += runLengthsSize;
    values_ = detail::createTypedEncodingView<T>(
        {pos, static_cast<size_t>(data.end() - pos)}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(values_);
  }

  ~RLEEncodingView() override {
    this->releaseVectorBuffer(runEnds_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
    NIMBLE_CHECK(it != runEnds_.end());
    return values_->readAt(static_cast<uint32_t>(it - runEnds_.begin()));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), offset);
    NIMBLE_CHECK(it != runEnds_.end());
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
      const auto runEnd = *it;
      const auto count = std::min(length - outputOffset, runEnd - offset);
      physicalType value;
      values_->readAt(runIndex, &value);
      std::fill(output + outputOffset, output + outputOffset + count, value);
      outputOffset += count;
      offset += count;
      ++it;
    }
  }

  Vector<uint32_t> runEnds_;
  std::unique_ptr<TypedEncodingView<T>> values_;
};

template <>
class RLEEncodingView<bool> final : public TypedEncodingView<bool> {
 public:
  RLEEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<bool>{data, pool, options},
        runEnds_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::RLE);
    const char* pos = data.data() + this->dataOffset_;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto runLengths = EncodingFactory().create(
        *this->pool_, {pos, runLengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);
    runEnds_.resize(runLengths->rowCount());
    runLengths->materialize(runLengths->rowCount(), runEnds_.data());
    uint32_t end = 0;
    for (auto& runEnd : runEnds_) {
      end += runEnd;
      runEnd = end;
    }
    NIMBLE_CHECK_EQ(end, this->rowCount_);

    pos += runLengthsSize;
    NIMBLE_CHECK_EQ(pos + sizeof(bool), data.end());
    initialValue_ = *reinterpret_cast<const bool*>(pos);
  }

  ~RLEEncodingView() override {
    this->releaseVectorBuffer(runEnds_);
  }

 private:
  bool readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
    NIMBLE_CHECK(it != runEnds_.end());
    const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
    return runIndex % 2 == 0 ? initialValue_ : !initialValue_;
  }

  void readPhysical(uint32_t offset, uint32_t length, bool* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), offset);
    NIMBLE_CHECK(it != runEnds_.end());
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
      const auto runEnd = *it;
      const auto count = std::min(length - outputOffset, runEnd - offset);
      std::fill(
          output + outputOffset,
          output + outputOffset + count,
          runIndex % 2 == 0 ? initialValue_ : !initialValue_);
      outputOffset += count;
      offset += count;
      ++it;
    }
  }

  Vector<uint32_t> runEnds_;
  bool initialValue_;
};

} // namespace facebook::nimble
