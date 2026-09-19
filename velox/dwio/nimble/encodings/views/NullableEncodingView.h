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
#include <vector>

#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::detail {

/// Maps logical rows to the compact non-null child encoding.
template <typename T>
class NullableEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;
  using TypedEncodingView<T>::read;

  NullableEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        isNonNull_{this->rowCount_, pool, options.bufferPool},
        nonNullOffsets_{0, pool, options.bufferPool} {
    NIMBLE_CHECK_LT(
        this->rowCount_,
        std::numeric_limits<uint32_t>::max(),
        "Nullable row count exceeds offset capacity");
    nonNullOffsets_.resize(static_cast<uint64_t>(this->rowCount_) + 1);
    NIMBLE_CHECK_LE(
        this->dataOffset_,
        data.size(),
        "Nullable encoding prefix exceeds payload bounds");
    const char* position = data.data() + this->dataOffset_;
    NIMBLE_CHECK_LE(
        sizeof(uint32_t),
        static_cast<size_t>(data.end() - position),
        "Nullable encoding is missing its value-stream length");
    const auto nonNullBytes = encoding::readUint32(position);
    NIMBLE_CHECK_LE(
        nonNullBytes,
        static_cast<size_t>(data.end() - position),
        "Nullable value encoding exceeds payload bounds");
    const std::string_view nonNullValues{position, nonNullBytes};
    position += nonNullBytes;
    const std::string_view nulls{
        position, static_cast<size_t>(data.end() - position)};

    nonNullValues_ =
        createTypedEncodingView<T>(nonNullValues, this->pool_, this->options_);
    auto nullsView =
        createTypedEncodingView<bool>(nulls, this->pool_, this->options_);
    NIMBLE_CHECK_EQ(
        nullsView->rowCount(),
        this->rowCount_,
        "Nullable null stream row count must match its parent");
    nullsView->read(0, this->rowCount_, isNonNull_.data());

    nonNullOffsets_[0] = 0;
    for (uint32_t row{0}; row < this->rowCount_; ++row) {
      nonNullOffsets_[row + 1] = nonNullOffsets_[row] + isNonNull_[row];
    }
    NIMBLE_CHECK_EQ(
        nonNullValues_->rowCount(),
        nonNullOffsets_[this->rowCount_],
        "Nullable value stream row count must match its null stream");
  }

  uint32_t read(
      std::span<const uint32_t> indices,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final {
    NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
    const auto numIndices = static_cast<uint32_t>(indices.size());
    ScopedVector<uint32_t> nonNullIndices{
        0, this->pool_, this->options_.bufferPool};
    ScopedVector<uint32_t> outputIndices{
        0, this->pool_, this->options_.bufferPool};
    nonNullIndices.reserve(indices.size());
    outputIndices.reserve(indices.size());

    auto* typedOutput = static_cast<physicalType*>(output);
    for (uint32_t outputIndex{0}; outputIndex < numIndices; ++outputIndex) {
      const auto sourceIndex = indices[outputIndex];
      this->checkReadRange(sourceIndex, 1);
      if (isNonNull_[sourceIndex]) {
        nonNullIndices.push_back(nonNullOffsets_[sourceIndex]);
        outputIndices.push_back(outputIndex);
      } else {
        typedOutput[outputIndex] = physicalType{};
        setNull(outputIndex);
      }
    }

    if (nonNullIndices.empty()) {
      return 0;
    }
    if (nonNullIndices.size() == numIndices) {
      nonNullValues_->readAt(
          std::span<const uint32_t>{
              nonNullIndices.data(), nonNullIndices.size()},
          typedOutput);
      return static_cast<uint32_t>(nonNullIndices.size());
    }

    ScopedVector<physicalType> values{
        0, this->pool_, this->options_.bufferPool};
    values.resize(nonNullIndices.size());
    nonNullValues_->readAt(
        std::span<const uint32_t>{nonNullIndices.data(), nonNullIndices.size()},
        values.data());
    for (size_t i{0}; i < values.size(); ++i) {
      typedOutput[outputIndices[i]] = values[i];
    }
    return static_cast<uint32_t>(nonNullIndices.size());
  }

  uint32_t read(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final {
    const auto numRows = this->checkReadRanges(ranges);
    auto* typedOutput = static_cast<physicalType*>(output);
    std::vector<RowRange> nonNullRanges;
    std::vector<uint32_t> outputIndices;
    nonNullRanges.reserve(numRows);
    outputIndices.reserve(numRows);

    uint32_t outputIndex{0};
    for (const auto& range : ranges) {
      auto sourceRow = range.startRow;
      while (sourceRow < range.endRow) {
        if (!isNonNull_[sourceRow]) {
          typedOutput[outputIndex] = physicalType{};
          setNull(outputIndex);
          ++sourceRow;
          ++outputIndex;
          continue;
        }

        const auto nonNullStart = nonNullOffsets_[sourceRow];
        do {
          outputIndices.push_back(outputIndex);
          ++sourceRow;
          ++outputIndex;
        } while (sourceRow < range.endRow && isNonNull_[sourceRow]);
        nonNullRanges.emplace_back(nonNullStart, nonNullOffsets_[sourceRow]);
      }
    }

    if (nonNullRanges.empty()) {
      return 0;
    }
    if (outputIndices.size() == numRows) {
      nonNullValues_->read(nonNullRanges, setNull, typedOutput);
      return numRows;
    }

    ScopedVector<physicalType> values{
        0, this->pool_, this->options_.bufferPool};
    values.resize(outputIndices.size());
    nonNullValues_->read(nonNullRanges, setNull, values.data());
    for (size_t i{0}; i < values.size(); ++i) {
      typedOutput[outputIndices[i]] = values[i];
    }
    return static_cast<uint32_t>(outputIndices.size());
  }

 protected:
  void readPhysical(uint32_t offset, uint32_t length, physicalType* /*output*/)
      const final {
    NIMBLE_UNSUPPORTED(
        "Nullable EncodingView requires a null-aware read: offset={}, length={}",
        offset,
        length);
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_UNSUPPORTED(
        "Nullable EncodingView requires a null-aware read: index={}", index);
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_UNSUPPORTED(
        "Nullable EncodingView requires a null-aware read: index={}", index);
  }

 private:
  std::unique_ptr<TypedEncodingView<T>> nonNullValues_;
  ScopedVector<bool> isNonNull_;
  ScopedVector<uint32_t> nonNullOffsets_;
};

} // namespace facebook::nimble::detail
