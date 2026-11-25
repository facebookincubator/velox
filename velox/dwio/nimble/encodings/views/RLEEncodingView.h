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
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class RLEEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;
  using TypedEncodingView<T>::read;

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
    values_ = detail::createTypedEncodingView<runValueType>(
        {pos, static_cast<size_t>(data.data() + data.size() - pos)},
        this->pool_,
        options);
    NIMBLE_CHECK_NOT_NULL(values_);
  }

  ~RLEEncodingView() override {
    this->releaseVectorBuffer(runEnds_);
  }

  /// Reads the value of each run of string rows once, with one read of the
  /// run values that takes `addStringBuffer`, so that it passes the buffers
  /// of the strings. Reads other values as the base view does.
  uint32_t read(
      std::span<const uint32_t> indices,
      const std::function<void(uint32_t)>& setNull,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const final {
    if constexpr (std::is_same_v<physicalType, std::string_view>) {
      if (addStringBuffer != nullptr && !indices.empty()) {
        NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
        this->checkReadIndices(indices);
        ScopedVector<uint32_t> runIndices{
            0, this->pool_, this->options_.bufferPool};
        ScopedVector<uint32_t> runRows{
            0, this->pool_, this->options_.bufferPool};
        runIndices.reserve(indices.size());
        runRows.reserve(indices.size());
        for (const auto index : indices) {
          const auto it =
              std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
          addRun(
              static_cast<uint32_t>(it - runEnds_.begin()),
              /*numRows=*/1,
              runIndices,
              runRows);
        }
        readRuns(
            runIndices,
            runRows,
            addStringBuffer,
            static_cast<std::string_view*>(output));
        return static_cast<uint32_t>(indices.size());
      }
    }
    return EncodingView::read(indices, setNull, addStringBuffer, output);
  }

  /// Does the same as the read of indices above for ordered, disjoint ranges.
  uint32_t read(
      std::span<const RowRange> ranges,
      const std::function<void(uint32_t)>& setNull,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const final {
    if constexpr (std::is_same_v<physicalType, std::string_view>) {
      if (addStringBuffer != nullptr) {
        const auto numRows = this->checkReadRanges(ranges);
        ScopedVector<uint32_t> runIndices{
            0, this->pool_, this->options_.bufferPool};
        ScopedVector<uint32_t> runRows{
            0, this->pool_, this->options_.bufferPool};
        const auto maxRuns = std::min<uint64_t>(numRows, runEnds_.size());
        runIndices.reserve(maxRuns);
        runRows.reserve(maxRuns);
        for (const auto& range : ranges) {
          auto it = std::upper_bound(
              runEnds_.begin(), runEnds_.end(), range.startRow);
          for (auto row = range.startRow; row < range.endRow; ++it) {
            const auto next = std::min(*it, range.endRow);
            addRun(
                static_cast<uint32_t>(it - runEnds_.begin()),
                next - row,
                runIndices,
                runRows);
            row = next;
          }
        }
        readRuns(
            runIndices,
            runRows,
            addStringBuffer,
            static_cast<std::string_view*>(output));
        return numRows;
      }
    }
    return TypedEncodingView<T>::read(ranges, setNull, addStringBuffer, output);
  }

 private:
  // Adds 'numRows' rows of the run 'runIndex' to the runs that a read of
  // strings reads, in the order of its output rows. Extends the last run when
  // it is the same, so that each run of consecutive rows is read once.
  static void addRun(
      uint32_t runIndex,
      uint32_t numRows,
      ScopedVector<uint32_t>& runIndices,
      ScopedVector<uint32_t>& runRows) {
    if (!runIndices.empty() && runIndices[runIndices.size() - 1] == runIndex) {
      runRows[runRows.size() - 1] += numRows;
      return;
    }
    runIndices.push_back(runIndex);
    runRows.push_back(numRows);
  }

  // Reads the value of each run in 'runIndices' with one read of 'values_'
  // that passes the buffers of the strings to 'addStringBuffer', and writes
  // it to the next 'runRows' rows of 'output'.
  void readRuns(
      const ScopedVector<uint32_t>& runIndices,
      const ScopedVector<uint32_t>& runRows,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      std::string_view* output) const {
    if (runIndices.empty()) {
      return;
    }
    ScopedVector<std::string_view> runValues{
        runIndices.size(), this->pool_, this->options_.bufferPool};
    values_->read(
        std::span<const uint32_t>{runIndices.data(), runIndices.size()},
        /*setNull=*/[](uint32_t) {},
        addStringBuffer,
        runValues.data());
    uint32_t outputOffset{0};
    for (size_t i{0}; i < runIndices.size(); ++i) {
      std::fill(
          output + outputOffset,
          output + outputOffset + runRows[i],
          runValues[i]);
      outputOffset += runRows[i];
    }
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
    NIMBLE_CHECK(it != runEnds_.end());
    physicalType value;
    values_->readAt(static_cast<uint32_t>(it - runEnds_.begin()), &value);
    return detail::castFromPhysicalType<T>(value);
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

  // RLE serializes floating run values with their logical type and all other
  // run values with their physical type.
  using runValueType =
      std::conditional_t<isFloatingPointType<T>(), T, physicalType>;

  Vector<uint32_t> runEnds_;
  std::unique_ptr<TypedEncodingView<runValueType>> values_;
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
    NIMBLE_CHECK_EQ(pos + sizeof(bool), data.data() + data.size());
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
