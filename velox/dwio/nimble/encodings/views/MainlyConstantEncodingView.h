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
#include <bit>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <folly/CPortability.h>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// Reads a MainlyConstant stream. With Encoding::Options::cacheEncodingView,
/// the view builds a bitmap of the common rows and a rank per 64-row word at
/// construction. Otherwise it builds neither and reads the stream of common
/// flags for each read.
template <typename T>
class MainlyConstantEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;
  using TypedEncodingView<T>::read;

  MainlyConstantEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        isCommonBits_{this->template getVectorBuffer<uint64_t>()},
        otherValueRanks_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::MainlyConstant);
    const char* pos = data.data() + this->dataOffset_;
    const auto isCommon =
        this->readSizedBytes(pos, "MainlyConstant common flags");
    otherValues_ = detail::createTypedEncodingView<T>(
        this->readSizedBytes(pos, "MainlyConstant other values"),
        this->pool_,
        options);
    NIMBLE_CHECK_NOT_NULL(otherValues_);

    if constexpr (std::is_same_v<physicalType, std::string_view>) {
      NIMBLE_CHECK_FILE_GE(
          this->remainingBytes(pos),
          sizeof(uint32_t),
          "Truncated MainlyConstant encoding.");
      NIMBLE_CHECK_FILE_EQ(
          this->remainingBytes(pos) - sizeof(uint32_t),
          encoding::peek<uint32_t>(pos),
          "Unexpected MainlyConstant view end.");
    } else {
      NIMBLE_CHECK_FILE_EQ(
          this->remainingBytes(pos),
          sizeof(physicalType),
          "Unexpected MainlyConstant view end.");
    }
    commonValue_ = encoding::read<physicalType>(pos);

    if (!options.cacheEncodingView) {
      isCommonView_ =
          detail::createTypedEncodingView<bool>(isCommon, this->pool_, options);
      NIMBLE_CHECK_NOT_NULL(isCommonView_);
      return;
    }
    buildLookupStructures(isCommon, options);
  }

  ~MainlyConstantEncodingView() override {
    this->releaseVectorBuffer(otherValueRanks_);
    this->releaseVectorBuffer(isCommonBits_);
  }

  /// Reads the other values of string rows with one read of the other values
  /// that takes `addStringBuffer`, so that it passes the buffers of their
  /// strings, and passes the input string buffer for the common value, which
  /// points into the input, or a copy of it when the view has no input string
  /// buffer. Reads other values as the base view does.
  uint32_t read(
      std::span<const uint32_t> indices,
      const std::function<void(uint32_t)>& setNull,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const final {
    if constexpr (std::is_same_v<physicalType, std::string_view>) {
      if (addStringBuffer != nullptr) {
        return readStrings(indices, addStringBuffer, output);
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
        return readStrings(ranges, addStringBuffer, output);
      }
    }
    return TypedEncodingView<T>::read(ranges, setNull, addStringBuffer, output);
  }

 private:
  void buildLookupStructures(
      std::string_view isCommon,
      const Encoding::Options& options) {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto isCommonEncoding = EncodingFactory{options}.create(
        *this->pool_, isCommon, noStringBufferFactory);
    NIMBLE_CHECK_NOT_NULL(isCommonEncoding);
    const auto numWords = velox::bits::nwords(this->rowCount_);
    isCommonBits_.resize(numWords);
    if (this->rowCount_ > 0) {
      isCommonEncoding->materializeBoolsAsBits(
          this->rowCount_, isCommonBits_.data(), /*begin=*/0);
      const auto numTailBits = this->rowCount_ & 63;
      if (numTailBits != 0) {
        isCommonBits_[numWords - 1] |= ~((1ULL << numTailBits) - 1);
      }
    }

    otherValueRanks_.resize(numWords);
    uint32_t numOtherValues{0};
    for (uint32_t word = 0; word < numWords; ++word) {
      otherValueRanks_[word] = numOtherValues;
      numOtherValues += std::popcount(~isCommonBits_[word]);
    }
    NIMBLE_CHECK_FILE_EQ(
        numOtherValues,
        otherValues_->rowCount(),
        "MainlyConstant other values do not match the common flags.");
  }

  // Returns whether the view holds the bitmap of the common rows and the rank
  // of every word.
  bool cachesLookupStructures() const {
    return this->options_.cacheEncodingView;
  }

  bool isCommon(uint32_t index) const {
    return velox::bits::isBitSet(isCommonBits_.data(), index);
  }

  // Returns the position of the row in 'otherValues_', which is the number of
  // non-common rows before it.
  uint32_t otherValueIndex(uint32_t index) const {
    const auto word = index >> 6;
    const auto precedingBits = (1ULL << (index & 63)) - 1;
    return otherValueRanks_[word] +
        std::popcount(~isCommonBits_[word] & precedingBits);
  }

  // Returns the first row in [row, end) whose isCommon differs from 'common',
  // or 'end' if all of them match.
  uint32_t runEnd(uint32_t row, uint32_t end, bool common) const {
    while (row < end) {
      const auto word = row >> 6;
      const auto differing =
          (common ? ~isCommonBits_[word] : isCommonBits_[word]) >> (row & 63);
      if (differing != 0) {
        return std::min<uint32_t>(row + std::countr_zero(differing), end);
      }
      row = (word + 1) << 6;
    }
    return end;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (cachesLookupStructures()) {
      if (isCommon(index)) {
        return detail::castFromPhysicalType<T>(commonValue_);
      }
      return otherValues_->readAt(otherValueIndex(index));
    }
    if (isCommonView_->readAt(index)) {
      return detail::castFromPhysicalType<T>(commonValue_);
    }
    return otherValues_->readAt(isCommonView_->count(index, /*value=*/false));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    forEachRun(
        offset,
        length,
        [&](uint32_t row, uint32_t count) {
          std::fill(output + row, output + row + count, commonValue_);
        },
        [&](uint32_t otherValue, uint32_t row, uint32_t count) {
          otherValues_->read(otherValue, count, output + row);
        });
  }

  // Calls onCommon(row, count) for each run of common rows among the 'length'
  // rows from 'offset', and onOther(otherValue, row, count) for each run of
  // other rows, whose first value is at 'otherValue' in 'otherValues_'. 'row'
  // counts from 'offset'. Without the lookup structures, reads the flags of
  // the rows alone and counts the non-common rows before them once, instead
  // of consulting structures over every row. Always inlined, so that
  // readPhysical() fills the runs in place, as a loop of its own would.
  template <typename OnCommon, typename OnOther>
  FOLLY_ALWAYS_INLINE void forEachRun(
      uint32_t offset,
      uint32_t length,
      const OnCommon& onCommon,
      const OnOther& onOther) const {
    if (length == 0) {
      return;
    }
    if (!cachesLookupStructures()) {
      ScopedVector<bool> isCommon{
          length, this->pool_, this->options_.bufferPool};
      isCommonView_->read(offset, length, isCommon.data());
      auto nextOtherValue = isCommonView_->count(offset, /*value=*/false);
      uint32_t row{0};
      while (row < length) {
        const bool common = isCommon[row];
        uint32_t next{row + 1};
        while (next < length && isCommon[next] == common) {
          ++next;
        }
        if (common) {
          onCommon(row, next - row);
        } else {
          onOther(nextOtherValue, row, next - row);
          nextOtherValue += next - row;
        }
        row = next;
      }
      return;
    }
    const auto end = offset + length;
    auto nextOtherValue = otherValueIndex(offset);
    uint32_t row{offset};
    while (row < end) {
      const bool common = isCommon(row);
      const auto next = runEnd(row, end, common);
      if (common) {
        onCommon(row - offset, next - row);
      } else {
        onOther(nextOtherValue, row - offset, next - row);
        nextOtherValue += next - row;
      }
      row = next;
    }
  }

  // Reads the rows of 'ranges' for a read given 'addStringBuffer'. Fills the
  // common rows, and reads the other rows with one read of 'otherValues_'
  // that passes the buffers of their strings.
  uint32_t readStrings(
      std::span<const RowRange> ranges,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const {
    const auto numRows = this->checkReadRanges(ranges);
    auto* stringOutput = static_cast<std::string_view*>(output);
    std::optional<std::string_view> commonValue;
    std::vector<RowRange> otherRanges;

    // The output row of the first value of each of 'otherRanges'.
    std::vector<uint32_t> otherOutputRows;
    uint32_t numOtherValues{0};
    uint32_t outputOffset{0};
    for (const auto& range : ranges) {
      forEachRun(
          range.startRow,
          range.numRows(),
          [&](uint32_t row, uint32_t count) {
            const auto value = addCommonValue(addStringBuffer, commonValue);
            std::fill(
                stringOutput + outputOffset + row,
                stringOutput + outputOffset + row + count,
                value);
          },
          [&](uint32_t otherValue, uint32_t row, uint32_t count) {
            otherRanges.emplace_back(otherValue, otherValue + count);
            otherOutputRows.emplace_back(outputOffset + row);
            numOtherValues += count;
          });
      outputOffset += range.numRows();
    }
    if (numOtherValues == 0) {
      return numRows;
    }
    ScopedVector<std::string_view> otherValues{
        numOtherValues, this->pool_, this->options_.bufferPool};
    otherValues_->read(
        otherRanges,
        /*setNull=*/[](uint32_t) {},
        addStringBuffer,
        otherValues.data());
    uint32_t otherOffset{0};
    for (size_t i{0}; i < otherRanges.size(); ++i) {
      const auto count = otherRanges[i].numRows();
      std::copy(
          otherValues.data() + otherOffset,
          otherValues.data() + otherOffset + count,
          stringOutput + otherOutputRows[i]);
      otherOffset += count;
    }
    return numRows;
  }

  // Does the same as the read of ranges above for the rows at 'indices'.
  uint32_t readStrings(
      std::span<const uint32_t> indices,
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      void* output) const {
    NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
    if (indices.empty()) {
      return 0;
    }
    this->checkReadIndices(indices);
    auto* stringOutput = static_cast<std::string_view*>(output);
    std::optional<std::string_view> commonValue;
    ScopedVector<uint32_t> otherIndices{
        0, this->pool_, this->options_.bufferPool};

    // The output row of each of 'otherIndices'.
    ScopedVector<uint32_t> otherOutputRows{
        0, this->pool_, this->options_.bufferPool};
    otherIndices.reserve(indices.size());
    otherOutputRows.reserve(indices.size());
    for (uint32_t i{0}; i < indices.size(); ++i) {
      const auto row = indices[i];
      const bool common =
          cachesLookupStructures() ? isCommon(row) : isCommonView_->readAt(row);
      if (common) {
        stringOutput[i] = addCommonValue(addStringBuffer, commonValue);
        continue;
      }
      otherIndices.push_back(
          cachesLookupStructures()
              ? otherValueIndex(row)
              : isCommonView_->count(row, /*value=*/false));
      otherOutputRows.push_back(i);
    }
    if (otherIndices.empty()) {
      return static_cast<uint32_t>(indices.size());
    }
    ScopedVector<std::string_view> otherValues{
        otherIndices.size(), this->pool_, this->options_.bufferPool};
    otherValues_->read(
        std::span<const uint32_t>{otherIndices.data(), otherIndices.size()},
        /*setNull=*/[](uint32_t) {},
        addStringBuffer,
        otherValues.data());
    for (size_t i{0}; i < otherIndices.size(); ++i) {
      stringOutput[otherOutputRows[i]] = otherValues[i];
    }
    return static_cast<uint32_t>(indices.size());
  }

  // Returns the common value for the rows of a read given 'addStringBuffer'.
  // The first call passes the buffer that holds a non-empty common value:
  // the input string buffer, or a copy of the value when the view has no input
  // string buffer. Later calls return the value that the first one stored in
  // 'commonValue'.
  std::string_view addCommonValue(
      const std::function<void(velox::BufferPtr)>& addStringBuffer,
      std::optional<std::string_view>& commonValue) const {
    if (!commonValue.has_value()) {
      std::string_view value{commonValue_};
      if (!value.empty()) {
        this->addInputStrings(
            addStringBuffer, std::span<std::string_view>{&value, 1});
      }
      commonValue = value;
    }
    return *commonValue;
  }

  // Holds one bit per row, set for rows that equal the common value. The tail
  // bits of the last word are set so that popcounts of the cleared bits only
  // count real rows. Empty without cacheEncodingView.
  Vector<uint64_t> isCommonBits_;
  // Holds the number of non-common rows before each 64-row word. Together with
  // the bits it maps a row to its position in 'otherValues_' in O(1). Both
  // vectors are fixed after construction, so concurrent reads are safe. Empty
  // without cacheEncodingView.
  Vector<uint32_t> otherValueRanks_;
  // Answers the flag and the number of non-common rows of single reads. Set
  // only without cacheEncodingView.
  std::unique_ptr<TypedEncodingView<bool>> isCommonView_;
  std::unique_ptr<TypedEncodingView<T>> otherValues_;
  physicalType commonValue_;
};

} // namespace facebook::nimble
