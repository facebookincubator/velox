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
/// Single-value reads require a non-null row. Uses null-aware reads for
/// batches.
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
    NIMBLE_CHECK_LE(
        this->dataOffset_,
        data.size(),
        "Nullable encoding prefix exceeds payload bounds");
    const char* position = data.data() + this->dataOffset_;
    NIMBLE_CHECK_LE(
        sizeof(uint32_t),
        static_cast<size_t>(data.data() + data.size() - position),
        "Nullable encoding is missing its value-stream length");
    const auto nonNullBytes = encoding::readUint32(position);
    NIMBLE_CHECK_LE(
        nonNullBytes,
        static_cast<size_t>(data.data() + data.size() - position),
        "Nullable value encoding exceeds payload bounds");
    const std::string_view nonNullValues{position, nonNullBytes};
    position += nonNullBytes;
    const std::string_view nulls{
        position, static_cast<size_t>(data.data() + data.size() - position)};

    nonNullValues_ =
        createTypedEncodingView<T>(nonNullValues, this->pool_, this->options_);
    auto nullsView =
        createTypedEncodingView<bool>(nulls, this->pool_, this->options_);
    NIMBLE_CHECK_EQ(
        nullsView->rowCount(),
        this->rowCount_,
        "Nullable null stream row count must match its parent");
    nullsView->read(0, this->rowCount_, isNonNull_.data());

    for (uint32_t row{0}; row < this->rowCount_; ++row) {
      numNonNulls_ += isNonNull_[row];
    }
    NIMBLE_CHECK_EQ(
        nonNullValues_->rowCount(),
        numNonNulls_,
        "Nullable value stream row count must match its null stream");
    // All-null and all-non-null streams need no offsets.
    if (numNonNulls_ == 0 || numNonNulls_ == this->rowCount_) {
      return;
    }

    nonNullOffsets_.resize(static_cast<uint64_t>(this->rowCount_) + 1);
    nonNullOffsets_[0] = 0;
    for (uint32_t row{0}; row < this->rowCount_; ++row) {
      nonNullOffsets_[row + 1] = nonNullOffsets_[row] + isNonNull_[row];
    }
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
        nonNullIndices.push_back(nonNullOffset(sourceIndex));
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
    if (numNonNulls_ == 0) {
      fillNulls(/*outputIndex=*/0, numRows, setNull, typedOutput);
      return 0;
    }

    // Describe each run of selected non-null rows once, so the cost follows
    // the number of runs and not the number of rows. Each run ends at a
    // selected null or at the end of a range.
    NonNullRuns runs{std::min<size_t>(
        numRows, ranges.size() + (this->rowCount_ - numNonNulls_))};
    uint32_t outputIndex{0};
    for (const auto& range : ranges) {
      const auto nonNullStart = nonNullOffset(range.startRow);
      const auto numRangeNonNull = nonNullOffset(range.endRow) - nonNullStart;
      if (numRangeNonNull == range.numRows()) {
        runs.addOrExtend(nonNullStart, numRangeNonNull, outputIndex);
        outputIndex += range.numRows();
      } else if (numRangeNonNull == 0) {
        fillNulls(outputIndex, range.numRows(), setNull, typedOutput);
        outputIndex += range.numRows();
      } else {
        outputIndex = readNulls(
            range, nonNullStart, outputIndex, setNull, typedOutput, runs);
      }
    }

    if (runs.numRows() == 0) {
      return 0;
    }
    if (runs.numRows() == numRows) {
      nonNullValues_->read(runs.ranges(), setNull, typedOutput);
      return numRows;
    }

    ScopedVector<physicalType> values{
        0, this->pool_, this->options_.bufferPool};
    values.resize(runs.numRows());
    nonNullValues_->read(runs.ranges(), setNull, values.data());
    scatterRuns(values.data(), runs, typedOutput);
    return runs.numRows();
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
    this->checkReadRange(index, 1);
    NIMBLE_CHECK(
        isNonNull_[index],
        "Nullable EncodingView physical reads require a non-null row");
    physicalType value;
    nonNullValues_->readAt(nonNullOffset(index), &value);
    return value;
  }

  T readTypedAt(uint32_t index) const final {
    return detail::castFromPhysicalType<T>(readPhysicalAt(index));
  }

 private:
  // Selected non-null rows grouped into runs. A run is a maximal stretch of
  // consecutive selected non-null rows. Each run has a range in the compact
  // child and an offset where it lands in the output.
  class NonNullRuns {
   public:
    explicit NonNullRuns(size_t maxNumRuns) {
      ranges_.reserve(maxNumRuns);
      outputOffsets_.reserve(maxNumRuns);
    }

    // Appends a run that cannot continue the previous one.
    void add(uint32_t start, uint32_t numRunRows, uint32_t outputIndex) {
      ranges_.emplace_back(start, start + numRunRows);
      outputOffsets_.push_back(outputIndex);
      numRows_ += numRunRows;
    }

    // Appends a run, or extends the previous run when this one continues it in
    // both the compact child and the output.
    void
    addOrExtend(uint32_t start, uint32_t numRunRows, uint32_t outputIndex) {
      if (continuesLastRun(start, outputIndex)) {
        ranges_.back().endRow += numRunRows;
        numRows_ += numRunRows;
        return;
      }
      add(start, numRunRows, outputIndex);
    }

    // Returns where each run sits in the compact child.
    const std::vector<RowRange>& ranges() const {
      return ranges_;
    }

    // Returns where each run lands in the output.
    const std::vector<uint32_t>& outputOffsets() const {
      return outputOffsets_;
    }

    // Returns the number of rows across all runs.
    uint32_t numRows() const {
      return numRows_;
    }

   private:
    // Returns whether a run that starts at compact child row 'start' and
    // output row 'outputIndex' continues the last run.
    bool continuesLastRun(uint32_t start, uint32_t outputIndex) const {
      return !ranges_.empty() && ranges_.back().endRow == start &&
          outputOffsets_.back() + ranges_.back().numRows() == outputIndex;
    }

    std::vector<RowRange> ranges_;
    std::vector<uint32_t> outputOffsets_;
    uint32_t numRows_{0};
  };

  // Returns the number of non-null rows before 'row'. 'row' may equal the row
  // count. Streams without mixed nulls keep no offsets because the answer is
  // implied.
  uint32_t nonNullOffset(uint32_t row) const {
    if (numNonNulls_ == this->rowCount_) {
      return row;
    }
    if (numNonNulls_ == 0) {
      return 0;
    }
    return nonNullOffsets_[row];
  }

  // Returns the first null row in [row, endRow), or 'endRow' if there is none.
  uint32_t nextNullRow(uint32_t row, uint32_t endRow) const {
    while (row < endRow && isNonNull_[row]) {
      ++row;
    }
    return row;
  }

  // Sets the nulls of a range that holds both nulls and non-nulls and adds its
  // runs of non-null rows to 'runs'. 'nonNullStart' is the compact child row of
  // the range's first non-null row. Returns the output index after the range.
  uint32_t readNulls(
      const RowRange& range,
      uint32_t nonNullStart,
      uint32_t outputIndex,
      const std::function<void(uint32_t)>& setNull,
      physicalType* typedOutput,
      NonNullRuns& runs) const {
    auto row = range.startRow;
    auto nonNullRow = nonNullStart;
    while (row < range.endRow) {
      if (!isNonNull_[row]) {
        do {
          typedOutput[outputIndex] = physicalType{};
          setNull(outputIndex);
          ++row;
          ++outputIndex;
        } while (row < range.endRow && !isNonNull_[row]);
        continue;
      }

      const auto numRunRows = nextNullRow(row + 1, range.endRow) - row;
      // Only a run that starts the range can continue the previous range's
      // run. Every other run follows a selected null, so the output has a gap.
      if (row == range.startRow) {
        runs.addOrExtend(nonNullRow, numRunRows, outputIndex);
      } else {
        runs.add(nonNullRow, numRunRows, outputIndex);
      }
      row += numRunRows;
      nonNullRow += numRunRows;
      outputIndex += numRunRows;
    }
    return outputIndex;
  }

  // Zeroes 'numNulls' consecutive output rows from 'outputIndex' and reports
  // each as null.
  static void fillNulls(
      uint32_t outputIndex,
      uint32_t numNulls,
      const std::function<void(uint32_t)>& setNull,
      physicalType* typedOutput) {
    std::fill_n(typedOutput + outputIndex, numNulls, physicalType{});
    for (uint32_t i{0}; i < numNulls; ++i) {
      setNull(outputIndex + i);
    }
  }

  // Copies each run from 'values', which holds the runs back to back, to its
  // output position.
  static void scatterRuns(
      const physicalType* values,
      const NonNullRuns& runs,
      physicalType* typedOutput) {
    // Runs of at most this many rows are copied inline.
    static constexpr uint32_t kMaxInlineCopyRows{8};
    const auto& ranges = runs.ranges();
    const auto& outputOffsets = runs.outputOffsets();
    const auto numRuns = ranges.size();
    for (size_t run{0}; run < numRuns; ++run) {
      const auto numRunRows = ranges[run].numRows();
      auto* target = typedOutput + outputOffsets[run];
      if (numRunRows > kMaxInlineCopyRows) {
        std::copy_n(values, numRunRows, target);
      } else {
        // Short runs dominate when nulls are dense, and std::copy_n on them
        // reads 5% to 12% slower than this loop at 12% to 50% nulls.
        for (uint32_t row{0}; row < numRunRows; ++row) {
          target[row] = values[row];
        }
      }
      values += numRunRows;
    }
  }

  std::unique_ptr<TypedEncodingView<T>> nonNullValues_;
  ScopedVector<bool> isNonNull_;
  ScopedVector<uint32_t> nonNullOffsets_;
  // Number of non-null rows. 'nonNullOffsets_' is built only when this lies
  // strictly between zero and the row count.
  uint32_t numNonNulls_{0};
};

} // namespace facebook::nimble::detail
