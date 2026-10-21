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
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::detail {

/// Maps logical rows to the compact non-null child encoding.
/// Single-value reads require a non-null row. Uses null-aware reads for
/// batches. With Encoding::Options::cacheEncodingView, the view keeps the null
/// flag and the number of preceding non-null rows of every row. Otherwise it
/// keeps neither, derives them for the rows of each read, and holds nothing but
/// the value stream and the null stream.
template <typename T>
class NullableEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;
  using TypedEncodingView<T>::read;

  NullableEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
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
    NIMBLE_CHECK_LE(
        numNonNullRows(),
        this->rowCount_,
        "Nullable value stream has more rows than its null stream");
    if (options.cacheEncodingView) {
      initializeCached(nulls);
    } else {
      initializeUncached(nulls);
    }
  }

  uint32_t read(
      std::span<const uint32_t> indices,
      const std::function<void(uint32_t)>& setNull,
      void* output) const final {
    NIMBLE_CHECK_LE(indices.size(), std::numeric_limits<uint32_t>::max());
    const auto numIndices = static_cast<uint32_t>(indices.size());
    if (numIndices == 0) {
      return 0;
    }
    auto* typedOutput = static_cast<physicalType*>(output);
    if (numNonNullRows() == 0) {
      fillNulls(/*outputIndex=*/0, numIndices, setNull, typedOutput);
      return 0;
    }
    if (numNonNullRows() == this->rowCount_) {
      // The compact child holds every row.
      nonNullValues_->readAt(indices, typedOutput);
      return numIndices;
    }
    std::optional<NullsWindow> nullsWindow;
    const auto nullsView = getNullsView(indices, nullsWindow);

    ScopedVector<uint32_t> nonNullIndices{
        0, this->pool_, this->options_.bufferPool};
    ScopedVector<uint32_t> outputIndices{
        0, this->pool_, this->options_.bufferPool};
    nonNullIndices.reserve(indices.size());
    outputIndices.reserve(indices.size());

    const auto addNonNull = [&](uint32_t outputIndex, uint32_t nonNullRow) {
      nonNullIndices.push_back(nonNullRow);
      outputIndices.push_back(outputIndex);
    };
    const auto addNull = [&](uint32_t outputIndex) {
      typedOutput[outputIndex] = physicalType{};
      setNull(outputIndex);
    };
    if (nullsView.has_value()) {
      for (uint32_t outputIndex{0}; outputIndex < numIndices; ++outputIndex) {
        const auto sourceIndex = indices[outputIndex];
        NIMBLE_DCHECK_LT(sourceIndex, this->rowCount_);
        if (nullsView->isNonNull(sourceIndex)) {
          addNonNull(outputIndex, nullsView->nonNullOffset(sourceIndex));
        } else {
          addNull(outputIndex);
        }
      }
    } else {
      // The rows are too few for one view, so each of them asks the null
      // stream directly.
      for (uint32_t outputIndex{0}; outputIndex < numIndices; ++outputIndex) {
        const auto sourceIndex = indices[outputIndex];
        if (nullsView_->readAt(sourceIndex)) {
          addNonNull(
              outputIndex, nullsView_->count(sourceIndex, /*value=*/true));
        } else {
          addNull(outputIndex);
        }
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
    if (numNonNullRows() == 0) {
      fillNulls(/*outputIndex=*/0, numRows, setNull, typedOutput);
      return 0;
    }
    if (ranges.empty()) {
      return 0;
    }
    if (numNonNullRows() == this->rowCount_) {
      // The compact child holds every row.
      nonNullValues_->read(ranges, setNull, typedOutput);
      return numRows;
    }

    // checkReadRanges() ordered the ranges, so the first start and the last end
    // bound them all.
    std::optional<NullsWindow> nullsWindow;
    const auto nullsView = getNullsView(ranges, numRows, nullsWindow);

    // Describe each run of selected non-null rows once, so the cost follows
    // the number of runs and not the number of rows. Each run ends at a
    // selected null or at the end of a range.
    NonNullRuns runs{std::min<size_t>(
        numRows, ranges.size() + (this->rowCount_ - numNonNullRows()))};
    uint32_t outputIndex{0};
    for (const auto& range : ranges) {
      std::optional<NullsWindow> rangeNullsWindow;
      const auto rangeNullsView = nullsView.has_value()
          ? *nullsView
          : createNullsWindow(range.startRow, range.endRow, rangeNullsWindow);
      const auto nonNullStart = rangeNullsView.nonNullOffset(range.startRow);
      const auto numRangeNonNull =
          rangeNullsView.nonNullOffset(range.endRow) - nonNullStart;
      if (numRangeNonNull == range.numRows()) {
        runs.addOrExtend(nonNullStart, numRangeNonNull, outputIndex);
        outputIndex += range.numRows();
      } else if (numRangeNonNull == 0) {
        fillNulls(outputIndex, range.numRows(), setNull, typedOutput);
        outputIndex += range.numRows();
      } else {
        outputIndex = readNulls(
            rangeNullsView,
            range,
            nonNullStart,
            outputIndex,
            setNull,
            typedOutput,
            runs);
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
    physicalType value;
    if (cachesNullFlags()) {
      NIMBLE_CHECK(
          cached_->view().isNonNull(index),
          "Nullable EncodingView physical reads require a non-null row");
      nonNullValues_->readAt(
          numNonNullRows() == this->rowCount_
              ? index
              : cached_->view().nonNullOffset(index),
          &value);
      return value;
    }
    NIMBLE_CHECK(
        nullsView_->readAt(index),
        "Nullable EncodingView physical reads require a non-null row");
    nonNullValues_->readAt(
        numNonNullRows() == this->rowCount_
            ? index
            : nullsView_->count(index, /*value=*/true),
        &value);
    return value;
  }

  T readTypedAt(uint32_t index) const final {
    return detail::castFromPhysicalType<T>(readPhysicalAt(index));
  }

 private:
  // The non-null flags of the rows [startRow, startRow + numRows) and the
  // number of non-null rows before each of them, as arrays that someone else
  // owns. Only provides immutable accessors.
  class NullsView {
   public:
    // 'nonNullFlags' holds one flag per row of the window, set for a non-null
    // row, which is 'numRows' flags, indexed from 'startRow'. 'nonNullOffsets'
    // holds the number of non-null rows between 'startRow' and each row of the
    // window, which is 'numRows' + 1 offsets, indexed from 'startRow'. The last
    // offset covers the whole window, so a range may end at the end of the
    // window. It is null when the stream has no mixed nulls, and then
    // nonNullOffset() must not be called. 'startNonNullRow' is the number of
    // non-null rows before 'startRow'.
    NullsView(
        const bool* nonNullFlags,
        const uint32_t* nonNullOffsets,
        uint32_t startRow,
        uint32_t numRows,
        uint32_t startNonNullRow)
        : nonNullFlags_{nonNullFlags},
          nonNullOffsets_{nonNullOffsets},
          startRow_{startRow},
          numRows_{numRows},
          startNonNullRow_{startNonNullRow} {}

    // Returns whether 'row', which must lie in the window, is non-null.
    bool isNonNull(uint32_t row) const {
      NIMBLE_DCHECK_GE(row, startRow_);
      NIMBLE_DCHECK_LT(row - startRow_, numRows_);
      return nonNullFlags_[row - startRow_];
    }

    // Returns the number of non-null rows before 'row', which may be the end of
    // the window. That is the position of the row in the compact child.
    uint32_t nonNullOffset(uint32_t row) const {
      NIMBLE_DCHECK_GE(row, startRow_);
      NIMBLE_DCHECK_LE(row - startRow_, numRows_);
      return startNonNullRow_ + nonNullOffsets_[row - startRow_];
    }

   private:
    const bool* const nonNullFlags_;
    const uint32_t* const nonNullOffsets_;
    const uint32_t startRow_;
    const uint32_t numRows_;
    const uint32_t startNonNullRow_;
  };

  // Owns the window that a read without cacheEncodingView builds from the null
  // stream. A window of up to kInlineRows rows, as a point read or a small
  // range needs, uses inline arrays and allocates nothing.
  class NullsWindow {
   public:
    // The inline arrays stay uninitialized because the constructor fills them.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    NullsWindow(
        const TypedEncodingView<bool>& nulls,
        uint32_t startRow,
        uint32_t endRow,
        velox::memory::MemoryPool* pool,
        velox::BufferPool* bufferPool)
        : startRow_{startRow},
          numRows_{endRow - startRow},
          startNonNullRow_{nulls.count(startRow, /*value=*/true)} {
      bool* flags = nonNulls_.data();
      uint32_t* offsets = nonNullOffsets_.data();
      if (numRows_ > kInlineRows) {
        nonNullsVector_.emplace(numRows_, pool, bufferPool);
        nonNullOffsetsVector_.emplace(
            static_cast<uint64_t>(numRows_) + 1, pool, bufferPool);
        flags = nonNullsVector_->data();
        offsets = nonNullOffsetsVector_->data();
      }
      nulls.read(startRow, numRows_, flags);
      offsets[0] = 0;
      for (uint32_t row{0}; row < numRows_; ++row) {
        offsets[row + 1] = offsets[row] + flags[row];
      }
      nonNullsPtr_ = flags;
      nonNullOffsetsPtr_ = offsets;
    }

    ~NullsWindow() = default;
    NullsWindow(const NullsWindow&) = delete;
    NullsWindow& operator=(const NullsWindow&) = delete;
    NullsWindow(NullsWindow&&) = delete;
    NullsWindow& operator=(NullsWindow&&) = delete;

    // Returns the view of the flags and the offsets, which stays valid as long
    // as this object lives.
    NullsView view() const {
      return NullsView{
          nonNullsPtr_,
          nonNullOffsetsPtr_,
          startRow_,
          numRows_,
          startNonNullRow_};
    }

   private:
    static constexpr uint32_t kInlineRows{64};

    const uint32_t startRow_;
    const uint32_t numRows_;
    const uint32_t startNonNullRow_;
    // Hold the window of a read of up to kInlineRows rows. The constructor
    // fills the part of these arrays that it uses before any read, so zeroing
    // them would only cost time.
    std::array<bool, kInlineRows> nonNulls_;
    std::array<uint32_t, kInlineRows + 1> nonNullOffsets_;
    // Hold the window of a larger read.
    std::optional<ScopedVector<bool>> nonNullsVector_;
    std::optional<ScopedVector<uint32_t>> nonNullOffsetsVector_;
    // Point to whichever of the arrays and the vectors holds the window.
    const bool* nonNullsPtr_{nullptr};
    const uint32_t* nonNullOffsetsPtr_{nullptr};
  };

  // The non-null flag of every row and the number of non-null rows before each
  // row, which a view with cacheEncodingView builds once.
  class CachedNulls {
   public:
    // Reads the flag of every row of 'nulls', which has 'numNonNulls' non-null
    // rows, and builds the number of non-null rows before each row.
    static std::unique_ptr<const CachedNulls> create(
        const TypedEncodingView<bool>& nulls,
        uint32_t numNonNulls,
        velox::memory::MemoryPool* pool,
        velox::BufferPool* bufferPool) {
      const auto numRows = nulls.rowCount();
      auto nonNulls =
          std::make_unique<ScopedVector<bool>>(numRows, pool, bufferPool);
      nulls.read(0, numRows, nonNulls->data());

      // All-null and all-non-null streams need no offsets.
      std::unique_ptr<ScopedVector<uint32_t>> nonNullOffsets;
      if (numNonNulls != 0 && numNonNulls != numRows) {
        nonNullOffsets = std::make_unique<ScopedVector<uint32_t>>(
            static_cast<uint64_t>(numRows) + 1, pool, bufferPool);
        (*nonNullOffsets)[0] = 0;
        for (uint32_t row{0}; row < numRows; ++row) {
          (*nonNullOffsets)[row + 1] =
              (*nonNullOffsets)[row] + (*nonNulls)[row];
        }
      }
      return std::unique_ptr<const CachedNulls>{new CachedNulls{
          numRows, std::move(nonNulls), std::move(nonNullOffsets)}};
    }

    // Returns the view of the flags and the offsets of all rows.
    const NullsView& view() const {
      return view_;
    }

   private:
    CachedNulls(
        uint32_t numRows,
        std::unique_ptr<ScopedVector<bool>> nonNulls,
        std::unique_ptr<ScopedVector<uint32_t>> nonNullOffsets)
        : nonNulls_{std::move(nonNulls)},
          nonNullOffsets_{std::move(nonNullOffsets)},
          view_{
              nonNulls_->data(),
              nonNullOffsets_ != nullptr ? nonNullOffsets_->data() : nullptr,
              /*startRow=*/0,
              numRows,
              /*startNonNullRow=*/0} {}

    // Holds one flag per row.
    const std::unique_ptr<ScopedVector<bool>> nonNulls_;
    // Holds the non-null rows before each row and one more entry for the end.
    // Null when the stream has no mixed nulls, because the answer is implied.
    const std::unique_ptr<ScopedVector<uint32_t>> nonNullOffsets_;
    // Views the arrays above.
    const NullsView view_;
  };

  // Reads the null flag of every row once and keeps the flags and the number of
  // non-null rows before each row, so that reads need no stream access.
  void initializeCached(std::string_view nulls) {
    const auto nullsView =
        createTypedEncodingView<bool>(nulls, this->pool_, this->options_);
    NIMBLE_CHECK_EQ(
        nullsView->rowCount(),
        this->rowCount_,
        "Nullable null stream row count must match its parent");
    cached_ = CachedNulls::create(
        *nullsView, numNonNullRows(), this->pool_, this->options_.bufferPool);
  }

  // Keeps the null stream as a view, so that a read derives the flags and the
  // non-null rows before them for its own rows and keeps nothing.
  void initializeUncached(std::string_view nulls) {
    nullsView_ =
        createTypedEncodingView<bool>(nulls, this->pool_, this->options_);
    NIMBLE_CHECK_NOT_NULL(nullsView_);
    NIMBLE_CHECK_EQ(
        nullsView_->rowCount(),
        this->rowCount_,
        "Nullable null stream row count must match its parent");
  }

  // Returns whether the view holds the flags and the non-null rows before each
  // row.
  bool cachesNullFlags() const {
    return cached_ != nullptr;
  }

  // Returns the number of non-null rows, which the value stream holds exactly.
  uint32_t numNonNullRows() const {
    return nonNullValues_->rowCount();
  }

  // Returns the view that a read of the rows in 'ranges', which hold 'numRows'
  // rows, uses: the cached view, or one window over the null stream, which
  // 'nullsWindow' keeps alive. Returns nothing for a few short ranges, which
  // read the null stream for each range because that costs less than one pass
  // over the span of the ranges.
  std::optional<NullsView> getNullsView(
      std::span<const RowRange> ranges,
      uint32_t numRows,
      std::optional<NullsWindow>& nullsWindow) const {
    if (cachesNullFlags()) {
      return cached_->view();
    }
    const auto startRow = ranges.front().startRow;
    const auto endRow = ranges.back().endRow;
    // A window for each range costs time that grows with the position of the
    // range and with its rows, and one window over the span costs time that
    // grows with the span. In streams of 4,096 and 65,536 rows, the windows win
    // for up to about 90 and 150 ranges of 16 rows, 10 and 75 ranges of 128
    // rows, and 3 and 55 ranges of 1,024 rows. These limits keep every
    // measured case within 1.25 times of the faster way: a window for each
    // range only when the ranges hold at most a third of the span, and for at
    // most this many ranges.
    constexpr size_t kMaxWindowedRanges{64};
    if (ranges.size() <= kMaxWindowedRanges &&
        static_cast<uint64_t>(numRows) * 3 <= endRow - startRow) {
      return std::nullopt;
    }
    return createNullsWindow(startRow, endRow, nullsWindow);
  }

  // Returns the view that a read of the rows 'indices' uses, as above. The
  // indices must lie within the row count. A window built from the null stream
  // rejects an index past it, and the cached view does not check.
  std::optional<NullsView> getNullsView(
      std::span<const uint32_t> indices,
      std::optional<NullsWindow>& nullsWindow) const {
    if (cachesNullFlags()) {
      return cached_->view();
    }
    // Each index costs time that grows with its position, so more indices than
    // this cost more than one pass over the span, whatever its length. The
    // break-even is about 190 indices in streams of 4,096 and of 65,536 rows.
    constexpr size_t kMaxDirectIndices{128};
    if (indices.size() <= kMaxDirectIndices) {
      return std::nullopt;
    }
    uint32_t startRow{std::numeric_limits<uint32_t>::max()};
    uint32_t endRow{0};
    for (const auto index : indices) {
      startRow = std::min(startRow, index);
      endRow = std::max(endRow, index + 1);
    }
    return createNullsWindow(startRow, endRow, nullsWindow);
  }

  // Builds the window of the rows [startRow, endRow) from the null stream in
  // 'nullsWindow', which keeps the arrays alive, and returns its view. Used by
  // reads without cacheEncodingView.
  NullsView createNullsWindow(
      uint32_t startRow,
      uint32_t endRow,
      std::optional<NullsWindow>& nullsWindow) const {
    return nullsWindow
        .emplace(
            *nullsView_,
            startRow,
            endRow,
            this->pool_,
            this->options_.bufferPool)
        .view();
  }

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

  // Returns the first null row in [row, endRow), or 'endRow' if there is none.
  static uint32_t
  nextNullRow(const NullsView& window, uint32_t row, uint32_t endRow) {
    while (row < endRow && window.isNonNull(row)) {
      ++row;
    }
    return row;
  }

  // Sets the nulls of a range that holds both nulls and non-nulls and adds its
  // runs of non-null rows to 'runs'. 'nonNullStart' is the compact child row of
  // the range's first non-null row. Returns the output index after the range.
  uint32_t readNulls(
      const NullsView& window,
      const RowRange& range,
      uint32_t nonNullStart,
      uint32_t outputIndex,
      const std::function<void(uint32_t)>& setNull,
      physicalType* typedOutput,
      NonNullRuns& runs) const {
    auto row = range.startRow;
    auto nonNullRow = nonNullStart;
    while (row < range.endRow) {
      if (!window.isNonNull(row)) {
        do {
          typedOutput[outputIndex] = physicalType{};
          setNull(outputIndex);
          ++row;
          ++outputIndex;
        } while (row < range.endRow && !window.isNonNull(row));
        continue;
      }

      const auto numRunRows = nextNullRow(window, row + 1, range.endRow) - row;
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
  // Holds the flag and the preceding non-null rows of every row. Set only with
  // cacheEncodingView, so a view without it holds only the two child views.
  std::unique_ptr<const CachedNulls> cached_;
  // Reads flags and counts non-null rows of single reads. Set only without
  // cacheEncodingView.
  std::unique_ptr<TypedEncodingView<bool>> nullsView_;
};

} // namespace facebook::nimble::detail
