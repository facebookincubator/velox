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
#include <span>
#include <type_traits>
#include <utility>

#include <folly/CPortability.h>

#include "velox/common/memory/RawVector.h"
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
    values_ = detail::createTypedEncodingView<runValueType>(
        {pos, static_cast<size_t>(data.data() + data.size() - pos)},
        this->pool_,
        options);
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
    const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
    // A run value stored as T is read typed, which skips the type-erased
    // readAt(index, void*) hop on every point read.
    if constexpr (std::is_same_v<runValueType, T>) {
      return values_->readAt(runIndex);
    } else {
      physicalType value;
      values_->readAt(runIndex, &value);
      return detail::castFromPhysicalType<T>(value);
    }
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), offset);
    NIMBLE_CHECK(it != runEnds_.end());

    // Reading run values in bulk replaces one virtual call per run with one
    // for the whole read, which only pays off when runs are short (so the
    // per-run dispatch dominates) and the read is long enough to amortise the
    // extra binary search this path needs to locate the last run. The length
    // gate is an absolute row count rather than a fraction of the section,
    // since SubIntSplitEncodingView reads fixed-size chunks regardless of
    // section size and a fraction-based gate would keep those off this path.
    if ((length >= kMinBulkRunValueLength ||
         length * kBulkRunValueDenominator >=
             this->rowCount_ * kBulkRunValueNumerator) &&
        runEnds_.size() * kMaxAverageRunLength >= this->rowCount_) {
      readRunsInBulk(
          offset, length, static_cast<uint32_t>(it - runEnds_.begin()), output);
      return;
    }

    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
      const auto count = std::min(length - outputOffset, *it - offset);
      physicalType value;
      values_->readAt(runIndex, &value);
      std::fill(output + outputOffset, output + outputOffset + count, value);
      outputOffset += count;
      offset += count;
      ++it;
    }
  }

  // Answers a whole range list from one forward walk of the run ends, rather
  // than a fresh binary search per range: worthwhile when a section has many
  // runs and ranges arrive roughly ascending, each starting close to where
  // the last ended, which is the case for a caller that sorts indices before
  // reading. Ascending order is not guaranteed by the interface, so a range
  // starting before the previous one ended falls back to a binary search.
  void readPhysicalRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const final {
    const uint32_t* const first = runEnds_.data();
    const uint32_t* const last = first + runEnds_.size();
    const uint32_t* run = first;
    // One past the highest row the cursor is known to be correct for.
    uint64_t walked = 0;
    for (const auto& range : ranges) {
      const uint32_t offset = range.startRow;
      const uint32_t length = range.numRows();
      if (length == 0) {
        continue;
      }
      // Long enough to amortise its own search, and readPhysical() has a
      // vectorised run fill this walk does not; the cursor stays where it
      // was, which is still behind this range and so still correct for the
      // next one.
      if (length >= kMinBulkRunValueLength) {
        readPhysical(offset, length, output);
        walked = static_cast<uint64_t>(offset) + length;
        output += length;
        continue;
      }
      this->checkReadRange(offset, length);
      run = offset >= walked ? advanceToRun(run, last, offset)
                             : std::upper_bound(first, last, offset);
      NIMBLE_CHECK(run != last);
      uint32_t produced = 0;
      uint32_t row = offset;
      while (produced < length) {
        const auto runIndex = static_cast<uint32_t>(run - first);
        const uint32_t count = std::min(length - produced, *run - row);
        physicalType value;
        values_->readAt(runIndex, &value);
        std::fill(output + produced, output + produced + count, value);
        produced += count;
        row += count;
        if (row == *run) {
          ++run;
        }
      }
      // Either the loop stopped inside a run, leaving *run > row, or it
      // exhausted one and stepped to the next, whose end is above row as
      // well. Both make the cursor correct for any later range starting at
      // `row` or beyond.
      walked = row;
      output += length;
    }
  }

  // First run end above `row`, searched forward from `run` in O(log gap)
  // rather than O(log runCount), so a nearby step avoids the cache misses a
  // fresh binary search over the whole table would pay.
  static const uint32_t*
  advanceToRun(const uint32_t* run, const uint32_t* last, uint32_t row) {
    if (run == last || *run > row) {
      return run;
    }
    size_t step = 1;
    const uint32_t* below = run;
    const uint32_t* above = below + 1;
    while (above < last && *above <= row) {
      below = above;
      step *= 2;
      above = static_cast<size_t>(last - below) > step ? below + step : last;
    }
    return std::upper_bound(below + 1, above, row);
  }

  // Kept out of line so that readPhysical()'s common path is unaffected by
  // this code's presence; only a call that actually takes this branch pays
  // for it.
  FOLLY_NOINLINE void readRunsInBulk(
      uint32_t offset,
      uint32_t length,
      uint32_t firstRun,
      physicalType* output) const {
    const auto lastIt =
        std::upper_bound(runEnds_.begin(), runEnds_.end(), offset + length - 1);
    NIMBLE_CHECK(lastIt != runEnds_.end());
    const auto runCount =
        static_cast<uint32_t>(lastIt - runEnds_.begin()) - firstRun + 1;

    // Held per thread rather than allocated per call: a bulk read reaches
    // this once per section, and a view is read concurrently.
    thread_local velox::raw_vector<physicalType> runValues;
    runValues.resize(runCount);
    values_->read(firstRun, runCount, runValues.data());

    // std::fill's vectorised body is guarded on the element count, which
    // mispredicts badly when runs are short, as they are on the sections this
    // path targets. Storing a fixed width unconditionally and advancing by
    // the run length avoids that branch; a short run overshoots into the next
    // run's output, which the next store overwrites, and the loop below stops
    // early enough that the overshoot never leaves the caller's buffer.
    constexpr uint32_t kLanes = 32 / sizeof(physicalType);

    uint32_t outputOffset{0};
    uint32_t run{0};
    while (outputOffset + kLanes <= length) {
      const uint32_t count =
          std::min(length - outputOffset, runEnds_[firstRun + run] - offset);
      physicalType* out = output + outputOffset;
      const physicalType value = runValues[run];
      // Fixed trip count, so this compiles to stores with no guard on it.
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        out[lane] = value;
      }
      if (count > kLanes) {
        std::fill(out + kLanes, out + count, value);
      }
      outputOffset += count;
      offset += count;
      ++run;
    }

    // Within one store of the end, where overshooting would write past the
    // caller's buffer.
    while (outputOffset < length) {
      const uint32_t count =
          std::min(length - outputOffset, runEnds_[firstRun + run] - offset);
      std::fill(
          output + outputOffset, output + outputOffset + count, runValues[run]);
      outputOffset += count;
      offset += count;
      ++run;
    }
  }

  // Fraction of the section a read must cover before the bulk run-value read
  // is worth its extra binary search.
  static constexpr uint32_t kBulkRunValueNumerator = 1;
  static constexpr uint32_t kBulkRunValueDenominator = 2;

  // Rows at or above which a read takes the bulk run-value path regardless of
  // the section's size, so a caller reading in fixed-size chunks still
  // qualifies.
  static constexpr uint32_t kMinBulkRunValueLength = 512;

  // Longest average run length for which the bulk run-value read still pays
  // off; sections with longer runs skip it.
  static constexpr uint32_t kMaxAverageRunLength = 32;
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
