/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <functional>

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/parquet/reader/Metadata.h"
#include "velox/dwio/parquet/reader/PageReader.h"

namespace facebook::velox::common {
class ScanSpec;
} // namespace facebook::velox::common

namespace facebook::velox::dwio::common {
class BufferedInput;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::parquet {

class ParquetParams : public dwio::common::FormatParams {
 public:
  ParquetParams(
      memory::MemoryPool& pool,
      dwio::common::SplitStats& stats,
      const FileMetaDataPtr metaData,
      const tz::TimeZone* sessionTimezone,
      TimestampPrecision timestampPrecision,
      bool nullStructIfAllFieldsMissing)
      : FormatParams(pool, stats),
        metaData_(metaData),
        sessionTimezone_(sessionTimezone),
        timestampPrecision_(timestampPrecision),
        nullStructIfAllFieldsMissing_(nullStructIfAllFieldsMissing) {}
  std::unique_ptr<dwio::common::FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const common::ScanSpec& scanSpec) override;

  TimestampPrecision timestampPrecision() const {
    return timestampPrecision_;
  }

  const FileMetaDataPtr& fileMetaData() const {
    return metaData_;
  }

  bool nullStructIfAllFieldsMissing() const {
    return nullStructIfAllFieldsMissing_;
  }

 private:
  const FileMetaDataPtr metaData_;
  const tz::TimeZone* sessionTimezone_;
  const TimestampPrecision timestampPrecision_;
  const bool nullStructIfAllFieldsMissing_;
};

/// Format-specific data created for each leaf column of a Parquet rowgroup.
class ParquetData : public dwio::common::FormatData {
 public:
  ParquetData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type,
      const FileMetaDataPtr fileMetadataPtr,
      memory::MemoryPool& pool,
      dwio::common::ColumnRuntimeStats& stats,
      const tz::TimeZone* sessionTimezone)
      : pool_(pool),
        type_(std::static_pointer_cast<const ParquetTypeWithId>(type)),
        fileMetaDataPtr_(fileMetadataPtr),
        maxDefine_(type_->maxDefine_),
        maxRepeat_(type_->maxRepeat_),
        rowsInRowGroup_(-1),
        stats_(stats),
        sessionTimezone_(sessionTimezone) {}

  /// Prepares to read data for 'index'th row group.
  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  /// True if 'this' is a physical column (has a column chunk per row group).
  bool isLeaf() const {
    return type_->isLeaf();
  }

  /// Marks 'this' as a lazily loaded column whose chunk may not be enqueued
  /// with its row group. 'onNeeded(rowGroup)' is called on the first read of
  /// a row group whose chunk is not enqueued; it must enqueue the chunk (see
  /// enqueueRowGroup()) before returning.
  void setLazyColumnCallback(std::function<void(uint32_t)> onNeeded);

  /// True if the chunk of 'index'th row group has been enqueued.
  bool isRowGroupEnqueued(uint32_t index) const {
    return index < enqueued_.size() && enqueued_[index];
  }

  /// Positions 'this' at 'index'th row group. loadRowGroup must be called
  /// first. The returned PositionProvider is empty and should not be used.
  /// Other formats may use it.
  dwio::common::PositionProvider seekToRowGroup(int64_t index) override;

  void filterRowGroups(
      const common::ScanSpec& scanSpec,
      uint64_t rowsPerRowGroup,
      const dwio::common::StatsContext& writerContext,
      FilterRowGroupsResult&) override;

  PageReader* reader() {
    ensureReader();
    VELOX_CHECK_NOT_NULL(
        reader_,
        "No PageReader for column {}: not a leaf or not positioned at a row group",
        type_->column());
    return reader_.get();
  }

  // Reads null flags for 'numValues' next top level rows. The first
  // 'numValues' bits of 'nulls' are set and the reader is advanced by
  // numValues'.
  void readNullsOnly(int32_t numValues, BufferPtr& nulls) {
    reader()->readNullsOnly(numValues, nulls);
  }

  bool hasNulls() const override {
    return maxDefine_ > 0;
  }

  /// Sets nulls to be returned by readNulls(). Nulls for non-leaf readers
  /// come from leaf repdefs which are gathered before descending the reader
  /// tree.
  void setNulls(BufferPtr& nulls, int32_t numValues) {
    if (nulls || numValues) {
      VELOX_CHECK_EQ(presetNullsConsumed_, presetNullsSize_);
    }
    presetNulls_ = nulls;
    presetNullsSize_ = numValues;
    presetNullsConsumed_ = 0;
    if (nulls) {
      nulls->setSize(bits::nbytes(numValues));
    }
  }

  int32_t presetNullsLeft() const {
    return presetNullsSize_ - presetNullsConsumed_;
  }

  void readNulls(
      vector_size_t numValues,
      const uint64_t* incomingNulls,
      BufferPtr& nulls,
      bool nullsOnly = false) override {
    // If the query accesses only nulls, read the nulls from the pages in
    // range. If nulls are preread, return those minus any skipped.
    if (presetNulls_) {
      VELOX_CHECK_LE(numValues, presetNullsSize_ - presetNullsConsumed_);
      if (!presetNullsConsumed_ && numValues == presetNullsSize_) {
        nulls = std::move(presetNulls_);
        presetNullsConsumed_ = numValues;
      } else {
        dwio::common::ensureCapacity<bool>(nulls, numValues, &pool_);
        auto bits = nulls->asMutable<uint64_t>();
        bits::copyBits(
            presetNulls_->as<uint64_t>(),
            presetNullsConsumed_,
            bits,
            0,
            numValues);
        nulls->setSize(bits::nbytes(numValues));
        presetNullsConsumed_ += numValues;
      }
      return;
    }
    if (nullsOnly) {
      readNullsOnly(numValues, nulls);
      return;
    }
    // There are no column-level nulls in Parquet, only page-level ones, so
    // this is always non-null.
    nulls = nullptr;
  }

  uint64_t skipNulls(uint64_t numValues, bool nullsOnly) override {
    // If we are seeking a column where nulls and data are read, the skip is
    // done in skip(). If we are reading nulls only, this is called with
    // 'nullsOnly' set and is responsible for reading however many nulls or
    // pages it takes to skip 'numValues' top level rows.
    if (nullsOnly) {
      reader()->skipNullsOnly(numValues);
    }
    if (presetNulls_) {
      VELOX_DCHECK_LE(numValues, presetNullsSize_ - presetNullsConsumed_);
      presetNullsConsumed_ += numValues;
    }
    return numValues;
  }

  uint64_t skip(uint64_t numRows) override {
    reader()->skip(numRows);
    return numRows;
  }

  /// Applies 'visitor' to the data in the column of 'this'. See
  /// PageReader::readWithVisitor().
  template <typename Visitor>
  void readWithVisitor(Visitor visitor) {
    reader()->readWithVisitor(visitor);
  }

  const VectorPtr& dictionaryValues(const TypePtr& type) {
    return reader()->dictionaryValues(type);
  }

  void clearDictionary() {
    reader()->clearDictionary();
  }

  bool hasDictionary() {
    return reader()->isDictionary();
  }

  bool isDeltaBinaryPacked() {
    return reader()->isDeltaBinaryPacked();
  }

  bool isDeltaByteArray() {
    return reader()->isDeltaByteArray();
  }

  bool parentNullsInLeaves() const override {
    return true;
  }

  // Returns the <offset, length> of the row group.
  std::pair<int64_t, int64_t> getRowGroupRegion(uint32_t index) const;

 private:
  /// True if 'filter' may have hits for the column of 'this' according to the
  /// stats in 'rowGroup'.
  bool rowGroupMatches(uint32_t rowGroupId, const common::Filter* filter);

 protected:
  memory::MemoryPool& pool_;
  std::shared_ptr<const ParquetTypeWithId> type_;
  const FileMetaDataPtr fileMetaDataPtr_;
  // Streams for this column in each of 'rowGroups_'. Will be created on or
  // ahead of first use, not at construction.
  std::vector<std::unique_ptr<dwio::common::SeekableInputStream>> streams_;
  // Whether the stream of each row group has been enqueued.
  std::vector<bool> enqueued_;

  const uint32_t maxDefine_;
  const uint32_t maxRepeat_;
  int64_t rowsInRowGroup_;
  dwio::common::ColumnRuntimeStats& stats_;
  const tz::TimeZone* sessionTimezone_;
  std::unique_ptr<PageReader> reader_;

  // Lazily loaded column support. seekToRowGroup() on a row group whose
  // stream was not enqueued records it here; the stream and PageReader are
  // created on first use by ensureReader().
  int64_t pendingRowGroup_{-1};
  std::function<void(uint32_t)> onLazyColumnNeeded_;

  void ensureReader();
  void createReader(uint32_t index);

  // Nulls derived from leaf repdefs for non-leaf readers.
  BufferPtr presetNulls_;

  // Number of valid bits in 'presetNulls_'.
  int32_t presetNullsSize_{0};

  // Count of leading skipped positions in 'presetNulls_'
  int32_t presetNullsConsumed_{0};
};

} // namespace facebook::velox::parquet
