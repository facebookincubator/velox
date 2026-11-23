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

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/SelectiveRepeatedColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetData.h"

namespace facebook::velox::parquet {

/// Enqueues row-group streams for a reader's logical descendants. For a struct
/// without logical children, follows its synthetic rep/def source instead.
void enqueueRowGroupRecursive(
    dwio::common::SelectiveColumnReader& reader,
    uint32_t index,
    dwio::common::BufferedInput& input);

/// Comtainer for the lengths of a repeated reader where the lengths are
/// pre-filled from repdefs.
class RepeatedLengths {
 public:
  void setLengths(BufferPtr lengths) {
    lengths_ = std::move(lengths);
    nextLengthIndex_ = 0;
  }

  BufferPtr& lengths() {
    return lengths_;
  }

  int32_t nextLengthIndex() const {
    return nextLengthIndex_;
  }

  void readLengths(int32_t* lengths, int32_t numLengths) {
    VELOX_CHECK_LE(
        nextLengthIndex_ + numLengths, lengths_->size() / sizeof(int32_t));
    memcpy(
        lengths,
        lengths_->as<int32_t>() + nextLengthIndex_,
        numLengths * sizeof(int32_t));
    nextLengthIndex_ += numLengths;
  }

 private:
  BufferPtr lengths_;
  int32_t nextLengthIndex_{0};
};

class MapColumnReader : public dwio::common::SelectiveMapColumnReader {
 public:
  MapColumnReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  void seekToRowGroup(int64_t index) override;

  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override;

  void setLengths(BufferPtr lengths) {
    lengths_.setLengths(lengths);
  }
  void readLengths(
      int32_t* lengths,
      int32_t numLengths,
      const uint64_t* /*nulls*/) override {
    lengths_.readLengths(lengths, numLengths);
  }

  /// Sets nulls and lengths of 'this' for the range of top level rows for which
  /// these have been decoded in 'leaf'.
  void setLengthsFromRepDefs(PageReader& leaf);

  /// advances 'this' to the end of the previously provided lengths/nulls. This
  /// is needed if lists are conditionally read from different structs that all
  /// end at different positions. Repeated children must use all lengths
  /// supplied before receiving new lengths.
  void skipUnreadLengths();

  void filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsContext&,
      dwio::common::FormatData::FilterRowGroupsResult&) const override;

 private:
  RepeatedLengths lengths_;
  RepeatedLengths keyLengths_;
  RepeatedLengths elementLengths_;
  LevelInfo levelInfo_;
};

class ListColumnReader : public dwio::common::SelectiveListColumnReader {
 public:
  ListColumnReader(
      const dwio::common::ColumnReaderOptions& columnReaderOptions,
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec);

  void seekToRowGroup(int64_t index) override;

  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input);

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override;

  void setLengths(BufferPtr lengths) {
    lengths_.setLengths(lengths);
  }
  void readLengths(
      int32_t* lengths,
      int32_t numLengths,
      const uint64_t* /*nulls*/) override {
    lengths_.readLengths(lengths, numLengths);
  }

  /// Sets nulls and lengths of 'this' for the range of top level rows for which
  /// these have been decoded in 'leaf'.
  void setLengthsFromRepDefs(PageReader& leaf);

  /// advances 'this' to the end of the previously provided lengths/nulls. This
  /// is needed if lists are conditionally read from different structs that all
  /// end at different positions. Repeated children must use all lengths
  /// supplied before receiving new lengths.
  void skipUnreadLengths();

  void filterRowGroups(
      uint64_t rowGroupSize,
      const dwio::common::StatsContext&,
      dwio::common::FormatData::FilterRowGroupsResult&) const override;

 private:
  RepeatedLengths lengths_;
  LevelInfo levelInfo_;
};

/// Prepares a complex Parquet reader for reading 'rows' at 'offset'. For a
/// direct child of the root struct, decodes repetition and definition levels
/// through the last requested row, populates nulls and lengths for its subtree,
/// and skips rows before 'offset'. Advances the reader's logical read offset to
/// 'offset' if needed. 'reader' must be a complex type reader. It may be inside
/// structs but may not be inside a repeated reader.
void prepareRepDefsAndOffset(
    dwio::common::SelectiveColumnReader& reader,
    int64_t offset,
    const RowSet& rows);

} // namespace facebook::velox::parquet
