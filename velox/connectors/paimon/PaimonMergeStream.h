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

#include "velox/common/base/TreeOfLosers.h"
#include "velox/connectors/common/FileSplitReader.h"
#include "velox/connectors/paimon/PaimonDeletionVectorReader.h"
#include "velox/connectors/paimon/PaimonRowKind.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::connector::paimon {

/// Layout of the rows the merge reads from primary-key files: the merge
/// columns of the table, which are the primary keys that are not partition
/// columns, the sequence fields, _SEQUENCE_NUMBER and _VALUE_KIND, and then
/// value columns. The merge columns before _VALUE_KIND order the rows of a
/// split, since Paimon Java orders the rows of a key by sequence fields and
/// then by sequence number.
struct PaimonMergeLayout {
  /// Builds the layout of the merge columns of 'tableHandle' followed by
  /// 'valueColumns', which must not repeat a merge column.
  /// 'lowerCaseSystemColumns' requests _SEQUENCE_NUMBER and _VALUE_KIND in
  /// lowercase, for file readers that read file column names as lowercase.
  static PaimonMergeLayout create(
      const PaimonTableHandle& tableHandle,
      const RowTypePtr& valueColumns,
      bool lowerCaseSystemColumns);

  /// Index of the _SEQUENCE_NUMBER column, the last column that orders rows.
  column_index_t sequenceNumberChannel() const {
    return static_cast<column_index_t>(compareFlags.size() - 1);
  }

  /// Index of the _VALUE_KIND column, which follows the columns that order
  /// rows.
  column_index_t valueKindChannel() const {
    return static_cast<column_index_t>(compareFlags.size());
  }

  /// Names and types of the columns, in the order above.
  RowTypePtr type;

  /// Number of leading key columns, which group the rows of a key.
  column_index_t numKeys{0};

  /// Compare flags of the leading columns that order rows.
  std::vector<CompareFlags> compareFlags;
};

/// Rows of one data file of a primary-key split in merge order, since Paimon
/// sorts a file by key and the rows of a key by sequence. Skips the rows
/// deleted by the deletion vector of the file.
class PaimonMergeStream final : public MergeStream {
 public:
  /// 'dataFile' must outlive the stream. 'reader' reads the columns of
  /// 'layout' from the file. 'deletionVectorReader' is null if the file has no
  /// deletion vector. Fails if the file does not have the number of rows its
  /// metadata counts.
  PaimonMergeStream(
      size_t fileIndex,
      const PaimonDataFile& dataFile,
      std::unique_ptr<connector::FileSplitReader> reader,
      std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader,
      std::shared_ptr<const PaimonMergeLayout> layout,
      uint64_t batchSize,
      memory::MemoryPool* pool);

  bool hasData() const override {
    return !atEnd_;
  }

  /// Orders rows by the merge columns, and rows that tie by file index.
  int32_t compare(const MergeStream& other) const override;

  /// Moves to the next row that is not deleted.
  void pop() {
    ++row_;
    skipDeletedRows();
  }

  /// Index of the file among the data files of the split.
  size_t fileIndex() const {
    return fileIndex_;
  }

  /// LSM level of the file.
  int32_t level() const {
    return dataFile_.level;
  }

  /// Position of the current row in the file.
  uint64_t fileRow() const {
    return batchStartRow_ + row_;
  }

  /// Batch of the current row. A read never reuses a batch that is still
  /// held, so rows of earlier batches stay readable while their batches are
  /// held.
  const RowVectorPtr& batch() const {
    return batch_;
  }

  /// Index of the current row in batch().
  vector_size_t row() const {
    return row_;
  }

  /// Returns true if the current row has the same primary key as row 'row' of
  /// 'batch', a batch of a stream of the same split. Compares the leading
  /// PaimonMergeLayout::numKeys columns.
  bool hasSameKey(const RowVector& batch, vector_size_t row) const;

  /// Kind of the current row.
  PaimonRowKind rowKind() const {
    return paimonRowKindFromValue(rowKinds_->valueAt(row_));
  }

  /// Returns true if the current row is a DELETE or UPDATE_BEFORE.
  bool isRetract() const;

 private:
  // Moves from row_ past deleted rows, reading batches as needed. Sets atEnd_
  // past the last row.
  void skipDeletedRows();

  // Reads the next batch. Returns false at the end of the file.
  bool loadBatch();

  // Position of the file in the split, which orders rows that tie.
  const size_t fileIndex_;
  // File the stream reads, named in error messages.
  const PaimonDataFile& dataFile_;
  // Reads the columns of 'layout_' from the file.
  const std::unique_ptr<connector::FileSplitReader> reader_;
  // Layout of the rows 'reader_' returns, shared by the streams of a split.
  const std::shared_ptr<const PaimonMergeLayout> layout_;
  // Maximum number of rows a read returns.
  const uint64_t batchSize_;
  // Pool of the batches the reads return.
  memory::MemoryPool* const pool_;
  // Reads the deletion vector of the file, or null if the file has none.
  const std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader_;
  // Bitmap of the rows of batch_ the deletion vector deletes, valid if
  // hasDeletedRows_.
  const BufferPtr deletedRows_;

  // True if the deletion vector may delete rows of batch_.
  bool hasDeletedRows_{false};

  // Rows of the last read, or null past the end of the file.
  RowVectorPtr batch_;
  // File position of the first row of batch_.
  uint64_t batchStartRow_{0};
  // Number of rows in batch_.
  vector_size_t numBatchRows_{0};
  // Index of the current row in batch_.
  vector_size_t row_{0};
  // Columns of batch_ that order rows, and its row kinds.
  std::vector<const BaseVector*> orderColumns_;
  const SimpleVector<int8_t>* rowKinds_{nullptr};
  // True once the file has no row left that is not deleted.
  bool atEnd_{false};
};

} // namespace facebook::velox::connector::paimon
