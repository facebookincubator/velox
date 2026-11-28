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

#include "velox/connectors/common/FileSplitReader.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/paimon/PaimonDeletionVectorReader.h"
#include "velox/connectors/paimon/PaimonMergeStream.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::paimon {

/// Sequential multi-file reader for rawConvertible Paimon splits. Opens one
/// file at a time and reuses FileSplitReader for per-file preparation and batch
/// reads, advancing past exhausted, empty or skipped files. The same ScanSpec
/// is reused across files. A file's deletion vector is loaded when the file is
/// opened and applied in next() via Mutation.deletedRows, the same contract as
/// IcebergSplitReader.
class PaimonSplitReader : public connector::FileSplitReader {
 public:
  /// @param fileSplit FileConnectorSplit for the first data file.
  /// @param paimonSplit Complete logical split of data files.
  /// @param partitionKeys Borrowed partition-column handles for the scan,
  ///        distinct from the partition values in paimonSplit.
  PaimonSplitReader(
      const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
      std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
      const std::shared_ptr<const PaimonTableHandle>& tableHandle,
      const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
          partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const connector::FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  ~PaimonSplitReader() override = default;

  /// Saves the metadata filter and borrows runtime stats for lazy per-file
  /// preparation. Files are opened by next(), not by this method.
  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStats& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  /// Reads from the current file. When a file is exhausted, transparently
  /// advances to the next file in the split. Returns 0 only when all files
  /// are exhausted.
  uint64_t next(uint64_t size, VectorPtr& output) override;

  /// Rebinds the borrowed counters when the reader moves to another data
  /// source.
  void setRuntimeStats(dwio::common::RuntimeStats& runtimeStats) {
    runtimeStats_ = &runtimeStats;
  }

  /// Builds the FileConnectorSplit that reads 'dataFile' of 'paimonSplit'.
  /// Reader options come from the split of the first file, so every split of
  /// a Paimon file must come from here.
  static std::shared_ptr<connector::FileConnectorSplit> makeFileConnectorSplit(
      const PaimonConnectorSplit& paimonSplit,
      const PaimonDataFile& dataFile);

 protected:
  size_t currentDataFileIndex() const {
    return currentFileIndex_;
  }

  // Returns the row selection of the current file, a bitmap with one bit per
  // row of the file, set for the rows to return. A file with a row selection
  // gets no deletion vector reader, so the selection must leave out the rows
  // its deletion vector deletes. Returns nullptr to return every row the
  // deletion vector does not delete.
  virtual const uint64_t* rowSelection() const {
    return nullptr;
  }

  // Opens a merge stream of each data file, in file order, which reads the
  // merge columns of 'tableHandle' and then 'valueColumns' 'batchSize' rows at
  // a time with no filters.
  std::vector<std::unique_ptr<PaimonMergeStream>> openMergeStreams(
      const PaimonTableHandle& tableHandle,
      const RowTypePtr& valueColumns,
      uint64_t batchSize);

 private:
  // Reuses the current reader or lazily prepares the next non-empty file.
  // Returns false when no files remain.
  bool ensureFileSplitReader();

  // Creates the deletion-vector reader of the current file if the file has a
  // deletion vector and no row selection.
  void maybeCreateDeletionVectorReader();

  // Returns the deleted-row bitmap for the next 'numRows' rows of the current
  // file, or nullptr if rowSelection() is null and the file has no deletion
  // vector or no deleted row at or after the next row.
  const uint64_t* deletedRows(uint64_t numRows);

  // Cleans up the current file reader after it is exhausted and advances
  // currentFileIndex_ so ensureFileSplitReader() opens the next file.
  void finishFileSplitReader();

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;

  // FileConnectorSplits built from paimonSplit_->dataFiles(), one per file.
  // Used to switch the base reader between files during iteration.
  std::vector<std::shared_ptr<connector::FileConnectorSplit>> fileSplits_;

  // Index of the current file, or the next file to prepare between readers.
  size_t currentFileIndex_{0};

  // Saved from prepareSplit() for re-use when advancing to subsequent files.
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  // Borrowed from the data source; rebound with setRuntimeStats() on takeover.
  dwio::common::RuntimeStats* runtimeStats_{nullptr};

  // Deletion-vector reader for the currently open data file, if present.
  std::unique_ptr<PaimonDeletionVectorReader> deletionVectorReader_;

  // Deleted rows of the current batch, passed to the row reader as
  // Mutation::deletedRows: a set bit i skips row i of the batch. Allocated on
  // the first batch with deletes and reused for later batches and files.
  BufferPtr deleteBitmap_;
};

} // namespace facebook::velox::connector::paimon
