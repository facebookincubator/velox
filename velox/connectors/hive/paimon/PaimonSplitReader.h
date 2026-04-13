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

#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::hive::paimon {

/// Reader for rawConvertible Paimon splits. Extends FileSplitReader to handle
/// Paimon-specific concerns:
///
///   - Multi-file iteration: A single Paimon split may contain multiple data
///     files (one per LSM level in a bucket). This reader iterates through
///     all files sequentially, transparently advancing to the next file when
///     the current one is exhausted.
///
///   - Deletion vectors: Roaring bitmaps marking deleted row positions within
///     a data file. Loaded in prepareSplit(), applied in next() via
///     Mutation.deletedRows (same pattern as IcebergSplitReader). (NYI)
///
///   - _rowkind system column: Change type (+I/-U/+U/-D) for changelog files
///     and input changelog mode. Handled in adaptColumns(). (NYI)
///
/// Also serves as the building block for PaimonMergeReader, which composes
/// multiple PaimonSplitReaders to perform merge-on-read for primary-key
/// tables with rawConvertible=false.
class PaimonSplitReader : public FileSplitReader {
 public:
  /// @param fileSplit FileConnectorSplit for the first data file (set by
  ///        FileDataSource::addSplit before calling createSplitReader).
  /// @param paimonSplit The full Paimon split. PaimonSplitReader builds
  ///        FileConnectorSplits for all data files internally.
  PaimonSplitReader(
      const std::shared_ptr<const FileConnectorSplit>& fileSplit,
      std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  ~PaimonSplitReader() override = default;

  /// Saves metadata filter and runtime stats for use by
  /// ensureFileSplitReader(). File validation is done in the constructor.
  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  /// Reads from the current file. When a file is exhausted, transparently
  /// advances to the next file in the split. Returns 0 only when all files
  /// are exhausted.
  uint64_t next(uint64_t size, VectorPtr& output) override;

 private:
  // Builds a FileConnectorSplit from a PaimonDataFile.
  std::shared_ptr<FileConnectorSplit> makeFileConnectorSplit(
      const PaimonDataFile& dataFile) const;

  // Ensures a file split reader is ready to produce rows. If the current
  // reader is exhausted, advances to the next non-empty file. Returns true
  // if a reader is ready, false when all files are exhausted.
  bool ensureFileSplitReader();

  // Cleans up the current file reader after it is exhausted and advances
  // currentFileIndex_ so ensureFileSplitReader() opens the next file.
  void finishFileSplitReader();

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;

  // FileConnectorSplits built from paimonSplit_->dataFiles(), one per file.
  // Used to switch the base reader between files during iteration.
  std::vector<std::shared_ptr<FileConnectorSplit>> fileSplits_;

  // Index of the next data file to open.
  size_t currentFileIndex_{0};

  // Saved from prepareSplit() for re-use when advancing to subsequent files.
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  dwio::common::RuntimeStatistics* runtimeStats_{nullptr};
};

} // namespace facebook::velox::connector::hive::paimon
