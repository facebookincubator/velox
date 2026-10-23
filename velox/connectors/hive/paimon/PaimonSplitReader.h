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
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::hive::paimon {

/// Sequential reader for rawConvertible Paimon splits. Opens files lazily,
/// delegates per-file preparation and reading to FileSplitReader, and advances
/// past exhausted or skipped files.
class PaimonSplitReader : public FileSplitReader {
 public:
  /// @param fileSplit FileConnectorSplit for the first data file.
  /// @param paimonSplit Complete logical split. The caller checks raw-read
  ///        eligibility and rejects deletion vectors and changelog files.
  /// @param partitionKeys Borrowed partition-column handles for the scan,
  ///        distinct from the partition values in paimonSplit.
  PaimonSplitReader(
      const std::shared_ptr<const FileConnectorSplit>& fileSplit,
      std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
      const std::shared_ptr<const PaimonTableHandle>& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
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

  /// Rebinds the borrowed counters when the reader moves to another data source.
  void setRuntimeStats(dwio::common::RuntimeStats& runtimeStats) {
    runtimeStats_ = &runtimeStats;
  }

 private:
  // Builds a FileConnectorSplit from a PaimonDataFile.
  std::shared_ptr<FileConnectorSplit> makeFileConnectorSplit(
      const PaimonDataFile& dataFile) const;

  // Reuses the current reader or lazily prepares the next non-empty file.
  // Returns false when no files remain.
  bool ensureFileSplitReader();

  // Cleans up the current file reader after it is exhausted and advances
  // currentFileIndex_ so ensureFileSplitReader() opens the next file.
  void finishFileSplitReader();

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;

  // FileConnectorSplits built from paimonSplit_->dataFiles(), one per file.
  // Used to switch the base reader between files during iteration.
  std::vector<std::shared_ptr<FileConnectorSplit>> fileSplits_;

  // Index of the current file, or the next file to prepare between readers.
  size_t currentFileIndex_{0};

  // Saved from prepareSplit() for re-use when advancing to subsequent files.
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  // Borrowed from the data source; rebound with setRuntimeStats() on takeover.
  dwio::common::RuntimeStats* runtimeStats_{nullptr};
};

} // namespace facebook::velox::connector::hive::paimon
