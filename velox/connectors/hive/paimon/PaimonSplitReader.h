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

/// Sequential multi-file reader for rawConvertible Paimon splits. Files are
/// opened lazily and advanced through EOF or legal empty files. The required
/// PaimonTableHandle supplies the fixed schema checked against each footer
/// before column adaptation; split values supply the partition constants.
/// Deletion vectors and changelog files are rejected before reading.
class PaimonSplitReader : public FileSplitReader {
 public:
  /// @param fileSplit FileConnectorSplit for the first data file (set by
  ///        FileDataSource::addSplit before calling createSplitReader).
  /// @param paimonSplit The full Paimon split. PaimonSplitReader builds
  ///        FileConnectorSplits for all data files internally. The caller must
  ///        validate the table read profile and split before construction.
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

  /// Saves metadata filter and runtime stats for use by
  /// ensureFileSplitReader(). Each footer is validated when its file is opened.
  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStats& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  /// Reads from the current file. When a file is exhausted, transparently
  /// advances to the next file in the split. Returns 0 only when all files
  /// are exhausted.
  uint64_t next(uint64_t size, VectorPtr& output) override;

  /// Rebinds the borrowed counters when a prepared data source is adopted.
  void setRuntimeStats(dwio::common::RuntimeStats& runtimeStats) {
    runtimeStats_ = &runtimeStats;
  }

 protected:
  void configureBaseReaderOptions() override;

 private:
  const std::shared_ptr<const PaimonTableHandle> paimonTableHandle_;

  // Builds a FileConnectorSplit from a PaimonDataFile.
  std::shared_ptr<FileConnectorSplit> makeFileConnectorSplit(
      const PaimonDataFile& dataFile) const;

  // Ensures a file split reader is ready to produce rows. If the current
  // reader is exhausted, advances to the next non-empty file. Returns true
  // if a reader is ready, false when all files are exhausted.
  bool ensureFileSplitReader();

  // Checks the current same-schema physical layout before column adaptation.
  void checkFileSchema(const RowTypePtr& fileType) const;

  // Cleans up the current file reader after it is exhausted and advances
  // currentFileIndex_ so ensureFileSplitReader() opens the next file.
  void finishFileSplitReader();

  // Partition handles must outlive a prepared data source's donor.
  const std::unordered_map<std::string, FileColumnHandlePtr>
      ownedPartitionKeys_;
  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;

  // FileConnectorSplits built from paimonSplit_->dataFiles(), one per file.
  // Used to switch the base reader between files during iteration.
  std::vector<std::shared_ptr<FileConnectorSplit>> fileSplits_;

  // Index of the next data file to open.
  size_t currentFileIndex_{0};

  // Saved from prepareSplit() for re-use when advancing to subsequent files.
  std::shared_ptr<common::MetadataFilter> metadataFilter_;
  dwio::common::RuntimeStats* runtimeStats_{nullptr};
};

} // namespace facebook::velox::connector::hive::paimon
