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

#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::hive::paimon {

/// Concatenates ordinary raw files, preserving duplicate rows. Each physical
/// reader owns a fresh ScanSpec, reader options and DWIO lifecycle. At most one
/// physical reader is active, and empty/filtered files yield between advances.
class PaimonSplitReader final : public FileScanReader {
 public:
  PaimonSplitReader(
      std::shared_ptr<const PaimonConnectorSplit> split,
      std::shared_ptr<const PaimonTableHandle> table,
      std::unordered_map<std::string, std::optional<std::string>>
          partitionValues,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* ctx,
      std::shared_ptr<const FileConfig> config,
      RowTypePtr outputType,
      std::shared_ptr<io::IoStatistics> dataIoStats,
      std::shared_ptr<io::IoStatistics> metadataIoStats,
      std::shared_ptr<IoStats> ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      dwio::common::RuntimeStats& stats,
      std::function<FileScanState()> makeScanState,
      folly::F14FastSet<std::string> remainingFilterColumns);

  ScanReadResult next(uint64_t maxRows, ContinueFuture& future) override;
  void resetFilterCaches() override;
  void addDynamicFilter(
      column_index_t channel,
      const std::shared_ptr<common::Filter>& filter) override;
  void updateRuntimeStats(dwio::common::RuntimeStats& stats) const override;
  int64_t estimatedRowSize() const override;
  bool allPrefetchIssued() const override;
  void resetSplit() override;
  void cancel() override;
  void setConnectorQueryCtx(const ConnectorQueryCtx* ctx) override;
  const FileConnectorSplit* currentFileSplit() const override;

 private:
  class PhysicalReader;
  void openFile();
  void finishFile();

  const std::shared_ptr<const PaimonConnectorSplit> split_;
  const std::shared_ptr<const PaimonTableHandle> table_;
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionValues_;
  const std::unordered_map<std::string, FileColumnHandlePtr>* const
      partitionKeys_;
  const ConnectorQueryCtx* const ctx_;
  const std::shared_ptr<const FileConfig> config_;
  const RowTypePtr outputType_;
  const std::shared_ptr<io::IoStatistics> dataIoStats_;
  const std::shared_ptr<io::IoStatistics> metadataIoStats_;
  const std::shared_ptr<IoStats> ioStats_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  dwio::common::RuntimeStats& stats_;
  const std::function<FileScanState()> makeScanState_;
  const folly::F14FastSet<std::string> remainingFilterColumns_;

  size_t fileIndex_{0};
  std::shared_ptr<FileConnectorSplit> fileSplit_;
  FileScanState state_;
  std::unique_ptr<FileSplitReader> reader_;
  VectorPtr output_;
  RowVectorPtr emptyOutput_;
};

} // namespace facebook::velox::connector::hive::paimon
