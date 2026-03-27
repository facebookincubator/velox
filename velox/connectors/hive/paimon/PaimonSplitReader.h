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

#include "velox/connectors/hive/SplitReader.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::hive::paimon {

/// Paimon-specific split reader that handles multi-file splits.
///
/// A single PaimonConnectorSplit may contain multiple data files (one per
/// LSM-tree level in a bucket). This reader iterates through them
/// sequentially, opening each file's reader when the previous one is
/// exhausted. Currently only supports append-only tables with
/// rawConvertible=true, where files can be read independently without
/// merge-on-read.
class PaimonSplitReader : public SplitReader {
 public:
  PaimonSplitReader(
      const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
      const std::shared_ptr<const PaimonConnectorSplit>& paimonSplit,
      const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<const HiveColumnHandle>>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  ~PaimonSplitReader() override = default;

  /// Creates a HiveConnectorSplit from a PaimonDataFile. Used by
  /// PaimonDataSource::addSplit() to convert the first file for the parent
  /// class, and internally when advancing to subsequent files.
  static std::shared_ptr<HiveConnectorSplit> toHiveConnectorSplit(
      const PaimonConnectorSplit& paimonSplit,
      const PaimonDataFile& dataFile);

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

 private:
  /// Validates that a data file is readable in the current mode (append-only).
  static void validateDataFile(const PaimonDataFile& dataFile);

  /// Opens the reader for the next data file. Resets the base reader/row
  /// reader and calls createReader/createRowReader for the new file.
  void openNextFile();

  const std::shared_ptr<const PaimonConnectorSplit> paimonSplit_;

  /// Index of the data file currently being read. Starts at 0 (first file
  /// is opened by the base class via prepareSplit).
  size_t currentFileIndex_{0};
};

} // namespace facebook::velox::connector::hive::paimon
