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

#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/common/Reader.h"
#include "velox/type/Type.h"

namespace facebook::velox::cache {
class AsyncDataCache;
}

namespace facebook::velox::connector {
class ConnectorQueryCtx;
}

namespace facebook::velox::dwio::common {
class BufferedInput;
}

namespace facebook::velox::connector::hive {

constexpr const char* kPath = "$path";
constexpr const char* kBucket = "$bucket";

struct HiveConnectorSplit;
class HiveTableHandle;
class HiveColumnHandle;

class SplitReader {
 public:
  static std::unique_ptr<SplitReader> create(
      std::shared_ptr<velox::connector::hive::HiveConnectorSplit> hiveSplit,
      std::shared_ptr<HiveTableHandle> hiveTableHandle,
      std::shared_ptr<common::ScanSpec> scanSpec,
      const RowTypePtr readerOutputType,
      std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
          partitionKeys,
      FileHandleFactory* fileHandleFactory,
      ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      std::shared_ptr<io::IoStatistics> ioStats);

  SplitReader(
      std::shared_ptr<velox::connector::hive::HiveConnectorSplit> hiveSplit,
      std::shared_ptr<HiveTableHandle> hiveTableHandle,
      std::shared_ptr<common::ScanSpec> scanSpec,
      const RowTypePtr readerOutputType,
      std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
          partitionKeys,
      FileHandleFactory* fileHandleFactory,
      ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      std::shared_ptr<io::IoStatistics> ioStats);

  virtual ~SplitReader() = default;

  /// This function is used by different table formats like Iceberg and Hudi to
  /// do additional preparations before reading the split, e.g. Open delete
  /// files or log files, and add column adapatations for metadata columns
  virtual void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats);

  virtual uint64_t next(int64_t size, VectorPtr& output);

  void resetFilterCaches();

  bool emptySplit() const;

  void resetSplit();

  int64_t estimatedRowSize() const;

  void updateRuntimeStats(dwio::common::RuntimeStatistics& stats) const;

  bool allPrefetchIssued() const;

  std::string toString() const;

 protected:
  // Different table formats may have different meatadata columns. This function
  // will be used to update the scanSpec for these columns.
  virtual std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const std::shared_ptr<const velox::RowType>& tableSchema);

  void setConstantValue(
      common::ScanSpec* FOLLY_NONNULL spec,
      const TypePtr& type,
      const velox::variant& value) const;

  void setNullConstantValue(
      common::ScanSpec* FOLLY_NONNULL spec,
      const TypePtr& type) const;

  void setPartitionValue(
      common::ScanSpec* FOLLY_NONNULL spec,
      const std::string& partitionKey,
      const std::optional<std::string>& value) const;

  std::shared_ptr<HiveConnectorSplit> hiveSplit_;
  std::shared_ptr<HiveTableHandle> hiveTableHandle_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  RowTypePtr readerOutputType_;
  std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
      partitionKeys_;
  std::unique_ptr<dwio::common::Reader> baseReader_;
  std::unique_ptr<dwio::common::RowReader> baseRowReader_;
  FileHandleFactory* const fileHandleFactory_;
  ConnectorQueryCtx* const connectorQueryCtx_;
  folly::Executor* const executor_;
  std::shared_ptr<io::IoStatistics> ioStats_;

 private:
  void configureReaderOptions();

  void configureRowReaderOptions(
      dwio::common::RowReaderOptions& options,
      const RowTypePtr& rowType);

  void parseSerdeParameters(
      const std::unordered_map<std::string, std::string>& serdeParameters);

  std::unique_ptr<dwio::common::BufferedInput> createBufferedInput(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOpts);

  dwio::common::ReaderOptions baseReaderOpts_;
  dwio::common::RowReaderOptions baseRowReaderOpts_;
  bool emptySplit_;
};

} // namespace facebook::velox::connector::hive
