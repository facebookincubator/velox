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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/SplitReader.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

namespace facebook::velox::connector::hive::iceberg {

struct IcebergDeleteFile;

class IcebergSplitReader : public SplitReader {
 public:
  IcebergSplitReader(
      const std::shared_ptr<const hive::HiveConnectorSplit>& hiveSplit,
      const HiveTableHandlePtr& hiveTableHandle,
      const std::unordered_map<std::string, HiveColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& ioStats,
      const std::shared_ptr<filesystems::File::IoStats>& fsStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  ~IcebergSplitReader() override = default;

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats) override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

  std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const std::shared_ptr<const velox::RowType>& tableSchema) const override;

 private:
  // The read offset to the beginning of the split in number of rows for the
  // current batch for the base data file
  uint64_t baseReadOffset_;
  // The file position for the first row in the split
  uint64_t splitOffset_;
  std::list<std::unique_ptr<PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;
  BufferPtr deleteBitmap_;
};
} // namespace facebook::velox::connector::hive::iceberg
