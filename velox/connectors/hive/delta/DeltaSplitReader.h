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
#include "velox/connectors/hive/HiveSplitReader.h"

namespace facebook::velox::connector::hive::delta {

class DeltaSplitReader : public HiveSplitReader {
 public:
  DeltaSplitReader(
      const std::shared_ptr<const HiveConnectorSplit>& hiveSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      const std::unordered_map<std::string, FileColumnHandlePtr>* infoColumns,
      std::vector<column_index_t> bucketChannels = {},
      const common::SubfieldFilters* subfieldFiltersForValidation = nullptr);

  ~DeltaSplitReader() override = default;

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

 private:
  /// Reconciles the physical file schema with the logical table schema.
  ///
  /// Sets constant values for info columns ($path, $file_size) and partition
  /// columns, maps present columns by name, and fills missing columns (schema
  /// evolution) with NULL constants.
  std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const override;
};

} // namespace facebook::velox::connector::hive::delta

