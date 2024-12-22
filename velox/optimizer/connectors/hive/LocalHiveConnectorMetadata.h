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

#include "velox/common/base/Fs.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"
#include "velox/optimizer/connectors/hive/HiveConnectorMetadata.h"

namespace facebook::velox::connector::hive {

class LocalHiveSplitSource : public SplitSource {
 public:
  LocalHiveSplitSource(
      std::vector<std::string> files,
      int32_t splitsPerFile,
      dwio::common::FileFormat format,
      const std::string& connectorId)
      : splitsPerFile_(splitsPerFile),
        format_(format),
        connectorId_(connectorId),
        files_(files) {}

  std::vector<SplitSource::SplitAndGroup> getSplits(
      uint64_t targetBytes) override;

 private:
  const int32_t splitsPerFile_;
  const dwio::common::FileFormat format_;
  const std::string connectorId_;
  std::vector<std::string> files_;
  std::vector<std::shared_ptr<connector::ConnectorSplit>> fileSplits_;
  int32_t currentFile_{-1};
  int32_t currentSplit_{0};
};

class LocalHiveConnectorMetadata;

class LocalHiveSplitManager : public ConnectorSplitManager {
 public:
  LocalHiveSplitManager(LocalHiveConnectorMetadata* metadata)
      : metadata_(metadata) {}
  std::vector<std::shared_ptr<const PartitionHandle>> listPartitions(
      const ConnectorTableHandlePtr& tableHandle) override;

  std::shared_ptr<SplitSource> getSplitSource(
      const ConnectorTableHandlePtr& tableHandle,
      std::vector<std::shared_ptr<const PartitionHandle>> partitions) override;

 private:
  LocalHiveConnectorMetadata* metadata_;
};

/// A HiveTableLayout backed by local files. Implements sampling by reading
/// local files and stores the file list inside 'this'.
class LocalHiveTableLayout : public HiveTableLayout {
 public:
  LocalHiveTableLayout(
      const std::string& name,
      const Table* table,
      connector::Connector* connector,
      std::vector<const Column*> columns,
      std::vector<const Column*> partitioning,
      std::vector<const Column*> orderColumns,
      std::vector<SortOrder> sortOrder,
      std::vector<const Column*> lookupKeys,
      std::vector<const Column*> hivePartitionColumns,
      dwio::common::FileFormat fileFormat,
      std::optional<int32_t> numBuckets = std::nullopt)
      : HiveTableLayout(
            name,
            table,
            connector,
            columns,
            partitioning,
            orderColumns,
            sortOrder,
            lookupKeys,
            hivePartitionColumns,
            fileFormat,
            numBuckets) {}

  std::pair<int64_t, int64_t> sample(
      const connector::ConnectorTableHandlePtr& handle,
      float pct,
      std::vector<core::TypedExprPtr> extraFilters,
      const std::vector<common::Subfield>& fields,
      HashStringAllocator* allocator = nullptr,
      std::vector<ColumnStatistics>* statistics = nullptr) const override;

  const std::vector<std::string>& files() const {
    return files_;
  }

  void setFiles(std::vector<std::string> files) {
    files_ = std::move(files);
  }

  /// Like sample() above, but fills 'builders' with the data.
  std::pair<int64_t, int64_t> sample(
      const connector::ConnectorTableHandlePtr& handle,
      float pct,
      const std::vector<common::Subfield>& fields,
      HashStringAllocator* allocator,
      std::vector<std::unique_ptr<dwrf::StatisticsBuilder>>* statsBuilders)
      const;

 private:
  std::vector<std::string> files_;
};

class LocalTable : public Table {
 public:
  LocalTable(const std::string& name, dwio::common::FileFormat format)
      : Table(name) {}

  std::unordered_map<std::string, std::unique_ptr<Column>>& columns() {
    return columns_;
  }
  const std::vector<const TableLayout*>& layouts() const override {
    return exportedLayouts_;
  }

  const std::unordered_map<std::string, const Column*>& columnMap()
      const override;

  void setType(const RowTypePtr& type) {
    type_ = type;
  }

  void makeDefaultLayout(
      std::vector<std::string> files,
      LocalHiveConnectorMetadata& metadata);

  uint64_t numRows() const override {
    return numRows_;
  }

  /// Samples  'samplePct' % rows of the table and sets the num distincts
  /// estimate for the columns. uses 'pool' for temporary data.
  void sampleNumDistincts(float samplePct, memory::MemoryPool* pool);

 private:
  // Serializes initialization, e.g. exportedColumns_.
  mutable std::mutex mutex_;

  // All columns. Filled by loadTable().
  std::unordered_map<std::string, std::unique_ptr<Column>> columns_;

  // Non-owning columns map used for exporting the column set as abstract
  // columns.
  mutable std::unordered_map<std::string, const Column*> exportedColumns_;

  ///  Table layouts. For a Hive table this is normally one layout with all
  ///  columns included.
  std::vector<std::unique_ptr<TableLayout>> layouts_;

  // Copy of 'llayouts_' for use in layouts().
  std::vector<const TableLayout*> exportedLayouts_;

  int64_t numRows_{0};
  int64_t numSampledRows_{0};

  friend class LocalHiveConnectorMetadata;
};

class LocalHiveConnectorMetadata : public HiveConnectorMetadata {
 public:
  LocalHiveConnectorMetadata(HiveConnector* hiveConector);

  void initialize() override;

  const Table* findTable(const std::string& name) override;

  ConnectorSplitManager* splitManager() override {
    return &splitManager_;
  }

  dwio::common::FileFormat fileFormat() const {
    return format_;
  }

  const std::shared_ptr<ConnectorQueryCtx>& connectorQueryCtx() const {
    return connectorQueryCtx_;
  }

  HiveConnector* hiveConnector() const {
    return hiveConnector_;
  }

  /// returns the set of known tables. This is not part of the
  /// ConnectorMetadata API. This This is only needed for running the
  /// DuckDB parser on testing queries since the latter needs a set of
  /// tables for name resolution.
  const std::unordered_map<std::string, std::unique_ptr<LocalTable>>& tables()
      const {
    return tables_;
  }

 private:
  void makeQueryCtx();
  void makeConnectorQueryCtx();
  void readTables(const std::string& path);

  void loadTable(const std::string& tableName, const fs::path& tablePath);

  std::shared_ptr<HiveConfig> hiveConfig_;
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> schemaPool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::shared_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  dwio::common::FileFormat format_;
  std::unordered_map<std::string, std::unique_ptr<LocalTable>> tables_;
  LocalHiveSplitManager splitManager_;
};

} // namespace facebook::velox::connector::hive
