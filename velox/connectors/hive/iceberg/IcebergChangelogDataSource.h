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
#include "velox/connectors/hive/iceberg/IcebergChangelogSplitInfo.h"
#include "velox/connectors/hive/iceberg/IcebergDataSource.h"

namespace facebook::velox::connector::hive::iceberg {

/// Wrapper data source for Iceberg changelog tables.
///
/// This class wraps an IcebergDataSource to provide changelog table support.
/// It delegates file reading to the wrapped IcebergDataSource (which reads
/// base table data) and transforms the output to changelog format with
/// operation, ordinal, snapshotid, and rowdata columns.
///
/// The wrapper pattern allows:
/// - Reusing existing IcebergDataSource for file reading
/// - Clean separation between data reading and changelog transformation
/// - Proper handling of base table schema vs changelog schema
class IcebergChangelogDataSource : public connector::DataSource {
 public:
  IcebergChangelogDataSource(
      std::unique_ptr<connector::DataSource> baseDataSource,
      RowTypePtr changelogOutputType,
      ColumnHandleMap assignments);

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override;

  std::optional<RowVectorPtr> next(uint64_t size, velox::ContinueFuture& future)
      override;

  const common::SubfieldFilters* getFilters() const override {
    return baseDataSource_->getFilters();
  }

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

  uint64_t getCompletedBytes() override;

  uint64_t getCompletedRows() override;

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

  bool allPrefetchIssued() const override;

  void setFromDataSource(
      std::unique_ptr<connector::DataSource> sourceUnique) override;

 private:
  /// Transforms data output to changelog format by wrapping data columns
  /// into rowdata and adding metadata columns (operation, ordinal, snapshotid)
  VectorPtr transformChangelogOutput(
      const VectorPtr& dataOutput,
      memory::MemoryPool* pool);

  /// Wrapped DataSource that reads base table data
  std::unique_ptr<connector::DataSource> baseDataSource_;

  /// Changelog output schema (operation, ordinal, snapshotid, rowdata)
  RowTypePtr changelogOutputType_;
  std::shared_ptr<ColumnHandleMap> changelogColumnHandles_;

  /// Save changelog split info from the split.
  std::shared_ptr<ChangelogSplitInfo> changelogSplitInfo_;
};

} // namespace facebook::velox::connector::hive::iceberg
