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

#include <optional>

#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergChangelogSplitReader.h"

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg-specific data source that extends HiveDataSource.
///
/// Provides Iceberg table format support by creating IcebergSplitReader
/// instances that handle positional delete files, schema evolution, and
/// Iceberg-specific metadata columns.
///
/// When the table handle has isChangelogQuery() == true, createSplitReader()
/// instantiates an IcebergChangelogSplitReader that reads data columns and
/// wraps each batch into the changelog output schema
/// (operation, ordinal, snapshotid, rowdata).
///
/// Changelog column demands are prepared once. Each split gets a fresh scan
/// spec, with filter selectivity carried over from the previous split.
class IcebergDataSource : public HiveDataSource {
 public:
  IcebergDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& assignments,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* ioExecutor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig);

  /// For changelog queries, intercepts dynamic filters on the three constant
  /// changelog columns (operation/ordinal/snapshotid) and accumulates them in
  /// changelogDynamicFilters_ so they can be applied at split-skipping time in
  /// IcebergChangelogSplitReader::prepareSplit(). rowdata dynamic filters are
  /// dropped (ROW-typed columns never produce pushable filters from HashProbe,
  /// so this is unreachable in practice; if somehow reached, forwarding to base
  /// would silently corrupt results). For non-changelog queries delegates to
  /// FileDataSource::addDynamicFilter().
  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

 protected:
  /// Creates an IcebergSplitReader (regular) or IcebergChangelogSplitReader
  /// (changelog) depending on the table handle's isChangelogQuery() flag.
  std::unique_ptr<FileSplitReader> createSplitReader() override;

 private:
  /// Column handles for the output columns as passed to the constructor.
  /// For regular queries these are the data column handles; for changelog
  /// queries these are the changelog output column handles
  /// (operation/ordinal/snapshotid/rowdata).
  std::shared_ptr<ColumnHandleMap> columnHandles_;

  /// Changelog-only column demands and current physical scan spec (nullopt
  /// for regular queries).
  std::optional<ChangelogScanContext> changelogScanContext_;

  /// Changelog-only: dynamic filters on the constant changelog columns
  /// (operation/ordinal/snapshotid) accumulated via addDynamicFilter().
  /// Passed by pointer into each IcebergChangelogSplitReader so that filters
  /// injected by HashProbe at runtime are applied during split-level skipping.
  common::SubfieldFilters changelogDynamicFilters_;
};

} // namespace facebook::velox::connector::hive::iceberg
