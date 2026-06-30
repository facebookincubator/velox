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

#include "velox/connectors/hive/HiveDataSource.h"

namespace facebook::velox::connector::hive::iceberg {

/// Iceberg-specific data source that extends HiveDataSource.
///
/// Provides Iceberg table format support by creating IcebergSplitReader
/// instances that handle positional delete files, schema evolution, and
/// Iceberg-specific metadata columns.
///
/// When the table handle has isChangelogQuery() == true, createSplitReader()
/// builds a data-scoped readerOutputType / scanSpec from the table handle's
/// dataColumnHandles and instantiates an IcebergChangelogSplitReader that
/// transforms each base-table batch into the changelog output schema
/// (operation, ordinal, snapshotid, rowdata).  No separate data-source class
/// is required for changelog queries.
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
};

} // namespace facebook::velox::connector::hive::iceberg
