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
      const HiveColumnHandleMap* partitionKeys,
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
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(
      uint64_t size,
      VectorPtr& output,
      dwio::common::RuntimeStatistics& runtimeStats) override;

 private:
  /// Adapts the data file schema to match the table schema expected by the
  /// query.
  ///
  /// This method reconciles differences between the physical data file schema
  /// and the logical table schema, handling various scenarios where columns may
  /// be missing, added, or need special treatment.
  ///
  /// @param fileType The schema read from the data file's metadata. This
  /// represents the actual columns physically present in the Parquet/ORC file.
  /// @param tableSchema The logical schema defined in the catalog (e.g., from
  /// DDL). This represents the current table schema that queries expect.
  ///
  /// @return A vector of column types adapted to match the query's
  /// expectations, with appropriate type conversions and constant values set
  /// for missing or special columns.
  ///
  /// The method handles the following scenarios for each column in the scan
  /// spec:
  ///
  /// 1. Info columns (e.g., $path, $data_sequence_number, $deleted)
  ///    These are virtual columns that provide metadata about the file itself.
  ///    Values are read from hiveSplit_->infoColumns map and set as constant
  ///    values in the scanSpec so they're materialized for all rows.
  ///
  /// 2. Regular columns present in File:
  ///    Column exists in both fileType and readerOutputType. Type is adapted
  ///    from fileType to match the expected output type, handling schema
  ///    evolution where column types may have changed.
  ///
  /// 3. Columns missing from File:
  ///    a) Partition columns (hive-migrated tables):
  ///       Column is marked as partition key in hiveSplit_->partitionKeys.
  ///       In Hive-written Iceberg tables, partition column values are stored
  ///       in partition metadata, not in the data file itself. Value is read
  ///       from partition metadata and set as a constant.
  ///    b) Schema evolution (newly added columns):
  ///       Column was added to the table schema after this data file was
  ///       written. Set as NULL constant since the old file doesn't contain
  ///       this column.
  std::vector<TypePtr> adaptColumns(
      const RowTypePtr& fileType,
      const RowTypePtr& tableSchema) const override;

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
