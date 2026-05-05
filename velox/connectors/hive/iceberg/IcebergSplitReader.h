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

#include <folly/container/F14Set.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

namespace facebook::velox::connector::hive::iceberg {

struct HiveIcebergSplit;
struct IcebergDeleteFile;

class IcebergSplitReader : public FileSplitReader {
 public:
  IcebergSplitReader(
      const std::shared_ptr<const HiveIcebergSplit>& icebergSplit,
      const FileTableHandlePtr& tableHandle,
      const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const RowTypePtr& readerOutputType,
      const std::shared_ptr<io::IoStatistics>& dataIoStats,
      const std::shared_ptr<io::IoStatistics>& metadataIoStats,
      const std::shared_ptr<IoStats>& ioStats,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const std::shared_ptr<common::ScanSpec>& scanSpec);

  ~IcebergSplitReader() override = default;

  void prepareSplit(
      std::shared_ptr<common::MetadataFilter> metadataFilter,
      dwio::common::RuntimeStatistics& runtimeStats,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override;

  uint64_t next(uint64_t size, VectorPtr& output) override;

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
  ///    Values are read from the split's infoColumns map and set as constant
  ///    values in the scanSpec so they're materialized for all rows.
  ///
  /// 2. Regular columns present in File:
  ///    Column exists in both fileType and readerOutputType. Type is adapted
  ///    from fileType to match the expected output type, handling schema
  ///    evolution where column types may have changed.
  ///
  /// 3. Columns missing from File:
  ///    a) Partition columns (hive-migrated tables):
  ///       Column is marked as partition key in splitPartitionKeys_.
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

  // Resolves the equality field IDs of an equality-delete file to the
  // corresponding column names and types in the table schema. In Iceberg,
  // field IDs for top-level columns are assigned sequentially starting from
  // 1, matching the column order in the table schema.
  std::pair<std::vector<std::string>, std::vector<TypePtr>>
  resolveEqualityColumns(const IcebergDeleteFile& deleteFile) const;

  // Discovers equality-delete columns that are not in the user's projection
  // and augments 'scanSpec_' and 'readerOutputType_' so they are physically
  // read and made available in the output RowVector. For partition columns
  // the partition value is set as a constant on the scan-spec child so the
  // augmentation works regardless of whether the data file physically
  // contains the partition column. Augmented columns are appended at the end
  // of 'readerOutputType_' so the upstream FileDataSource's positional
  // projection naturally drops them from the operator output.
  void configureEqualityDeleteColumns();

  // Names of scan-spec children that 'configureEqualityDeleteColumns'
  // pre-installed a partition-value constant on for the current split.
  // Mirrors the Java 'PARTITION_KEY' column-type distinction in
  // 'IcebergUtil.getColumns(fields, schema, partitionSpec, typeManager)':
  // for an equality-delete column that is also an identity-partition
  // column, the value MUST come from the split's partition metadata, never
  // from the data file body — even if the file body physically contains
  // the column. 'adaptColumns' uses this set to skip clearing the partition
  // constant on Branch 1, which would otherwise leave the column stuck as
  // a constant null and break equality-delete matching after schema
  // evolution adds new columns at indices that shift the augmented column
  // out of its file-natural position.
  folly::F14FastSet<std::string> equalityAugmentedPartitionColumns_;

  const std::shared_ptr<const HiveIcebergSplit> icebergSplit_;

  /// Read offset to the beginning of the split in number of rows for the
  /// current batch for the base data file.
  uint64_t baseReadOffset_;
  /// File position for the first row in the split.
  uint64_t splitOffset_;
  std::list<std::unique_ptr<PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;
  BufferPtr deleteBitmap_;

  /// Readers for Iceberg V3 deletion vectors (Puffin-encoded roaring bitmaps).
  std::list<std::unique_ptr<DeletionVectorReader>> deletionVectorReaders_;

  /// Readers for equality delete files.
  std::list<std::unique_ptr<EqualityDeleteFileReader>>
      equalityDeleteFileReaders_;
};
} // namespace facebook::velox::connector::hive::iceberg
