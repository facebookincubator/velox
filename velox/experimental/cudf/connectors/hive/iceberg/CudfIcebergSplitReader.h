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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfEqualityDeleteFileReader.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

/// GPU-accelerated per-split reader for Iceberg tables.
///
/// Derives from `CudfSplitReader` and adds Iceberg delete semantics:
///   - Deletion vectors (V3): Roaring bitmap blob read via
///     `CudfDeletionVectorReader` and applied on GPU using cuco and cudf
///   - Positional deletes (V2): Host-side bitmap read via upstream
///     `PositionalDeleteFileReader` and applied on GPU using cudf
///   - Equality deletes (V2): Read via `CudfEqualityDeleteFileReader`
///     and applied on GPU using cudf
class CudfIcebergSplitReader : public CudfSplitReader {
 public:
  CudfIcebergSplitReader(
      std::shared_ptr<CudfHiveConnectorSplit> split,
      std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit,
      std::shared_ptr<const velox_hive::HiveTableHandle> tableHandle,
      const RowTypePtr& outputType,
      const std::vector<std::string>& readColumnNames,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      bool useExperimentalCudfReader,
      cudf::ast::expression const* subfieldFilterExpr);

  /// Override to setup delete file readers and column projection.
  /// @param runtimeStats Reference to the DataSource's runtime statistics,
  /// passed through to delete file readers so they can accumulate stats
  /// directly into the DataSource's stats object.
  void prepareSplit(dwio::common::RuntimeStatistics& runtimeStats) override;

 protected:
  // Override to create the chunked parquet reader.
  void createCudfReader(rmm::device_async_resource_ref output_mr) override;

  // Override to apply Iceberg deletes after reading a cudf table chunk.
  std::optional<cudf::io::table_with_metadata> readNextChunk(
      rmm::device_async_resource_ref output_mr) override;

 private:
  // Determine the memory resource to construct the `chunked_parquet_reader`
  // and when calling `readNextChunk` for the hybrid scan reader.
  rmm::device_async_resource_ref determineCudfMemoryResource();

  // Setup delete file readers for positional deletes, equality deletes, and
  // deletion vectors.
  // @param runtimeStats Reference to the DataSource's runtime statistics,
  // passed to delete file readers for stats accumulation.
  void setupDeleteFileReaders(dwio::common::RuntimeStatistics& runtimeStats);

  // Apply deletion vector (V3) to the input cudf table.
  void applyDeletionVector(cudf::table_view input);

  // Apply positional deletes (V2) to the input cudf table.
  void applyPositionalDeletes(cudf::table_view input);

  // Apply equality deletes (V2) to the input cudf table.
  void applyEqualityDeletes(cudf::table_view input);

  // Setup column projection to include any equality delete key columns
  // that are not already in the output projection.
  void setupColumnProjection();

  // Adapts the data file schema to match the table schema expected by the
  // query.
  //
  // This method reconciles differences between the physical data file schema
  // and the logical table schema, handling various scenarios where columns may
  // be missing, added, or need special treatment.
  //
  // Classifies each output column into one of:
  //
  // 1. Info columns:
  //    Column is synthesized using a constant value (e.g., $file_size) from
  //    the split metadata. Recorded for post-read injection as a constant.
  //
  // 2. Regular columns present in file:
  //    Column exists in both fileType and readerOutputType. Type is adapted
  //    from fileType to match the expected output type, handling schema
  //    evolution where column types may have changed.
  //
  // 3. Columns missing from file:
  //    a) Partition columns (Hive-migrated tables):
  //       Column is a partition key in the Iceberg split.
  //       In Hive-written Iceberg tables, partition column values are stored
  //       in partition metadata, not in the data file itself. Value is read
  //       from partition metadata and set as a constant.
  //    b) Schema evolution (newly added columns - NOT detected here):
  //       Column was added to the table schema after this data file was
  //       written. Injected as NULL constant since the old file doesn't contain
  //       this column by `buildOutputTable()` after reading the table's
  //       schema.
  void adaptColumns();

  // Assemble the final output table. Reorder read columns, inject
  // info/partition constants, fill typed NULL columns for schema-evolution
  // columns and drop equality delete key columns.
  std::unique_ptr<cudf::table> buildOutputTable(
      std::unique_ptr<cudf::table>&& table,
      rmm::device_async_resource_ref mr);

  // Lazily builds `fileColumnNames_` and `fileColumnIndex_` from the cuDF
  // table's schema info for reuse across chunks
  void ensureFileColumnIndex(
      const std::vector<cudf::io::column_name_info>& schemaInfo);

  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit_;
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;

  // cuDF-accelerated reader for Iceberg V3 deletion vector (Puffin-encoded
  // roaring bitmaps).
  std::unique_ptr<CudfDeletionVectorReader> deletionVectorReader_;

  // Positional delete file readers
  std::list<std::unique_ptr<velox_iceberg::PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;

  // Equality delete file readers.
  std::list<std::unique_ptr<CudfEqualityDeleteFileReader>>
      equalityDeleteFileReaders_;

  // Extra equality delete key columns appended to readColumnNames_ that
  // are not part of the output projection.
  std::vector<std::string> extraEqualityColumns_;

  // Describes a column whose value is known from the split metadata (an info
  // column or a Hive-migrated partition column) and is injected as a constant
  // after reading the table.
  struct InjectedColumn {
    std::string value; // info-column / partition-key string value
    TypePtr veloxType;
  };

  // Columns to inject after reading, keyed by output column name.
  // This is populated by `adaptColumns()`.
  std::unordered_map<std::string, InjectedColumn> injectedColumns_;

  // Column names from the cudf reader, in reader order, and a name ->
  // position lookup over them.
  std::vector<std::string> fileColumnNames_;
  std::unordered_map<std::string, size_t> fileColumnIndex_;

  // Tracks the absolute row offset within the data file. Each chunk advances
  // this by the number of rows read (before deletes).
  uint64_t baseReadOffset_{0};

  BufferPtr deleteBitmap_{nullptr};
  std::shared_ptr<rmm::device_buffer> deviceBitmap_;
  std::unique_ptr<cudf::column> deleteMask_;
  // Current mutable view into the deleteMask_ column.
  cudf::mutable_column_view deleteMaskView_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
