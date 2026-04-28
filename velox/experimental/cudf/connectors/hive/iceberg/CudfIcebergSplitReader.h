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
  /// Override to create the chunked parquet reader.
  void createCudfReader(rmm::device_async_resource_ref output_mr) override;

  /// Override to apply Iceberg deletes after reading a cudf table chunk.
  std::optional<std::unique_ptr<cudf::table>> readNextChunk(
      rmm::device_async_resource_ref output_mr) override;

 private:
  /// Determine the memory resource to construct the `chunked_parquet_reader`
  /// and when calling `readNextChunk` for the hybrid scan reader.
  rmm::device_async_resource_ref determineCudfMemoryResource();

  /// Setup delete file readers for positional deletes, equality deletes, and
  /// deletion vectors.
  /// @param runtimeStats Reference to the DataSource's runtime statistics,
  /// passed to delete file readers for stats accumulation.
  void setupDeleteFileReaders(dwio::common::RuntimeStatistics& runtimeStats);

  /// Apply deletion vector (V3) to the input cudf table.
  void applyDeletionVector(cudf::table_view input);

  /// Apply positional deletes (V2) to the input cudf table.
  void applyPositionalDeletes(cudf::table_view input);

  /// Apply equality deletes (V2) to the input cudf table.
  void applyEqualityDeletes(cudf::table_view input);

  /// Setup column projection to include any equality delete key columns
  /// that are not already in the output projection.
  void setupColumnProjection();

  /// Adapts the data file schema to match the table schema expected by the
  /// query.
  ///
  /// This method reconciles differences between the physical data file schema
  /// and the logical table schema, handling various scenarios where columns may
  /// be missing, added, or need special treatment.
  ///
  /// Classifies each output column into one of:
  ///
  /// 1. Info columns:
  ///    Column is a synthesized info column (e.g., $file_size) from the
  ///    split metadata. Recorded for post-read injection as a constant.
  ///
  /// 2. Columns present in File:
  ///    Column exists in the parquet file and will be read normally.
  ///    Type coercion is handled by the cudf parquet reader options.
  ///
  /// 3. Columns missing from File:
  ///    a) Partition columns (Hive-migrated tables):
  ///       Column is a partition key in the Iceberg split. In Hive-written
  ///       Iceberg tables, partition column values are stored in partition
  ///       metadata, not in the data file itself. Recorded for post-read
  ///       injection as a constant.
  ///    b) Schema evolution (newly added columns):
  ///       Column was added to the table schema after this data file was
  ///       written. Recorded for post-read injection as a typed NULL column.
  ///
  /// Columns are recorded in injectedColumns_ and removed from readColumnNames_
  /// so they are injected after reading using `injectMissingColumns()`.
  void adaptColumns();

  /// Inject partition columns and schema-evolution NULL columns into the
  /// cudf table after reading. Returns a new table with all output columns
  /// in the correct order.
  std::unique_ptr<cudf::table> injectMissingColumns(
      std::unique_ptr<cudf::table>&& table,
      rmm::device_async_resource_ref mr);

  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit_;
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;

  /// cuCollections-accelerated deletion vector (roaring bitmap) reader
  std::unique_ptr<CudfDeletionVectorReader> deletionVectorReader_;

  /// Positional delete file readers
  std::list<std::unique_ptr<velox_iceberg::PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;

  /// Equality delete file readers.
  std::list<std::unique_ptr<CudfEqualityDeleteFileReader>>
      equalityDeleteFileReaders_;

  /// Extra equality delete key columns appended to readColumnNames_ that
  /// are not part of the output projection.
  std::vector<std::string> extraEqualityColumns_;

  /// Describes a column that must be injected after reading because it is
  /// not present in the parquet file (partition column or schema evolution).
  struct InjectedColumn {
    size_t outputIndex; // position in the final output schema
    std::string name;
    std::optional<std::string>
        partitionValue; // nullopt = NULL (schema evolution)
    TypePtr veloxType;
  };

  /// Columns to inject after reading. Populated by adaptColumns().
  std::vector<InjectedColumn> injectedColumns_;

  /// Tracks the absolute row offset within the data file. Each chunk advances
  /// this by the number of rows read (before deletes).
  uint64_t baseReadOffset_{0};

  BufferPtr deleteBitmap_{nullptr};
  std::shared_ptr<rmm::device_buffer> deviceDeleteBitmap_;
  std::unique_ptr<cudf::column> deleteMask_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
