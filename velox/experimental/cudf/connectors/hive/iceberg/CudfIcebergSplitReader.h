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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergExpressionTransformers.hpp"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

#include <list>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
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

 protected:
  // Sets up delete file readers and column projection after base state reset.
  void prepareSplitInternal(
      dwio::common::RuntimeStatistics& runtimeStats) override;

  // Override to only setup cuDF reader if we have columns to read.
  void setupReader() override;

  // Returns the filter to push down to the cuDF reader.
  cudf::ast::expression const* pushdownFilter() const override;

  // Override to determine the memory resource to construct cuDF reader.
  rmm::device_async_resource_ref determineCudfMemoryResource() const override;

  // Override to apply Iceberg deletes after reading a cudf table chunk.
  std::optional<std::unique_ptr<cudf::table>> readNextChunk() override;

 private:
  // Clear delete readers and column injection
  void resetSplit();

  // Setup delete file readers for positional and equality deletes,
  // and deletion vectors.
  // @param runtimeStats DataSource's runtime statistics, passed to delete
  // file readers for accumulation.
  void setupDeleteFileReaders(dwio::common::RuntimeStatistics& runtimeStats);

  // Applies deletion vector (V3).
  void applyDeletionVector(cudf::column_view rowIndex);

  // Applies positional deletes (V2).
  void applyPositionalDeletes(
      std::size_t startRow,
      std::size_t numRows,
      cudf::column_view rowIndex);

  // Reads positional delete positions for a file row range.
  void readPositionalDeleteBitmap(std::size_t startRow, std::size_t numRows);

  // Apply equality deletes (V2).
  void applyEqualityDeletes(cudf::table_view input);

  // Returns whether cuDF must prepend absolute file row positions.
  bool needPrependedRowIndex() const;

  // Prepares a filter over columns read from the data file.
  void prepareSubfieldFilter();

  // Removes and returns the prepended row-index column.
  std::unique_ptr<cudf::column> extractRowIndex(
      std::unique_ptr<cudf::table>& table) const;

  // Creates contiguous absolute file row positions.
  std::unique_ptr<cudf::column> makeRowIndex(cudf::size_type numRows) const;

  // Returns the file row range covered by `rowIndex`.
  std::pair<std::size_t, std::size_t> rowRange(
      cudf::column_view rowIndex) const;

  // Setup column projection to include any equality delete key columns
  // that are not already in the output projection.
  void setupEqualityColumnKeys();

  // Read metadata and cache `splitRowCount_` and `fileColumnNames_`
  void cacheSchemaFromMetadata();

  // Returns the row range covered by the split.
  std::pair<std::size_t, std::size_t> computeSplitRowRange() const;

  // Adapts the data file schema to match the table schema expected by the
  // query. Classifies each output and filter-only column into one of:
  //
  // 1. Info columns:
  //    Synthesized from split metadata (e.g. $file_size). Recorded for
  //    post-read injection as a constant.
  //
  // 2. Partition columns (Hive-migrated tables):
  //    Value comes from the split's `partitionKeys`, not the data file.
  //    Recorded for post-read injection as a constant.
  //
  // 3. Columns missing from the file (schema evolution):
  //    Newly added columns absent from `fileColumnNames_`. Recorded for
  //    post-read injection as a typed NULL.
  //
  // 4. Columns present in the file:
  //    Left in `readColumnNames_` for the parquet reader.
  //
  // Injected names (1-3) are removed from `readColumnNames_`. `outputIndex` is
  // the column's position in the pre-strip `readColumnNames_` layout (output,
  // then filter-only), which `buildOutputTable` restores for remaining /
  // deferred subfield filters.
  void adaptColumns();

  // Assemble the final output table.
  //
  // Interleaves info/partition/schema-evolution constants at their
  // `outputIndex` among the file columns. Uses `rowCountOverride` when every
  // projected column is injected (no file columns to size the constants).
  // @param table Input table
  // @param mr Memory resource to allocate injected columns
  // @param rowCountOverride Surviving row count when the input has no physical
  // columns (injected-only). `nullopt` means use `table->num_rows()`.
  std::unique_ptr<cudf::table> buildOutputTable(
      std::unique_ptr<cudf::table>&& table,
      rmm::device_async_resource_ref mr,
      std::optional<cudf::size_type> rowCountOverride) const;

  // Describes a column that must be injected after reading because it is
  // not present in the parquet file (partition, info, or schema evolution).
  struct InjectedColumn {
    // Position in the assembled table / pre-strip `readColumnNames_` layout.
    size_t outputIndex;
    std::string name;
    std::optional<std::string>
        partitionValue; // nullopt = NULL (schema evolution)
    TypePtr veloxType;
  };

  // Builds a cudf scalar for an injected column from its optional string value;
  // a `nullopt` value yields a typed NULL.
  std::unique_ptr<cudf::scalar> makeInjectedScalar(
      const InjectedColumn& col) const;

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

  // Columns to inject after reading.
  std::vector<InjectedColumn> injectedColumns_;

  // Whether every projected column is injected
  bool noColumnsToRead_{false};
  bool syntheticTableProduced_{false};

  // Whether the original subfield filter is deferred to post table read.
  bool deferSubfieldFilter_{false};

  // Transformed filter pushed to the cuDF reader with injected columns removed.
  std::unique_ptr<CudfIcebergExpressionTransformer> pushdownFilter_;

  // Top-level column names and total row count from the file metadata
  std::unordered_set<std::string> fileColumnNames_;

  // Tracks the absolute row range covered by the split.
  std::size_t baseReadOffset_{0};
  std::size_t splitRowCount_{0};

  // Bitmaps for positional deletes
  BufferPtr deleteBitmap_{nullptr};
  std::shared_ptr<rmm::device_buffer> deviceBitmap_;

  // Deletion mask column updated by each deletion mechanism
  std::unique_ptr<cudf::column> deleteMask_;
  cudf::mutable_column_view deleteMaskView_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
