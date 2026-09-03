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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergFilterTransform.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

#include <list>
#include <optional>
#include <string>
#include <string_view>
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
      const cudf::ast::expression* subfieldFilterAst,
      const common::SubfieldFilters* subfieldFilters);

 protected:
  // Sets up delete file readers and column projection after base state reset.
  void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats) override;

  // Override to report a split the filter rejects as skipped.
  bool isSplitSkipped() const override;

  // Override to only setup cuDF reader if we have columns to read.
  void setupReader() override;

  // Skip Parquet pushdown when the subfield filter must run after reading.
  cudf::ast::expression const* pushdownFilter() const override;

  // Override to determine the memory resource to construct cuDF reader.
  rmm::device_async_resource_ref determineCudfMemoryResource() const override;

  // Override to apply Iceberg deletes after reading a cudf table chunk.
  std::optional<std::unique_ptr<cudf::table>> readNextChunk() override;

 private:
  // Clear delete readers and column injection
  void resetSplit();

  // Selects applicable positional delete, equality delete, and deletion vector
  // files that apply to the split without opening any files.
  void classifyDeleteFiles();

  // Setup delete file readers for selected positional and equality deletes,
  // and deletion vectors.
  // @param runtimeStats DataSource's runtime statistics, passed to delete
  // file readers for accumulation.
  void setupDeleteFileReaders(dwio::common::RuntimeStats& runtimeStats);

  // Applicable equality delete file, together with the columns its field IDs
  // key on.
  struct EqualityDeleteFile {
    // Owned by `icebergSplit_`.
    const velox_iceberg::IcebergDeleteFile* file;
    std::vector<std::string> keyNames;
    std::vector<TypePtr> keyTypes;
  };

  // Resolves the equality field IDs of a delete file to the names and types of
  // the columns they key on.
  EqualityDeleteFile equalityDeleteKeys(
      const velox_iceberg::IcebergDeleteFile& deleteFile) const;

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

  // Decides whether the subfield filter is pushed into the data-file reader as
  // is, pushed as a filter over the columns read from the data file, or
  // deferred to post table read.
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

  // Returns whether timestamp partition values are read as local time.
  bool readTimestampAsLocalTime() const;

  // Returns the filter on a top-level column, or null when it is not involved
  // in the filter, or filters only a subfield of it.
  const common::Filter* topLevelColumnFilter(std::string_view name) const;

  // Evaluates the query's filter on an injected column against the constant
  // value that column holds for the whole split.
  ConstantFilterFold foldInjectedColumn(const InjectedColumn& col) const;

  // Returns the deferred filter to apply to the assembled table or null when
  // the pushed filter already applies the complete filter.
  const cudf::ast::expression* deferredFilter() const;

  // Returns whether nothing can be pushed, so the whole filter is deferred.
  bool deferEverything() const;

  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit_;
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;

  // Subfield filters the pushed AST was built from, used to fold the filter on
  // an injected column against the constant that column holds. Owned by the
  // data source, which outlives the split reader.
  const common::SubfieldFilters* subfieldFilters_;

  // Delete files that apply to the split, owned by `icebergSplit_`.
  std::vector<const velox_iceberg::IcebergDeleteFile*> positionalDeleteFiles_;
  std::vector<EqualityDeleteFile> equalityDeleteFiles_;
  const velox_iceberg::IcebergDeleteFile* deletionVectorFile_{nullptr};

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

  // Whether the filter rejects this split entirely.
  bool skipSplit_{false};

  // Filter over file-backed columns pushed to the Parquet reader. Empty when
  // the original filter was not transformed or has a `nullptr` root when
  // nothing can be pushed.
  std::optional<TransformedFilter> transformedPushdownFilter_;

  // Transform of the logical filter, held only when a `PushdownFilterBuilder`
  // has transformed it differently from the pushed filter.
  std::optional<TransformedFilter> transformedLogicalFilter_;

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
