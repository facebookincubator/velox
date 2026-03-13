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

#include <folly/Executor.h>
#include <map>
#include <memory>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive::iceberg {

struct IcebergMetadataColumn;

/// Reads a positional update file and applies column-level updates to the base
/// data during merge-on-read. The update file schema is:
///   (file_path: VARCHAR, pos: BIGINT, <updated_col_1>, <updated_col_2>, ...)
///
/// NOTE: "Positional updates" are a Velox-side merge-on-read extension. The
/// Iceberg V2 spec does not define a native update file format; standard
/// Iceberg achieves updates via delete + insert (rewriting the full row into
/// a new data file). This extension avoids full-row rewrites by carrying only
/// the changed column values alongside the (file_path, pos) position key.
///
/// For each row in the update file matching the base data file path, this
/// reader records the updated column values indexed by the in-file row
/// position. During read, updated values are overlaid onto the base data output
/// vector at the matching positions.
///
/// Mirrors PositionalDeleteFileReader but instead of producing a delete bitmap,
/// builds a sparse map of position → source row index in the read batch, then
/// copies updated values into the output vector.
class PositionalUpdateFileReader {
 public:
  /// Constructs a reader for a single positional update file.
  ///
  /// @param updateFile Metadata about the update file (path, format, bounds).
  /// @param baseFilePath Path of the base data file being read. Used to filter
  ///   update records that apply to this specific data file.
  /// @param updateColumnNames Ordered names of the columns being updated.
  /// @param updateColumnTypes Ordered types of the columns being updated,
  ///   matching the update file's schema for those columns.
  PositionalUpdateFileReader(
      const IcebergDeleteFile& updateFile,
      const std::string& baseFilePath,
      const std::vector<std::string>& updateColumnNames,
      const std::vector<TypePtr>& updateColumnTypes,
      FileHandleFactory* fileHandleFactory,
      const ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      const std::shared_ptr<const HiveConfig>& hiveConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      dwio::common::RuntimeStatistics& runtimeStats,
      uint64_t splitOffset,
      const std::string& connectorId);

  /// Applies pending column-level updates to the base data output vector.
  ///
  /// The overall flow for each batch is:
  ///   1. Read update positions and values from the update file for the current
  ///      batch range via readUpdatePositions().
  ///   2. For each pending update position:
  ///      a. If a deleteBitmap is present and the position is marked as
  ///      deleted,
  ///         skip the update (deletes take precedence over updates).
  ///      b. Map the original file position to an output row index, accounting
  ///         for deleted rows that have been removed from the output.
  ///      c. Copy updated column values from the update file batch into the
  ///         output vector at the computed row index.
  ///
  /// @param baseReadOffset The read offset from the beginning of the split in
  ///   number of rows for the current batch.
  /// @param output The base data output vector. Updated column values are
  ///   overlaid at matching positions.
  /// @param deleteBitmap Optional deletion bitmap from positional delete files.
  ///   When present (i.e., when the split has associated positional delete
  ///   files), bit i is set if the row at file position i was deleted. Deleted
  ///   positions are skipped during update application, and the bitmap is used
  ///   to compute the compacted output row index for surviving rows.
  /// @param batchSize The pre-deletion batch size (number of rows read from
  ///   the file before deletions are applied). Used with deleteBitmap to
  ///   compute the correct output row index.
  void applyUpdates(
      uint64_t baseReadOffset,
      const RowVectorPtr& output,
      const BufferPtr& deleteBitmap,
      uint64_t batchSize);

  /// Returns true when there is no more update data to read from this file.
  bool noMoreData();

 private:
  // -- Internal helpers (not part of the public contract) --

  /// Reads update records from the file and populates pendingUpdates_ for
  /// positions within the current batch range.
  void readUpdatePositions(
      uint64_t baseReadOffset,
      int64_t rowNumberUpperBound);

  /// Returns true if we have read enough update positions for the current
  /// batch.
  bool readFinishedForBatch(int64_t rowNumberUpperBound);

  const IcebergDeleteFile updateFile_;
  const std::string baseFilePath_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* connectorQueryCtx_;
  const std::shared_ptr<const HiveConfig> hiveConfig_;
  const std::shared_ptr<io::IoStatistics> ioStatistics_;
  const std::shared_ptr<IoStats> ioStats_;
  memory::MemoryPool* const pool_;

  std::shared_ptr<IcebergMetadataColumn> filePathColumn_;
  std::shared_ptr<IcebergMetadataColumn> posColumn_;
  uint64_t splitOffset_;

  /// Cached output row type for readUpdatePositions() to avoid
  /// reconstructing it on every call. Schema: (pos, update_col_1, ...).
  RowTypePtr outputRowType_;

  /// Names and types of the columns being updated.
  std::vector<std::string> updateColumnNames_;
  std::vector<TypePtr> updateColumnTypes_;

  std::shared_ptr<HiveConnectorSplit> updateSplit_;
  std::unique_ptr<dwio::common::RowReader> updateRowReader_;

  /// Holds the raw output from reading the update file.
  VectorPtr updatePositionsOutput_;
  uint64_t updatePositionsOffset_{0};
  uint64_t totalNumRowsScanned_{0};

  /// Maps base-file row position → (source batch, index within that batch)
  /// from which the updated column values should be copied. Each entry keeps
  /// its own shared_ptr to the source batch so that the underlying data remains
  /// valid even when the reader advances to subsequent batches.
  std::map<int64_t, std::pair<RowVectorPtr, vector_size_t>> pendingUpdates_;
};

} // namespace facebook::velox::connector::hive::iceberg
