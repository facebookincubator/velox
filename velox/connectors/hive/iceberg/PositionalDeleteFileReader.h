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
#include <memory>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive::iceberg {

/// Reads a positional delete file and produces a deletion bitmap for the base
/// data during merge-on-read. The delete file schema per the Iceberg V2 spec
/// is:
///   (file_path: VARCHAR, pos: BIGINT)
///
/// For each row in the delete file matching the base data file path, this
/// reader records the deleted row position. During read, these positions are
/// converted into a bitmap where set bits indicate rows that should be excluded
/// from the output.
///
/// The reader processes delete positions incrementally, batch by batch.
/// Positions from the delete file that fall within the current batch range are
/// converted to bitmap bits. Leftover positions beyond the current batch are
/// preserved and carried over to the next readDeletePositions() call.
class PositionalDeleteFileReader {
 public:
  /// Constructs a reader for a single positional delete file.
  ///
  /// Opens the delete file, sets up a filter on file_path = baseFilePath to
  /// read only delete positions relevant to the current base data file, and
  /// applies testFilters() to skip the entire delete file when possible (e.g.,
  /// when the delete file contains no entries for this base file).
  ///
  /// @param deleteFile Metadata about the delete file (path, format, bounds,
  ///   record count).
  /// @param baseFilePath Path of the base data file being read. Used to filter
  ///   delete records that apply to this specific data file.
  /// @param fileHandleFactory Factory for creating file handles.
  /// @param connectorQueryCtx Query context providing memory pool, session
  ///   properties, and filesystem token.
  /// @param executor Executor for async I/O operations.
  /// @param fileConfig Hive connector configuration.
  /// @param ioStatistics Shared I/O statistics counters.
  /// @param ioStats Shared I/O stats tracker.
  /// @param runtimeStats Runtime statistics to record skipped bytes when the
  ///   delete file is filtered out entirely.
  /// @param splitOffset Row number offset of the current split within the base
  ///   data file. Delete positions are absolute within the file, so this offset
  ///   is used to translate them to split-relative positions.
  /// @param connectorId Connector ID for constructing the internal split.
  PositionalDeleteFileReader(
      const IcebergDeleteFile& deleteFile,
      const std::string& baseFilePath,
      FileHandleFactory* fileHandleFactory,
      const ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      dwio::common::RuntimeStatistics& runtimeStats,
      uint64_t splitOffset,
      const std::string& connectorId);

  /// Reads delete positions for the current batch and sets corresponding bits
  /// in the deletion bitmap.
  ///
  /// Processes delete positions in the range
  /// [splitOffset + baseReadOffset, splitOffset + baseReadOffset + size).
  /// Positions from the delete file that fall within this range have their
  /// corresponding bits set in deleteBitmap. Positions beyond the range are
  /// buffered for subsequent calls.
  ///
  /// @param baseReadOffset The read offset from the beginning of the split in
  ///   number of rows for the current batch.
  /// @param size The number of rows in the current batch, before deletion.
  /// @param deleteBitmap Output bitmap buffer where set bits mark rows to
  ///   delete. Bit positions are relative to baseReadOffset within the split.
  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      BufferPtr deleteBitmap);

  /// Returns true when all delete positions have been read and consumed from
  /// this file.
  bool noMoreData();

 private:
  // Converts delete positions from deletePositionsVector into set bits in the
  // deleteBitmapBuffer for positions within
  // [splitOffset + baseReadOffset, rowNumberUpperBound).
  void updateDeleteBitmap(
      VectorPtr deletePositionsVector,
      uint64_t baseReadOffset,
      int64_t rowNumberUpperBound,
      BufferPtr deleteBitmapBuffer);

  // Returns true if enough delete positions have been read for the current
  // batch, either because EOF or the next unprocessed position >=
  // rowNumberUpperBound.
  bool readFinishedForBatch(int64_t rowNumberUpperBound);

  const IcebergDeleteFile deleteFile_;
  const std::string baseFilePath_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<const FileConfig> fileConfig_;
  const std::shared_ptr<io::IoStatistics> ioStatistics_;
  const std::shared_ptr<IoStats> ioStats_;
  memory::MemoryPool* const pool_;

  // Iceberg metadata column descriptors for the delete file schema.
  const std::shared_ptr<IcebergMetadataColumn> filePathColumn_;
  const std::shared_ptr<IcebergMetadataColumn> posColumn_;

  // Row number offset of the current split within the base data file.
  const uint64_t splitOffset_;

  // Internal split and row reader for the delete file. Reset to nullptr when
  // the delete file is fully consumed or skipped by testFilters().
  std::shared_ptr<HiveConnectorSplit> deleteSplit_;
  std::unique_ptr<dwio::common::RowReader> deleteRowReader_;

  // Holds the raw output from reading the delete file. Contains a RowVector
  // with a single pos column. Positions are absolute row numbers within the
  // base data file.
  VectorPtr deletePositionsOutput_;

  // Index into deletePositionsOutput_ indicating how far delete positions
  // have been consumed and converted into bitmap bits.
  uint64_t deletePositionsOffset_;

  // Total number of rows read from this delete file, including rows filtered
  // out by the file_path filter. Used to detect end-of-file.
  uint64_t totalNumRowsScanned_;
};

} // namespace facebook::velox::connector::hive::iceberg
