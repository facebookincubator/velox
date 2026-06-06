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
#include <unordered_set>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive::iceberg {

/// Reads an Iceberg equality delete file and filters base data rows whose
/// equality delete column values match any row in the delete file.
///
/// Iceberg equality delete files contain rows with values for one or more
/// columns (identified by equalityFieldIds). A base data row is deleted if
/// its values match ALL specified columns of ANY row in the delete file.
///
/// Unlike positional deletes (which set bits before reading), equality deletes
/// require reading the base data first, then probing each row against the
/// delete set. The reader eagerly loads all delete key tuples from the file
/// into an in-memory hash set during construction.
///
/// The equality delete column names are resolved from equalityFieldIds via
/// the table schema provided by the caller.
class EqualityDeleteFileReader {
 public:
  /// Constructs a reader for a single equality delete file.
  ///
  /// Eagerly reads the entire delete file and builds an in-memory hash set
  /// of delete key tuples. The delete file is fully consumed during
  /// construction.
  ///
  /// @param deleteFile Metadata about the equality delete file. Must have
  ///   content == FileContent::kEqualityDeletes and non-empty
  ///   equalityFieldIds.
  /// @param equalityColumnNames Ordered column names corresponding to
  ///   equalityFieldIds, resolved by the caller from the table schema.
  /// @param equalityColumnTypes Ordered column types corresponding to
  ///   equalityFieldIds.
  /// @param baseFilePath Path of the base data file being read.
  /// @param fileHandleFactory Factory for creating file handles.
  /// @param connectorQueryCtx Query context for memory and config.
  /// @param executor IO executor for async reads.
  /// @param hiveConfig Hive configuration.
  /// @param ioStatistics IO statistics collector.
  /// @param ioStats IO stats tracker.
  /// @param runtimeStats Runtime statistics for recording skipped bytes.
  /// @param connectorId Connector identifier.
  EqualityDeleteFileReader(
      const IcebergDeleteFile& deleteFile,
      const std::vector<std::string>& equalityColumnNames,
      const std::vector<TypePtr>& equalityColumnTypes,
      const std::string& baseFilePath,
      FileHandleFactory* fileHandleFactory,
      const ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      const std::shared_ptr<const FileConfig>& fileConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      dwio::common::RuntimeStatistics& runtimeStats,
      const std::string& connectorId);

  /// Applies equality deletes to the output vector by setting bits in the
  /// delete bitmap for rows whose equality column values match any delete
  /// key tuple.
  ///
  /// @param output The base data output vector to filter.
  /// @param deleteBitmap Output bitmap. Bit i is set if row i matches an
  ///   equality delete. The bitmap must be pre-allocated to cover at least
  ///   output->size() rows.
  void applyDeletes(const RowVectorPtr& output, BufferPtr deleteBitmap);

  /// Returns the number of delete key tuples loaded from the file.
  size_t numDeleteKeys() const {
    return deleteKeyHashes_.size();
  }

  /// Returns true if this reader has no delete keys (file was skipped or
  /// empty). When true, applyDeletes() is a no-op.
  bool empty() const {
    return deleteKeyHashes_.empty();
  }

 private:
  // Resolves column indices for the given row type, caching the result in
  // outputColumnIndices_ for reuse across rows.
  const std::vector<column_index_t>& resolveOutputColumnIndices(
      const RowVectorPtr& row) const;

  // Hashes a single row's equality delete columns into a uint64_t key.
  uint64_t hashRow(
      const RowVectorPtr& row,
      vector_size_t index,
      const std::vector<column_index_t>& colIndices) const;

  // Checks whether two rows are equal on all equality delete columns.
  bool equalRows(
      const RowVectorPtr& left,
      vector_size_t leftIndex,
      const RowVectorPtr& right,
      vector_size_t rightIndex) const;

  // Column names and types for equality delete comparison.
  std::vector<std::string> equalityColumnNames_;
  std::vector<TypePtr> equalityColumnTypes_;

  // Column indices in the delete file output vector.
  std::vector<column_index_t> deleteColumnIndices_;

  // Cached column indices for the output (probe) row type. Resolved lazily
  // on first applyDeletes() call to avoid repeated name lookups per row.
  mutable std::vector<column_index_t> outputColumnIndices_;

  // All rows read from the equality delete file, stored for equality
  // comparison during probing.
  std::vector<RowVectorPtr> deleteRows_;

  // Hash multimap storing (hash → (batch index, row index)) for all delete
  // key tuples. Used for O(1) average-case probing.
  struct DeleteKeyEntry {
    size_t batchIndex;
    vector_size_t rowIndex;
  };
  std::unordered_multimap<uint64_t, DeleteKeyEntry> deleteKeyHashes_;

  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox::connector::hive::iceberg
