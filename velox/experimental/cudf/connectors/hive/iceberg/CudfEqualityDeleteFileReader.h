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

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;
namespace velox_hive = ::facebook::velox::connector::hive;

/// Reads an Iceberg equality delete file and filters base data rows on GPU
/// whose equality delete column values match any row in the delete file.
///
/// Iceberg equality delete files contain rows with values for one or more
/// columns (identified by equalityFieldIds). A base data row is deleted if
/// its values match ALL specified columns of ANY row in the delete file.
///
/// Unlike positional deletes (which set bits before reading), equality deletes
/// require reading the base data first, then probing each row against the
/// delete set. The reader eagerly loads all delete key tuples from the file
/// into an in-memory hash set during construction via the upstream reader.
///
/// The equality delete column names are resolved from equalityFieldIds via
/// the table schema provided by the caller.
class CudfEqualityDeleteFileReader {
 public:
  /// Constructs a reader for a single equality delete file.
  ///
  /// Eagerly reads the entire delete file via the upstream
  /// EqualityDeleteFileReader and builds an in-memory hash set of delete key
  /// tuples. The delete file is fully consumed during construction.
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
  CudfEqualityDeleteFileReader(
      const velox_iceberg::IcebergDeleteFile& deleteFile,
      const std::vector<std::string>& equalityColumnNames,
      const std::vector<TypePtr>& equalityColumnTypes,
      const std::string& baseFilePath,
      FileHandleFactory* fileHandleFactory,
      const ConnectorQueryCtx* connectorQueryCtx,
      folly::Executor* executor,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      dwio::common::RuntimeStatistics& runtimeStats,
      const std::string& connectorId);

  /// Applies equality deletes to a GPU table by marking deleted rows in the
  /// row mask. Rows whose equality column values match any delete key tuple
  /// are marked as deleted.
  ///
  /// @param table The cudf table view to filter.
  /// @param rowMask Device buffer of booleans; rows not deleted are marked
  ///   true.
  /// @param stream CUDA stream for kernel launches.
  void applyDeletes(
      cudf::table_view table,
      std::shared_ptr<rmm::device_buffer> rowMask,
      rmm::cuda_stream_view stream);

  /// Applies equality deletes to the output vector by setting bits in the
  /// delete bitmap for rows whose equality column values match any delete
  /// key tuple. Delegates to the upstream reader.
  ///
  /// @param output The base data output vector to filter.
  /// @param deleteBitmap Output bitmap. Bit i is set if row i matches an
  ///   equality delete.
  void applyDeletes(const RowVectorPtr& output, BufferPtr deleteBitmap);

  /// Returns the number of delete key tuples loaded from the file.
  size_t numDeleteKeys() const;

  /// Returns true if no delete keys were loaded (file was skipped or empty).
  /// When true, applyDeletes() is a no-op.
  bool empty() const;

 private:
  // Upstream reader used to read the delete file and apply deletes on CPU.
  std::unique_ptr<velox_iceberg::EqualityDeleteFileReader> upstreamReader_;

  // Column names for equality comparison.
  std::vector<std::string> equalityColumnNames_;

  // Column types for equality comparison.
  std::vector<TypePtr> equalityColumnTypes_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
