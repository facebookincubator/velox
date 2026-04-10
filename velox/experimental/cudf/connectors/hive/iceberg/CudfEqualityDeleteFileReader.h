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
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/Statistics.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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
/// require reading the base data first, then evaluating each row against the
/// delete set. The reader eagerly loads all delete key tuples from the file
/// during construction and lazily builds a cudf::ast::tree encoding the
/// delete predicates on the first call to applyDeletes.
///
/// The equality delete column names are resolved from equalityFieldIds via
/// the table schema provided by the caller.
class CudfEqualityDeleteFileReader {
 public:
  /// Constructs a reader for a single equality delete file.
  ///
  /// Eagerly reads the entire delete file and stores the delete rows as
  /// Velox RowVectors. The AST expression tree is built lazily on the first
  /// call to applyDeletes, once the input table's column layout is known.
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

  /// Applies equality deletes to a GPU table by evaluating a cudf AST
  /// expression and updating the row mask. Rows whose equality column values
  /// match any delete key tuple are marked as deleted (false in the mask).
  ///
  /// Builds the AST tree lazily on the first call using inputColumnNames to
  /// resolve equality column indices in the input table.
  ///
  /// @param table The cudf table view to filter.
  /// @param inputColumnNames Column names of the input table, used to map
  ///   equality column names to column indices.
  /// @param rowMask Device buffer of booleans; surviving rows are true.
  /// @param stream CUDA stream for kernel launches.
  void applyDeletes(
      cudf::table_view table,
      const std::vector<std::string>& inputColumnNames,
      std::shared_ptr<rmm::device_buffer> rowMask,
      rmm::cuda_stream_view stream);

  /// Returns the total number of delete rows loaded from the file.
  size_t numDeleteKeys() const;

  /// Returns true if no delete rows were loaded (file was skipped or empty).
  /// When true, applyDeletes() is a no-op.
  bool empty() const;

 private:
  /// Builds the cudf::ast::tree encoding all delete predicates.
  ///
  /// For each delete row, creates NULL_EQUAL comparisons for each equality
  /// column ANDed together, then ORs all row predicates into a single
  /// expression.
  ///
  /// @param inputColumnNames Column names of the input table for index
  ///   resolution.
  void buildAstTree(const std::vector<std::string>& inputColumnNames);

  /// Column names for equality comparison.
  std::vector<std::string> equalityColumnNames_;

  /// Column types for equality comparison.
  std::vector<TypePtr> equalityColumnTypes_;

  /// All rows read from the equality delete file, stored for AST construction.
  std::vector<RowVectorPtr> deleteRows_;

  /// Total number of individual delete key rows across all batches.
  size_t numDeleteKeys_{0};

  /// AST expression tree encoding all delete predicates. Built lazily on the
  /// first call to applyDeletes.
  std::unique_ptr<cudf::ast::tree> astTree_;

  /// Scalars owned by the AST tree. Must outlive the tree since ast::literal
  /// stores references to them.
  std::vector<std::unique_ptr<cudf::scalar>> ownedScalars_;

  /// Memory pool for Velox allocations.
  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
