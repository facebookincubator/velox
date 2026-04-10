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
      cudf::ast::expression const* subfieldFilterExpr,
      std::unique_ptr<exec::ExprSet>* remainingFilterExprSet,
      std::shared_ptr<CudfExpression> cudfExpressionEvaluator,
      std::atomic<uint64_t>* totalRemainingFilterTime);

  void prepareSplit() override;

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
  void setupDeleteFileReaders();

  /// Apply positional deletes (V2) to the input cudf table.
  std::unique_ptr<cudf::table> applyPositionalDeletes(
      cudf::table_view input,
      rmm::device_async_resource_ref output_mr);

  /// Apply equality deletes (V2) to the input cudf table on GPU.
  std::unique_ptr<cudf::table> applyEqualityDeletes(
      cudf::table_view input,
      rmm::device_async_resource_ref output_mr);

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

  /// Tracks the absolute row offset within the data file. Each chunk advances
  /// this by the number of rows read (before deletes).
  uint64_t baseReadOffset_{0};

  BufferPtr deleteBitmap_{nullptr};
  std::shared_ptr<rmm::device_buffer> deviceDeleteBitmap_;
  std::shared_ptr<rmm::device_buffer> rowMask_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
