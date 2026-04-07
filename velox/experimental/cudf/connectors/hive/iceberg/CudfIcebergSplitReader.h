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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"

#include <list>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

/// GPU-accelerated per-split reader for Iceberg tables.
///
/// Derives from CudfSplitReader and adds Iceberg delete semantics:
///   - Deletion vectors (V3): applied on the GPU using cuco roaring_bitmap
///     and cudf::apply_boolean_mask after reading each chunk.
///   - Positional deletes (V2): host-side bitmap from upstream
///     PositionalDeleteFileReader, filtered on GPU via apply_boolean_mask.
///   - Equality deletes (V2): host-side bitmap from upstream
///     EqualityDeleteFileReader, filtered on GPU via apply_boolean_mask.
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
      bool useExperimentalReader,
      cudf::ast::expression const* subfieldFilterExpr,
      std::unique_ptr<exec::ExprSet>* remainingFilterExprSet,
      std::shared_ptr<CudfExpression> cudfExpressionEvaluator,
      std::atomic<uint64_t>* totalRemainingFilterTime);

  void prepareSplit() override;

 protected:
  /// Override to apply Iceberg deletes after reading a chunk.
  std::optional<std::unique_ptr<cudf::table>> readNextChunk() override;

  /// NOTE: setupCudfDataSourceAndOptions() and createCudfReader() are NOT
  /// overridden. The base CudfSplitReader implementations (taken from the more
  /// mature CudfHiveDataSource) handle both Hive and Iceberg cases:
  ///   - Path cleaning (file:, s3a:) is done in addSplit() when building
  ///     CudfHiveConnectorSplit, so split_->filePath is already clean.
  ///   - Subfield filter is safely skipped when subfieldFilterExpr_ is nullptr.
  ///   - Reader creation is identical between Hive and Iceberg.

 private:
  void loadDeletionVector();
  void setupDeleteFileReaders();

  std::unique_ptr<cudf::table> applyPositionalDeletes(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::unique_ptr<cudf::table> applyEqualityDeletes(
      cudf::table_view input,
      const RowVectorPtr& rowVector,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit_;
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;

  std::unique_ptr<CudfDeletionVectorReader> dvReader_;
  std::list<std::unique_ptr<velox_iceberg::PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;
  std::list<std::unique_ptr<velox_iceberg::EqualityDeleteFileReader>>
      equalityDeleteFileReaders_;

  // Tracks the absolute row offset within the data file. Each chunk advances
  // this by the number of rows read (before deletes).
  uint64_t baseReadOffset_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
