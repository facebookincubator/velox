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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/PositionalDeleteFileReader.h"
#include "velox/dwio/common/Statistics.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <list>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

using CudfParquetReader = cudf::io::chunked_parquet_reader;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;

namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

/// GPU-accelerated per-split reader for Iceberg tables.
///
/// Reads parquet data files via cudf's chunked_parquet_reader and handles all
/// Iceberg delete semantics:
///   - Deletion vectors (V3): applied on the GPU using cuco roaring_bitmap
///     and cudf::apply_boolean_mask after reading each chunk.
///   - Positional deletes (V2): host-side bitmap from upstream
///     PositionalDeleteFileReader, filtered on GPU via apply_boolean_mask.
///   - Equality deletes (V2): host-side bitmap from upstream
///     EqualityDeleteFileReader, filtered on GPU via apply_boolean_mask.
class CudfIcebergSplitReader : public NvtxHelper {
 public:
  CudfIcebergSplitReader(
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
      const std::shared_ptr<IoStats>& ioStats);

  /// Prepare the split: open cudf reader, load DV, set up delete readers.
  void prepareSplit();

  /// Read the next chunk and apply all deletes. Returns nullopt when done.
  std::optional<RowVectorPtr> next(uint64_t size);

  bool emptySplit() const {
    return !splitReader_;
  }

  dwio::common::RuntimeStatistics& runtimeStats() {
    return runtimeStats_;
  }

  size_t completedRows() const {
    return completedRows_;
  }

 private:
  void setupCudfDataSourceAndOptions();
  void createCudfReader();
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
  std::shared_ptr<const velox_hive::HiveTableHandle> tableHandle_;
  const RowTypePtr outputType_;
  std::vector<std::string> readColumnNames_;

  FileHandleFactory* fileHandleFactory_;
  folly::Executor* executor_;
  const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx_;
  std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;
  memory::MemoryPool* pool_;

  cudf::io::parquet_reader_options readerOptions_;
  std::shared_ptr<cudf::io::datasource> dataSource_;
  CudfParquetReaderPtr splitReader_;
  rmm::cuda_stream_view stream_;

  std::unique_ptr<CudfDeletionVectorReader> dvReader_;

  std::list<std::unique_ptr<velox_iceberg::PositionalDeleteFileReader>>
      positionalDeleteFileReaders_;
  std::list<std::unique_ptr<velox_iceberg::EqualityDeleteFileReader>>
      equalityDeleteFileReaders_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;
  dwio::common::RuntimeStatistics runtimeStats_;
  size_t completedRows_{0};

  // Tracks the absolute row offset within the data file. Each chunk advances
  // this by the number of rows read (before deletes).
  uint64_t baseReadOffset_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
