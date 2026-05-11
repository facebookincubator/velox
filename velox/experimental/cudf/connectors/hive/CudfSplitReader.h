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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

using CudfParquetReader = cudf::io::chunked_parquet_reader;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;

using CudfHybridScanReader =
    cudf::io::parquet::experimental::hybrid_scan_reader;
using CudfHybridScanReaderPtr = std::unique_ptr<CudfHybridScanReader>;

class CudfSplitReader : public NvtxHelper {
 public:
  CudfSplitReader(
      std::shared_ptr<CudfHiveConnectorSplit> split,
      std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
          tableHandle,
      const RowTypePtr& outputType,
      const std::vector<std::string>& readColumnNames,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      bool useExperimentalCudfReader,
      cudf::ast::expression const* subfieldFilterExpr);

  virtual ~CudfSplitReader() = default;

  /// Prepare the split: open cudf reader, set up data source and options.
  /// @param runtimeStats Reference to the DataSource's runtime statistics
  virtual void prepareSplit(dwio::common::RuntimeStatistics& runtimeStats);

  /// Read the next raw cudf table chunk. Returns nullopt when done.
  virtual std::optional<std::unique_ptr<cudf::table>> next(uint64_t size);

  /// Get the stream.
  rmm::cuda_stream_view stream() const {
    return stream_;
  }

 protected:
  /// Create the chunked parquet reader.
  virtual void createCudfReader(rmm::device_async_resource_ref output_mr);

  /// Setup the cuDF data source
  void setupCudfDataSource();

  /// Clear splitReaders and datasources after split has been fully processed.
  void resetSplit();

  /// Read the next raw chunk from the parquet reader (regular or hybrid).
  /// Returns nullopt when no more data.
  virtual std::optional<std::unique_ptr<cudf::table>> readNextChunk(
      rmm::device_async_resource_ref output_mr);

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;
  const RowTypePtr outputType_;
  std::vector<std::string> readColumnNames_;

  FileHandleFactory* fileHandleFactory_;
  folly::Executor* executor_;
  const ConnectorQueryCtx* connectorQueryCtx_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  std::shared_ptr<cudf::io::datasource> dataSource_;
  rmm::cuda_stream_view stream_;

 private:
  /// Setup the cuDF reader options
  void setupReaderOptions();

  /// Create the experimental hybrid scan reader.
  void createExperimentalReader();

  std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  memory::MemoryPool* pool_;

  // cuDF split reader stuff.
  cudf::io::parquet_reader_options readerOptions_;
  CudfParquetReaderPtr splitReader_;
  CudfHybridScanReaderPtr exptSplitReader_;
  std::unique_ptr<HybridScanState> hybridScanState_;
  bool useExperimentalCudfReader_;

  dwio::common::ReaderOptions baseReaderOpts_;

  cudf::ast::expression const* subfieldFilterExpr_;

  struct TotalScanTimeCallbackData {
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };

  static void totalScanTimeCalculator(void* userData);
};

} // namespace facebook::velox::cudf_velox::connector::hive
