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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderIOHelpers.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>

#include <functional>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

using CudfParquetReader =
    cudf::io::parquet::experimental::hybrid_scan_multifile;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;

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
      cudf::ast::expression const* subfieldFilterExpr);

  virtual ~CudfSplitReader();

  using PushdownFilterBuilder = std::function<cudf::ast::expression const*(
      const cudf::io::parquet::FileMetaData&)>;

  /// Sets a builder for a split-specific pushdown filter. The builder is
  /// invoked after the Parquet footer is read and before reader options are
  /// configured. The returned expression must remain alive while the split is
  /// being read.
  void setPushdownFilterBuilder(PushdownFilterBuilder builder) {
    pushdownFilterBuilder_ = std::move(builder);
  }

  /// Prepare the split: open cudf reader, set up data source and options.
  /// @param runtimeStats Reference to the DataSource's runtime statistics
  void prepareSplit(dwio::common::RuntimeStats& runtimeStats);

  /// Read the next raw cudf table chunk. Returns nullopt when done.
  virtual std::optional<std::unique_ptr<cudf::table>> next(uint64_t size);

  /// Rebinds the query context of a reader prepared in the background to the
  /// context owned by the driver that reads it. Must be called before reading
  /// from a reader that outlives the context it was prepared with.
  void setConnectorQueryCtx(const ConnectorQueryCtx* connectorQueryCtx);

  /// Get the stream.
  rmm::cuda_stream_view stream() const {
    return stream_;
  }

 protected:
  // Performs split-specific setup after base reader state is reset.
  virtual void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats);

  // Return the split-specific filter to push down to the cuDF reader.
  virtual cudf::ast::expression const* pushdownFilter() const;

  // Determine the output memory resource for the cuDF reader.
  virtual rmm::device_async_resource_ref determineCudfMemoryResource() const;

  // Read the next table chunk from the parquet reader. Returns nullopt when no
  // more data.
  virtual std::optional<std::unique_ptr<cudf::table>> readNextChunk();

  // Setup the cuDF data source
  void setupCudfDataSource();

  // Create the parquet reader and select the row group passes to read.
  void createCudfReader();

  // Read file metadatas.
  void fileMetaDatas();

  // Return the logical subfield filter used after reading.
  cudf::ast::expression const* subfieldFilter() const;

  // Return whether the pushdown filter was built for the current split.
  bool hasSplitSpecificPushdownFilter() const;

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

  rmm::cuda_stream_view stream_;

  // Parquet metadata(s) for the current split(s).
  std::vector<cudf::io::parquet::FileMetaData> fileMetaData_;

  // Whether to prepend a row index column to the output.
  bool prependRowIndex_{false};

 private:
  // Tracks how far the row group passes of the current split have been read.
  struct RowGroupPassState {
    // Row groups to read, one entry per pass, in read order.
    std::vector<std::vector<std::vector<cudf::size_type>>> passes;

    // The pass being materialized.
    size_t currentPass{0};

    // Whether the chunked read of the current pass has been set up.
    bool isChunkingSetup{false};

    // Owns the device data of the current pass.
    ByteRangeFetch fetch;
  };

  // Clear splitReaders and datasources after split has been fully processed.
  void resetSplit();

  // Setup the cuDF reader options
  void setupReaderOptions();

  // Setup Parquet column and offset indexes for the cudf split reader.
  void setupPageIndexes();

  // Return the row groups to read, grouped into passes bounded by the pass
  // read limit. Empty when the split has no row groups left after pruning.
  std::vector<std::vector<std::vector<cudf::size_type>>> selectRowGroupPasses()
      const;

  // Start the reads of the column chunks of the current pass without waiting
  // for them. Does nothing when they are already in flight or complete.
  void startCurrentPassFetch();

  // Wait for the column chunks of the current pass, fetching them first if
  // that has not started yet, and set up its chunked read.
  void setupChunkingForCurrentPass(rmm::device_async_resource_ref mr);

  // Release the column chunk data of the current pass, canceling its reads
  // when they have not started yet and waiting for them otherwise.
  void releaseCurrentPassData();

  std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  memory::MemoryPool* pool_;

  // cuDF split reader stuff.
  std::shared_ptr<cudf::io::datasource> dataSource_;
  cudf::io::parquet_reader_options readerOptions_;
  CudfParquetReaderPtr splitReader_;
  std::unique_ptr<RowGroupPassState> passState_;

  // Chunk and pass read limits resolved from the session when the reader is
  // created.
  std::size_t chunkReadLimit_{0};
  std::size_t passReadLimit_{0};

  dwio::common::ReaderOptions baseReaderOpts_;
  cudf::ast::expression const* subfieldFilterExpr_;
  cudf::ast::expression const* pushdownFilterExpr_;
  PushdownFilterBuilder pushdownFilterBuilder_;
  bool hasSplitSpecificPushdownFilter_{false};

  struct TotalScanTimeCallbackData {
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };

  static void totalScanTimeCalculator(void* userData);
};

} // namespace facebook::velox::cudf_velox::connector::hive
