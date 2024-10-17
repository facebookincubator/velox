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

#include "velox/dwio/pagefile/writer/PageFileWriter.h"
#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/io/file.h>
// #include <arrow/table.h>
// #include "velox/common/config/Config.h"
// #include "velox/common/testutil/TestValue.h"
// #include "velox/core/QueryConfig.h"
// #include "velox/dwio/parquet/writer/arrow/Properties.h"
// #include "velox/dwio/parquet/writer/arrow/Writer.h"
// #include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::pagefile {

// using facebook::velox::parquet::arrow::ArrowWriterProperties;
// using facebook::velox::parquet::arrow::Compression;
// using facebook::velox::parquet::arrow::WriterProperties;
// using facebook::velox::parquet::arrow::arrow::FileWriter;
//

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool,
    RowTypePtr schema)
    : pool_(std::move(pool)),
      generalPool_{pool_->addLeafChild(".general")},
      stream_(std::make_shared<ArrowDataBufferSink>(
          std::move(sink),
          *generalPool_,
          options.bufferGrowRatio)),
      arrowContext_(std::make_shared<ArrowContext>()),
      schema_(std::move(schema)) {
  // validateSchemaRecursive(schema_);

  if (options.flushPolicyFactory) {
    flushPolicy_ = options.flushPolicyFactory();
  } else {
    flushPolicy_ = std::make_unique<DefaultFlushPolicy>();
  }
  // arrowContext_->properties =
      // getArrowParquetWriterOptions(options, flushPolicy_);
  // setMemoryReclaimers();
}

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options,
    RowTypePtr schema)
    : Writer{
          std::move(sink),
          options,
          options.memoryPool->addAggregateChild(fmt::format(
              "writer_node_{}",
              folly::to<std::string>(folly::Random::rand64()))),
          std::move(schema)} {}

dwio::common::StripeProgress getStripeProgress(
    uint64_t stagingRows,
    int64_t stagingBytes) {
  return dwio::common::StripeProgress{
    .stripeRowCount = stagingRows, .stripeSizeEstimate = stagingBytes};
}


void Writer::write(const VectorPtr& data) {
  VELOX_USER_CHECK(
      data->type()->equivalent(*schema_),
      "The file schema type should be equal with the input rowvector type.");

  ArrowArray array;
  ArrowSchema schema;
  exportToArrow(data, array, generalPool_.get(), options_);
  exportToArrow(data, schema, options_);

  auto arrowSchema = ::arrow::ImportSchema(&schema).ValueOrDie();
  auto recordBatch = ::arrow::ImportRecordBatch(&array, arrowSchema);
  if (!arrowContext_->schema) {
    arrowContext_->schema = arrowSchema;
  }

  auto bytes = data->estimateFlatSize();
  auto numRows = data->size();
  if (flushPolicy_->shouldFlush(getStripeProgress(
          arrowContext_->stagingRows, arrowContext_->stagingBytes))) {
    flush();
   }

  arrowContext_->stagingBatches.push_back(*recordBatch);
  arrowContext_->stagingRows += numRows;
  arrowContext_->stagingBytes += bytes;
}
void Writer::flush() {
  if (arrowContext_->stagingRows > 0) {
    std::shared_ptr<::arrow::io::FileOutputStream> outputStream;
    auto ipcWriterResult = ::arrow::ipc::MakeFileWriter(stream_, arrowContext_->schema);
    if (!ipcWriterResult.ok()) {
      // Handle the error
      throw std::runtime_error(ipcWriterResult.status().ToString());
    }
    std::shared_ptr<::arrow::ipc::RecordBatchWriter> ipcWriter = *ipcWriterResult;

    // Write each RecordBatch to the file
    for (const auto& recordBatch : arrowContext_->stagingBatches) {
      auto writeStatus = ipcWriter->WriteRecordBatch(*recordBatch);
      if (!writeStatus.ok()) {
        throw std::runtime_error(writeStatus.ToString());
      }
    }

    // Finalize and close the writer and output stream
    // Finalize and close the IPC writer
    auto closeWriterStatus = ipcWriter->Close();
    if (!closeWriterStatus.ok()) {
      throw std::runtime_error(closeWriterStatus.ToString());
    }
    // Flush the sink (ArrowDataBufferSink) to ensure everything is written to FileSink
    auto flushStatus = stream_->Flush();
    if (!flushStatus.ok()) {
      throw std::runtime_error(flushStatus.ToString());
    }

    // Clear the staging data
    arrowContext_->stagingBatches.clear();
    arrowContext_->stagingRows = 0;
    arrowContext_->stagingBytes = 0;
  }
}


void Writer::close() {
  flush();
  arrowContext_->stagingBatches.clear();
}

/// Aborts the writing by closing the writer and dropping everything.
/// Data can no longer be written.
void Writer::abort() {
  stream_->abort();
  arrowContext_.reset();
}


} // namespace facebook::velox::pagefile
