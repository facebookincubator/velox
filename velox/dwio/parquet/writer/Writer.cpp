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

#include "velox/vector/arrow/Bridge.h"

#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>
#include "velox/dwio/parquet/writer/Writer.h"

namespace facebook::velox::parquet {

// Utility for buffering Arrow output with a DataBuffer.
class ArrowDataBufferSink : public arrow::io::OutputStream {
 public:
  /// @param growRatio Growth factor used when invoking the reserve() method of
  /// DataSink, thereby helping to minimize frequent memcpy operations.
  ArrowDataBufferSink(
      std::unique_ptr<dwio::common::FileSink> sink,
      memory::MemoryPool& pool,
      double growRatio)
      : sink_(std::move(sink)), growRatio_(growRatio), buffer_(pool) {}

  arrow::Status Write(const std::shared_ptr<arrow::Buffer>& data) override {
    auto requestCapacity = buffer_.size() + data->size();
    if (requestCapacity > buffer_.capacity()) {
      buffer_.reserve(growRatio_ * (requestCapacity));
    }
    buffer_.append(
        buffer_.size(),
        reinterpret_cast<const char*>(data->data()),
        data->size());
    return arrow::Status::OK();
  }

  arrow::Status Write(const void* data, int64_t nbytes) override {
    auto requestCapacity = buffer_.size() + nbytes;
    if (requestCapacity > buffer_.capacity()) {
      buffer_.reserve(growRatio_ * (requestCapacity));
    }
    buffer_.append(buffer_.size(), reinterpret_cast<const char*>(data), nbytes);
    return arrow::Status::OK();
  }

  arrow::Status Flush() override {
    bytesFlushed_ += buffer_.size();
    sink_->write(std::move(buffer_));
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    return bytesFlushed_ + buffer_.size();
  }

  arrow::Status Close() override {
    ARROW_RETURN_NOT_OK(Flush());
    sink_->close();
    return arrow::Status::OK();
  }

  bool closed() const override {
    return sink_->isClosed();
  }

 private:
  std::unique_ptr<dwio::common::FileSink> sink_;
  const double growRatio_;
  dwio::common::DataBuffer<char> buffer_;
  int64_t bytesFlushed_ = 0;
};

struct ArrowContext {
  std::unique_ptr<::parquet::arrow::FileWriter> writer;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<::parquet::WriterProperties> properties;
  uint64_t stagingRows = 0;
  int64_t stagingBytes = 0;
  // columns, Arrays
  std::vector<std::vector<std::shared_ptr<arrow::Array>>> stagingChunks;
};

::parquet::Compression::type getArrowParquetCompression(
    common::CompressionKind compression) {
  if (compression == common::CompressionKind_SNAPPY) {
    return ::parquet::Compression::SNAPPY;
  } else if (compression == common::CompressionKind_GZIP) {
    return ::parquet::Compression::GZIP;
  } else if (compression == common::CompressionKind_ZSTD) {
    return ::parquet::Compression::ZSTD;
  } else if (compression == common::CompressionKind_NONE) {
    return ::parquet::Compression::UNCOMPRESSED;
  } else if (compression == common::CompressionKind_LZ4) {
    return ::parquet::Compression::LZ4_HADOOP;
  } else {
    VELOX_FAIL("Unsupported compression {}", compression);
  }
}

std::shared_ptr<::parquet::WriterProperties> getArrowParquetWriterOptions(
    const parquet::WriterOptions& options,
    const std::unique_ptr<DefaultFlushPolicy>& flushPolicy) {
  auto builder = ::parquet::WriterProperties::Builder();
  ::parquet::WriterProperties::Builder* properties = &builder;
  if (!options.enableDictionary) {
    properties = properties->disable_dictionary();
  }
  properties =
      properties->compression(getArrowParquetCompression(options.compression));
  properties = properties->data_pagesize(options.dataPageSize);
  properties = properties->max_row_group_length(
      static_cast<int64_t>(flushPolicy->rowsInRowGroup()));
  return properties->build();
}

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options,
    std::shared_ptr<memory::MemoryPool> pool)
    : pool_(std::move(pool)),
      generalPool_{pool_->addLeafChild(".general")},
      stream_(std::make_shared<ArrowDataBufferSink>(
          std::move(sink),
          *generalPool_,
          options.bufferGrowRatio)),
      arrowContext_(std::make_shared<ArrowContext>()) {
  if (options.flushPolicyFactory) {
    flushPolicy_ = options.flushPolicyFactory();
  } else {
    flushPolicy_ = std::make_unique<DefaultFlushPolicy>();
  }
  arrowContext_->properties =
      getArrowParquetWriterOptions(options, flushPolicy_);
}

Writer::Writer(
    std::unique_ptr<dwio::common::FileSink> sink,
    const WriterOptions& options)
    : Writer{
          std::move(sink),
          options,
          options.memoryPool->addAggregateChild(fmt::format(
              "writer_node_{}",
              folly::to<std::string>(folly::Random::rand64())))} {}

void Writer::flush() {
  if (arrowContext_->stagingRows > 0) {
    if (!arrowContext_->writer) {
      auto arrowProperties =
          ::parquet::ArrowWriterProperties::Builder().build();
      PARQUET_THROW_NOT_OK(::parquet::arrow::FileWriter::Open(
          *arrowContext_->schema.get(),
          arrow::default_memory_pool(),
          stream_,
          arrowContext_->properties,
          arrowProperties,
          &arrowContext_->writer));
    }

    auto fields = arrowContext_->schema->fields();
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunks;
    for (int colIdx = 0; colIdx < fields.size(); colIdx++) {
      auto dataType = fields.at(colIdx)->type();
      auto chunk =
          arrow::ChunkedArray::Make(
              std::move(arrowContext_->stagingChunks.at(colIdx)), dataType)
              .ValueOrDie();
      chunks.push_back(chunk);
    }
    auto table = arrow::Table::Make(
        arrowContext_->schema,
        std::move(chunks),
        static_cast<int64_t>(arrowContext_->stagingRows));
    PARQUET_THROW_NOT_OK(arrowContext_->writer->WriteTable(
        *table, static_cast<int64_t>(flushPolicy_->rowsInRowGroup())));
    PARQUET_THROW_NOT_OK(stream_->Flush());
    for (auto& chunk : arrowContext_->stagingChunks) {
      chunk.clear();
    }
    arrowContext_->stagingRows = 0;
    arrowContext_->stagingBytes = 0;
  }
}

dwio::common::StripeProgress getStripeProgress(
    uint64_t stagingRows,
    int64_t stagingBytes) {
  return dwio::common::StripeProgress{
      .stripeRowCount = stagingRows, .stripeSizeEstimate = stagingBytes};
}

/**
 * This method would cache input `ColumnarBatch` to make the size of row group
 * big. It would flush when:
 * - the cached numRows bigger than `rowsInRowGroup_`
 * - the cached bytes bigger than `bytesInRowGroup_`
 *
 * This method assumes each input `ColumnarBatch` have same schema.
 */
void Writer::write(const VectorPtr& data) {
  ArrowArray array;
  ArrowSchema schema;
  exportToArrow(data, array, generalPool_.get());
  exportToArrow(data, schema);
  PARQUET_ASSIGN_OR_THROW(
      auto recordBatch, arrow::ImportRecordBatch(&array, &schema));
  if (!arrowContext_->schema) {
    arrowContext_->schema = recordBatch->schema();
    for (int colIdx = 0; colIdx < arrowContext_->schema->num_fields();
         colIdx++) {
      arrowContext_->stagingChunks.push_back(
          std::vector<std::shared_ptr<arrow::Array>>());
    }
  }

  auto bytes = data->estimateFlatSize();
  auto numRows = data->size();
  if (flushPolicy_->shouldFlush(getStripeProgress(
          arrowContext_->stagingRows, arrowContext_->stagingBytes))) {
    flush();
  }

  for (int colIdx = 0; colIdx < recordBatch->num_columns(); colIdx++) {
    arrowContext_->stagingChunks.at(colIdx).push_back(
        recordBatch->column(colIdx));
  }
  arrowContext_->stagingRows += numRows;
  arrowContext_->stagingBytes += bytes;
}

bool Writer::isCodecAvailable(common::CompressionKind compression) {
  return arrow::util::Codec::IsAvailable(
      getArrowParquetCompression(compression));
}

void Writer::newRowGroup(int32_t numRows) {
  PARQUET_THROW_NOT_OK(arrowContext_->writer->NewRowGroup(numRows));
}

void Writer::close() {
  flush();

  if (arrowContext_->writer) {
    PARQUET_THROW_NOT_OK(arrowContext_->writer->Close());
    arrowContext_->writer.reset();
  }
  PARQUET_THROW_NOT_OK(stream_->Close());

  arrowContext_->stagingChunks.clear();
}

void Writer::abort() {
  VELOX_NYI("abort function for Parquet writer is not supported yet");
}

parquet::WriterOptions getParquetOptions(
    const dwio::common::WriterOptions& options) {
  parquet::WriterOptions parquetOptions;
  parquetOptions.memoryPool = options.memoryPool;
  if (options.compressionKind.has_value()) {
    parquetOptions.compression = options.compressionKind.value();
  }
  return parquetOptions;
}

std::unique_ptr<dwio::common::Writer> ParquetWriterFactory::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    const dwio::common::WriterOptions& options) {
  auto parquetOptions = getParquetOptions(options);
  return std::make_unique<Writer>(std::move(sink), parquetOptions);
}

} // namespace facebook::velox::parquet
