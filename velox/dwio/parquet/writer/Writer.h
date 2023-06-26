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

#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/DataSink.h"

#include "velox/core/Context.h"
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/vector/ComplexVector.h"

#include <parquet/arrow/writer.h> // @manual

namespace facebook::velox::parquet {

// Utility for capturing Arrow output into a DataBuffer.
class DataBufferSink : public arrow::io::OutputStream {
 public:
  explicit DataBufferSink(
      dwio::common::DataSink* sink,
      memory::MemoryPool& pool,
      uint32_t growRatio = 1)
      : sink_(sink), buffer_(pool), growRatio_(growRatio) {}

  arrow::Status Write(const std::shared_ptr<arrow::Buffer>& data) override {
    buffer_.append(
        buffer_.size(),
        reinterpret_cast<const char*>(data->data()),
        data->size(),
        growRatio_);
    return arrow::Status::OK();
  }

  arrow::Status Write(const void* data, int64_t nbytes) override {
    buffer_.append(
        buffer_.size(),
        reinterpret_cast<const char*>(data),
        nbytes,
        growRatio_);
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

  dwio::common::DataBuffer<char>& dataBuffer() {
    return buffer_;
  }

 private:
  dwio::common::DataSink* sink_;
  dwio::common::DataBuffer<char> buffer_;
  uint32_t growRatio_ = 1;
  int64_t bytesFlushed_ = 0;
};

// Writes Velox vectors into  a DataSink using Arrow Parquet writer.
class Writer {
 public:
  // Constructts a writer with output to 'sink'. A new row group is
  // started every 'rowsInRowGroup' top level rows. 'pool' is used for
  // temporary memory. 'properties' specifies Parquet-specific
  // options.
  Writer(
      std::unique_ptr<dwio::common::DataSink> sink,
      memory::MemoryPool& pool,
      int32_t rowsInRowGroup,
      std::shared_ptr<::parquet::WriterProperties> properties =
          ::parquet::WriterProperties::Builder().build(),
      std::shared_ptr<velox::core::QueryCtx> queryCtx =
          std::make_shared<velox::core::QueryCtx>(nullptr))
      : rowsInRowGroup_(rowsInRowGroup),
        pool_(pool),
        finalSink_(std::move(sink)),
        properties_(std::move(properties)),
        queryCtx_(std::move(queryCtx)) {}

  // Appends 'data' into the writer.
  void write(const RowVectorPtr& data);

  void flush();

  // Forces a row group boundary before the data added by next write().
  void newRowGroup(int32_t numRows);

  // Closes 'this', After close, data can no longer be added and the completed
  // Parquet file is flushed into 'sink' provided at construction. 'sink' stays
  // live until destruction of 'this'.
  void close();

 private:
  const int32_t rowsInRowGroup_;

  // Pool for 'stream_'.
  memory::MemoryPool& pool_;

  // Final destination of output.
  std::unique_ptr<dwio::common::DataSink> finalSink_;

  // Temporary Arrow stream for capturing the output.
  std::shared_ptr<DataBufferSink> stream_;

  std::unique_ptr<::parquet::arrow::FileWriter> arrowWriter_;

  std::shared_ptr<::parquet::WriterProperties> properties_;
  std::shared_ptr<velox::core::QueryCtx> queryCtx_;
};

} // namespace facebook::velox::parquet
