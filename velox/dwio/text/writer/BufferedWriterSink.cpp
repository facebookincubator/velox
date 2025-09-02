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

#include "velox/dwio/text/writer/BufferedWriterSink.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::text {

BufferedWriterSink::BufferedWriterSink(
    std::unique_ptr<dwio::common::FileSink> sink,
    std::shared_ptr<memory::MemoryPool> pool,
    uint64_t flushBufferSize,
    const std::shared_ptr<dwio::common::WriterOptions>& options)
    : sink_(std::move(sink)),
      pool_(std::move(pool)),
      flushBufferSize_(flushBufferSize),
      buf_(std::make_unique<dwio::common::DataBuffer<char>>(*pool_)),
      options_(options),
      compressor_(nullptr) {
  if (options_ && options_->compressionKind.has_value()) {
    compressor_ = dwio::common::compression::createCompressor(
        options_->compressionKind.value(),
        getTextCompressionOptions(options_->compressionKind.value()));
  }

  reserveBuffer();
}

BufferedWriterSink::~BufferedWriterSink() {
  VELOX_CHECK_EQ(
      buf_->size(),
      0,
      "Unexpected buffer data on BufferedWriterSink destruction");
}

void BufferedWriterSink::write(char value) {
  write(&value, 1);
}

void BufferedWriterSink::write(const char* data, uint64_t size) {
  // TODO: Handle case when size is greater than flushBufferSize_
  VELOX_CHECK_GE(
      flushBufferSize_,
      size,
      "write data size exceeds flush buffer size limit");

  if (buf_->size() + size > flushBufferSize_) {
    flush();
  }
  buf_->append(buf_->size(), data, size);
}

void BufferedWriterSink::flush() {
  if (buf_->size() == 0) {
    return;
  }

  if (!options_ || !options_->compressionKind.has_value() ||
      options_->compressionKind.value() == common::CompressionKind_NONE) {
    sink_->write(std::move(*buf_));
    reserveBuffer();
    return;
  }

  dwio::common::ensureCapacity<char>(
      compressionBufferPtr_, buf_->size(), pool_.get(), false, false);
  char* dest = compressionBufferPtr_->asMutable<char>();
  const auto compressedSize =
      compressor_->compress(buf_->data(), dest, buf_->size());
  buf_->clear();
  buf_->append(buf_->size(), dest, compressedSize);
  sink_->write(std::move(*buf_));
  reserveBuffer();
}

void BufferedWriterSink::abort() {
  // TODO Add a flag to indicate sink is aborted to
  //  prevent flush and write operations after aborted.
  buf_->clear();
  sink_->close();
}

void BufferedWriterSink::close() {
  flush();
  buf_->clear();
  sink_->close();
}

void BufferedWriterSink::reserveBuffer() {
  VELOX_CHECK_NOT_NULL(buf_);
  VELOX_CHECK_EQ(buf_->size(), 0);
  VELOX_CHECK_EQ(buf_->capacity(), 0);
  buf_->reserve(flushBufferSize_);
}

} // namespace facebook::velox::text
