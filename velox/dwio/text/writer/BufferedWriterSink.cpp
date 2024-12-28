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

namespace facebook::velox::text {

BufferedWriterSink::BufferedWriterSink(
    std::unique_ptr<dwio::common::FileSink> sink,
    std::shared_ptr<memory::MemoryPool> pool,
    int64_t flushCount)
    : sink_(std::move(sink)), pool_(std::move(pool)), flushCount_(flushCount) {
  initializeBuffer();
}

BufferedWriterSink::~BufferedWriterSink() {
  VELOX_CHECK_EQ(
      buf_->size(),
      0,
      "There are still data in data buffer when BufferedWriterSink is destroyed");
}

void BufferedWriterSink::write(char value) {
  if (buf_->size() >= flushCount_) {
    flush();
  }
  buf_->append(value);
}

void BufferedWriterSink::write(const char* data, uint64_t size) {
  VELOX_CHECK_GE(flushCount_, size, "size is larger than flushCount");
  // TODO Add logic for when size is larger than flushCount_

  if (buf_->size() + size > flushCount_) {
    flush();
  }
  buf_->append(buf_->size(), data, size);
}

void BufferedWriterSink::flush() {
  if (buf_->size() == 0) {
    return;
  }

  sink_->write(std::move(*buf_));
  initializeBuffer();
}

void BufferedWriterSink::close() {
  flush();
  buf_->clear();
  sink_->close();
}

void BufferedWriterSink::abort() {
  buf_->clear();
  sink_->close();
}

void BufferedWriterSink::initializeBuffer() {
  if (buf_ == nullptr) {
    buf_ = std::make_unique<dwio::common::DataBuffer<char>>(*pool_);
    buf_->reserve(flushCount_);
  }
}

} // namespace facebook::velox::text
