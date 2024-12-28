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

#include "velox/dwio/common/FileSink.h"

namespace facebook::velox::text {

/// Takes character(s) and writes into a 'sink'.
/// It will buffer the characters(s) in memory before flushing to the sink.
/// The upper limit character count is specified by 'flushCount'.
class BufferedWriterSink {
 public:
  BufferedWriterSink(
      std::unique_ptr<dwio::common::FileSink> sink,
      std::shared_ptr<memory::MemoryPool> pool,
      int64_t flushCount);

  ~BufferedWriterSink();

  void write(char value);
  void write(const char* data, uint64_t size);
  void flush();
  void close();
  void abort();

 private:
  const std::unique_ptr<dwio::common::FileSink> sink_;
  const std::shared_ptr<memory::MemoryPool> pool_;
  // The upper limit character count.
  const int64_t flushCount_;
  std::unique_ptr<dwio::common::DataBuffer<char>> buf_;

  void initializeBuffer();
};

} // namespace facebook::velox::text
