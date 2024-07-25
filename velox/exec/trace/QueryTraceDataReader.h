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

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {
class TraceInputStream final : public ByteInputStream {
 public:
  TraceInputStream(std::unique_ptr<ReadFile>&& file, BufferPtr buffer)
      : file_(std::move(file)), size_(file_->size()), buffer_(buffer) {
    next(true);
  }

  /// True if all of the file has been read into vectors.
  bool atEnd() const override {
    return offset_ >= size_ && ranges()[0].position >= ranges()[0].size;
  }

 private:
  void next(bool throwIfPastEnd) override;

  const std::unique_ptr<ReadFile> file_;
  const uint64_t size_;
  const BufferPtr buffer_;
  // Offset of first byte not in 'buffer_'
  uint64_t offset_{0};
};

class QueryTraceDataReader {
 public:
  explicit QueryTraceDataReader(std::string path);

  /// Read stream from an item of stream_ and deserialized into the batch,
  /// return false if there is no more data.
  bool read(RowVectorPtr& batch) const;

 private:
  void loadSummary();

  const std::string path_;
  RowTypePtr type_{nullptr};
  memory::MemoryPool* const pool_{
      memory::MemoryManager::getInstance()->tracePool()};
  std::unique_ptr<TraceInputStream> stream_;
  const serializer::presto::PrestoVectorSerde::PrestoOptions readOptions_{
      true,
      common::CompressionKind_NONE,
      true /*nulsFirst*/};
};
} // namespace facebook::velox::exec
