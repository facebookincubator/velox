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

#include "velox/common/file/FileInputStream.h"
#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {

class QueryTraceDataReader {
 public:
  explicit QueryTraceDataReader(std::string path);

  /// Read stream from an item of stream_ and deserialized into the batch,
  /// return false if there is no more data.
  bool read(RowVectorPtr& batch) const;

 private:
  RowTypePtr traceDataType(
      const std::shared_ptr<filesystems::FileSystem>& fs) const;
  std::unique_ptr<common::FileInputStream> getFileInputStream(
      const std::shared_ptr<filesystems::FileSystem>& fs) const;

  const std::string path_;
  const serializer::presto::PrestoVectorSerde::PrestoOptions readOptions_{
      true,
      common::CompressionKind_ZSTD, // TODO: Use trace config.
      /*nulsFirst*/ true};

  memory::MemoryPool* const pool_{
      memory::MemoryManager::getInstance()->tracePool()};
  RowTypePtr type_;
  std::unique_ptr<common::FileInputStream> stream_;
};
} // namespace facebook::velox::exec
