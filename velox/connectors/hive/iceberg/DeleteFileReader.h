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

#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveDataSource.h"

#include <folly/Executor.h>
#include <memory>

namespace facebook::velox::cache {
class AsyncDataCache;
}

namespace facebook::velox::connector {
class ConnectorQueryCtx;
}

namespace facebook::velox::core {
class ExpressionEvaluator;
}

namespace facebook::velox::connector::hive::iceberg {

class IcebergDeleteFile;

class DeleteFileReader {
 public:
  DeleteFileReader(
      const IcebergDeleteFile& deleteFile,
      const std::string& baseFilePath,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      ConnectorQueryCtx* connectorQueryCtx,
      uint64_t splitOffset,
      const std::string& connectorId);

  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      int8_t* deleteBitmap);

  bool endOfFile();

 private:
  void createPositionalDeleteDataSource(
      const IcebergDeleteFile& deleteFile,
      const std::string& connectorId);

  void readDeletePositionsVec(
      uint64_t baseReadOffset,
      int64_t rowNumberUpperBound,
      int8_t* deleteBitmap);

  bool readFinishedForBatch(int64_t rowNumberUpperBound);

  const IcebergDeleteFile& deleteFile_;
  const std::string& baseFilePath_;
  FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  ConnectorQueryCtx* const connectorQueryCtx_;
  uint64_t splitOffset_;

  std::unique_ptr<HiveDataSource> deleteDataSource_;
  VectorPtr deletePositionsVector_;
  uint64_t deletePositionsOffset_;
  bool endOfFile_;
};

} // namespace facebook::velox::connector::hive::iceberg