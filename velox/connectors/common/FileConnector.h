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

#include "velox/common/caching/FileHandle.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/common/FileConfig.h"

namespace facebook::velox::connector::file {

/// Owns file-access resources shared by file-based connectors. Derived
/// connectors provide table-specific scan, write and capability entry points.
class FileConnector : public Connector {
 public:
  folly::Executor* ioExecutor() const override {
    return ioExecutor_;
  }

  FileHandleCacheStats fileHandleCacheStats() {
    return fileHandleFactory_.cacheStats();
  }

  /// Clears the file handle cache for operational purposes.
  FileHandleCacheStats clearFileHandleCache() {
    return fileHandleFactory_.clearCache();
  }

 protected:
  FileConnector(
      const std::string& id,
      std::shared_ptr<FileConfig> fileConfig,
      folly::Executor* ioExecutor);

  const std::shared_ptr<FileConfig> fileConfig_;
  FileHandleFactory fileHandleFactory_;
  folly::Executor* ioExecutor_;
};

} // namespace facebook::velox::connector::file
