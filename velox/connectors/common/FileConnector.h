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

#include "velox/connectors/Connector.h"
#include "velox/connectors/common/FileConnectorConfig.h"
#include "velox/connectors/common/FileHandle.h"

namespace facebook::velox::connector {

/// Base connector for file-based table formats. Manages the file handle cache,
/// IO executor, and connector configuration. Subclasses (Hive, Iceberg, Paimon)
/// provide format-specific DataSource and DataSink implementations.
class FileConnector : public Connector {
 public:
  FileConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor);

  bool canAddDynamicFilter() const override {
    return true;
  }

  bool supportsSplitPreload() const override {
    return true;
  }

  folly::Executor* ioExecutor() const override {
    return ioExecutor_;
  }

  FileHandleCacheStats fileHandleCacheStats() {
    return fileHandleFactory_.cacheStats();
  }

  FileHandleCacheStats clearFileHandleCache() {
    return fileHandleFactory_.clearCache();
  }

 protected:
  /// Protected constructor for subclasses that provide their own config type
  /// (e.g., HiveConfig). The fileConfig must be pre-built by the subclass.
  FileConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      std::shared_ptr<FileConnectorConfig> fileConfig,
      folly::Executor* executor);

  const std::shared_ptr<FileConnectorConfig> fileConnectorConfig_;
  FileHandleFactory fileHandleFactory_;
  folly::Executor* const ioExecutor_;
};

} // namespace facebook::velox::connector
