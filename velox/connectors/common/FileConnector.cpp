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

#include "velox/connectors/common/FileConnector.h"

namespace facebook::velox::connector {

namespace {

FileHandleFactory makeFileHandleFactory(
    const FileConnectorConfig& config) {
  return FileHandleFactory(
      config.isFileHandleCacheEnabled()
          ? std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(
                config.numCacheFileHandles(),
                config.fileHandleExpirationDurationMs())
          : nullptr,
      std::make_unique<FileHandleGenerator>(config.config()));
}

} // namespace

FileConnector::FileConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* executor)
    : FileConnector(
          id,
          std::move(config),
          std::make_shared<FileConnectorConfig>(connectorConfig()),
          executor) {}

FileConnector::FileConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    std::shared_ptr<FileConnectorConfig> fileConfig,
    folly::Executor* executor)
    : Connector(id, std::move(config)),
      fileConnectorConfig_(std::move(fileConfig)),
      fileHandleFactory_(makeFileHandleFactory(*fileConnectorConfig_)),
      ioExecutor_(executor) {}

} // namespace facebook::velox::connector
