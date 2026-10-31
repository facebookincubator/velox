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

namespace facebook::velox::connector::file {

FileConnector::FileConnector(
    const std::string& id,
    std::shared_ptr<FileConfig> fileConfig,
    folly::Executor* ioExecutor)
    : Connector(id, fileConfig->config()),
      fileConfig_(std::move(fileConfig)),
      fileHandleFactory_(
          fileConfig_->isFileHandleCacheEnabled()
              ? std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(
                    fileConfig_->numCacheFileHandles(),
                    fileConfig_->fileHandleExpirationDurationMs())
              : nullptr,
          std::make_unique<FileHandleGenerator>(fileConfig_->config())),
      ioExecutor_(ioExecutor) {
  if (fileConfig_->isFileHandleCacheEnabled()) {
    LOG(INFO) << "File connector " << connectorId()
              << " created with maximum of "
              << fileConfig_->numCacheFileHandles()
              << " cached file handles with expiration of "
              << fileConfig_->fileHandleExpirationDurationMs() << "ms.";
  } else {
    LOG(INFO) << "File connector " << connectorId()
              << " created with file handle cache disabled";
  }
}

} // namespace facebook::velox::connector::file
