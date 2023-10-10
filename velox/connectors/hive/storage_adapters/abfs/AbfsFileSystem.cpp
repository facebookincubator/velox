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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsFileSystem.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/core/Config.h"

#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <glog/logging.h>

namespace facebook::velox::filesystems::abfs {
class AbfsConfig {
 public:
  AbfsConfig(const Config* config) : config_(config) {}

  std::string connectionString(const std::string& path) const {
    auto abfsAccount = AbfsAccount(path);
    auto key = abfsAccount.credKey();
    VELOX_USER_CHECK(
        config_->isValueExists(key), "Failed to find storage credentials");

    return abfsAccount.connectionString(config_->get(key).value());
  }

 private:
  const Config* FOLLY_NONNULL config_;
};

class AbfsFileSystem::Impl {
 public:
  explicit Impl(const Config* config) : abfsConfig_(config) {
    LOG(INFO) << "Init Azure Blob file system";
  }

  ~Impl() {
    LOG(INFO) << "Dispose Azure Blob file system";
  }

  const std::string connectionString(const std::string& path) const {
    // Extract account name
    return abfsConfig_.connectionString(path);
  }

 private:
  const AbfsConfig abfsConfig_;
  std::shared_ptr<folly::Executor> ioExecutor_;
};

AbfsFileSystem::AbfsFileSystem(const std::shared_ptr<const Config>& config)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get());
}

std::string AbfsFileSystem::name() const {
  return "ABFS";
}

std::unique_ptr<ReadFile> AbfsFileSystem::openFileForRead(
    std::string_view path,
    const FileOptions& /*unused*/) {
  auto abfsfile = std::make_unique<AbfsReadFile>(
      std::string(path), impl_->connectionString(std::string(path)));
  abfsfile->initialize();
  return abfsfile;
}
} // namespace facebook::velox::filesystems::abfs
