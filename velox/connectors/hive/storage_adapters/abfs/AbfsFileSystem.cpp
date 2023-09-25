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
#include "AbfsFileSystem.h"
#include "AbfsReadFile.h"
#include "AbfsUtil.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/core/Config.h"

#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <glog/logging.h>

namespace facebook::velox::filesystems::abfs {
folly::once_flag abfsInitiationFlag;

class AbfsConfig {
 public:
  AbfsConfig(const Config* config) : config_(config) {}

  std::string connectionString(const std::string& path) const {
    if (!connectionString().empty()) {
      // This is only used for testing.
      return connectionString();
    }
    auto abfsAccount = AbfsAccount(path);
    auto key = abfsAccount.credKey();
    VELOX_USER_CHECK(
        config_->isValueExists(key), "Failed to find storage credentials");

    return abfsAccount.connectionString(useHttps(), config_->get(key).value());
  }

  bool useHttps() const {
    return config_->get<bool>(AbfsFileSystem::kReaderAbfsUseHttps, true);
  }

  std::string connectionString() const {
    return config_->get<std::string>(
        AbfsFileSystem::kReadAbfsConnectionStr, "");
  }

 private:
  const Config* FOLLY_NONNULL config_;
};

class AbfsFileSystem::Impl {
 public:
  explicit Impl(const Config* config) : abfsConfig_(config) {
    const size_t originCount = initCounter_++;
    LOG(INFO) << "Init ABFS file system (" << std::to_string(originCount)
              << ")";
  }

  ~Impl() {
    const size_t newCount = --initCounter_;
    LOG(INFO) << "Dispose ABFS file system(" << std::to_string(newCount) << ")";
  }

  const std::string connectionString(const std::string& path) const {
    // extract account name
    return abfsConfig_.connectionString(path);
  }

 private:
  const AbfsConfig abfsConfig_;
  std::shared_ptr<folly::Executor> ioExecutor_;
  static std::atomic<size_t> initCounter_;
};

std::atomic<size_t> AbfsFileSystem::Impl::initCounter_(0);
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

static std::function<std::shared_ptr<FileSystem>(
    std::shared_ptr<const Config>,
    std::string_view)>
    filesystemGenerator = [](std::shared_ptr<const Config> properties,
                             std::string_view filePath) {
      static std::shared_ptr<FileSystem> filesystem;
      folly::call_once(abfsInitiationFlag, [&properties]() {
        filesystem = std::make_shared<AbfsFileSystem>(properties);
      });
      return filesystem;
    };

void registerAbfsFileSystem() {
  LOG(INFO) << "Register ABFS";
  registerFileSystem(isAbfsFile, filesystemGenerator);
}
} // namespace facebook::velox::filesystems::abfs