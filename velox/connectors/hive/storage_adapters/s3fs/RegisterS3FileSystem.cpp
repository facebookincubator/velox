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

#ifdef VELOX_ENABLE_S3
#include "folly/concurrency/ConcurrentHashMap.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"
#include "velox/core/Config.h"
#include "velox/dwio/common/FileSink.h"
#endif

#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"

namespace facebook::velox::filesystems {

#ifdef VELOX_ENABLE_S3
std::mutex mtx1, mtx2;
/// Multiple S3 filesystems are supported.
/// Key is the endpoint value specified in the config using hive.s3.endpoint.
/// If the endpoint is empty, it will default to AWS S3.
/// We cannot use folly::ConcurrentHashMap to store the filesystem instances.
/// This is because we need to delete the filesystems during
/// finalizeS3FileSystem and folly::ConcurrentHashMap does not guarantee the
/// entries are cleared on invoking clear().
static std::unordered_map<std::string, std::shared_ptr<FileSystem>> filesystems;

std::function<std::shared_ptr<
    FileSystem>(std::shared_ptr<const Config>, std::string_view)>
fileSystemGenerator() {
  static auto filesystemGenerator = [](std::shared_ptr<const Config> properties,
                                       std::string_view filePath) {
    static folly::
        ConcurrentHashMap<std::string, std::shared_ptr<folly::once_flag>>
            s3InitiationFlags;
    std::string s3Identity("aws-s3");
    if (properties) {
      auto endpoint = connector::hive::HiveConfig::s3Endpoint(properties.get());
      if (!endpoint.empty()) {
        // The identity is the endpoint.
        s3Identity = endpoint;
      }
    }
    std::unique_lock<std::mutex> lk(mtx1, std::defer_lock);
    /// If the init flag for a given s3 endpoint is not found,
    /// create one for init use. It's a singleton.
    if (s3InitiationFlags.find(s3Identity) == s3InitiationFlags.end()) {
      lk.lock();
      if (s3InitiationFlags.find(s3Identity) == s3InitiationFlags.end()) {
        std::shared_ptr<folly::once_flag> initiationFlagPtr =
            std::make_shared<folly::once_flag>();
        s3InitiationFlags.insert(s3Identity, initiationFlagPtr);
      }
      lk.unlock();
    }
    folly::call_once(
        *s3InitiationFlags[s3Identity], [&properties, &s3Identity]() {
          std::shared_ptr<S3FileSystem> fs;
          if (properties != nullptr) {
            initializeS3(properties.get());
            fs = std::make_shared<S3FileSystem>(properties);
          } else {
            auto config = std::make_shared<core::MemConfig>();
            initializeS3(config.get());
            fs = std::make_shared<S3FileSystem>(config);
          }
          std::lock_guard ins(mtx2);
          filesystems.emplace(s3Identity, fs);
        });
    return filesystems[s3Identity];
  };
  return filesystemGenerator;
}

std::function<std::unique_ptr<velox::dwio::common::FileSink>(
    const std::string&,
    const velox::dwio::common::FileSink::Options& options)>
s3WriteFileSinkGenerator() {
  static auto s3WriteFileSink =
      [](const std::string& fileURI,
         const velox::dwio::common::FileSink::Options& options)
      -> std::unique_ptr<dwio::common::WriteFileSink> {
    if (isS3File(fileURI)) {
      auto fileSystem =
          filesystems::getFileSystem(fileURI, options.connectorProperties);
      return std::make_unique<dwio::common::WriteFileSink>(
          fileSystem->openFileForWrite(fileURI, {{}, options.pool}),
          fileURI,
          options.metricLogger,
          options.stats);
    }
    return nullptr;
  };

  return s3WriteFileSink;
}
#endif

void registerS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  if (filesystems.empty()) {
    registerFileSystem(isS3File, fileSystemGenerator());
    dwio::common::FileSink::registerFactory(s3WriteFileSinkGenerator());
  }
#endif
}

void finalizeS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  bool singleUseCount = true;
  for (const auto& [id, fs] : filesystems) {
    singleUseCount &= (fs.use_count() == 1);
  }
  VELOX_CHECK(singleUseCount, "Cannot finalize S3FileSystem while in use");
  filesystems.clear();
  finalizeS3();
#endif
}

} // namespace facebook::velox::filesystems
