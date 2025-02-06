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
#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h" // @manual
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h" // @manual
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h" // @manual
#include "velox/dwio/common/FileSink.h"
#endif

namespace facebook::velox::filesystems {

#ifdef VELOX_ENABLE_S3
using FileSystemMap = folly::Synchronized<
    std::unordered_map<std::string, std::shared_ptr<FileSystem>>>;

/// Multiple S3 filesystems are supported.
/// Key is the endpoint value specified in the config using hive.s3.endpoint.
/// If the endpoint is empty, it will default to AWS S3 Library.
/// Different S3 buckets can be accessed with different client configurations.
/// This allows for different endpoints, data read and write strategies.
/// The bucket specific option is set by replacing the hive.s3. prefix on an
/// option with hive.s3.bucket.BUCKETNAME., where BUCKETNAME is the name of the
/// bucket. When connecting to a bucket, all options explicitly set will
/// override the base hive.s3. values.

FileSystemMap& fileSystems() {
  static FileSystemMap instances;
  return instances;
}

std::shared_ptr<FileSystem> fileSystemGenerator(
    std::shared_ptr<const config::ConfigBase> properties,
    std::string_view s3Path) {
  std::string bucketName, key;
  getBucketAndKeyFromPath(getPath(s3Path), bucketName, key);
  auto identity = S3Config::identity(bucketName, properties);

  // Check if an instance exists with a read lock (shared).
  auto fs = fileSystems().withRLock(
      [&](auto& instanceMap) -> std::shared_ptr<FileSystem> {
        auto iterator = instanceMap.find(identity);
        if (iterator != instanceMap.end()) {
          return iterator->second;
        }
        return nullptr;
      });
  if (fs != nullptr) {
    return fs;
  }

  return fileSystems().withWLock(
      [&](auto& instanceMap) -> std::shared_ptr<FileSystem> {
        // Repeat the checks with a write lock.
        auto iterator = instanceMap.find(identity);
        if (iterator != instanceMap.end()) {
          return iterator->second;
        }

        auto logLevel =
            properties->get(S3Config::kS3LogLevel, std::string("FATAL"));
        initializeS3(logLevel);
        auto fs = std::make_shared<S3FileSystem>(bucketName, properties);
        instanceMap.insert({identity, fs});
        return fs;
      });
}

std::unique_ptr<velox::dwio::common::FileSink> s3WriteFileSinkGenerator(
    const std::string& fileURI,
    const velox::dwio::common::FileSink::Options& options) {
  if (isS3File(fileURI)) {
    auto fileSystem =
        filesystems::getFileSystem(fileURI, options.connectorProperties);
    return std::make_unique<dwio::common::WriteFileSink>(
        fileSystem->openFileForWrite(fileURI, {{}, options.pool, std::nullopt}),
        fileURI,
        options.metricLogger,
        options.stats);
  }
  return nullptr;
}
#endif

void registerS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  fileSystems().withWLock([&](auto& instanceMap) {
    if (instanceMap.empty()) {
      registerFileSystem(isS3File, std::function(fileSystemGenerator));
      dwio::common::FileSink::registerFactory(
          std::function(s3WriteFileSinkGenerator));
    }
  });
#endif
}

void finalizeS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  bool singleUseCount = true;
  fileSystems().withWLock([&](auto& instanceMap) {
    for (const auto& [id, fs] : instanceMap) {
      singleUseCount &= (fs.use_count() == 1);
    }
    VELOX_CHECK(singleUseCount, "Cannot finalize S3FileSystem while in use");
    instanceMap.clear();
  });

  finalizeS3();
#endif
}

} // namespace facebook::velox::filesystems
