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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"
#include "velox/dwio/common/FileSink.h"
#endif

#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"

namespace facebook::velox::filesystems {

#ifdef VELOX_ENABLE_S3
using FileSystemMap = folly::Synchronized<
    std::unordered_map<std::string, std::shared_ptr<FileSystem>>>;

/// Multiple S3 filesystems are supported.
/// Key is the endpoint value specified in the config using hive.s3.endpoint.
/// If the endpoint is empty, it will default to AWS S3.
FileSystemMap& fileSystems() {
  static FileSystemMap instances;
  return instances;
}

std::string getS3Identity(std::shared_ptr<core::MemConfig>& config) {
  HiveConfig hiveConfig = HiveConfig(config);
  auto endpoint = hiveConfig.s3Endpoint();
  if (!endpoint.empty()) {
    // The identity is the endpoint.
    return endpoint;
  }
  // Default key value.
  return "aws-s3-key";
}

std::shared_ptr<FileSystem> fileSystemGenerator(
    std::shared_ptr<const Config> properties,
    std::string_view /*filePath*/) {
  std::shared_ptr<core::MemConfig> config = std::make_shared<core::MemConfig>();
  if (properties) {
    *config = core::MemConfig(properties->values());
  }
  const auto s3Identity = getS3Identity(config);

  return fileSystems().withWLock(
      [&](auto& instanceMap) -> std::shared_ptr<FileSystem> {
        initializeS3(config.get());
        if (instanceMap.count(s3Identity) == 0) {
          auto fs = std::make_shared<S3FileSystem>(properties);
          instanceMap.insert({s3Identity, fs});
          return fs;
        }
        return instanceMap[s3Identity];
      });
}

std::unique_ptr<velox::dwio::common::FileSink> s3WriteFileSinkGenerator(
    const std::string& fileURI,
    const velox::dwio::common::FileSink::Options& options) {
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
}
#endif

void registerS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  if (fileSystems()->empty()) {
    registerFileSystem(isS3File, std::function(fileSystemGenerator));
    dwio::common::FileSink::registerFactory(
        std::function(s3WriteFileSinkGenerator));
  }
#endif
}

void finalizeS3FileSystem() {
#ifdef VELOX_ENABLE_S3
  bool singleUseCount = true;
  fileSystems().withRLock([&](auto& instanceMap) {
    for (const auto& [id, fs] : instanceMap) {
      singleUseCount &= (fs.use_count() == 1);
    }
  });

  VELOX_CHECK(singleUseCount, "Cannot finalize S3FileSystem while in use");
  fileSystems()->clear();
  finalizeS3();
#endif
}

} // namespace facebook::velox::filesystems
