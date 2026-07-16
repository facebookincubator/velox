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

#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h" // @manual

#ifdef VELOX_ENABLE_ABFS
#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsFileSystem.h" // @manual
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h" // @manual
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h" // @manual
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h" // @manual
#include "velox/dwio/common/FileSink.h"
#endif

namespace facebook::velox::filesystems {

#ifdef VELOX_ENABLE_ABFS

std::shared_ptr<FileSystem> abfsFileSystemGenerator(
    std::shared_ptr<const config::ConfigBase> properties,
    std::string_view filePath) {
  // Cache FileSystem instances per unique set of Azure account configurations
  // to support multiple catalogs with different Azure storage accounts.
  // Each catalog may have different account credentials and configurations.
  static folly::Synchronized<
      std::unordered_map<std::vector<CacheKey>, std::shared_ptr<FileSystem>>>
      filesystemCache;

  auto cacheKeys = extractCacheKeyFromConfig(*properties);

  return filesystemCache.withWLock(
      [&](auto& cache) -> std::shared_ptr<FileSystem> {
        auto it = cache.find(cacheKeys);
        if (it != cache.end()) {
          return it->second;
        }
        auto filesystem = std::make_shared<AbfsFileSystem>(properties);
        cache[cacheKeys] = filesystem;
        return filesystem;
      });
}

std::unique_ptr<velox::dwio::common::FileSink> abfsWriteFileSinkGenerator(
    const std::string& fileURI,
    const velox::dwio::common::FileSink::Options& options) {
  if (isAbfsFile(fileURI)) {
    auto fileSystem =
        filesystems::getFileSystem(fileURI, options.connectorProperties);
    return std::make_unique<dwio::common::WriteFileSink>(
        fileSystem->openFileForWrite(fileURI),
        fileURI,
        options.metricLogger,
        options.stats,
        options.fileSystemStats);
  }
  return nullptr;
}
#endif

void registerAbfsFileSystem() {
#ifdef VELOX_ENABLE_ABFS
  registerFileSystem(isAbfsFile, std::function(abfsFileSystemGenerator));
  dwio::common::FileSink::registerFactory(
      std::function(abfsWriteFileSinkGenerator));
#endif
}

void registerAzureClientProvider(const config::ConfigBase& config) {
#ifdef VELOX_ENABLE_ABFS

  for (const auto& [accountNameWithSuffix, authType] :
       extractCacheKeyFromConfig(config)) {
    auto factory =
        AzureClientProviderFactories::getDefaultProviderFactory(authType);

    if (factory) {
      // Extract just the account name (before first dot) for registration
      auto accountName =
          accountNameWithSuffix.substr(0, accountNameWithSuffix.find('.'));
      AzureClientProviderFactories::registerFactory(accountName, factory);
    } else {
      VELOX_USER_FAIL(
          "Unsupported auth type {}, supported auth types are SharedKey, OAuth and SAS.",
          authType);
    }
  }
#endif
}

void registerAzureClientProviderFactory(
    const std::string& account,
    const AzureClientProviderFactory& factory) {
#ifdef VELOX_ENABLE_ABFS
  AzureClientProviderFactories::registerFactory(account, factory);
#endif
}

} // namespace facebook::velox::filesystems
