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

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureBlobClient.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProvider.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureDataLakeFileClient.h"

#include <functional>
#include <memory>
#include <string>

namespace facebook::velox::filesystems {

using AzureClientProviderFactory =
    std::function<std::unique_ptr<AzureClientProvider>(
        const std::shared_ptr<AbfsPath>& path,
        const config::ConfigBase& config)>;

/// Handles the registration and retrieval of Azure client providers.
/// If no AzureClientProvider implementation is registered for a specific
/// account, DefaultAzureClientProviderFactory is used to create the client
/// based on the configuration provided in AbfsConfig.h.
class AzureClientProviderFactories {
 public:
  static void registerFactory(
      const std::string& account,
      const AzureClientProviderFactory& factory);

  static bool clientFactoryRegistered(const std::string& account);

  static std::unique_ptr<AzureBlobClient> getBlobClient(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  static std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);
};

} // namespace facebook::velox::filesystems
