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

/// Handles the registration of Azure client providers and the creation of
/// AzureBlobClient and AzureDataLakeFileClient instances.
/// For accounts with a custom AzureClientProvider implementation registered,
/// use the `getBlobClient` and `getDataLakeFileClient` methods in the custom
/// provider to get the clients. If no implementation is registered for a
/// specific account, use the default client providers in
/// `DefaultAzureClientProviders.h` to create clients based on the
/// authentication configuration in `AbfsConfig.h`. Supported authentication
/// methods are SharedKey, OAuth, and fixed SAS token.
class AzureClientProviderFactories {
 public:
  /// Registers a factory for creating AzureClientProvider instances.
  static void registerFactory(
      const std::string& account,
      const AzureClientProviderFactory& factory);

  /// Checks if a factory is registered for the given account.
  static bool clientFactoryRegistered(const std::string& account);

  /// Uses the registered AzureClientProviderFactory to create an
  /// AzureBlobClient. If no factory is registered for the account
  /// specified in `abfsPath`, falls back to the default client providers in
  /// `DefaultAzureClientProviders.h`, based on the authentication configuration
  /// in `config`. A temporary AzureClientProvider instance will be created to
  /// get the client.
  static std::unique_ptr<AzureBlobClient> getBlobClient(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);

  /// Uses the registered AzureClientProviderFactory to create an
  /// AzureDataLakeFileClient. If no factory is registered for the account
  /// specified in `abfsPath`, falls back to the default client providers in
  /// `DefaultAzureClientProviders.h`, based on the authentication configuration
  /// in `config`. A temporary AzureClientProvider instance will be created to
  /// get the client.
  static std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config);
};

} // namespace facebook::velox::filesystems
