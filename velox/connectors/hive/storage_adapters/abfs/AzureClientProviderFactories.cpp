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

#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "velox/connectors/hive/storage_adapters/abfs/DefaultAzureClientProvider.h"

#include <folly/Synchronized.h>
#include <unordered_map>

namespace facebook::velox::filesystems {

namespace {

folly::Synchronized<
    std::unordered_map<std::string, AzureClientProviderFactory>>&
azureClientFactoryRegistry() {
  static folly::Synchronized<
      std::unordered_map<std::string, AzureClientProviderFactory>>
      factories;
  return factories;
}

} // namespace

void AzureClientProviderFactories::registerFactory(
    const std::string& account,
    const AzureClientProviderFactory& factory) {
  azureClientFactoryRegistry().withWLock([&](auto& factories) {
    auto it = factories.find(account);
    VELOX_USER_CHECK(
        it == factories.end(),
        "AzureClientProvider for account '{}' is already registered.",
        account);
    factories[account] = factory;
  });
}

bool AzureClientProviderFactories::clientFactoryRegistered(
    const std::string& account) {
  return azureClientFactoryRegistry().withRLock([&](const auto& factories) {
    return factories.find(account) != factories.end();
  });
}

std::unique_ptr<AzureBlobClient> AzureClientProviderFactories::getBlobClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  if (clientFactoryRegistered(abfsPath->accountName())) {
    const auto factory =
        azureClientFactoryRegistry().withRLock([&](const auto& factories) {
          return factories.at(abfsPath->accountName());
        });
    return factory(abfsPath, config)->getBlobClient();
  }
  return DefaultAzureClientProvider(abfsPath, config).getBlobClient();
}

std::unique_ptr<AzureDataLakeFileClient>
AzureClientProviderFactories::getDataLakeFileClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  if (clientFactoryRegistered(abfsPath->accountName())) {
    const auto factory =
        azureClientFactoryRegistry().withRLock([&](const auto& factories) {
          return factories.at(abfsPath->accountName());
        });
    return factory(abfsPath, config)->getDataLakeFileClient();
  }
  return DefaultAzureClientProvider(abfsPath, config).getDataLakeFileClient();
}

} // namespace facebook::velox::filesystems
