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

#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviders.h"

#include <folly/Synchronized.h>

namespace facebook::velox::filesystems {

namespace {
folly::Synchronized<
    std::unordered_map<std::string, std::shared_ptr<AzureClientProvider>>>&
azureClientProviderRegistry() {
  static folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<AzureClientProvider>>>
      providers;
  return providers;
}
} // namespace

void AzureClientProviders::registerProvider(
    const std::string& account,
    std::shared_ptr<AzureClientProvider> provider) {
  azureClientProviderRegistry().withWLock([&](auto& providers) {
    auto it = providers.find(account);
    VELOX_USER_CHECK(
        it != providers.end(),
        "AzureClientProvider for account '{}' is already registered.",
        account);
    providers[account] = std::move(provider);
  });
}

bool AzureClientProviders::providerRegistered(const std::string& account) {
  return azureClientProviderRegistry().withRLock([&](const auto& providers) {
    return providers.find(account) != providers.end();
  });
}

std::unique_ptr<AzureBlobClient> AzureClientProviders::getBlobClient(
    const AbfsConfig& config) {
  const auto& account = config.accountName();
  if (providerRegistered(account)) {
    const auto provider = azureClientProviderRegistry().withRLock(
        [&](const auto& providers) { return providers.at(account); });
    return provider->getBlobClient(
        config.getUrl(true), config.fileSystem(), config.filePath());
  }
  return config.getReadFileClient();
}

std::unique_ptr<AzureDataLakeFileClient>
AzureClientProviders::getDataLakeFileClient(const AbfsConfig& config) {
  const auto& account = config.accountName();
  if (providerRegistered(account)) {
    const auto provider = azureClientProviderRegistry().withRLock(
        [&](const auto& providers) { return providers.at(account); });
    return provider->getDataLakeFileClient(
        config.getUrl(false), config.fileSystem(), config.filePath());
  }
  return config.getWriteFileClient();
}

} // namespace facebook::velox::filesystems
