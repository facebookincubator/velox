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
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviders.h"

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

void factoryNotRegistered(const std::string& account) {
  VELOX_USER_FAIL(
      "No AzureClientProviderFactory registered for account '{}'. Please use "
      "`registerAzureClientProviderFactory` to register a factory for the "
      "account before using it.",
      account);
}

} // namespace

void AzureClientProviderFactories::registerFactory(
    const std::string& account,
    const AzureClientProviderFactory& factory) {
  azureClientFactoryRegistry().withWLock([&](auto& factories) {
    auto it = factories.find(account);
    factories[account] = factory;
  });
}

std::optional<AzureClientProviderFactory>
AzureClientProviderFactories::getClientFactory(const std::string& account) {
  return azureClientFactoryRegistry().withRLock(
      [&](const auto& factories) -> std::optional<AzureClientProviderFactory> {
        if (auto it = factories.find(account); it != factories.end()) {
          return it->second;
        }
        return std::nullopt;
      });
}

std::unique_ptr<AzureBlobClient>
AzureClientProviderFactories::getReadFileClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  if (const auto factory = getClientFactory(abfsPath->accountName())) {
    return factory.value()(abfsPath->accountName())
        ->getReadFileClient(abfsPath, config);
  }
  factoryNotRegistered(abfsPath->accountName());
  VELOX_UNREACHABLE();
}

std::unique_ptr<AzureDataLakeFileClient>
AzureClientProviderFactories::getWriteFileClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  if (const auto factory = getClientFactory(abfsPath->accountName())) {
    return factory.value()(abfsPath->accountName())
        ->getWriteFileClient(abfsPath, config);
  }
  factoryNotRegistered(abfsPath->accountName());
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::filesystems
