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

#include <fmt/format.h>
#include <folly/Synchronized.h>

#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h"

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

const std::unordered_map<std::string, AzureClientProviderFactory>&
defaultProviderRegistry() {
  static const std::unordered_map<std::string, AzureClientProviderFactory>
      kRegistry = {
          {kAzureSharedKeyAuthType,
           [](const std::string& /*account*/) {
             return std::make_unique<SharedKeyAzureClientProvider>();
           }},
          {kAzureOAuthAuthType,
           [](const std::string& /*account*/) {
             return std::make_unique<OAuthAzureClientProvider>();
           }},
          {kAzureSASAuthType,
           [](const std::string& /*account*/) {
             return std::make_unique<FixedSasAzureClientProvider>();
           }},
      };
  return kRegistry;
}

} // namespace

void AzureClientProviderFactories::registerFactory(
    const std::string& account,
    const AzureClientProviderFactory& factory) {
  azureClientFactoryRegistry().withWLock([&](auto& factories) {
    auto [_, inserted] = factories.insert_or_assign(account, factory);
    LOG_IF(INFO, !inserted) << "AzureClientProviderFactory for account '"
                            << account << "' has been overridden.";
  });
}

AzureClientProviderFactory
AzureClientProviderFactories::getDefaultProviderFactory(
    const std::string& authType) {
  const auto& registry = defaultProviderRegistry();
  auto it = registry.find(authType);

  if (it != registry.end()) {
    return it->second;
  }

  return nullptr;
}

AzureClientProviderFactory AzureClientProviderFactories::getClientFactory(
    const std::string& account,
    const config::ConfigBase& config) {
  return azureClientFactoryRegistry().withRLock(
      [&](const auto& factories) -> AzureClientProviderFactory {
        if (auto it = factories.find(account); it != factories.end()) {
          return it->second;
        }
        LOG(INFO) << "No AzureClientProviderFactory registered for account '"
                  << account << "', creating default provider from config.";

        // Extract auth type from config to avoid capturing non-copyable
        // ConfigBase. This allows the returned factory to be safely stored and
        // called later.
        auto authTypeKey = fmt::format(
            "{}.{}{}", kAzureAccountAuthType, account, kAzureAccountNameSuffix);
        VELOX_USER_CHECK(
            config.valueExists(authTypeKey),
            "No AzureClientProviderFactory registered for account '{}' and no "
            "auth type found in config key '{}'. "
            "Please either register a factory using `registerAzureClientProvider` "
            "or `registerAzureClientProviderFactory`, or provide auth type in config.",
            account,
            authTypeKey);

        auto authType = config.get<std::string>(authTypeKey).value();

        // Return a factory based on the auth type that doesn't depend on config
        auto factory = getDefaultProviderFactory(authType);
        VELOX_USER_CHECK(
            factory != nullptr,
            "Unsupported auth type '{}' for account '{}'. "
            "Supported auth types are SharedKey, OAuth and SAS.",
            authType,
            account);

        return factory;
      });
}

std::unique_ptr<AzureBlobClient>
AzureClientProviderFactories::getReadFileClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  auto factory = getClientFactory(abfsPath->accountName(), config);
  return factory(abfsPath->accountName())->getReadFileClient(abfsPath, config);
}

std::unique_ptr<AzureDataLakeFileClient>
AzureClientProviderFactories::getWriteFileClient(
    const std::shared_ptr<AbfsPath>& abfsPath,
    const config::ConfigBase& config) {
  auto factory = getClientFactory(abfsPath->accountName(), config);
  return factory(abfsPath->accountName())->getWriteFileClient(abfsPath, config);
}

} // namespace facebook::velox::filesystems
