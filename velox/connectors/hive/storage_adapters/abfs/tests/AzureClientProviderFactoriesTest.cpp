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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviders.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

using namespace facebook::velox;
using namespace facebook::velox::filesystems;

namespace {

class DummyAzureClientProvider final : public AzureClientProvider {
 public:
  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("DummyAzureClientProvider: Not implemented.");
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("DummyAzureClientProvider: Not implemented.");
  }
};

void registerOAuthClientProviderFactory(const std::string& accountName) {
  registerAzureClientProviderFactory(accountName, [](const auto& accountName) {
    return std::make_unique<OAuthAzureClientProvider>();
  });
}

void registerFixedSasClientProviderFactory(const std::string& accountName) {
  registerAzureClientProviderFactory(accountName, [](const auto& accountName) {
    return std::make_unique<FixedSasAzureClientProvider>();
  });
}

void registerSharedKeyClientProviderFactory(const std::string& accountName) {
  registerAzureClientProviderFactory(accountName, [](const auto& accountName) {
    return std::make_unique<SharedKeyAzureClientProvider>();
  });
}

} // namespace

TEST(AzureClientProviderFactoriesTest, createFromConfig) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@efg.dfs.core.windows.net/file/test.txt");

  {
    // OAuth auth type.
    registerOAuthClientProviderFactory("efg");
    const config::ConfigBase config(
        {{"fs.azure.account.oauth2.client.id.efg.dfs.core.windows.net", "123"},
         {"fs.azure.account.oauth2.client.secret.efg.dfs.core.windows.net",
          "456"},
         {"fs.azure.account.oauth2.client.endpoint.efg.dfs.core.windows.net",
          "https://login.microsoftonline.com/{TENANTID}/oauth2/token"}},
        false);
    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SharedKey auth type.
    registerSharedKeyClientProviderFactory("efg");
    const config::ConfigBase config(
        {{"fs.azure.account.key.efg.dfs.core.windows.net", "456"}}, false);
    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SAS auth type.
    registerFixedSasClientProviderFactory("efg");
    const config::ConfigBase config(
        {{"fs.azure.sas.fixed.token.efg.dfs.core.windows.net", "456"}}, false);
    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }
}

TEST(AzureClientProviderFactoriesTest, registerAzureClientFactory) {
  static const std::string path = "abfs://test@efg.dfs.core.windows.net/test";
  const auto abfsPath = std::make_shared<AbfsPath>(path);

  registerAzureClientProviderFactory(
      "efg",
      [](const std::string& account) -> std::unique_ptr<AzureClientProvider> {
        return std::make_unique<DummyAzureClientProvider>();
      });

  ASSERT_TRUE(
      AzureClientProviderFactories::getClientFactory("efg").has_value());
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getReadFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getWriteFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");

  ASSERT_FALSE(
      AzureClientProviderFactories::getClientFactory("efg2").has_value());
}
