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
#include "velox/connectors/hive/storage_adapters/abfs/DefaultAzureClientProviders.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

using namespace facebook::velox;
using namespace facebook::velox::filesystems;

namespace {

class DummyAzureClientProvider final : public AzureClientProvider {
 public:
  DummyAzureClientProvider(
      const std::shared_ptr<AbfsPath>& abfsPath,
      const config::ConfigBase& config)
      : AzureClientProvider(abfsPath) {}

  std::unique_ptr<AzureBlobClient> getBlobClient() override {
    VELOX_FAIL("Not implemented.");
  }

  std::unique_ptr<AzureDataLakeFileClient> getDataLakeFileClient() override {
    VELOX_FAIL("Not implemented.");
  }
};

} // namespace

TEST(AzureClientProviderFactoriesTest, createFromConfig) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@efg.dfs.core.windows.net/file/test.txt");

  {
    // OAuth auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "OAuth"},
         {"fs.azure.account.oauth2.client.id.efg.dfs.core.windows.net", "123"},
         {"fs.azure.account.oauth2.client.secret.efg.dfs.core.windows.net",
          "456"},
         {"fs.azure.account.oauth2.client.endpoint.efg.dfs.core.windows.net",
          "https://login.microsoftonline.com/{TENANTID}/oauth2/token"}},
        false);
    ASSERT_NE(
        AzureClientProviderFactories::getBlobClient(abfsPath, config), nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getDataLakeFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SharedKey auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SharedKey"},
         {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
        false);
    ASSERT_NE(
        AzureClientProviderFactories::getBlobClient(abfsPath, config), nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getDataLakeFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SAS auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.fixed.token.efg.dfs.core.windows.net", "456"}},
        false);
    ASSERT_NE(
        AzureClientProviderFactories::getBlobClient(abfsPath, config), nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getDataLakeFileClient(abfsPath, config),
        nullptr);
  }

  {
    // Invalid auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "Custom"},
         {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
        false);
    const std::string error =
        "Unsupported auth type Custom, supported auth types are SharedKey, OAuth and SAS.";
    VELOX_ASSERT_USER_THROW(
        AzureClientProviderFactories::getBlobClient(abfsPath, config), error);
    VELOX_ASSERT_USER_THROW(
        AzureClientProviderFactories::getDataLakeFileClient(abfsPath, config),
        error);
  }
}

TEST(AzureClientProviderFactoriesTest, registerAzureClientFactory) {
  static const std::string path = "abfs://test@efg.dfs.core.windows.net/test";
  const auto abfsPath = std::make_shared<AbfsPath>(path);

  registerAzureClientProviderFactory(
      "efg",
      [](const std::shared_ptr<AbfsPath>& path,
         const config::ConfigBase& config)
          -> std::unique_ptr<AzureClientProvider> {
        return std::make_unique<DummyAzureClientProvider>(path, config);
      });

  ASSERT_TRUE(AzureClientProviderFactories::clientFactoryRegistered("efg"));
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getBlobClient(
          abfsPath, config::ConfigBase({})),
      "Not implemented.");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getDataLakeFileClient(
          abfsPath, config::ConfigBase({})),
      "Not implemented.");

  ASSERT_FALSE(AzureClientProviderFactories::clientFactoryRegistered("efg2"));
}
