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
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h"
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

} // namespace

TEST(AzureClientProviderFactoriesTest, registerFromConfig) {
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
    registerAzureClientProvider(config);

    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SharedKey auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SharedKey"},
         {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
        false);
    registerAzureClientProvider(config);

    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SAS auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.fixed.token.efg.dfs.core.windows.net", "456"}},
        false);
    registerAzureClientProvider(config);

    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // Invalid auth type.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "Custom"},
         {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
        false);
    VELOX_ASSERT_THROW(
        registerAzureClientProvider(config),
        "Unsupported auth type Custom, supported auth types are SharedKey, OAuth and SAS.");
  }

  {
    // Invalid config key - missing suffix.
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.efg", "SharedKey"},
         {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
        false);
    VELOX_ASSERT_THROW(
        registerAzureClientProvider(config),
        "Invalid Azure account auth type key: fs.azure.account.auth.type.efg");
  }
}

TEST(AzureClientProviderFactoriesTest, registerCustomFactory) {
  static const std::string path = "abfs://test@efg.dfs.core.windows.net/test";
  const auto abfsPath = std::make_shared<AbfsPath>(path);

  registerAzureClientProviderFactory(
      "efg",
      [](const std::string& account) -> std::unique_ptr<AzureClientProvider> {
        return std::make_unique<DummyAzureClientProvider>();
      });

  ASSERT_NO_THROW(
      AzureClientProviderFactories::getClientFactory(
          "efg", config::ConfigBase({})));
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getReadFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getWriteFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getClientFactory(
          "efg2", config::ConfigBase({})),
      "No AzureClientProviderFactory registered for account 'efg2' and no "
      "auth type found in config key 'fs.azure.account.auth.type.efg2.dfs.core.windows.net'");
}

TEST(
    AzureClientProviderFactoriesTest,
    defaultProviderFromConfigWithoutRegistration) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@testaccount.dfs.core.windows.net/file/test.txt");

  {
    // OAuth auth type - should work without explicit registration
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.testaccount.dfs.core.windows.net",
          "OAuth"},
         {"fs.azure.account.oauth2.client.id.testaccount.dfs.core.windows.net",
          "123"},
         {"fs.azure.account.oauth2.client.secret.testaccount.dfs.core.windows.net",
          "456"},
         {"fs.azure.account.oauth2.client.endpoint.testaccount.dfs.core.windows.net",
          "https://login.microsoftonline.com/{TENANTID}/oauth2/token"}},
        false);

    // Should create client without prior registration
    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SharedKey auth type - should work without explicit registration
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.testaccount.dfs.core.windows.net",
          "SharedKey"},
         {"fs.azure.account.key.testaccount.dfs.core.windows.net",
          "dGVzdGtleQ=="}},
        false);

    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // SAS auth type - should work without explicit registration
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.testaccount.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.fixed.token.testaccount.dfs.core.windows.net",
          "sv=2021-06-08&ss=b&srt=sco&sp=rwdlac"}},
        false);

    ASSERT_NE(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        nullptr);
    ASSERT_NE(
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config),
        nullptr);
  }

  {
    // Invalid auth type - should fail with clear error
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.testaccount.dfs.core.windows.net",
          "InvalidAuth"},
         {"fs.azure.account.key.testaccount.dfs.core.windows.net", "456"}},
        false);

    VELOX_ASSERT_THROW(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        "Unsupported auth type 'InvalidAuth' for account 'testaccount'");
  }

  {
    // Missing auth type config - should fail with clear error
    const config::ConfigBase config(
        {{"fs.azure.account.key.testaccount.dfs.core.windows.net", "456"}},
        false);

    VELOX_ASSERT_THROW(
        AzureClientProviderFactories::getReadFileClient(abfsPath, config),
        "No AzureClientProviderFactory registered for account 'testaccount' and no auth type found in config key 'fs.azure.account.auth.type.testaccount.dfs.core.windows.net'");
  }
}

TEST(AzureClientProviderFactoriesTest, registeredFactoryTakesPrecedence) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@precedencetest.dfs.core.windows.net/file/test.txt");

  // Register a custom factory
  registerAzureClientProviderFactory(
      "precedencetest",
      [](const std::string& account) -> std::unique_ptr<AzureClientProvider> {
        return std::make_unique<DummyAzureClientProvider>();
      });

  // Even with valid config, registered factory should take precedence
  const config::ConfigBase config(
      {{"fs.azure.account.auth.type.precedencetest.dfs.core.windows.net",
        "SharedKey"},
       {"fs.azure.account.key.precedencetest.dfs.core.windows.net", "456"}},
      false);

  // Should use the registered DummyAzureClientProvider, not create from config
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getReadFileClient(abfsPath, config),
      "DummyAzureClientProvider: Not implemented.");
}

TEST(AzureClientProviderFactoriesTest, multipleAccountsSingleConfig) {
  const auto abfsPath1 = std::make_shared<AbfsPath>(
      "abfss://abc@account1.dfs.core.windows.net/file/test.txt");
  const auto abfsPath2 = std::make_shared<AbfsPath>(
      "abfss://abc@account2.dfs.core.windows.net/file/test.txt");

  // Even with valid config, registered factory should take precedence.
  const config::ConfigBase config(
      {{"fs.azure.account.auth.type.account1.dfs.core.windows.net",
        "SharedKey"},
       {"fs.azure.account.key.account1.dfs.core.windows.net", "123"},
       {"fs.azure.account.auth.type.account2.dfs.core.windows.net",
        "SharedKey"},
       {"fs.azure.account.key.account2.dfs.core.windows.net", "456"}},
      false);

  ASSERT_NE(
      AzureClientProviderFactories::getReadFileClient(abfsPath1, config),
      nullptr);
  ASSERT_NE(
      AzureClientProviderFactories::getWriteFileClient(abfsPath1, config),
      nullptr);
  ASSERT_NE(
      AzureClientProviderFactories::getReadFileClient(abfsPath2, config),
      nullptr);
  ASSERT_NE(
      AzureClientProviderFactories::getWriteFileClient(abfsPath2, config),
      nullptr);
}
