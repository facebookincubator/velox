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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsAsyncRuntime.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

using namespace facebook::velox;
using namespace facebook::velox::filesystems;

namespace {

class TestAzureBlobClient final : public AzureBlobClient {
 public:
  Azure::Response<Azure::Storage::Blobs::Models::BlobProperties> getProperties()
      override {
    VELOX_FAIL("TestAzureBlobClient: Not implemented.");
  }

  Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult> download(
      const Azure::Storage::Blobs::DownloadBlobOptions& options) override {
    VELOX_FAIL("TestAzureBlobClient: Not implemented.");
  }

  std::string getUrl() override {
    return "test";
  }
};

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

class LegacyAzureClientProvider final : public AzureClientProvider {
 public:
  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    return std::make_unique<TestAzureBlobClient>();
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("LegacyAzureClientProvider: Not implemented.");
  }
};

struct ProviderCallCounts {
  int factory{0};
  int sync{0};
  int fiber{0};
  std::string account;
  const Azure::Storage::Blobs::BlobClientOptions* options{nullptr};
};

class PairAzureClientProvider final : public AzureClientProvider {
 public:
  explicit PairAzureClientProvider(
      std::shared_ptr<ProviderCallCounts> callCounts)
      : callCounts_(std::move(callCounts)) {}

  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    ++callCounts_->sync;
    syncCreated_ = true;
    return std::make_unique<TestAzureBlobClient>();
  }

  std::unique_ptr<AzureBlobClient> getReadFileClientWithOptions(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config,
      const Azure::Storage::Blobs::BlobClientOptions& options) override {
    VELOX_CHECK(syncCreated_);
    ++callCounts_->fiber;
    callCounts_->options = &options;
    return std::make_unique<TestAzureBlobClient>();
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("PairAzureClientProvider: Not implemented.");
  }

 private:
  const std::shared_ptr<ProviderCallCounts> callCounts_;
  bool syncCreated_{false};
};

} // namespace

TEST(AzureClientProviderFactoriesTest, readFileClientsWithoutFiberOptions) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@pairednull.dfs.core.windows.net/file/test.txt");
  const auto callCounts = std::make_shared<ProviderCallCounts>();
  registerAzureClientProviderFactory(
      "pairednull", [callCounts](const std::string& account) {
        ++callCounts->factory;
        callCounts->account = account;
        return std::make_unique<PairAzureClientProvider>(callCounts);
      });

  auto clients = AzureClientProviderFactories::getReadFileClients(
      abfsPath, config::ConfigBase({}), nullptr);

  EXPECT_NE(clients.sync, nullptr);
  EXPECT_EQ(clients.fiber, nullptr);
  EXPECT_TRUE(clients.asyncUnsupportedReason.empty());
  EXPECT_EQ(clients.providerContext, "registered provider");
  EXPECT_EQ(callCounts->factory, 1);
  EXPECT_EQ(callCounts->sync, 1);
  EXPECT_EQ(callCounts->fiber, 0);
  EXPECT_EQ(callCounts->account, "pairednull");
}

TEST(AzureClientProviderFactoriesTest, readFileClientsWithFiberOptions) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@pairedoptions.dfs.core.windows.net/file/test.txt");
  const auto callCounts = std::make_shared<ProviderCallCounts>();
  registerAzureClientProviderFactory(
      "pairedoptions", [callCounts](const std::string& account) {
        ++callCounts->factory;
        callCounts->account = account;
        return std::make_unique<PairAzureClientProvider>(callCounts);
      });
  const Azure::Storage::Blobs::BlobClientOptions options;
  auto authService = std::make_shared<AbfsAsyncAuthService>(1, 1);
  const AzureAsyncReadContext context{options, authService};

  auto clients = AzureClientProviderFactories::getReadFileClients(
      abfsPath, config::ConfigBase({}), &context);

  EXPECT_NE(clients.sync, nullptr);
  EXPECT_NE(clients.fiber, nullptr);
  EXPECT_TRUE(clients.asyncUnsupportedReason.empty());
  EXPECT_EQ(clients.providerContext, "registered provider");
  EXPECT_EQ(callCounts->factory, 1);
  EXPECT_EQ(callCounts->sync, 1);
  EXPECT_EQ(callCounts->fiber, 1);
  EXPECT_EQ(callCounts->account, "pairedoptions");
  EXPECT_EQ(callCounts->options, &options);
}

TEST(
    AzureClientProviderFactoriesTest,
    legacyProviderDoesNotSupportFiberOptions) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://abc@pairedlegacy.dfs.core.windows.net/file/test.txt");
  int factoryCalls{0};
  registerAzureClientProviderFactory(
      "pairedlegacy", [&factoryCalls](const std::string& account) {
        ++factoryCalls;
        return std::make_unique<LegacyAzureClientProvider>();
      });
  const Azure::Storage::Blobs::BlobClientOptions options;
  auto authService = std::make_shared<AbfsAsyncAuthService>(1, 1);
  const AzureAsyncReadContext context{options, authService};

  auto clients = AzureClientProviderFactories::getReadFileClients(
      abfsPath, config::ConfigBase({}), &context);

  EXPECT_NE(clients.sync, nullptr);
  EXPECT_EQ(clients.fiber, nullptr);
  EXPECT_EQ(
      clients.asyncUnsupportedReason,
      "Azure client provider does not support read clients with custom options.");
  EXPECT_EQ(factoryCalls, 1);
}

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
          abfsPath, config::ConfigBase({})));
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getReadFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getWriteFileClient(
          abfsPath, config::ConfigBase({})),
      "DummyAzureClientProvider: Not implemented.");

  // Unregistered account on a non-public Azure cloud — the fallback auth-type
  // key must use the real suffix from the URL, not .dfs.core.windows.net.
  const auto unregisteredPath =
      std::make_shared<AbfsPath>("abfs://test@efg2.dfs.core.foobar.net/test");
  VELOX_ASSERT_THROW(
      AzureClientProviderFactories::getClientFactory(
          unregisteredPath, config::ConfigBase({})),
      "No AzureClientProviderFactory registered for account 'efg2' and no "
      "auth type found in config key 'fs.azure.account.auth.type.efg2.dfs.core.foobar.net'");
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
