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

#include "connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsConfig.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"

#include <azure/storage/blobs/blob_sas_builder.hpp>

#include <chrono>
#include "gtest/gtest.h"

using namespace facebook::velox::filesystems;
using namespace facebook::velox;

namespace {

class MyFixedAbfsSasTokenProvider : public AbfsSasTokenProvider {
 public:
  void initialize(
      const config::ConfigBase& conf,
      const std::string& accountNameWithSuffix) override {
    const auto firstDot = accountNameWithSuffix.find_first_of(".");
    ASSERT_NE(firstDot, std::string::npos);
    account_ = accountNameWithSuffix.substr(0, firstDot);
  }

  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    return fmt::format("sas={}_sas_token", account_);
  }

 private:
  std::string account_;
};

class MyDynamicAbfsSasTokenProvider : public AbfsSasTokenProvider {
 public:
  MyDynamicAbfsSasTokenProvider(int64_t expiration)
      : expirationSeconds_(expiration) {}

  void initialize(
      const config::ConfigBase& conf,
      const std::string& accountNameWithSuffix) override {
    const auto firstDot = accountNameWithSuffix.find_first_of(".");
    ASSERT_NE(firstDot, std::string::npos);
    account_ = accountNameWithSuffix.substr(0, firstDot);
  }

  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    const auto lastSlash = path.find_last_of("/");
    const auto containerName = path.substr(0, lastSlash);
    const auto blobName = path.substr(lastSlash + 1);

    Azure::Storage::Sas::BlobSasBuilder sasBuilder;
    sasBuilder.ExpiresOn = Azure::DateTime::clock::now() +
        std::chrono::seconds(expirationSeconds_);
    sasBuilder.BlobContainerName = containerName;
    sasBuilder.BlobName = blobName;
    sasBuilder.Resource = Azure::Storage::Sas::BlobSasResource::Blob;
    sasBuilder.SetPermissions(
        Azure::Storage::Sas::BlobSasPermissions::Read &
        Azure::Storage::Sas::BlobSasPermissions::Write);

    std::string sasToken =
        sasBuilder.GenerateSasToken(Azure::Storage::StorageSharedKeyCredential(
            "test",
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="));

    // Remove the leading '?' from the SAS token.
    if (sasToken[0] == '?') {
      sasToken = sasToken.substr(1);
    }

    return sasToken;
  }

 private:
  int64_t expirationSeconds_;
  std::string account_;
};

} // namespace

TEST(AbfsUtilsTest, isAbfsFile) {
  EXPECT_FALSE(isAbfsFile("abfs:"));
  EXPECT_FALSE(isAbfsFile("abfss:"));
  EXPECT_FALSE(isAbfsFile("abfs:/"));
  EXPECT_FALSE(isAbfsFile("abfss:/"));
  EXPECT_TRUE(isAbfsFile("abfs://test@test.dfs.core.windows.net/test"));
  EXPECT_TRUE(isAbfsFile("abfss://test@test.dfs.core.windows.net/test"));
}

TEST(AbfsConfigTest, authType) {
  const config::ConfigBase config(
      {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "Custom"},
       {"fs.azure.account.key.efg.dfs.core.windows.net", "456"}},
      false);
  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@efg.dfs.core.windows.net/test.txt", config),
      "Unsupported auth type Custom, supported auth types are SharedKey, OAuth and SAS.");
}

TEST(AbfsConfigTest, clientSecretOAuth) {
  const config::ConfigBase config(
      {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "OAuth"},
       {"fs.azure.account.auth.type.bar1.dfs.core.windows.net", "OAuth"},
       {"fs.azure.account.auth.type.bar2.dfs.core.windows.net", "OAuth"},
       {"fs.azure.account.auth.type.bar3.dfs.core.windows.net", "OAuth"},
       {"fs.azure.account.oauth2.client.id.efg.dfs.core.windows.net", "test"},
       {"fs.azure.account.oauth2.client.secret.efg.dfs.core.windows.net",
        "test"},
       {"fs.azure.account.oauth2.client.endpoint.efg.dfs.core.windows.net",
        "https://login.microsoftonline.com/{TENANTID}/oauth2/token"},
       {"fs.azure.account.oauth2.client.id.bar2.dfs.core.windows.net", "test"},
       {"fs.azure.account.oauth2.client.id.bar3.dfs.core.windows.net", "test"},
       {"fs.azure.account.oauth2.client.secret.bar3.dfs.core.windows.net",
        "test"}},
      false);
  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@bar1.dfs.core.windows.net/test.txt", config),
      "Config fs.azure.account.oauth2.client.id.bar1.dfs.core.windows.net not found");
  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@bar2.dfs.core.windows.net/test.txt", config),
      "Config fs.azure.account.oauth2.client.secret.bar2.dfs.core.windows.net not found");
  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@bar3.dfs.core.windows.net/test.txt", config),
      "Config fs.azure.account.oauth2.client.endpoint.bar3.dfs.core.windows.net not found");
  auto abfsConfig =
      AbfsConfig("abfss://abc@efg.dfs.core.windows.net/file/test.txt", config);
  EXPECT_EQ(abfsConfig.tenentId(), "{TENANTID}");
  EXPECT_EQ(abfsConfig.authorityHost(), "https://login.microsoftonline.com/");
  auto readClient = abfsConfig.getReadFileClient();
  EXPECT_EQ(
      readClient->GetUrl(),
      "https://efg.blob.core.windows.net/abc/file/test.txt");
  auto writeClient = abfsConfig.getWriteFileClient();
  // GetUrl retrieves the value from the internal blob client, which represents
  // the blob's path as well.
  EXPECT_EQ(
      writeClient->getUrl(),
      "https://efg.blob.core.windows.net/abc/file/test.txt");
}

TEST(AbfsConfigTest, sasToken) {
  const config::ConfigBase config(
      {{"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SAS"},
       {"fs.azure.account.auth.type.bar.dfs.core.windows.net", "SAS"},
       {"fs.azure.sas.fixed.token.bar.dfs.core.windows.net", "sas=test"}},
      false);
  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@efg.dfs.core.windows.net/test.txt", config),
      "Config fs.azure.sas.fixed.token.efg.dfs.core.windows.net not found");
  auto abfsConfig =
      AbfsConfig("abfs://abc@bar.dfs.core.windows.net/file", config);
  auto readClient = abfsConfig.getReadFileClient();
  EXPECT_EQ(
      readClient->GetUrl(),
      "http://bar.blob.core.windows.net/abc/file?sas=test");
  auto writeClient = abfsConfig.getWriteFileClient();
  // GetUrl retrieves the value from the internal blob client, which represents
  // the blob's path as well.
  EXPECT_EQ(
      writeClient->getUrl(),
      "http://bar.blob.core.windows.net/abc/file?sas=test");

  // Honor the registered SAS token provider.
  registerAbfsSasTokenProvider(
      "efg", [] { return std::make_unique<MyFixedAbfsSasTokenProvider>(); });
  abfsConfig = AbfsConfig("abfs://abc@efg.dfs.core.windows.net/file", config);
  readClient = abfsConfig.getReadFileClient();
  EXPECT_EQ(
      readClient->GetUrl(),
      "http://efg.blob.core.windows.net/abc/file?sas=efg_sas_token");
  writeClient = abfsConfig.getWriteFileClient();
  EXPECT_EQ(
      writeClient->getUrl(),
      "http://efg.blob.core.windows.net/abc/file?sas=efg_sas_token");

  VELOX_ASSERT_USER_THROW(
      registerAbfsSasTokenProvider(
          "efg",
          [] { return std::make_unique<MyFixedAbfsSasTokenProvider>(); }),
      "SAS key generator for efg already registered");
}

TEST(AbfsConfigTest, dynamicSasToken) {
  {
    const std::string account = "account1";
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.token.renew.period.for.streams", "1"}},
        false);
    registerAbfsSasTokenProvider(account, [] {
      return std::make_unique<MyDynamicAbfsSasTokenProvider>(3);
    });

    auto abfsConfig = AbfsConfig(
        fmt::format("abfs://abc@{}.dfs.core.windows.net/file", account),
        config);
    auto readClient = abfsConfig.getReadFileClient();
    auto writeClient = abfsConfig.getWriteFileClient();

    auto readUrl = readClient->GetUrl();
    auto writeUrl = writeClient->getUrl();

    // Let the current time pass 3 seconds to ensure the SAS token is expired.
    std::this_thread::sleep_for(std::chrono::seconds(3)); // NOLINT

    auto newReadUrl = readClient->GetUrl();
    ASSERT_NE(readUrl, newReadUrl);
    // The SAS token should be reused.
    ASSERT_EQ(newReadUrl, readClient->GetUrl());

    auto newWriteUrl = writeClient->getUrl();
    ASSERT_NE(writeUrl, newWriteUrl);
    // The SAS token should be reused.
    ASSERT_EQ(newWriteUrl, writeClient->getUrl());
  }

  {
    // SAS token expired by setting the renewal period to 120 seconds.
    const std::string account = "account2";
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.account2.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.token.renew.period.for.streams", "120"}},
        false);
    registerAbfsSasTokenProvider(account, [] {
      return std::make_unique<MyDynamicAbfsSasTokenProvider>(60);
    });

    auto abfsConfig = AbfsConfig(
        fmt::format("abfs://abc@{}.dfs.core.windows.net/file", account),
        config);
    auto readClient = abfsConfig.getReadFileClient();
    auto writeClient = abfsConfig.getWriteFileClient();

    auto readUrl = readClient->GetUrl();
    auto writeUrl = writeClient->getUrl();

    // Let the current time pass 3 seconds to ensure the timestamp in the SAS
    // token is updated.
    std::this_thread::sleep_for(std::chrono::seconds(3)); // NOLINT

    // Sas token should be renewed because the time left is less than the
    // renewal period.
    ASSERT_NE(readUrl, readClient->GetUrl());
    ASSERT_NE(writeUrl, writeClient->getUrl());
  }
}

TEST(AbfsConfigTest, sharedKey) {
  const config::ConfigBase config(
      {{"fs.azure.account.key.efg.dfs.core.windows.net", "123"},
       {"fs.azure.account.auth.type.efg.dfs.core.windows.net", "SharedKey"},
       {"fs.azure.account.key.foobar.dfs.core.windows.net", "456"},
       {"fs.azure.account.key.bar.dfs.core.windows.net", "789"}},
      false);

  auto abfsConfig =
      AbfsConfig("abfs://abc@efg.dfs.core.windows.net/file", config);
  EXPECT_EQ(abfsConfig.fileSystem(), "abc");
  EXPECT_EQ(abfsConfig.filePath(), "file");
  EXPECT_EQ(
      abfsConfig.connectionString(),
      "DefaultEndpointsProtocol=http;AccountName=efg;AccountKey=123;EndpointSuffix=core.windows.net;");

  auto abfssConfig = AbfsConfig(
      "abfss://abc@foobar.dfs.core.windows.net/sf_1/store_sales/ss_sold_date_sk=2450816/part-00002-a29c25f1-4638-494e-8428-a84f51dcea41.c000.snappy.parquet",
      config);
  EXPECT_EQ(abfssConfig.fileSystem(), "abc");
  EXPECT_EQ(
      abfssConfig.filePath(),
      "sf_1/store_sales/ss_sold_date_sk=2450816/part-00002-a29c25f1-4638-494e-8428-a84f51dcea41.c000.snappy.parquet");
  EXPECT_EQ(
      abfssConfig.connectionString(),
      "DefaultEndpointsProtocol=https;AccountName=foobar;AccountKey=456;EndpointSuffix=core.windows.net;");

  // Test with special character space.
  auto abfssConfigWithSpecialCharacters = AbfsConfig(
      "abfss://foo@bar.dfs.core.windows.net/main@dir/sub dir/test.txt", config);

  EXPECT_EQ(abfssConfigWithSpecialCharacters.fileSystem(), "foo");
  EXPECT_EQ(
      abfssConfigWithSpecialCharacters.filePath(), "main@dir/sub dir/test.txt");

  VELOX_ASSERT_USER_THROW(
      std::make_unique<AbfsConfig>(
          "abfss://foo@otheraccount.dfs.core.windows.net/test.txt", config),
      "Config fs.azure.account.key.otheraccount.dfs.core.windows.net not found");
}
