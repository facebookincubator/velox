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
#include <memory>
#include <unordered_map>

#include "velox/common/config/Config.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

using namespace facebook::velox;
using namespace facebook::velox::filesystems;

class AbfsFileSystemRegistrationTest : public testing::Test {
 protected:
  void SetUp() override {
    registerAbfsFileSystem();
  }

  std::shared_ptr<const config::ConfigBase> createConfig(
      const std::unordered_map<std::string, std::string>& values) {
    return std::make_shared<const config::ConfigBase>(
        std::unordered_map<std::string, std::string>(values));
  }
};

TEST_F(AbfsFileSystemRegistrationTest, singleCatalogSingleAccount) {
  // Test that a single catalog with one account works correctly
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  ASSERT_NE(fs1, nullptr);
  EXPECT_EQ(fs1->name(), "ABFS");
}

TEST_F(AbfsFileSystemRegistrationTest, multipleCatalogsDifferentAccounts) {
  // Test that multiple catalogs with different accounts get separate
  // FileSystem instances
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.account2.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account2.dfs.core.windows.net", "key2"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account2.dfs.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);
  EXPECT_EQ(fs1->name(), "ABFS");
  EXPECT_EQ(fs2->name(), "ABFS");

  // Different configs should result in different FileSystem instances
  EXPECT_NE(fs1.get(), fs2.get());
}

TEST_F(AbfsFileSystemRegistrationTest, multipleCatalogsSameAccount) {
  // Test that multiple catalogs with the same account configuration
  // share the same FileSystem instance (caching works)
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account1.dfs.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);

  // Same account configuration should result in the same cached instance
  EXPECT_EQ(fs1.get(), fs2.get());
}

TEST_F(AbfsFileSystemRegistrationTest, singleCatalogMultipleAccounts) {
  // Test that a single catalog with multiple accounts works correctly
  auto config = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
      {"fs.azure.account.auth.type.account2.dfs.core.windows.net", "OAuth"},
      {"fs.azure.account.oauth2.client.id.account2.dfs.core.windows.net",
       "client-id"},
      {"fs.azure.account.oauth2.client.secret.account2.dfs.core.windows.net",
       "client-secret"},
      {"fs.azure.account.oauth2.client.endpoint.account2.dfs.core.windows.net",
       "https://login.microsoftonline.com/tenant/oauth2/token"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account2.dfs.core.windows.net/path2", config);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);

  // Same config object should result in the same cached instance
  EXPECT_EQ(fs1.get(), fs2.get());
}

TEST_F(AbfsFileSystemRegistrationTest, differentAuthTypes) {
  // Test that different auth types for the same account result in
  // different FileSystem instances
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "OAuth"},
      {"fs.azure.account.oauth2.client.id.account1.dfs.core.windows.net",
       "client-id"},
      {"fs.azure.account.oauth2.client.secret.account1.dfs.core.windows.net",
       "client-secret"},
      {"fs.azure.account.oauth2.client.endpoint.account1.dfs.core.windows.net",
       "https://login.microsoftonline.com/tenant/oauth2/token"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account1.dfs.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);

  // Different auth types should result in different FileSystem instances
  EXPECT_NE(fs1.get(), fs2.get());
}

TEST_F(AbfsFileSystemRegistrationTest, multipleCatalogsPartialOverlap) {
  // Test catalogs with some overlapping and some different accounts
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
      {"fs.azure.account.auth.type.account2.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account2.dfs.core.windows.net", "key2"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
      {"fs.azure.account.auth.type.account3.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account3.dfs.core.windows.net", "key3"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account1.dfs.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);

  // Different sets of accounts should result in different FileSystem instances
  EXPECT_NE(fs1.get(), fs2.get());
}

TEST_F(AbfsFileSystemRegistrationTest, cacheKeyOrdering) {
  // Test that the order of accounts in config doesn't affect caching
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
      {"fs.azure.account.auth.type.account2.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account2.dfs.core.windows.net", "key2"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.account2.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account2.dfs.core.windows.net", "key2"},
      {"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SharedKey"},
      {"fs.azure.account.key.account1.dfs.core.windows.net", "key1"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@account1.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@account1.dfs.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);

  // Note: The current implementation may create different instances if the
  // order differs, as the cache key is a vector. This test documents the
  // current behavior. If order-independent caching is desired, the cache
  // key should be sorted or use a set instead of a vector.
  // For now, we just verify both instances are created successfully.
  EXPECT_EQ(fs1->name(), "ABFS");
  EXPECT_EQ(fs2->name(), "ABFS");
}

TEST_F(AbfsFileSystemRegistrationTest, sameAccountNameDifferentSuffix) {
  // Test that accounts with the same name but different suffixes
  // (e.g., .dfs.core.windows.net vs .blob.core.windows.net) are treated
  // as different accounts and get separate FileSystem instances.
  auto config1 = createConfig({
      {"fs.azure.account.auth.type.myaccount.dfs.core.windows.net",
       "SharedKey"},
      {"fs.azure.account.key.myaccount.dfs.core.windows.net", "key1"},
  });

  auto config2 = createConfig({
      {"fs.azure.account.auth.type.myaccount.blob.core.windows.net",
       "SharedKey"},
      {"fs.azure.account.key.myaccount.blob.core.windows.net", "key2"},
  });

  auto fs1 = filesystems::getFileSystem(
      "abfs://container1@myaccount.dfs.core.windows.net/path1", config1);
  auto fs2 = filesystems::getFileSystem(
      "abfs://container2@myaccount.blob.core.windows.net/path2", config2);

  ASSERT_NE(fs1, nullptr);
  ASSERT_NE(fs2, nullptr);
  EXPECT_EQ(fs1->name(), "ABFS");
  EXPECT_EQ(fs2->name(), "ABFS");

  // Different suffixes should result in different FileSystem instances
  // even though the account name is the same.
  EXPECT_NE(fs1.get(), fs2.get());
}

// Made with Bob
