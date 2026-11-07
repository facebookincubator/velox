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

#include "velox/connectors/hive/storage_adapters/abfs/DynamicSasTokenClientProvider.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsAsyncRuntime.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

#include "gtest/gtest.h"

#include <azure/storage/blobs/blob_sas_builder.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

using namespace facebook::velox::filesystems;
using namespace facebook::velox;

namespace {

constexpr auto kWaitTimeout = std::chrono::seconds(5);

bool waitForAuthWaiters(
    const std::shared_ptr<AbfsAsyncAuthService>& authService,
    size_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (authService->metrics().waitingRefreshes == expected) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

class MyDynamicAbfsSasTokenProvider : public SasTokenProvider {
 public:
  MyDynamicAbfsSasTokenProvider(int64_t expiration)
      : expirationSeconds_(expiration) {}

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

    std::string sasToken = sasBuilder.GenerateSasToken(
        Azure::Storage::StorageSharedKeyCredential(
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
};

class SyntheticSasTokenProvider final : public SasTokenProvider {
 public:
  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    ++calls_;
    return "se=2099-01-01T00%3A00%3A00Z&marker=synthetic";
  }

  size_t calls() const {
    return calls_;
  }

 private:
  std::atomic<size_t> calls_{0};
};

class SequencedSasTokenProvider final : public SasTokenProvider {
 public:
  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    std::unique_lock lock(mutex_);
    ++calls_;
    if (calls_ == 1) {
      return "se=2099-01-01T00%3A00%3A00Z&marker=initial";
    }
    refreshStarted_ = true;
    refreshStartedCondition_.notify_all();
    releaseRefreshCondition_.wait(lock, [&] { return refreshReleased_; });
    return "se=9999-01-01T00%3A00%3A00Z&marker=renewed";
  }

  bool waitForRefresh() {
    std::unique_lock lock(mutex_);
    return refreshStartedCondition_.wait_for(
        lock, kWaitTimeout, [&] { return refreshStarted_; });
  }

  void releaseRefresh() {
    std::lock_guard lock(mutex_);
    refreshReleased_ = true;
    releaseRefreshCondition_.notify_all();
  }

  size_t calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable refreshStartedCondition_;
  std::condition_variable releaseRefreshCondition_;
  bool refreshStarted_{false};
  bool refreshReleased_{false};
  size_t calls_{0};
};

class BlockingFailingSasTokenProvider final : public SasTokenProvider {
 public:
  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    std::unique_lock lock(mutex_);
    ++calls_;
    callbackStarted_ = true;
    callbackStartedCondition_.notify_all();
    releaseCallbackCondition_.wait(lock, [&] { return callbackReleased_; });
    throw std::runtime_error("synthetic provider failure");
  }

  bool waitForCallback() {
    std::unique_lock lock(mutex_);
    return callbackStartedCondition_.wait_for(
        lock, kWaitTimeout, [&] { return callbackStarted_; });
  }

  void releaseCallback() {
    std::lock_guard lock(mutex_);
    callbackReleased_ = true;
    releaseCallbackCondition_.notify_all();
  }

  size_t calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable callbackStartedCondition_;
  std::condition_variable releaseCallbackCondition_;
  bool callbackStarted_{false};
  bool callbackReleased_{false};
  size_t calls_{0};
};

class FailingRecordingTransport final
    : public Azure::Core::Http::HttpTransport {
 public:
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(
      Azure::Core::Http::Request& request,
      const Azure::Core::Context& context) override {
    ++requests_;
    throw std::runtime_error("synthetic transport failure");
  }

  size_t requests() const {
    return requests_;
  }

 private:
  std::atomic<size_t> requests_{0};
};

} // namespace

TEST(DynamicSasTokenClientProviderTest, asyncClientRefreshesOnFirstOperation) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://filesystem@account.dfs.core.windows.net/path");
  const config::ConfigBase config(
      {{"fs.azure.sas.token.renew.period.for.streams", "120"}}, false);
  auto tokenProvider = std::make_shared<SyntheticSasTokenProvider>();
  DynamicSasTokenClientProvider provider(tokenProvider);
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  auto transport = std::make_shared<FailingRecordingTransport>();
  Azure::Storage::Blobs::BlobClientOptions options;
  options.Transport.Transport = transport;
  options.Retry.MaxRetries = 0;
  const AzureAsyncReadContext context{options, runtime.authService()};

  auto client = provider.getReadFileClientForAsync(abfsPath, config, context);

  ASSERT_NE(client, nullptr);
  EXPECT_EQ(tokenProvider->calls(), 0);
  EXPECT_EQ(client->getUrl(), abfsPath->getUrl(true));
  EXPECT_EQ(tokenProvider->calls(), 0);

  auto operation = runtime.submit(
      "account.dfs.core.windows.net",
      [&client](const folly::CancellationToken&) { client->getProperties(); });
  EXPECT_THROW(std::move(operation).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(tokenProvider->calls(), 1);
  EXPECT_EQ(transport->requests(), 1);
}

TEST(DynamicSasTokenClientProviderTest, asyncClientUsesConfiguredBlobEndpoint) {
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://filesystem@account.dfs.core.windows.net/path");
  const config::ConfigBase config(
      {{kAzureBlobEndpoint, "http://127.0.0.1:1234/account/"}}, false);
  auto tokenProvider = std::make_shared<SyntheticSasTokenProvider>();
  DynamicSasTokenClientProvider provider(tokenProvider);
  AbfsAsyncRuntime runtime(AbfsAsyncRuntimeOptions{});
  const Azure::Storage::Blobs::BlobClientOptions options;
  const AzureAsyncReadContext context{options, runtime.authService()};

  auto client = provider.getReadFileClientForAsync(abfsPath, config, context);

  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->getUrl(), "http://127.0.0.1:1234/account/filesystem/path");
  EXPECT_EQ(tokenProvider->calls(), 0);
}

TEST(DynamicSasTokenClientProviderTest, asyncClientSharesForcedRefresh) {
  constexpr size_t kWaiters = 64;
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://filesystem@account.dfs.core.windows.net/path");
  const config::ConfigBase config(
      {{"fs.azure.sas.token.renew.period.for.streams", "3000000000"}}, false);
  auto tokenProvider = std::make_shared<SequencedSasTokenProvider>();
  DynamicSasTokenClientProvider provider(tokenProvider);
  AbfsAsyncRuntimeOptions runtimeOptions;
  runtimeOptions.maxActiveRequests = kWaiters;
  AbfsAsyncRuntime runtime(runtimeOptions);
  auto authService = runtime.authService();
  auto transport = std::make_shared<FailingRecordingTransport>();
  Azure::Storage::Blobs::BlobClientOptions options;
  options.Transport.Transport = transport;
  options.Retry.MaxRetries = 0;
  const AzureAsyncReadContext context{options, authService};
  auto client = provider.getReadFileClientForAsync(abfsPath, config, context);

  auto initial = runtime.submit(
      "account.dfs.core.windows.net",
      [&client](const folly::CancellationToken&) { client->getProperties(); });
  EXPECT_THROW(std::move(initial).get(kWaitTimeout), std::runtime_error);
  EXPECT_EQ(tokenProvider->calls(), 1);

  std::vector<folly::SemiFuture<folly::Unit>> refreshes;
  for (size_t waiter = 0; waiter < kWaiters; ++waiter) {
    refreshes.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&client](const folly::CancellationToken&) {
          client->getProperties();
        }));
  }
  auto releaseGuard =
      folly::makeGuard([tokenProvider] { tokenProvider->releaseRefresh(); });
  ASSERT_TRUE(tokenProvider->waitForRefresh());
  ASSERT_TRUE(waitForAuthWaiters(authService, kWaiters));
  EXPECT_EQ(tokenProvider->calls(), 2);

  tokenProvider->releaseRefresh();
  releaseGuard.dismiss();
  for (auto& refresh : refreshes) {
    EXPECT_THROW(std::move(refresh).get(kWaitTimeout), std::runtime_error);
  }
  EXPECT_EQ(tokenProvider->calls(), 2);
  EXPECT_EQ(transport->requests(), kWaiters + 1);
}

TEST(DynamicSasTokenClientProviderTest, asyncClientFansOutProviderFailure) {
  constexpr size_t kWaiters = 8;
  const auto abfsPath = std::make_shared<AbfsPath>(
      "abfss://filesystem@account.dfs.core.windows.net/path");
  const config::ConfigBase config({});
  auto tokenProvider = std::make_shared<BlockingFailingSasTokenProvider>();
  DynamicSasTokenClientProvider provider(tokenProvider);
  AbfsAsyncRuntimeOptions runtimeOptions;
  runtimeOptions.maxActiveRequests = kWaiters;
  AbfsAsyncRuntime runtime(runtimeOptions);
  auto authService = runtime.authService();
  auto transport = std::make_shared<FailingRecordingTransport>();
  Azure::Storage::Blobs::BlobClientOptions options;
  options.Transport.Transport = transport;
  const AzureAsyncReadContext context{options, authService};
  auto client = provider.getReadFileClientForAsync(abfsPath, config, context);
  std::vector<folly::SemiFuture<folly::Unit>> refreshes;

  for (size_t waiter = 0; waiter < kWaiters; ++waiter) {
    refreshes.push_back(runtime.submit(
        "account.dfs.core.windows.net",
        [&client](const folly::CancellationToken&) {
          client->getProperties();
        }));
  }
  auto releaseGuard =
      folly::makeGuard([tokenProvider] { tokenProvider->releaseCallback(); });
  ASSERT_TRUE(tokenProvider->waitForCallback());
  ASSERT_TRUE(waitForAuthWaiters(authService, kWaiters));

  tokenProvider->releaseCallback();
  releaseGuard.dismiss();
  for (auto& refresh : refreshes) {
    try {
      std::move(refresh).get(kWaitTimeout);
      FAIL() << "Expected provider failure";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "synthetic provider failure");
    }
  }
  EXPECT_EQ(tokenProvider->calls(), 1);
  EXPECT_EQ(transport->requests(), 0);
}

TEST(DynamicSasTokenClientProviderTest, dynamicSasToken) {
  {
    const std::string account = "account1";
    const config::ConfigBase config(
        {{"fs.azure.account.auth.type.account1.dfs.core.windows.net", "SAS"},
         {"fs.azure.sas.token.renew.period.for.streams", "1"}},
        false);
    registerAzureClientProviderFactory(account, [](const std::string&) {
      auto sasTokenProvider =
          std::make_shared<MyDynamicAbfsSasTokenProvider>(3);
      return std::make_unique<DynamicSasTokenClientProvider>(sasTokenProvider);
    });

    auto abfsPath = std::make_shared<AbfsPath>(
        fmt::format("abfs://abc@{}.dfs.core.windows.net/file", account));
    auto readClient =
        AzureClientProviderFactories::getReadFileClient(abfsPath, config);
    auto writeClient =
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config);

    auto readUrl = readClient->getUrl();
    auto writeUrl = writeClient->getUrl();

    // Let the current time pass 3 seconds to ensure the SAS token is expired.
    std::this_thread::sleep_for(std::chrono::seconds(3)); // NOLINT

    auto newReadUrl = readClient->getUrl();
    ASSERT_NE(readUrl, newReadUrl);
    // The SAS token should be reused.
    ASSERT_EQ(newReadUrl, readClient->getUrl());

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
    registerAzureClientProviderFactory(account, [](const std::string&) {
      auto sasTokenProvider =
          std::make_shared<MyDynamicAbfsSasTokenProvider>(60);
      return std::make_unique<DynamicSasTokenClientProvider>(sasTokenProvider);
    });

    auto abfsPath = std::make_shared<AbfsPath>(
        fmt::format("abfs://abc@{}.dfs.core.windows.net/file", account));
    auto readClient =
        AzureClientProviderFactories::getReadFileClient(abfsPath, config);
    auto writeClient =
        AzureClientProviderFactories::getWriteFileClient(abfsPath, config);

    auto readUrl = readClient->getUrl();
    auto writeUrl = writeClient->getUrl();

    // Let the current time pass 3 seconds to ensure the timestamp in the SAS
    // token is updated.
    std::this_thread::sleep_for(std::chrono::seconds(3)); // NOLINT

    // Sas token should be renewed because the time left is less than the
    // renewal period.
    ASSERT_NE(readUrl, readClient->getUrl());
    ASSERT_NE(writeUrl, writeClient->getUrl());
  }
}
