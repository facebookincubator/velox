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

#include <azure/core/io/body_stream.hpp>
#include <folly/ScopeGuard.h>
#include <folly/fibers/Baton.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <string_view>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/config/Config.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsFileSystem.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"

#include "connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h"
#include "connectors/hive/storage_adapters/abfs/DynamicSasTokenClientProvider.h"
#include "connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsWriteFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"
#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"
#include "velox/connectors/hive/storage_adapters/abfs/tests/MockDataLakeFileClient.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/exec/tests/utils/PortUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::filesystems;
using namespace facebook::velox::common::testutil;
using ::facebook::velox::common::Region;

namespace {

constexpr int kOneMB = 1'048'576;

struct RecordedDownload {
  int64_t offset{0};
  int64_t length{0};
};

struct InMemoryReadState {
  std::string data;
  std::vector<RecordedDownload> downloads;
  std::atomic<size_t> fiberClientCreations{0};
  int32_t fiberMaxRetries{-1};
  std::function<void(std::chrono::milliseconds, const Azure::Core::Context&)>
      retryDelayCallback;
  bool blockDownloads{false};
  size_t numDownloadsBeforeRelease{1};
  folly::Baton<> downloadsStarted;
  std::mutex releaseMutex;
  std::vector<std::shared_ptr<folly::fibers::Baton>> releaseWaiters;
  std::atomic<size_t> activeDownloads{0};
  std::atomic<size_t> peakActiveDownloads{0};
  bool failDownload{false};
  bool timeoutDownload{false};
  std::optional<size_t> responseBodyBytes;
  size_t maxBodyReadChunk{std::numeric_limits<size_t>::max()};
  std::function<void()> beforeDownload;

  void releaseBlockedDownloads() {
    std::vector<std::shared_ptr<folly::fibers::Baton>> waiters;
    {
      std::lock_guard lock(releaseMutex);
      waiters = releaseWaiters;
    }
    for (const auto& waiter : waiters) {
      waiter->post();
    }
  }
};

class FragmentedBodyStream final : public Azure::Core::IO::BodyStream {
 public:
  FragmentedBodyStream(const uint8_t* data, size_t length, size_t maxReadChunk)
      : data_(data), length_(length), maxReadChunk_(maxReadChunk) {}

  int64_t Length() const override {
    return length_;
  }

 private:
  size_t OnRead(
      uint8_t* buffer,
      size_t count,
      const Azure::Core::Context& context) override {
    const auto bytes = std::min({count, maxReadChunk_, length_ - offset_});
    std::memcpy(buffer, data_ + offset_, bytes);
    offset_ += bytes;
    return bytes;
  }

  const uint8_t* data_;
  size_t length_;
  size_t maxReadChunk_;
  size_t offset_{0};
};

class InMemoryAzureBlobClient final : public AzureBlobClient {
 public:
  explicit InMemoryAzureBlobClient(std::shared_ptr<InMemoryReadState> state)
      : state_(std::move(state)) {}

  Azure::Response<Azure::Storage::Blobs::Models::BlobProperties> getProperties()
      override {
    VELOX_FAIL("Unexpected getProperties call.");
  }

  Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult> download(
      const Azure::Storage::Blobs::DownloadBlobOptions& options) override {
    const auto& range = options.Range.Value();
    const auto offset = range.Offset;
    const auto length = range.Length.Value();

    VELOX_CHECK_GE(offset, 0);
    VELOX_CHECK_GE(length, 0);
    VELOX_CHECK_LE(offset + length, static_cast<int64_t>(state_->data.size()));

    state_->downloads.push_back({offset, length});
    if (state_->beforeDownload) {
      auto beforeDownload = std::move(state_->beforeDownload);
      beforeDownload();
    }
    if (state_->failDownload) {
      throw std::runtime_error("In-memory Blob download failed");
    }
    if (state_->timeoutDownload) {
      throw Azure::Core::Http::TransportException(
          "In-memory Blob download timed out");
    }
    if (state_->blockDownloads) {
      auto releaseWaiter = std::make_shared<folly::fibers::Baton>();
      {
        std::lock_guard lock(state_->releaseMutex);
        state_->releaseWaiters.push_back(releaseWaiter);
      }
      const auto active = ++state_->activeDownloads;
      auto peak = state_->peakActiveDownloads.load();
      while (active > peak &&
             !state_->peakActiveDownloads.compare_exchange_weak(peak, active)) {
      }
      if (active >= state_->numDownloadsBeforeRelease) {
        state_->downloadsStarted.post();
      }
      releaseWaiter->wait();
      --state_->activeDownloads;
    }

    const auto bodyBytes = std::min(
        static_cast<size_t>(length),
        state_->responseBodyBytes.value_or(static_cast<size_t>(length)));

    Azure::Storage::Blobs::Models::DownloadBlobResult result;
    const auto* body =
        reinterpret_cast<const uint8_t*>(state_->data.data() + offset);
    if (state_->maxBodyReadChunk == std::numeric_limits<size_t>::max()) {
      result.BodyStream =
          std::make_unique<Azure::Core::IO::MemoryBodyStream>(body, bodyBytes);
    } else {
      result.BodyStream = std::make_unique<FragmentedBodyStream>(
          body, bodyBytes, state_->maxBodyReadChunk);
    }
    return Azure::Response<Azure::Storage::Blobs::Models::DownloadBlobResult>(
        std::move(result), nullptr);
  }

  std::string getUrl() override {
    return std::string{kUrl};
  }

 private:
  static constexpr std::string_view kUrl =
      "http://127.0.0.1:80/container/test-file";

  std::shared_ptr<InMemoryReadState> state_;
};

class InMemoryAzureClientProvider final : public AzureClientProvider {
 public:
  explicit InMemoryAzureClientProvider(std::shared_ptr<InMemoryReadState> state)
      : state_(std::move(state)) {}

  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    return std::make_unique<InMemoryAzureBlobClient>(state_);
  }

  std::unique_ptr<AzureBlobClient> getReadFileClientWithOptions(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config,
      const Azure::Storage::Blobs::BlobClientOptions& options) override {
    ++state_->fiberClientCreations;
    state_->fiberMaxRetries = options.Retry.MaxRetries;
    state_->retryDelayCallback = options.Retry.RetryDelayCallback;
    return std::make_unique<InMemoryAzureBlobClient>(state_);
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("Unexpected getWriteFileClient call.");
  }

 private:
  std::shared_ptr<InMemoryReadState> state_;
};

class SyncOnlyInMemoryAzureClientProvider final : public AzureClientProvider {
 public:
  explicit SyncOnlyInMemoryAzureClientProvider(
      std::shared_ptr<InMemoryReadState> state)
      : state_(std::move(state)) {}

  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    return std::make_unique<InMemoryAzureBlobClient>(state_);
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    VELOX_FAIL("Unexpected getWriteFileClient call.");
  }

 private:
  std::shared_ptr<InMemoryReadState> state_;
};

class TestAzureClientProvider final : public AzureClientProvider {
 public:
  std::unique_ptr<AzureBlobClient> getReadFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    if (config.valueExists(
            "fs.azure.sas.fixed.token.test.dfs.core.windows.net")) {
      return FixedSasAzureClientProvider().getReadFileClient(path, config);
    }
    return SharedKeyAzureClientProvider().getReadFileClient(path, config);
  }

  std::unique_ptr<AzureBlobClient> getReadFileClientWithOptions(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config,
      const Azure::Storage::Blobs::BlobClientOptions& options) override {
    if (config.valueExists(
            "fs.azure.sas.fixed.token.test.dfs.core.windows.net")) {
      return FixedSasAzureClientProvider().getReadFileClientWithOptions(
          path, config, options);
    }
    return SharedKeyAzureClientProvider().getReadFileClientWithOptions(
        path, config, options);
  }

  std::unique_ptr<AzureDataLakeFileClient> getWriteFileClient(
      const std::shared_ptr<AbfsPath>& path,
      const config::ConfigBase& config) override {
    return std::make_unique<MockDataLakeFileClient>();
  }
};

class CountingSasTokenProvider final : public SasTokenProvider {
 public:
  explicit CountingSasTokenProvider(std::string token)
      : token_(std::move(token)) {}

  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    ++calls_;
    return token_;
  }

  size_t calls() const {
    return calls_;
  }

 private:
  const std::string token_;
  std::atomic<size_t> calls_{0};
};

class BlockingSecondSasTokenProvider final : public SasTokenProvider {
 public:
  explicit BlockingSecondSasTokenProvider(std::string token)
      : token_(std::move(token)) {}

  std::string getSasToken(
      const std::string& fileSystem,
      const std::string& path,
      const std::string& operation) override {
    std::unique_lock lock(mutex_);
    ++calls_;
    if (calls_ == 2) {
      secondCallStarted_ = true;
      secondCallStartedCondition_.notify_all();
      releaseSecondCallCondition_.wait(
          lock, [&] { return secondCallReleased_; });
    }
    return token_;
  }

  bool waitForSecondCall() {
    std::unique_lock lock(mutex_);
    return secondCallStartedCondition_.wait_for(
        lock, std::chrono::seconds(5), [&] { return secondCallStarted_; });
  }

  void releaseSecondCall() {
    std::lock_guard lock(mutex_);
    secondCallReleased_ = true;
    releaseSecondCallCondition_.notify_all();
  }

  size_t calls() const {
    std::lock_guard lock(mutex_);
    return calls_;
  }

 private:
  const std::string token_;
  mutable std::mutex mutex_;
  std::condition_variable secondCallStartedCondition_;
  std::condition_variable releaseSecondCallCondition_;
  bool secondCallStarted_{false};
  bool secondCallReleased_{false};
  size_t calls_{0};
};

class BlockingSasTokenProvider final : public SasTokenProvider {
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
    return "se=9999-01-01T00%3A00%3A00Z&marker=synthetic";
  }

  bool waitForCallback() {
    std::unique_lock lock(mutex_);
    return callbackStartedCondition_.wait_for(
        lock, std::chrono::seconds(5), [&] { return callbackStarted_; });
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

std::shared_ptr<const config::ConfigBase> nativeAsyncConfig(
    size_t maxActiveRequests = 4,
    size_t maxQueuedRequests = 4) {
  return std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {"fs.azure.async-read.enabled", "true"},
          {"fs.azure.async-read.disable-retries-for-test", "true"},
          {"fs.azure.async-read.event-threads", "1"},
          {"fs.azure.async-read.max-active-requests",
           std::to_string(maxActiveRequests)},
          {"fs.azure.async-read.max-queued-requests",
           std::to_string(maxQueuedRequests)},
      });
}

std::unique_ptr<ReadFile> openAsyncInMemoryFile(
    std::string account,
    const std::shared_ptr<InMemoryReadState>& state,
    size_t maxActiveRequests = 4,
    size_t maxQueuedRequests = 4) {
  registerAzureClientProviderFactory(account, [state](const std::string&) {
    return std::make_unique<InMemoryAzureClientProvider>(state);
  });
  AbfsFileSystem fileSystem(
      nativeAsyncConfig(maxActiveRequests, maxQueuedRequests));
  FileOptions options;
  options.fileSize = state->data.size();
  return fileSystem.openFileForRead(
      fmt::format(
          "abfs://container@{}.dfs.core.windows.net/test-file", account),
      options);
}

} // namespace

class AbfsFileSystemTest : public testing::Test {
 public:
  std::shared_ptr<AzuriteServer> azuriteServer_;
  std::unique_ptr<AbfsFileSystem> abfs_;

  static void SetUpTestCase() {
    registerAbfsFileSystem();
    registerAzureClientProviderFactory("test", [](const std::string&) {
      return std::make_unique<TestAzureClientProvider>();
    });
  }

  void SetUp() override {
    auto port = facebook::velox::exec::test::getFreePort();
    azuriteServer_ = std::make_shared<AzuriteServer>(port);
    azuriteServer_->start();
    auto tempFile = createFile();
    azuriteServer_->addFile(tempFile->getPath());
    abfs_ = std::make_unique<AbfsFileSystem>(azuriteServer_->hiveConfig());
  }

  void TearDown() override {
    // azuriteServer_ is left null if SetUp() threw before it could be
    // constructed (e.g. the azurite-blob executable wasn't found).
    // TearDown() runs unconditionally after SetUp(), even on failure, so
    // it must not assume construction succeeded.
    if (azuriteServer_ != nullptr) {
      azuriteServer_->stop();
    }
  }

  static std::string generateRandomData(int size) {
    static constexpr std::string_view kCharacters =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<size_t> distribution(
        0, kCharacters.size() - 1);

    std::string data(size, ' ');

    for (int i = 0; i < size; ++i) {
      data[i] = kCharacters[distribution(generator)];
    }

    return data;
  }

 private:
  static std::shared_ptr<TempFilePath> createFile() {
    auto tempFile = TempFilePath::create();
    tempFile->append("aaaaa");
    tempFile->append("bbbbb");
    tempFile->append(std::string(kOneMB, 'c'));
    tempFile->append("ddddd");
    return tempFile;
  }
};

namespace {

void readData(ReadFile* readFile) {
  ASSERT_EQ(readFile->size(), 15 + kOneMB);
  char buffer1[5];
  ASSERT_EQ(readFile->pread(10 + kOneMB, 5, &buffer1), "ddddd");
  char buffer2[10];
  ASSERT_EQ(readFile->pread(0, 10, &buffer2), "aaaaabbbbb");
  char buffer3[kOneMB];
  ASSERT_EQ(readFile->pread(10, kOneMB, buffer3), std::string(kOneMB, 'c'));
  ASSERT_EQ(readFile->size(), 15 + kOneMB);
  char buffer4[10];
  const std::string_view arf = readFile->pread(5, 10, &buffer4);
  const std::string zarf = readFile->pread(kOneMB, 15);
  auto buf = std::make_unique<char[]>(8);
  const std::string_view warf = readFile->pread(4, 8, buf.get());
  const std::string_view warfFromBuf(buf.get(), 8);
  ASSERT_EQ(arf, "bbbbbccccc");
  ASSERT_EQ(zarf, "ccccccccccddddd");
  ASSERT_EQ(warf, "abbbbbcc");
  ASSERT_EQ(warfFromBuf, "abbbbbcc");

  char buff1[10];
  char buff2[10];
  std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buff1, 10),
      folly::Range<char*>(nullptr, kOneMB - 5),
      folly::Range<char*>(buff2, 10)};
  ASSERT_EQ(10 + kOneMB - 5 + 10, readFile->preadv(0, buffers));
  ASSERT_EQ(std::string_view(buff1, sizeof(buff1)), "aaaaabbbbb");
  ASSERT_EQ(std::string_view(buff2, sizeof(buff2)), "cccccddddd");

  std::vector<folly::IOBuf> iobufs(2);
  std::vector<Region> regions = {{0, 10}, {10, 5}};
  ASSERT_EQ(
      10 + 5,
      readFile->preadv(
          {regions.data(), regions.size()}, {iobufs.data(), iobufs.size()}));
  ASSERT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(iobufs[0].writableData()),
          iobufs[0].length()),
      "aaaaabbbbb");
  ASSERT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(iobufs[1].writableData()),
          iobufs[1].length()),
      "ccccc");
}

} // namespace

TEST_F(AbfsFileSystemTest, readFile) {
  auto readFile = abfs_->openFileForRead(azuriteServer_->fileURI());
  readData(readFile.get());
}

TEST_F(AbfsFileSystemTest, openFileForReadWithOptions) {
  FileOptions options;
  options.fileSize = 15 + kOneMB;
  auto readFile = abfs_->openFileForRead(azuriteServer_->fileURI(), options);
  readData(readFile.get());
}

TEST_F(AbfsFileSystemTest, nativeAsyncSharedKeyReadFile) {
  auto asyncFileSystem =
      std::make_unique<AbfsFileSystem>(azuriteServer_->hiveConfig(
          {{"fs.azure.async-read.enabled", "true"},
           {"fs.azure.async-read.disable-retries-for-test", "true"},
           {"fs.azure.async-read.event-threads", "1"},
           {"fs.azure.async-read.max-active-requests", "4"},
           {"fs.azure.async-read.max-queued-requests", "4"},
           {"fs.azure.async-read.max-connections-per-endpoint", "2"}}));
  auto readFile = asyncFileSystem->openFileForRead(azuriteServer_->fileURI());

  ASSERT_TRUE(readFile->hasPreadvAsync());
  readData(readFile.get());

  char firstBuffer[5];
  char secondBuffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(firstBuffer, sizeof(firstBuffer)),
      folly::Range<char*>(nullptr, kOneMB),
      folly::Range<char*>(secondBuffer, sizeof(secondBuffer)),
  };
  auto future = readFile->preadvAsync(5, buffers);

  EXPECT_EQ(std::move(future).get(), kOneMB + 10);
  EXPECT_EQ(std::string_view(firstBuffer, sizeof(firstBuffer)), "bbbbb");
  EXPECT_EQ(std::string_view(secondBuffer, sizeof(secondBuffer)), "ddddd");
}

TEST_F(AbfsFileSystemTest, nativeAsyncFixedSasReadFile) {
  const auto sasToken = azuriteServer_->readSasToken();
  const std::unordered_map<std::string, std::string> sasConfig = {
      {"fs.azure.sas.fixed.token.test.dfs.core.windows.net", sasToken}};
  AbfsFileSystem syncFileSystem(azuriteServer_->hiveConfig(sasConfig));
  auto syncReadFile = syncFileSystem.openFileForRead(azuriteServer_->fileURI());
  const auto expected = syncReadFile->pread(0, 10);

  auto asyncConfig = sasConfig;
  asyncConfig["fs.azure.async-read.enabled"] = "true";
  asyncConfig["fs.azure.async-read.disable-retries-for-test"] = "true";
  asyncConfig["fs.azure.async-read.event-threads"] = "1";
  asyncConfig["fs.azure.async-read.max-active-requests"] = "4";
  asyncConfig["fs.azure.async-read.max-queued-requests"] = "4";
  AbfsFileSystem asyncFileSystem(azuriteServer_->hiveConfig(asyncConfig));
  auto asyncReadFile =
      asyncFileSystem.openFileForRead(azuriteServer_->fileURI());

  ASSERT_TRUE(asyncReadFile->hasPreadvAsync());
  char buffer[10];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};
  auto future = asyncReadFile->preadvAsync(0, buffers);

  EXPECT_EQ(std::move(future).get(), sizeof(buffer));
  EXPECT_EQ(std::string_view(buffer, sizeof(buffer)), expected);
}

TEST_F(AbfsFileSystemTest, nativeAsyncDynamicSasReadFile) {
  auto tokenProvider = std::make_shared<CountingSasTokenProvider>(
      azuriteServer_->readSasToken());
  registerAzureClientProviderFactory(
      "test", [tokenProvider](const std::string&) {
        return std::make_unique<DynamicSasTokenClientProvider>(tokenProvider);
      });
  auto restoreProvider = folly::makeGuard([] {
    registerAzureClientProviderFactory("test", [](const std::string&) {
      return std::make_unique<TestAzureClientProvider>();
    });
  });
  auto config = azuriteServer_->hiveConfig(
      {{"fs.azure.async-read.enabled", "true"},
       {"fs.azure.async-read.disable-retries-for-test", "true"},
       {"fs.azure.async-read.event-threads", "1"},
       {"fs.azure.async-read.max-active-requests", "4"},
       {"fs.azure.async-read.max-queued-requests", "4"},
       {"fs.azure.sas.token.renew.period.for.streams", "120"}});
  AbfsFileSystem fileSystem(config);
  FileOptions options;
  options.fileSize = 15 + kOneMB;

  auto readFile =
      fileSystem.openFileForRead(azuriteServer_->fileURI(), options);

  ASSERT_TRUE(readFile->hasPreadvAsync());
  EXPECT_EQ(tokenProvider->calls(), 0);
  char buffer[10];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};
  auto future = readFile->preadvAsync(0, buffers);

  EXPECT_EQ(std::move(future).get(), sizeof(buffer));
  EXPECT_EQ(std::string_view(buffer, sizeof(buffer)), "aaaaabbbbb");
  EXPECT_EQ(tokenProvider->calls(), 1);
}

TEST_F(AbfsFileSystemTest, dynamicSasRefreshDoesNotBlockWarmSiblingRead) {
  auto tokenProvider = std::make_shared<BlockingSecondSasTokenProvider>(
      azuriteServer_->readSasToken());
  registerAzureClientProviderFactory(
      "test", [tokenProvider](const std::string&) {
        return std::make_unique<DynamicSasTokenClientProvider>(tokenProvider);
      });
  auto restoreProvider = folly::makeGuard([] {
    registerAzureClientProviderFactory("test", [](const std::string&) {
      return std::make_unique<TestAzureClientProvider>();
    });
  });
  auto config = azuriteServer_->hiveConfig(
      {{"fs.azure.async-read.enabled", "true"},
       {"fs.azure.async-read.disable-retries-for-test", "true"},
       {"fs.azure.async-read.event-threads", "1"},
       {"fs.azure.async-read.max-active-requests", "4"},
       {"fs.azure.async-read.max-queued-requests", "4"},
       {"fs.azure.sas.token.renew.period.for.streams", "120"}});
  AbfsFileSystem fileSystem(config);
  FileOptions options;
  options.fileSize = 15 + kOneMB;
  auto warmFile =
      fileSystem.openFileForRead(azuriteServer_->fileURI(), options);
  auto refreshingFile =
      fileSystem.openFileForRead(azuriteServer_->fileURI(), options);

  char warmBuffer[5];
  const std::vector<folly::Range<char*>> warmBuffers = {
      folly::Range<char*>(warmBuffer, sizeof(warmBuffer))};
  EXPECT_EQ(
      std::move(warmFile->preadvAsync(0, warmBuffers)).get(),
      sizeof(warmBuffer));
  EXPECT_EQ(tokenProvider->calls(), 1);

  char refreshingBuffer[5];
  const std::vector<folly::Range<char*>> refreshingBuffers = {
      folly::Range<char*>(refreshingBuffer, sizeof(refreshingBuffer))};
  auto refreshing = refreshingFile->preadvAsync(5, refreshingBuffers);
  auto releaseGuard =
      folly::makeGuard([tokenProvider] { tokenProvider->releaseSecondCall(); });
  ASSERT_TRUE(tokenProvider->waitForSecondCall());

  char siblingBuffer[5];
  const std::vector<folly::Range<char*>> siblingBuffers = {
      folly::Range<char*>(siblingBuffer, sizeof(siblingBuffer))};
  auto sibling = warmFile->preadvAsync(10 + kOneMB, siblingBuffers);
  EXPECT_EQ(
      std::move(sibling).get(std::chrono::seconds(1)), sizeof(siblingBuffer));
  EXPECT_EQ(std::string_view(siblingBuffer, sizeof(siblingBuffer)), "ddddd");

  tokenProvider->releaseSecondCall();
  releaseGuard.dismiss();
  EXPECT_EQ(std::move(refreshing).get(), sizeof(refreshingBuffer));
  EXPECT_EQ(
      std::string_view(refreshingBuffer, sizeof(refreshingBuffer)), "bbbbb");
  EXPECT_EQ(tokenProvider->calls(), 2);
}

TEST(AbfsReadFileTest, dynamicSasDestructionDuringRefreshIsSafe) {
  const std::string account{"unit-dynamic-lifetime"};
  auto tokenProvider = std::make_shared<BlockingSasTokenProvider>();
  registerAzureClientProviderFactory(
      account, [tokenProvider](const std::string&) {
        return std::make_unique<DynamicSasTokenClientProvider>(tokenProvider);
      });
  auto configValues = nativeAsyncConfig()->rawConfigsCopy();
  configValues[kAzureBlobEndpoint] = "http://127.0.0.1:1/account";
  auto fileSystem = std::make_unique<AbfsFileSystem>(
      std::make_shared<const config::ConfigBase>(std::move(configValues)));
  FileOptions options;
  options.fileSize = 5;
  auto readFile = fileSystem->openFileForRead(
      fmt::format(
          "abfs://container@{}.dfs.core.windows.net/test-file", account),
      options);
  char buffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(0, buffers);
  auto releaseGuard =
      folly::makeGuard([tokenProvider] { tokenProvider->releaseCallback(); });
  ASSERT_TRUE(tokenProvider->waitForCallback());
  readFile.reset();
  fileSystem.reset();

  tokenProvider->releaseCallback();
  releaseGuard.dismiss();
  EXPECT_THROW(std::move(future).get(), std::runtime_error);
  EXPECT_EQ(tokenProvider->calls(), 1);
}

TEST(AbfsReadFileTest, dynamicSasCallerCancellationDuringRefresh) {
  const std::string account{"unit-dynamic-cancellation"};
  auto tokenProvider = std::make_shared<BlockingSasTokenProvider>();
  registerAzureClientProviderFactory(
      account, [tokenProvider](const std::string&) {
        return std::make_unique<DynamicSasTokenClientProvider>(tokenProvider);
      });
  auto configValues = nativeAsyncConfig()->rawConfigsCopy();
  configValues[kAzureBlobEndpoint] = "http://127.0.0.1:1/account";
  AbfsFileSystem fileSystem(
      std::make_shared<const config::ConfigBase>(std::move(configValues)));
  FileOptions options;
  options.fileSize = 5;
  auto readFile = fileSystem.openFileForRead(
      fmt::format(
          "abfs://container@{}.dfs.core.windows.net/test-file", account),
      options);
  char buffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(0, buffers);
  auto releaseGuard =
      folly::makeGuard([tokenProvider] { tokenProvider->releaseCallback(); });
  ASSERT_TRUE(tokenProvider->waitForCallback());
  future.cancel();

  EXPECT_THROW(
      std::move(future).get(std::chrono::seconds(1)),
      folly::FutureCancellation);
  tokenProvider->releaseCallback();
  releaseGuard.dismiss();
  EXPECT_EQ(tokenProvider->calls(), 1);
}

TEST(AbfsReadFileTest, preadvUsesSingleDownloadForBuffersWithGaps) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789abcdefghijklmn";

  registerAzureClientProviderFactory("unit", [state](const std::string&) {
    return std::make_unique<InMemoryAzureClientProvider>(state);
  });

  AbfsReadFile readFile{
      "abfs://container@unit.dfs.core.windows.net/test-file",
      config::ConfigBase({})};
  FileOptions options;
  options.fileSize = state->data.size();
  readFile.initialize(options);

  char firstBuffer[5];
  char secondBuffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(firstBuffer, sizeof(firstBuffer)),
      folly::Range<char*>(nullptr, 7),
      folly::Range<char*>(secondBuffer, sizeof(secondBuffer)),
  };

  ASSERT_EQ(17, readFile.preadv(2, buffers));
  ASSERT_EQ(state->downloads.size(), 1);
  EXPECT_EQ(state->downloads[0].offset, 2);
  EXPECT_EQ(state->downloads[0].length, 17);
  EXPECT_EQ(std::string_view(firstBuffer, sizeof(firstBuffer)), "23456");
  EXPECT_EQ(std::string_view(secondBuffer, sizeof(secondBuffer)), "efghi");
}

TEST(AbfsReadFileTest, preadvUsesSingleDownloadForDefaultCoalescedGap) {
  constexpr size_t kLeadingReadSize = 4;
  constexpr size_t kGapSize = static_cast<size_t>(512) * 1'024;
  constexpr size_t kTrailingReadSize = 4;

  auto state = std::make_shared<InMemoryReadState>();
  state->data = std::string(kLeadingReadSize, 'a') +
      std::string(kGapSize, 'x') + std::string(kTrailingReadSize, 'b');

  registerAzureClientProviderFactory(
      "unit-large-gap", [state](const std::string&) {
        return std::make_unique<InMemoryAzureClientProvider>(state);
      });

  AbfsReadFile readFile{
      "abfs://container@unit-large-gap.dfs.core.windows.net/test-file",
      config::ConfigBase({})};
  FileOptions options;
  options.fileSize = state->data.size();
  readFile.initialize(options);

  char firstBuffer[kLeadingReadSize];
  char secondBuffer[kTrailingReadSize];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(firstBuffer, sizeof(firstBuffer)),
      folly::Range<char*>(nullptr, kGapSize),
      folly::Range<char*>(secondBuffer, sizeof(secondBuffer)),
  };

  ASSERT_EQ(state->data.size(), readFile.preadv(0, buffers));
  ASSERT_EQ(state->downloads.size(), 1);
  EXPECT_EQ(state->downloads[0].offset, 0);
  EXPECT_EQ(state->downloads[0].length, state->data.size());
  EXPECT_EQ(std::string_view(firstBuffer, sizeof(firstBuffer)), "aaaa");
  EXPECT_EQ(std::string_view(secondBuffer, sizeof(secondBuffer)), "bbbb");
}

TEST(AbfsReadFileTest, asyncDisabledPreservesLegacyShortBodyResult) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  state->responseBodyBytes = 4;
  registerAzureClientProviderFactory(
      "unit-sync-short-body", [state](const std::string&) {
        return std::make_unique<InMemoryAzureClientProvider>(state);
      });
  AbfsReadFile readFile{
      "abfs://container@unit-sync-short-body.dfs.core.windows.net/test-file",
      config::ConfigBase({})};
  FileOptions options;
  options.fileSize = state->data.size();
  readFile.initialize(options);
  std::array<char, 5> buffer;
  buffer.fill('x');
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer.data(), buffer.size())};

  EXPECT_EQ(readFile.preadv(0, buffers), buffer.size());
  EXPECT_EQ(std::string_view(buffer.data(), 4), "0123");
  EXPECT_EQ(buffer.back(), 'x');
}

TEST(AbfsReadFileTest, nativeAsyncPreadvUsesSingleDownloadWithGaps) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789abcdefghijklmn";

  registerAzureClientProviderFactory("unit-async", [state](const std::string&) {
    return std::make_unique<InMemoryAzureClientProvider>(state);
  });
  auto config = std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {"fs.azure.async-read.enabled", "true"},
          {"fs.azure.async-read.disable-retries-for-test", "true"},
          {"fs.azure.async-read.event-threads", "1"},
          {"fs.azure.async-read.max-active-requests", "4"},
          {"fs.azure.async-read.max-queued-requests", "4"},
      });
  AbfsFileSystem fileSystem(config);
  FileOptions options;
  options.fileSize = state->data.size();
  auto readFile = fileSystem.openFileForRead(
      "abfs://container@unit-async.dfs.core.windows.net/test-file", options);

  ASSERT_TRUE(readFile->hasPreadvAsync());
  EXPECT_EQ(state->fiberMaxRetries, 0);
  EXPECT_TRUE(state->retryDelayCallback);
  char firstBuffer[5];
  char secondBuffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(firstBuffer, sizeof(firstBuffer)),
      folly::Range<char*>(nullptr, 7),
      folly::Range<char*>(secondBuffer, sizeof(secondBuffer)),
  };

  auto future = readFile->preadvAsync(2, buffers);

  EXPECT_EQ(std::move(future).get(), 17);
  ASSERT_EQ(state->downloads.size(), 1);
  EXPECT_EQ(state->downloads[0].offset, 2);
  EXPECT_EQ(state->downloads[0].length, 17);
  EXPECT_EQ(std::string_view(firstBuffer, sizeof(firstBuffer)), "23456");
  EXPECT_EQ(std::string_view(secondBuffer, sizeof(secondBuffer)), "efghi");
}

TEST(AbfsReadFileTest, asyncDisabledDoesNotRequestFiberClient) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  registerAzureClientProviderFactory(
      "unit-disabled", [state](const std::string&) {
        return std::make_unique<InMemoryAzureClientProvider>(state);
      });
  AbfsFileSystem fileSystem(
      std::make_shared<const config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
  FileOptions options;
  options.fileSize = state->data.size();

  auto readFile = fileSystem.openFileForRead(
      "abfs://container@unit-disabled.dfs.core.windows.net/test-file", options);

  EXPECT_FALSE(readFile->hasPreadvAsync());
  EXPECT_EQ(state->fiberClientCreations, 0);
}

TEST(AbfsReadFileTest, asyncEnabledRequiresTestRetryGate) {
  VELOX_ASSERT_USER_THROW(
      AbfsFileSystem(
          std::make_shared<const config::ConfigBase>(
              std::unordered_map<std::string, std::string>{
                  {"fs.azure.async-read.enabled", "true"}})),
      "Stage 3 test-only configuration gate");
}

TEST(AbfsReadFileTest, asyncAuthResourcesAreBounded) {
  auto tooManyWorkers = nativeAsyncConfig()->rawConfigsCopy();
  tooManyWorkers["fs.azure.async-read.auth-threads"] = "3";
  EXPECT_THROW(
      AbfsFileSystem(
          std::make_shared<const config::ConfigBase>(
              std::move(tooManyWorkers))),
      std::invalid_argument);

  auto emptyQueue = nativeAsyncConfig()->rawConfigsCopy();
  emptyQueue["fs.azure.async-read.max-queued-auth-refreshes"] = "0";
  EXPECT_THROW(
      AbfsFileSystem(
          std::make_shared<const config::ConfigBase>(std::move(emptyQueue))),
      std::invalid_argument);
}

TEST(AbfsReadFileTest, asyncEnabledRejectsUnsupportedProvider) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  registerAzureClientProviderFactory(
      "unit-sync-only", [state](const std::string&) {
        return std::make_unique<SyncOnlyInMemoryAzureClientProvider>(state);
      });
  AbfsFileSystem fileSystem(nativeAsyncConfig());
  FileOptions options;
  options.fileSize = state->data.size();

  VELOX_ASSERT_USER_THROW(
      fileSystem.openFileForRead(
          "abfs://container@unit-sync-only.dfs.core.windows.net/test-file",
          options),
      "account 'unit-sync-only' with auth context 'registered provider'");
}

TEST(AbfsReadFileTest, registeredProviderContextTakesPrecedenceOverAuthConfig) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  registerAzureClientProviderFactory(
      "unit-provider-context", [state](const std::string&) {
        return std::make_unique<SyncOnlyInMemoryAzureClientProvider>(state);
      });
  auto config = nativeAsyncConfig()->rawConfigsCopy();
  config
      ["fs.azure.account.auth.type.unit-provider-context.dfs.core.windows.net"] =
          "SharedKey";
  config["fs.azure.account.key.unit-provider-context.dfs.core.windows.net"] =
      "key";
  AbfsFileSystem fileSystem(
      std::make_shared<const config::ConfigBase>(std::move(config)));
  FileOptions options;
  options.fileSize = state->data.size();

  VELOX_ASSERT_USER_THROW(
      fileSystem.openFileForRead(
          "abfs://container@unit-provider-context.dfs.core.windows.net/test-file",
          options),
      "account 'unit-provider-context' with auth context 'registered provider'");
}

TEST(AbfsReadFileTest, asyncEnabledReportsUnsupportedOAuthContext) {
  auto config = nativeAsyncConfig()->rawConfigsCopy();
  config["fs.azure.account.auth.type.unit-oauth.dfs.core.windows.net"] =
      "OAuth";
  config["fs.azure.account.oauth2.client.id.unit-oauth.dfs.core.windows.net"] =
      "client";
  config
      ["fs.azure.account.oauth2.client.secret.unit-oauth.dfs.core.windows.net"] =
          "secret";
  config
      ["fs.azure.account.oauth2.client.endpoint.unit-oauth.dfs.core.windows.net"] =
          "https://login.microsoftonline.com/tenant/oauth2/token";
  AbfsFileSystem fileSystem(
      std::make_shared<const config::ConfigBase>(std::move(config)));

  VELOX_ASSERT_USER_THROW(
      fileSystem.openFileForRead(
          "abfss://container@unit-oauth.dfs.core.windows.net/test-file"),
      "account 'unit-oauth' with auth context 'OAuth'");
}

TEST(AbfsReadFileTest, nativeAsyncZeroLengthSkipsDownload) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  auto readFile = openAsyncInMemoryFile("unit-zero-length", state);
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(static_cast<char*>(nullptr), size_t{0})};

  auto future = readFile->preadvAsync(3, buffers);

  EXPECT_EQ(std::move(future).get(), 0);
  EXPECT_TRUE(state->downloads.empty());
}

TEST(AbfsReadFileTest, nativeAsyncAllNullBuffersUseOneDownload) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  auto readFile = openAsyncInMemoryFile("unit-all-null", state);
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(nullptr, 3),
      folly::Range<char*>(nullptr, 4),
  };

  auto future = readFile->preadvAsync(2, buffers);

  EXPECT_EQ(std::move(future).get(), 7);
  ASSERT_EQ(state->downloads.size(), 1);
  EXPECT_EQ(state->downloads[0].offset, 2);
  EXPECT_EQ(state->downloads[0].length, 7);
}

TEST(AbfsReadFileTest, nativeAsyncRejectsUnrepresentableOffset) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0";
  auto readFile = openAsyncInMemoryFile("unit-offset-overflow", state);
  char buffer[1];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1, buffers);

  VELOX_ASSERT_USER_THROW(
      std::move(future).get(),
      "ABFS read offset exceeds the Azure Blob range limit");
}

TEST(AbfsReadFileTest, nativeAsyncErrorsSettleThroughFuture) {
  auto failureState = std::make_shared<InMemoryReadState>();
  failureState->data = "0123456789";
  failureState->failDownload = true;
  auto failureReadFile = openAsyncInMemoryFile("unit-failure", failureState);
  char failureBuffer[5];
  const std::vector<folly::Range<char*>> failureBuffers = {
      folly::Range<char*>(failureBuffer, sizeof(failureBuffer))};

  auto failureFuture = failureReadFile->preadvAsync(0, failureBuffers);

  EXPECT_THROW(std::move(failureFuture).get(), std::runtime_error);

  auto shortBodyState = std::make_shared<InMemoryReadState>();
  shortBodyState->data = "0123456789";
  shortBodyState->responseBodyBytes = 4;
  auto shortBodyReadFile =
      openAsyncInMemoryFile("unit-short-body", shortBodyState);
  char shortBodyBuffer[5];
  const std::vector<folly::Range<char*>> shortBodyBuffers = {
      folly::Range<char*>(shortBodyBuffer, sizeof(shortBodyBuffer))};

  auto shortBodyFuture = shortBodyReadFile->preadvAsync(0, shortBodyBuffers);

  EXPECT_THROW(std::move(shortBodyFuture).get(), std::runtime_error);
}

TEST(AbfsReadFileTest, nativeAsyncTimeoutSettlesThroughFuture) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  state->timeoutDownload = true;
  auto readFile = openAsyncInMemoryFile("unit-timeout", state);
  char buffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(0, buffers);

  EXPECT_THROW(std::move(future).get(), Azure::Core::Http::TransportException);
}

TEST(AbfsReadFileTest, nativeAsyncFragmentedBodyMatchesSyncScatter) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789abcdefghijklmn";
  state->maxBodyReadChunk = 2;
  registerAzureClientProviderFactory(
      "unit-fragmented", [state](const std::string&) {
        return std::make_unique<InMemoryAzureClientProvider>(state);
      });
  AbfsFileSystem syncFileSystem(
      std::make_shared<const config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
  FileOptions options;
  options.fileSize = state->data.size();
  auto syncReadFile = syncFileSystem.openFileForRead(
      "abfs://container@unit-fragmented.dfs.core.windows.net/test-file",
      options);
  char syncFirst[3];
  char syncSecond[5];
  const std::vector<folly::Range<char*>> syncBuffers = {
      folly::Range<char*>(syncFirst, sizeof(syncFirst)),
      folly::Range<char*>(nullptr, 4),
      folly::Range<char*>(syncSecond, sizeof(syncSecond)),
  };
  ASSERT_EQ(syncReadFile->preadv(1, syncBuffers), 12);

  auto asyncReadFile = openAsyncInMemoryFile("unit-fragmented", state);
  char asyncFirst[3];
  char asyncSecond[5];
  const std::vector<folly::Range<char*>> asyncBuffers = {
      folly::Range<char*>(asyncFirst, sizeof(asyncFirst)),
      folly::Range<char*>(nullptr, 4),
      folly::Range<char*>(asyncSecond, sizeof(asyncSecond)),
  };

  auto future = asyncReadFile->preadvAsync(1, asyncBuffers);

  EXPECT_EQ(std::move(future).get(), 12);
  EXPECT_EQ(
      std::string_view(asyncFirst, sizeof(asyncFirst)),
      std::string_view(syncFirst, sizeof(syncFirst)));
  EXPECT_EQ(
      std::string_view(asyncSecond, sizeof(asyncSecond)),
      std::string_view(syncSecond, sizeof(syncSecond)));
  ASSERT_EQ(state->downloads.size(), 2);
  EXPECT_EQ(state->downloads[0].length, 12);
  EXPECT_EQ(state->downloads[1].length, 12);
}

TEST(AbfsReadFileTest, synchronousReadFromRuntimeThreadFailsThroughFuture) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  auto readFile = openAsyncInMemoryFile("unit-reentrant-sync", state);
  state->beforeDownload = [&readFile] {
    char nestedBuffer[1];
    const std::vector<folly::Range<char*>> nestedBuffers = {
        folly::Range<char*>(nestedBuffer, sizeof(nestedBuffer))};
    readFile->preadv(0, nestedBuffers);
  };
  char buffer[1];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(0, buffers);

  VELOX_ASSERT_THROW(
      std::move(future).get(),
      "Synchronous ABFS reads cannot run on an async runtime thread");
}

TEST(AbfsReadFileTest, nativeAsyncRetainsFileUntilPendingReadCompletes) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  state->blockDownloads = true;
  auto readFile = openAsyncInMemoryFile("unit-file-lifetime", state);
  char buffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(2, buffers);
  ASSERT_TRUE(state->downloadsStarted.try_wait_for(std::chrono::seconds(5)));
  EXPECT_FALSE(future.isReady());
  readFile.reset();
  state->releaseBlockedDownloads();

  EXPECT_EQ(std::move(future).get(), sizeof(buffer));
  EXPECT_EQ(std::string_view(buffer, sizeof(buffer)), "23456");
}

TEST(AbfsReadFileTest, nativeAsyncCancellationPropagatesToRuntime) {
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  state->blockDownloads = true;
  auto readFile = openAsyncInMemoryFile("unit-future-cancellation", state);
  char buffer[5];
  const std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buffer, sizeof(buffer))};

  auto future = readFile->preadvAsync(2, buffers);
  ASSERT_TRUE(state->downloadsStarted.try_wait_for(std::chrono::seconds(5)));
  future.cancel();
  state->releaseBlockedDownloads();

  EXPECT_THROW(std::move(future).get(), folly::FutureCancellation);
}

TEST(AbfsReadFileTest, nativeAsyncExceedsRuntimeThreadCount) {
  constexpr size_t kNumReads = 4;
  auto state = std::make_shared<InMemoryReadState>();
  state->data = "0123456789";
  state->blockDownloads = true;
  state->numDownloadsBeforeRelease = kNumReads;
  auto readFile =
      openAsyncInMemoryFile("unit-concurrency", state, kNumReads, kNumReads);
  std::array<std::array<char, 1>, kNumReads> buffers;
  std::vector<folly::SemiFuture<uint64_t>> futures;
  futures.reserve(kNumReads);
  for (size_t index = 0; index < kNumReads; ++index) {
    const std::vector<folly::Range<char*>> ranges = {
        folly::Range<char*>(buffers[index].data(), buffers[index].size())};
    futures.push_back(readFile->preadvAsync(index, ranges));
  }

  ASSERT_TRUE(state->downloadsStarted.try_wait_for(std::chrono::seconds(5)));
  EXPECT_EQ(state->peakActiveDownloads, kNumReads);
  state->releaseBlockedDownloads();

  for (size_t index = 0; index < kNumReads; ++index) {
    EXPECT_EQ(std::move(futures[index]).get(), 1);
    EXPECT_EQ(buffers[index][0], state->data[index]);
  }
}

TEST_F(AbfsFileSystemTest, openFileForReadWithInvalidOptions) {
  FileOptions options;
  options.fileSize = -kOneMB;
  VELOX_ASSERT_THROW(
      abfs_->openFileForRead(azuriteServer_->fileURI(), options),
      "File size must be non-negative");
}

TEST_F(AbfsFileSystemTest, fileHandleWithProperties) {
  FileHandleFactory factory(
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(1),
      std::make_unique<FileHandleGenerator>(azuriteServer_->hiveConfig()));
  FileProperties properties = {15 + kOneMB, 1};
  FileHandleKey key{azuriteServer_->fileURI()};
  auto fileHandleProperties = factory.generate(key, &properties);
  readData(fileHandleProperties->file.get());

  auto fileHandleWithoutProperties = factory.generate(key);
  readData(fileHandleWithoutProperties->file.get());
}

TEST_F(AbfsFileSystemTest, multipleThreadsWithReadFile) {
  std::atomic<bool> startThreads = false;
  std::vector<std::thread> threads;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> sleepTimesInMicroseconds = {0, 500, 5000};
  std::uniform_int_distribution<std::size_t> distribution(
      0, sleepTimesInMicroseconds.size() - 1);
  for (int i = 0; i < 10; i++) {
    const auto sleepTime = sleepTimesInMicroseconds[distribution(generator)];
    auto thread = std::thread([&, sleepTime] {
      while (!startThreads) {
        std::this_thread::yield();
      }
      std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
      auto readFile = abfs_->openFileForRead(azuriteServer_->fileURI());
      readData(readFile.get());
    });
    threads.emplace_back(std::move(thread));
  }
  startThreads = true;
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(AbfsFileSystemTest, missingFile) {
  const std::string abfsFile = azuriteServer_->URI() + "test.txt";
  VELOX_ASSERT_RUNTIME_THROW_CODE(
      abfs_->openFileForRead(abfsFile), error_code::kFileNotFound, "404");
}

TEST(AbfsWriteFileTest, openFileForWriteTest) {
  std::string_view kAbfsFile =
      "abfs://test@test.dfs.core.windows.net/test/writetest.txt";
  std::unique_ptr<AzureDataLakeFileClient> mockClient =
      std::make_unique<MockDataLakeFileClient>();
  auto mockClientPath =
      reinterpret_cast<MockDataLakeFileClient*>(mockClient.get())->path();
  AbfsWriteFile abfsWriteFile(kAbfsFile, mockClient);
  EXPECT_EQ(abfsWriteFile.size(), 0);
  std::string dataContent;
  uint64_t totalSize = 0;
  std::string randomData = AbfsFileSystemTest::generateRandomData(kOneMB);
  for (int i = 0; i < 8; ++i) {
    abfsWriteFile.append(randomData);
    dataContent += randomData;
  }
  totalSize = randomData.size() * 8;
  abfsWriteFile.flush();
  EXPECT_EQ(abfsWriteFile.size(), totalSize);

  randomData = AbfsFileSystemTest::generateRandomData(9 * kOneMB);
  dataContent += randomData;
  abfsWriteFile.append(randomData);
  totalSize += randomData.size();
  randomData = AbfsFileSystemTest::generateRandomData(2 * kOneMB);
  dataContent += randomData;
  totalSize += randomData.size();
  abfsWriteFile.append(randomData);
  abfsWriteFile.flush();
  EXPECT_EQ(abfsWriteFile.size(), totalSize);
  abfsWriteFile.flush();
  abfsWriteFile.close();
  VELOX_ASSERT_THROW(abfsWriteFile.append("abc"), "File is not open");

  std::unique_ptr<AzureDataLakeFileClient> mockClientCopy =
      std::make_unique<MockDataLakeFileClient>(mockClientPath);
  VELOX_ASSERT_THROW(
      AbfsWriteFile(kAbfsFile, mockClientCopy), "File already exists");
  MockDataLakeFileClient readClient(mockClientPath);
  auto fileContent = readClient.readContent();
  ASSERT_EQ(fileContent.size(), dataContent.size());
  ASSERT_EQ(fileContent, dataContent);
}

TEST_F(AbfsFileSystemTest, renameNotImplemented) {
  VELOX_ASSERT_THROW(
      abfs_->rename("text", "text2"), "rename for abfs not implemented");
}

TEST_F(AbfsFileSystemTest, notImplemented) {
  VELOX_ASSERT_THROW(abfs_->remove("text"), "remove for abfs not implemented");
  VELOX_ASSERT_THROW(abfs_->exists("text"), "exists for abfs not implemented");
  VELOX_ASSERT_THROW(abfs_->list("dir"), "list for abfs not implemented");
  VELOX_ASSERT_THROW(abfs_->mkdir("dir"), "mkdir for abfs not implemented");
  VELOX_ASSERT_THROW(abfs_->rmdir("dir"), "rmdir for abfs not implemented");
}

TEST_F(AbfsFileSystemTest, clientProviderFactoryNotRegistered) {
  const std::string abfsFile =
      std::string("abfs://test@test1.dfs.core.windows.net/test");
  VELOX_ASSERT_THROW(
      abfs_->openFileForRead(abfsFile),
      "No AzureClientProviderFactory registered for account 'test1'");
}

TEST_F(AbfsFileSystemTest, registerAbfsFileSink) {
  const std::vector<std::string> paths = {
      "abfs://test@test.dfs.core.windows.net/test",
      "abfss://test@test.dfs.core.windows.net/test"};
  std::unordered_map<std::string, std::string> config(
      {{"fs.azure.account.key.test.dfs.core.windows.net", "NDU2"}});
  auto hiveConfig =
      std::make_shared<const config::ConfigBase>(std::move(config));
  for (const auto& path : paths) {
    auto sink = dwio::common::FileSink::create(
        path, {.connectorProperties = hiveConfig});
    auto writeFileSink = dynamic_cast<dwio::common::WriteFileSink*>(sink.get());
    auto writeFile = writeFileSink->toWriteFile();
    auto abfsWriteFile = dynamic_cast<AbfsWriteFile*>(writeFile.get());
    ASSERT_TRUE(abfsWriteFile != nullptr);
  }
}
