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

#include "velox/common/io/AsyncIoContext.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <random>

#include <fmt/format.h>

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::io;
using namespace facebook::velox::common::testutil;

// Tests that run regardless of io_uring availability.
TEST(AsyncIoConfigTest, defaultValues) {
  AsyncIoConfig config;
  EXPECT_EQ(config.numThreads, 4);
  EXPECT_EQ(config.maxSubmit, 512);
  EXPECT_EQ(config.maxCapacity, 1'024);
  EXPECT_EQ(config.minCapacity, 512);
  EXPECT_EQ(config.registeredFds, 4'096);
  EXPECT_FALSE(config.sqpoll);
  EXPECT_EQ(config.sqpollIdleMs, 2'000);
}

TEST(AsyncIoConfigTest, toString) {
  struct TestCase {
    AsyncIoConfig config;
    std::vector<std::string> expectedSubstrings;
    std::string debugString() const {
      return config.toString();
    }
  };
  std::vector<TestCase> testCases = {
      {AsyncIoConfig{},
       {"numThreads=4", "maxSubmit=512", "maxCapacity=1024", "sqpoll=false"}},
      {AsyncIoConfig{.numThreads = 8, .sqpoll = true, .sqpollIdleMs = 500},
       {"numThreads=8", "sqpoll=true", "sqpollIdleMs=500"}},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    const auto str = testCase.config.toString();
    for (const auto& expected : testCase.expectedSubstrings) {
      EXPECT_NE(str.find(expected), std::string::npos)
          << "missing: " << expected;
    }
  }
}

// Parameterized over sqpoll mode: false = coop+defer, true = sqpoll.
class AsyncIoContextTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!AsyncIoContext::available()) {
      GTEST_SKIP() << "io_uring not available on this kernel";
    }
    if (GetParam()) {
      // SQPOLL requires CAP_SYS_NICE or root — skip if unprivileged.
      if (geteuid() != 0) {
        GTEST_SKIP() << "sqpoll mode requires root or CAP_SYS_NICE";
      }
    }
    memory::MemoryManager::testingSetInstance({});
    pool_ = memory::memoryManager()->addLeafPool("AsyncIoContextTest");
    AsyncIoConfig config;
    config.numThreads = 2;
    config.sqpoll = GetParam();
    ctx_ = std::make_shared<AsyncIoContext>(config);
  }

  void* allocateAligned(int64_t size) {
    return pool_->allocateAligned(size, kPageSize);
  }

  void freeAligned(void* buffer, int64_t size) {
    pool_->freeAligned(buffer, size, kPageSize);
  }

  std::shared_ptr<TempFilePath> createTestFile(const std::string& data) {
    auto tempFile = TempFilePath::create();
    {
      LocalWriteFile writeFile(
          tempFile->getPath(),
          /*shouldCreateParentDirectories=*/false,
          /*shouldThrowOnFileAlreadyExists=*/false);
      writeFile.append(data);
      writeFile.close();
    }
    return tempFile;
  }

  std::string modeName() const {
    return GetParam() ? "sqpoll" : "non-sqpoll";
  }

  static constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<AsyncIoContext> ctx_;
};

TEST_P(AsyncIoContextTest, read) {
  SCOPED_TRACE(modeName());

  constexpr int kNumPages = 16;
  std::string fileData(kNumPages * kPageSize, '\0');
  for (int i = 0; i < kNumPages; ++i) {
    std::memset(fileData.data() + i * kPageSize, 'A' + (i % 26), kPageSize);
  }
  auto tempFile = createTestFile(fileData);

  const int fd = ::open(tempFile->getPath().c_str(), O_RDONLY | O_DIRECT);
  ASSERT_GE(fd, 0);

  struct TestCase {
    off_t offset;
    size_t size;
    std::string debugString() const {
      return fmt::format("offset={}, size={}", offset, size);
    }
  };
  std::vector<TestCase> testCases = {
      {0, kPageSize},
      {0, 2 * kPageSize},
      {static_cast<off_t>(kPageSize), kPageSize},
      {0, kNumPages * kPageSize},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    auto* buf = allocateAligned(testCase.size);

    const int bytesRead = folly::coro::blockingWait(
        ctx_->coPread(fd, testCase.offset, buf, testCase.size));

    EXPECT_EQ(bytesRead, testCase.size);
    EXPECT_EQ(
        std::memcmp(buf, fileData.data() + testCase.offset, testCase.size), 0);

    freeAligned(buf, testCase.size);
  }

  ::close(fd);
}

TEST_P(AsyncIoContextTest, preadv) {
  SCOPED_TRACE(modeName());

  constexpr int kMaxIovecs = 4;
  std::string fileData(kMaxIovecs * kPageSize, '\0');
  for (int i = 0; i < kMaxIovecs; ++i) {
    std::memset(fileData.data() + i * kPageSize, 'A' + i, kPageSize);
  }
  auto tempFile = createTestFile(fileData);

  const int fd = ::open(tempFile->getPath().c_str(), O_RDONLY | O_DIRECT);
  ASSERT_GE(fd, 0);

  for (int iovcnt : {1, 2, 4}) {
    SCOPED_TRACE(fmt::format("iovcnt={}", iovcnt));

    std::vector<void*> buffers(iovcnt);
    std::vector<struct iovec> iovecs(iovcnt);
    for (int i = 0; i < iovcnt; ++i) {
      buffers[i] = allocateAligned(kPageSize);
      iovecs[i] = {buffers[i], kPageSize};
    }

    const int bytesRead =
        folly::coro::blockingWait(ctx_->coPreadv(fd, 0, iovecs.data(), iovcnt));

    EXPECT_EQ(bytesRead, iovcnt * kPageSize);
    for (int i = 0; i < iovcnt; ++i) {
      const std::string expected(kPageSize, 'A' + i);
      EXPECT_EQ(std::memcmp(buffers[i], expected.data(), kPageSize), 0);
    }

    for (auto* buf : buffers) {
      freeAligned(buf, kPageSize);
    }
  }

  ::close(fd);
}

TEST_P(AsyncIoContextTest, concurrentReads) {
  SCOPED_TRACE(modeName());

  constexpr int kNumPages = 16;
  std::string fileData(kNumPages * kPageSize, '\0');
  for (int i = 0; i < kNumPages; ++i) {
    std::memset(fileData.data() + i * kPageSize, 'A' + (i % 26), kPageSize);
  }
  auto tempFile = createTestFile(fileData);

  const int fd = ::open(tempFile->getPath().c_str(), O_RDONLY | O_DIRECT);
  ASSERT_GE(fd, 0);

  // Stress test: repeatedly launch batches of concurrent coPread and
  // coPreadv calls for 10 seconds, randomly mixing both interfaces.
  constexpr auto kDuration = std::chrono::seconds(10);
  constexpr int kBatchSize = 32;
  const auto deadline = std::chrono::steady_clock::now() + kDuration;
  std::mt19937 rng(42);
  int totalOps = 0;

  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<folly::coro::Task<int>> tasks;
    std::vector<void*> buffers;
    std::vector<int> pageIndices;
    // Whether each task uses preadv (true) or pread (false).
    std::vector<bool> isPreadv;
    // For preadv tasks, the iovec arrays must outlive the coroutine.
    std::vector<std::vector<struct iovec>> iovecStorage;

    for (int i = 0; i < kBatchSize; ++i) {
      const int pageIdx = rng() % kNumPages;
      const bool usePreadv = rng() % 2 == 0;
      isPreadv.push_back(usePreadv);
      pageIndices.push_back(pageIdx);

      if (usePreadv) {
        const int iovcnt =
            std::min(1 + static_cast<int>(rng() % 3), kNumPages - pageIdx);
        std::vector<struct iovec> iovecs(iovcnt);
        for (int j = 0; j < iovcnt; ++j) {
          auto* buf = allocateAligned(kPageSize);
          buffers.push_back(buf);
          iovecs[j] = {buf, kPageSize};
        }
        iovecStorage.push_back(std::move(iovecs));
        auto& storedIovecs = iovecStorage.back();
        tasks.emplace_back(ctx_->coPreadv(
            fd, pageIdx * kPageSize, storedIovecs.data(), storedIovecs.size()));
      } else {
        auto* buf = allocateAligned(kPageSize);
        buffers.push_back(buf);
        tasks.emplace_back(
            ctx_->coPread(fd, pageIdx * kPageSize, buf, kPageSize));
      }
    }

    auto results = folly::coro::blockingWait(
        folly::coro::collectAllRange(std::move(tasks)));

    // Verify results.
    int bufIdx = 0;
    int iovecIdx = 0;
    for (int i = 0; i < kBatchSize; ++i) {
      if (isPreadv[i]) {
        const auto& iovecs = iovecStorage[iovecIdx++];
        EXPECT_EQ(results[i], iovecs.size() * kPageSize);
        for (size_t j = 0; j < iovecs.size(); ++j) {
          const std::string expected(
              kPageSize, 'A' + ((pageIndices[i] + j) % 26));
          EXPECT_EQ(
              std::memcmp(buffers[bufIdx], expected.data(), kPageSize), 0);
          freeAligned(buffers[bufIdx], kPageSize);
          ++bufIdx;
        }
      } else {
        EXPECT_EQ(results[i], kPageSize);
        const std::string expected(kPageSize, 'A' + (pageIndices[i] % 26));
        EXPECT_EQ(std::memcmp(buffers[bufIdx], expected.data(), kPageSize), 0);
        freeAligned(buffers[bufIdx], kPageSize);
        ++bufIdx;
      }
    }

    totalOps += kBatchSize;
  }

  LOG(INFO) << "concurrentReads completed " << totalOps << " ops in 10s ("
            << modeName() << ")";

  ::close(fd);
}

TEST_P(AsyncIoContextTest, misalignedBuffer) {
  SCOPED_TRACE(modeName());
  auto* aligned = static_cast<char*>(allocateAligned(2 * kPageSize));
  void* misaligned = aligned + 1;

  VELOX_ASSERT_THROW(
      folly::coro::blockingWait(
          ctx_->coPread(/*fd=*/0, /*offset=*/0, misaligned, kPageSize)),
      "page-aligned");

  freeAligned(aligned, 2 * kPageSize);
}

TEST_P(AsyncIoContextTest, misalignedOffset) {
  SCOPED_TRACE(modeName());
  auto* buf = allocateAligned(kPageSize);

  VELOX_ASSERT_THROW(
      folly::coro::blockingWait(
          ctx_->coPread(/*fd=*/0, /*offset=*/1, buf, kPageSize)),
      "page-aligned");

  freeAligned(buf, kPageSize);
}

INSTANTIATE_TEST_SUITE_P(
    IoModes,
    AsyncIoContextTest,
    ::testing::Values(false, true),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "sqpoll" : "nonSqpoll";
    });
