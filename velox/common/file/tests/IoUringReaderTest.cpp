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

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <folly/ScopeGuard.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/testutil/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;

namespace {

IoUringReader::Stats makeStats(
    uint64_t readCalls,
    uint64_t regions,
    uint64_t minRegionsPerRead,
    uint64_t maxRegionsPerRead,
    uint64_t batches,
    uint64_t minBatchSize,
    uint64_t maxBatchSize,
    uint64_t registeredFileUpdates = 0) {
  IoUringReader::Stats stats;
  stats.readCalls = readCalls;
  stats.regions = regions;
  stats.minRegionsPerRead = minRegionsPerRead;
  stats.maxRegionsPerRead = maxRegionsPerRead;
  stats.batches = batches;
  stats.minBatchSize = minBatchSize;
  stats.maxBatchSize = maxBatchSize;
  stats.registeredFileUpdates = registeredFileUpdates;
  return stats;
}

void expectStatsEqual(
    const IoUringReader::Stats& actual,
    const IoUringReader::Stats& expected) {
  EXPECT_EQ(actual.readCalls, expected.readCalls);
  EXPECT_EQ(actual.regions, expected.regions);
  EXPECT_EQ(actual.minRegionsPerRead, expected.minRegionsPerRead);
  EXPECT_EQ(actual.maxRegionsPerRead, expected.maxRegionsPerRead);
  EXPECT_EQ(actual.batches, expected.batches);
  EXPECT_EQ(actual.minBatchSize, expected.minBatchSize);
  EXPECT_EQ(actual.maxBatchSize, expected.maxBatchSize);
  EXPECT_EQ(actual.registeredFileUpdates, expected.registeredFileUpdates);
}

TEST(IoUringReaderStatsTest, mergeAndToString) {
  struct TestCase {
    const char* name;
    IoUringReader::Stats initial;
    std::vector<IoUringReader::Stats> merged;
    IoUringReader::Stats expected;
    std::string expectedString;
  };

  const std::vector<TestCase> testCases = {
      {
          "empty",
          IoUringReader::Stats{},
          {},
          IoUringReader::Stats{},
          "readCalls=0, regions=0, regionsPerReadAvg=0, regionsPerReadMin=0, regionsPerReadMax=0, batches=0, batchSizeAvg=0, batchSizeMin=0, batchSizeMax=0, registeredFileUpdates=0",
      },
      {
          "mergeEmptyStats",
          makeStats(1, 4, 4, 4, 1, 4, 4),
          {IoUringReader::Stats{}},
          makeStats(1, 4, 4, 4, 1, 4, 4),
          "readCalls=1, regions=4, regionsPerReadAvg=4, regionsPerReadMin=4, regionsPerReadMax=4, batches=1, batchSizeAvg=4, batchSizeMin=4, batchSizeMax=4, registeredFileUpdates=0",
      },
      {
          "mergeNonEmptyStats",
          makeStats(1, 7, 7, 7, 2, 3, 4, 1),
          {makeStats(2, 5, 2, 3, 2, 2, 3, 3)},
          makeStats(3, 12, 2, 7, 4, 2, 4, 4),
          "readCalls=3, regions=12, regionsPerReadAvg=4, regionsPerReadMin=2, regionsPerReadMax=7, batches=4, batchSizeAvg=3, batchSizeMin=2, batchSizeMax=4, registeredFileUpdates=4",
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    auto stats = testCase.initial;
    for (const auto& other : testCase.merged) {
      stats.merge(other);
    }
    expectStatsEqual(stats, testCase.expected);
    EXPECT_EQ(stats.toString(), testCase.expectedString);
  }
}

#if FOLLY_HAS_LIBURING

struct FdGuard {
  explicit FdGuard(int descriptor) : fd(descriptor) {}

  explicit FdGuard(std::string_view path) {
    const std::string ownedPath(path);
    fd = ::open(ownedPath.c_str(), O_RDONLY);
  }

  FdGuard(const FdGuard&) = delete;
  FdGuard& operator=(const FdGuard&) = delete;

  FdGuard(FdGuard&& other) noexcept : fd(std::exchange(other.fd, -1)) {}

  ~FdGuard() {
    if (fd >= 0) {
      ::close(fd);
    }
  }

  int fd{-1};
};

template <typename Ranges>
folly::Range<const folly::Range<char*>*> asConstRange(Ranges& ranges) {
  return {ranges.data(), ranges.size()};
}

struct ReadRegion {
  uint64_t offset;
  size_t length;
};

constexpr size_t kPageSize{4'096};
constexpr size_t kTestDataSize{4 * 1'024 * 1'024};
constexpr size_t kRegisteredFileSlotTestDataSize{256 * 1'024};

std::vector<ReadRegion> makeReadRegions(size_t count) {
  std::vector<ReadRegion> regions;
  regions.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    regions.push_back({i * kPageSize, kPageSize});
  }
  return regions;
}

std::string makeTestData(size_t size) {
  std::string data(size, '\0');
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>('a' + (i % 26));
  }
  return data;
}

IoUringReader::Options makeReaderOptions(
    int32_t queueDepth = 128,
    int32_t registeredFileSlots = 128) {
  IoUringReader::Options options;
  options.submissionQueueDepth = queueDepth;
  options.completionQueueDepth = queueDepth;
  options.registeredFileSlots = registeredFileSlots;
  return options;
}

uint64_t expectedRegisteredFileUpdates(uint64_t count) {
#ifdef IOSQE_FIXED_FILE
  return count;
#else
  return 0;
#endif
}

IoUringReader::Stats expectedStatsForRegionCounts(
    const std::vector<size_t>& regionCounts,
    size_t submissionQueueDepth) {
  IoUringReader::Stats expected;
  for (const auto regionCount : regionCounts) {
    auto oneRead = makeStats(
        /*readCalls=*/1,
        /*regions=*/regionCount,
        /*minRegionsPerRead=*/regionCount,
        /*maxRegionsPerRead=*/regionCount,
        /*batches=*/(regionCount + submissionQueueDepth - 1) /
            submissionQueueDepth,
        /*minBatchSize=*/regionCount <= submissionQueueDepth
            ? regionCount
            : std::min(
                  regionCount % submissionQueueDepth, submissionQueueDepth),
        /*maxBatchSize=*/std::min(regionCount, submissionQueueDepth),
        /*registeredFileUpdates=*/expectedRegisteredFileUpdates(1));
    if (regionCount > submissionQueueDepth &&
        regionCount % submissionQueueDepth == 0) {
      oneRead.minBatchSize = submissionQueueDepth;
    }
    expected.merge(oneRead);
  }
  return expected;
}

class IoUringReaderOptionsGuard {
 public:
  explicit IoUringReaderOptionsGuard(IoUringReader::Options options) {
    IoUringReader::setOptions(options);
  }

  ~IoUringReaderOptionsGuard() {
    IoUringReader::setOptions(IoUringReader::Options{});
  }
};

class IoUringReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempFile_ = TempFilePath::create();
    LocalWriteFile writeFile(
        tempFile_->getPath(),
        /*shouldCreateParentDirectories=*/false,
        /*shouldThrowOnFileAlreadyExists=*/false);
    writeFile.append(testData_);
    writeFile.close();
  }

  FdGuard openFile() const {
    FdGuard fd(tempFile_->getPath());
    EXPECT_GE(fd.fd, 0);
    return fd;
  }

  void verifyRead(const std::vector<char>& buffer, uint64_t offset = 10) const {
    EXPECT_EQ(
        std::memcmp(buffer.data(), testData_.data() + offset, buffer.size()),
        0);
  }

  template <typename Reader>
  void readAndVerify(
      Reader& reader,
      int fd,
      const std::vector<ReadRegion>& readRegions,
      bool useRegisteredFiles = true) const {
    std::vector<std::vector<char>> reads;
    std::vector<common::Region> regions;
    std::vector<folly::Range<char*>> buffers;
    reads.reserve(readRegions.size());
    regions.reserve(readRegions.size());
    buffers.reserve(readRegions.size());

    uint64_t expectedBytes{0};
    for (const auto& region : readRegions) {
      reads.emplace_back(region.length);
      regions.emplace_back(region.offset, region.length);
      buffers.emplace_back(reads.back().data(), reads.back().size());
      expectedBytes += region.length;
    }

    const auto bytesRead = reader.read(
        fd,
        folly::Range<const common::Region*>(regions.data(), regions.size()),
        asConstRange(buffers),
        useRegisteredFiles);
    VELOX_CHECK_EQ(bytesRead, expectedBytes);
    for (size_t i = 0; i < reads.size(); ++i) {
      VELOX_CHECK_EQ(
          std::memcmp(
              reads[i].data(),
              testData_.data() + readRegions[i].offset,
              reads[i].size()),
          0);
    }
  }

  const std::string testData_{makeTestData(kTestDataSize)};
  std::shared_ptr<TempFilePath> tempFile_;
};

TEST_F(IoUringReaderTest, basic) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto fd = openFile();
  ASSERT_GE(fd.fd, 0);

  struct TestCase {
    const char* name;
    std::vector<ReadRegion> regions;
  };

  const std::vector<TestCase> testCases = {
      {
          "singlePage",
          {{0, kPageSize}},
      },
      {
          "adjacentPages",
          {{0, kPageSize}, {kPageSize, kPageSize}, {2 * kPageSize, kPageSize}},
      },
      {
          "outOfOrderPages",
          {{3 * kPageSize, kPageSize}, {kPageSize, kPageSize}, {0, kPageSize}},
      },
      {
          "sparsePages",
          {{0, kPageSize},
           {31 * kPageSize, kPageSize},
           {127 * kPageSize, kPageSize}},
      },
      {
          "partialPageRanges",
          {
              {kPageSize / 2, kPageSize},
              {2 * kPageSize - 128, 256},
              {4 * kPageSize + 17, kPageSize / 2},
          },
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    const auto options = makeReaderOptions();
    auto reader = std::make_unique<IoUringReader>(options);

    readAndVerify(*reader, fd.fd, testCase.regions);
    const auto stats = reader->stats();
    EXPECT_EQ(stats.readCalls, 1);
    EXPECT_EQ(stats.regions, testCase.regions.size());
    EXPECT_EQ(stats.minRegionsPerRead, testCase.regions.size());
    EXPECT_EQ(stats.maxRegionsPerRead, testCase.regions.size());
    EXPECT_EQ(stats.batches, 1);
    EXPECT_EQ(stats.minBatchSize, testCase.regions.size());
    EXPECT_EQ(stats.maxBatchSize, testCase.regions.size());
  }
}

TEST_F(IoUringReaderTest, getIoUringReaderStatsReturnsMergedStats) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto fd = openFile();
  ASSERT_GE(fd.fd, 0);

  enum class Mode {
    kSingleThread,
    kMultipleThreads,
  };

  struct TestCase {
    const char* name;
    Mode mode;
    std::vector<size_t> regionCounts;
  };

  const std::vector<TestCase> testCases = {
      {
          "singleThread",
          Mode::kSingleThread,
          {1, 2},
      },
      {
          "multipleThreads",
          Mode::kMultipleThreads,
          {1, 2, 3, 4},
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    ThreadLocalIoUringReader::testingClear();
    const auto options = makeReaderOptions();
    uint64_t numReaders{0};
    expectStatsEqual(getIoUringReaderStats(numReaders), IoUringReader::Stats{});
    ASSERT_EQ(numReaders, 0);

    if (testCase.mode == Mode::kSingleThread) {
      std::vector<std::unique_ptr<IoUringReader>> readers;
      readers.reserve(testCase.regionCounts.size());
      for (const auto regionCount : testCase.regionCounts) {
        readers.push_back(std::make_unique<IoUringReader>(options));
        readAndVerify(*readers.back(), fd.fd, makeReadRegions(regionCount));
      }

      const auto stats = getIoUringReaderStats(numReaders);
      EXPECT_EQ(numReaders, testCase.regionCounts.size());
      expectStatsEqual(
          stats,
          expectedStatsForRegionCounts(
              testCase.regionCounts, options.submissionQueueDepth));

      readers.clear();
      expectStatsEqual(
          getIoUringReaderStats(numReaders), IoUringReader::Stats{});
      EXPECT_EQ(numReaders, 0);
      continue;
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::exception_ptr error;
    size_t completed{0};
    bool release{false};
    std::vector<std::thread> threads;
    threads.reserve(testCase.regionCounts.size());
    for (size_t i = 0; i < testCase.regionCounts.size(); ++i) {
      threads.emplace_back([&, i]() {
        try {
          IoUringReader reader(options);
          readAndVerify(
              reader, fd.fd, makeReadRegions(testCase.regionCounts[i]));
          {
            std::lock_guard<std::mutex> l(mutex);
            ++completed;
          }
          condition.notify_all();
          std::unique_lock<std::mutex> l(mutex);
          condition.wait(l, [&]() { return release; });
        } catch (...) {
          std::lock_guard<std::mutex> l(mutex);
          if (error == nullptr) {
            error = std::current_exception();
          }
          ++completed;
          condition.notify_all();
        }
      });
    }

    {
      std::unique_lock<std::mutex> l(mutex);
      condition.wait(
          l, [&]() { return completed == testCase.regionCounts.size(); });
    }

    if (error == nullptr) {
      const auto stats = getIoUringReaderStats(numReaders);
      EXPECT_EQ(numReaders, testCase.regionCounts.size());
      expectStatsEqual(
          stats,
          expectedStatsForRegionCounts(
              testCase.regionCounts, options.submissionQueueDepth));
    }

    {
      std::lock_guard<std::mutex> l(mutex);
      release = true;
    }
    condition.notify_all();
    for (auto& thread : threads) {
      thread.join();
    }

    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    expectStatsEqual(getIoUringReaderStats(numReaders), IoUringReader::Stats{});
    EXPECT_EQ(numReaders, 0);
  }
}

TEST_F(IoUringReaderTest, queueDepth) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto fd = openFile();
  ASSERT_GE(fd.fd, 0);

  struct TestCase {
    std::string name;
    int32_t submissionQueueDepth;
    int32_t completionQueueDepth;
    size_t regionCount;
    size_t expectedBatches;
  };

  auto expectedBatchCount = [](size_t regionCount, int32_t queueDepth) {
    return (regionCount + queueDepth - 1) / queueDepth;
  };

  std::vector<TestCase> testCases;
  for (int32_t queueDepth = 2; queueDepth <= 128; queueDepth *= 2) {
    const auto singleBatchRegions = static_cast<size_t>(queueDepth);
    const auto multipleBatchRegions = static_cast<size_t>(queueDepth + 3);
    testCases.push_back(
        {std::string("equalQueuesSingleBatch") + std::to_string(queueDepth),
         /*submissionQueueDepth=*/queueDepth,
         /*completionQueueDepth=*/queueDepth,
         /*regionCount=*/singleBatchRegions,
         /*expectedBatches=*/
         expectedBatchCount(singleBatchRegions, queueDepth)});
    testCases.push_back(
        {std::string("equalQueuesMultipleBatches") + std::to_string(queueDepth),
         /*submissionQueueDepth=*/queueDepth,
         /*completionQueueDepth=*/queueDepth,
         /*regionCount=*/multipleBatchRegions,
         /*expectedBatches=*/
         expectedBatchCount(multipleBatchRegions, queueDepth)});
    if (queueDepth < 128) {
      testCases.push_back(
          {std::string("largerCompletionQueueSingleBatch") +
               std::to_string(queueDepth),
           /*submissionQueueDepth=*/queueDepth,
           /*completionQueueDepth=*/queueDepth * 2,
           /*regionCount=*/singleBatchRegions,
           /*expectedBatches=*/
           expectedBatchCount(singleBatchRegions, queueDepth)});
      testCases.push_back(
          {std::string("largerCompletionQueueMultipleBatches") +
               std::to_string(queueDepth),
           /*submissionQueueDepth=*/queueDepth,
           /*completionQueueDepth=*/queueDepth * 2,
           /*regionCount=*/multipleBatchRegions,
           /*expectedBatches=*/
           expectedBatchCount(multipleBatchRegions, queueDepth)});
    }
  }

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    IoUringReader::Options options;
    options.submissionQueueDepth = testCase.submissionQueueDepth;
    options.completionQueueDepth = testCase.completionQueueDepth;
    options.registeredFileSlots = 4;

    std::unique_ptr<IoUringReader> reader;
    ASSERT_NO_THROW(reader = std::make_unique<IoUringReader>(options));

    readAndVerify(*reader, fd.fd, makeReadRegions(testCase.regionCount));
    EXPECT_EQ(reader->stats().batches, testCase.expectedBatches);
    expectStatsEqual(
        reader->stats(),
        expectedStatsForRegionCounts(
            {testCase.regionCount}, testCase.submissionQueueDepth));
  }
}

TEST_F(
    IoUringReaderTest,
    unregisteredReadsSupportMixedModesAndMultipleBatches) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  auto fd = openFile();
  IoUringReader reader(makeReaderOptions(2, 4));
  readAndVerify(reader, fd.fd, makeReadRegions(7), false);
  EXPECT_EQ(reader.stats().registeredFileUpdates, 0u);
  readAndVerify(reader, fd.fd, makeReadRegions(7));
  EXPECT_EQ(
      reader.stats().registeredFileUpdates, expectedRegisteredFileUpdates(1));
  readAndVerify(reader, fd.fd, makeReadRegions(7), false);
  EXPECT_EQ(
      reader.stats().registeredFileUpdates, expectedRegisteredFileUpdates(1));
}

TEST_F(IoUringReaderTest, unregisteredReadReleasesKernelFileOwnership) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
#ifdef IOSQE_FIXED_FILE
  const auto path = tempFile_->getPath() + "-lease";
  SCOPE_EXIT {
    ::unlink(path.c_str());
  };
  {
    LocalWriteFile writer(path);
    writer.append(testData_);
    writer.close();
  }
  FdGuard leaseProbe(::open(path.c_str(), O_RDWR));
  const int probe = leaseProbe.fd;
  ASSERT_GE(probe, 0);
  if (::fcntl(probe, F_SETLEASE, F_WRLCK) != 0) {
    GTEST_SKIP() << "filesystem does not support exclusive file leases";
  }
  ASSERT_EQ(::fcntl(probe, F_SETLEASE, F_UNLCK), 0);
  {
    IoUringReader registered(makeReaderOptions());
    {
      FdGuard fd(path);
      readAndVerify(registered, fd.fd, makeReadRegions(1));
    }
    EXPECT_EQ(::fcntl(probe, F_SETLEASE, F_WRLCK), -1);
    EXPECT_EQ(errno, EAGAIN);
  }
  ASSERT_EQ(::fcntl(probe, F_SETLEASE, F_WRLCK), 0);
  ASSERT_EQ(::fcntl(probe, F_SETLEASE, F_UNLCK), 0);
  IoUringReader unregistered(makeReaderOptions());
  {
    FdGuard fd(path);
    readAndVerify(unregistered, fd.fd, makeReadRegions(1), false);
    ASSERT_EQ(::unlink(path.c_str()), 0);
  }
  EXPECT_EQ(unregistered.stats().registeredFileUpdates, 0u);
  EXPECT_EQ(::fcntl(probe, F_SETLEASE, F_WRLCK), 0);
  EXPECT_EQ(::fcntl(probe, F_SETLEASE, F_UNLCK), 0);
#else
  GTEST_SKIP() << "fixed-file registration is unavailable";
#endif
}

TEST_F(IoUringReaderTest, invalidBatchDoesNotLeaveQueuedReads) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  auto fd = openFile();
  IoUringReader reader(makeReaderOptions(2));
  std::vector<char> destination(8, 'z');
  const std::vector<common::Region> regions{{0, 8}, {0, 8}};
  const std::vector<folly::Range<char*>> buffers{
      {destination.data(), destination.size()}, {destination.data(), 7}};
  VELOX_ASSERT_THROW(
      reader.read(fd.fd, regions, buffers, false),
      "io_uring destination length must match region length");
  readAndVerify(reader, fd.fd, makeReadRegions(3), false);
  EXPECT_EQ(destination, std::vector<char>(8, 'z'));
}

TEST_F(IoUringReaderTest, partialSubmissionAndInterruptionCompleteEveryRead) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  auto fd = openFile();
  IoUringReader reader(makeReaderOptions(4));
  static thread_local size_t calls;
  calls = 0;
  reader.testingSetIoUringCalls(
      [](io_uring* ring, unsigned waitCount) {
        if (++calls == 1) {
          return -EINTR;
        }
        if (calls == 2) {
          const auto tail = ring->sq.sqe_tail;
          ring->sq.sqe_tail = ring->sq.sqe_head + 1;
          const int result = ::io_uring_submit_and_wait(ring, waitCount);
          ring->sq.sqe_tail = tail;
          return result;
        }
        return ::io_uring_submit_and_wait(ring, waitCount);
      },
      nullptr);
  readAndVerify(reader, fd.fd, makeReadRegions(7), false);
  EXPECT_GE(calls, 4u);
  reader.testingSetIoUringCalls(nullptr, nullptr);
  readAndVerify(reader, fd.fd, makeReadRegions(3));
}

TEST_F(IoUringReaderTest, failedSubmissionDiscardsQueuedReadsBeforeFdReuse) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  auto fd = openFile();
  IoUringReader reader(makeReaderOptions(4));
  reader.testingSetIoUringCalls(
      [](io_uring*, unsigned) { return -EIO; }, nullptr);
  std::vector<char> staleDestination(8, 'z');
  const std::vector<common::Region> regions{{0, staleDestination.size()}};
  const std::vector<folly::Range<char*>> buffers{
      {staleDestination.data(), staleDestination.size()}};
  VELOX_ASSERT_THROW(
      reader.read(fd.fd, regions, buffers, false),
      "io_uring submission failed");
  const int reusedFd = fd.fd;
  ::close(std::exchange(fd.fd, -1));
  auto replacement = openFile();
  ASSERT_GE(replacement.fd, 0);
  if (replacement.fd != reusedFd) {
    ASSERT_EQ(::dup2(replacement.fd, reusedFd), reusedFd);
    ::close(std::exchange(replacement.fd, reusedFd));
  }
  reader.testingSetIoUringCalls(nullptr, nullptr);
  readAndVerify(reader, replacement.fd, makeReadRegions(4), false);
  EXPECT_EQ(staleDestination, std::vector<char>(8, 'z'));
}

TEST_F(IoUringReaderTest, failureAfterPartialSubmissionRecoversForNextBatch) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  auto fd = openFile();
  IoUringReader reader(makeReaderOptions(4));
  static thread_local size_t calls;
  calls = 0;
  reader.testingSetIoUringCalls(
      [](io_uring* ring, unsigned waitCount) {
        if (++calls != 1) {
          return -EIO;
        }
        const auto tail = ring->sq.sqe_tail;
        ring->sq.sqe_tail = ring->sq.sqe_head + 1;
        const int result = ::io_uring_submit_and_wait(ring, waitCount);
        ring->sq.sqe_tail = tail;
        return result;
      },
      nullptr);
  VELOX_ASSERT_THROW(
      readAndVerify(reader, fd.fd, makeReadRegions(4), false),
      "io_uring submission failed");
  reader.testingSetIoUringCalls(nullptr, nullptr);
  readAndVerify(reader, fd.fd, makeReadRegions(4), false);
}

TEST_F(IoUringReaderTest, completionWaitFailureRecoversOutstandingRead) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  int descriptors[2];
  ASSERT_EQ(::pipe(descriptors), 0);
  FdGuard source(descriptors[0]);
  FdGuard destination(descriptors[1]);
  static thread_local int writer;
  writer = destination.fd;
  IoUringReader reader(makeReaderOptions(4));
  reader.testingSetIoUringCalls(
      [](io_uring* ring, unsigned) { return ::io_uring_submit(ring); },
      [](io_uring*, io_uring_cqe**, unsigned) {
        EXPECT_EQ(::write(writer, "12345678", 8), 8);
        return -EIO;
      });
  std::vector<char> buffer(8);
  const std::vector<common::Region> regions{
      {std::numeric_limits<uint64_t>::max(), 8}};
  const std::vector<folly::Range<char*>> buffers{
      {buffer.data(), buffer.size()}};
  VELOX_ASSERT_THROW(
      reader.read(source.fd, regions, buffers, false),
      "io_uring completion wait failed");
  std::fill(buffer.begin(), buffer.end(), 'z');
  reader.testingSetIoUringCalls(nullptr, nullptr);
  auto fd = openFile();
  readAndVerify(reader, fd.fd, makeReadRegions(4), false);
  EXPECT_EQ(buffer, std::vector<char>(8, 'z'));
}

TEST_F(IoUringReaderTest, completionErrorDrainsBatchBeforeNextRead) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }
  IoUringReader reader(makeReaderOptions(4));
  VELOX_ASSERT_THROW(
      readAndVerify(reader, 1'000'000, makeReadRegions(4), false),
      "io_uring batch pread failed");
  auto fd = openFile();
  readAndVerify(reader, fd.fd, makeReadRegions(4), false);
}

TEST_F(IoUringReaderTest, reusesRegisteredFileSlots) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto read = [&](IoUringReader& reader, int fd, const std::string& expected) {
    std::vector<char> buffer(8);
    std::vector<common::Region> regions = {
        common::Region(10, buffer.size()),
    };
    std::vector<folly::Range<char*>> buffers = {
        folly::Range<char*>(buffer.data(), buffer.size()),
    };

    const auto bytesRead = reader.read(
        fd,
        folly::Range<const common::Region*>(regions.data(), regions.size()),
        asConstRange(buffers));
    EXPECT_EQ(bytesRead, buffer.size());
    EXPECT_EQ(
        std::memcmp(buffer.data(), expected.data() + 10, buffer.size()), 0);
  };

  auto makeFileData = [](size_t fileIndex) {
    std::string data(kRegisteredFileSlotTestDataSize, '\0');
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<char>('a' + ((fileIndex * 7 + i) % 26));
    }
    return data;
  };

  struct TestCase {
    const char* name;
    int32_t registeredFileSlots;
    size_t fileCount;
    std::vector<size_t> accessOrder;
    uint64_t expectedFileUpdates;
  };

  const std::vector<TestCase> testCases = {
      {
          "reuseWhenTableFits",
          /*registeredFileSlots=*/2,
          /*fileCount=*/2,
          /*accessOrder=*/{0, 1, 0, 1},
          /*expectedFileUpdates=*/2,
      },
      {
          "singleSlotAlternates",
          /*registeredFileSlots=*/1,
          /*fileCount=*/2,
          /*accessOrder=*/{0, 1, 0, 1},
          /*expectedFileUpdates=*/4,
      },
      {
          "evictWhenTableIsFull",
          /*registeredFileSlots=*/2,
          /*fileCount=*/3,
          /*accessOrder=*/{0, 1, 0, 1, 2, 0, 1, 1},
          /*expectedFileUpdates=*/5,
      },
      {
          "largerTableFitsAllFiles",
          /*registeredFileSlots=*/4,
          /*fileCount=*/3,
          /*accessOrder=*/{0, 1, 2, 0, 2, 1},
          /*expectedFileUpdates=*/3,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    std::vector<std::shared_ptr<TempFilePath>> tempFiles;
    std::vector<std::string> data;
    std::vector<FdGuard> fds;
    tempFiles.reserve(testCase.fileCount);
    data.reserve(testCase.fileCount);
    fds.reserve(testCase.fileCount);
    for (size_t i = 0; i < testCase.fileCount; ++i) {
      data.push_back(makeFileData(i));
      auto tempFile = TempFilePath::create();
      {
        LocalWriteFile writeFile(
            tempFile->getPath(),
            /*shouldCreateParentDirectories=*/false,
            /*shouldThrowOnFileAlreadyExists=*/false);
        writeFile.append(data.back());
        writeFile.close();
      }
      fds.emplace_back(tempFile->getPath());
      ASSERT_GE(fds.back().fd, 0);
      tempFiles.push_back(std::move(tempFile));
    }

    auto reader = std::make_unique<IoUringReader>(makeReaderOptions(
        /*queueDepth=*/128, testCase.registeredFileSlots));

    for (const auto fileIndex : testCase.accessOrder) {
      ASSERT_LT(fileIndex, testCase.fileCount);
      read(*reader, fds[fileIndex].fd, data[fileIndex]);
    }

    const auto stats = reader->stats();
    EXPECT_EQ(stats.readCalls, testCase.accessOrder.size());
    EXPECT_EQ(stats.regions, testCase.accessOrder.size());
    EXPECT_EQ(
        stats.registeredFileUpdates,
        expectedRegisteredFileUpdates(testCase.expectedFileUpdates));
  }
}

TEST_F(IoUringReaderTest, singleThreadRead) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto fd = openFile();
  ASSERT_GE(fd.fd, 0);

  IoUringReaderOptionsGuard optionsGuard(makeReaderOptions());

  std::vector<char> first(8);
  std::vector<common::Region> regions = {
      common::Region(10, first.size()),
  };
  std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(first.data(), first.size()),
  };

  const auto bytesRead = ThreadLocalIoUringReader::get().read(
      fd.fd,
      folly::Range<const common::Region*>(regions.data(), regions.size()),
      asConstRange(buffers));
  EXPECT_EQ(bytesRead, first.size());
  verifyRead(first);
}

TEST_F(IoUringReaderTest, concurrentRead) {
  if (!IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  auto fd = openFile();
  ASSERT_GE(fd.fd, 0);

  IoUringReader::Options options;
  options.submissionQueueDepth = 64;
  options.completionQueueDepth = 64;
  options.registeredFileSlots = 64;
  IoUringReaderOptionsGuard optionsGuard(options);

  constexpr size_t kThreadCount{8};
  constexpr size_t kMaxRegionsPerRead{16};
  constexpr size_t kMaxReadLength{kPageSize};
  constexpr auto kRunDuration = std::chrono::seconds(10);

  auto makeRandomReadRegions = [](std::mt19937_64& rng) {
    std::uniform_int_distribution<size_t> regionCountDistribution(
        1, kMaxRegionsPerRead);
    std::uniform_int_distribution<size_t> readLengthDistribution(
        1, kMaxReadLength);

    std::vector<ReadRegion> readRegions;
    const auto regionCount = regionCountDistribution(rng);
    readRegions.reserve(regionCount);
    for (size_t i = 0; i < regionCount; ++i) {
      const auto length = readLengthDistribution(rng);
      std::uniform_int_distribution<uint64_t> offsetDistribution(
          0, kTestDataSize - length);
      readRegions.push_back({offsetDistribution(rng), length});
    }
    return readRegions;
  };

  std::atomic<bool> stop{false};
  std::vector<uint64_t> completedReads(kThreadCount, 0);
  std::vector<uint64_t> completedRegions(kThreadCount, 0);
  std::vector<std::exception_ptr> exceptions(kThreadCount);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, this, i]() {
      try {
        std::mt19937_64 rng(0x123456789abcdef0ULL + i);
        auto& reader = ThreadLocalIoUringReader::get();
        while (!stop.load(std::memory_order_relaxed)) {
          const auto readRegions = makeRandomReadRegions(rng);
          readAndVerify(reader, fd.fd, readRegions);
          ++completedReads[i];
          completedRegions[i] += readRegions.size();
        }
      } catch (...) {
        exceptions[i] = std::current_exception();
        stop.store(true, std::memory_order_relaxed);
      }
    });
  }

  std::this_thread::sleep_for(kRunDuration);
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& exception : exceptions) {
    if (exception) {
      std::rethrow_exception(exception);
    }
  }

  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_GT(completedReads[i], 0) << "thread " << i;
    EXPECT_GT(completedRegions[i], 0) << "thread " << i;
  }
}

#endif

TEST(IoUringReaderUnavailableTest, threadLocalGetThrowsWhenUnavailable) {
  if (IoUringReader::available()) {
    GTEST_SKIP() << "io_uring is available";
  }

  VELOX_ASSERT_THROW(
      ThreadLocalIoUringReader::get(),
      "io_uring batch reads requested but io_uring is unavailable");
}

TEST(IoUringReaderOptionsTest, deferTaskRunRequiresSingleIssuer) {
  IoUringReader::Options options;
  options.singleIssuer = false;
  options.coopTaskRun = true;
  options.deferTaskRun = true;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring deferTaskRun option requires singleIssuer option");
}

TEST(IoUringReaderOptionsTest, invalidQueueDepth) {
  IoUringReader::Options options;
  options.submissionQueueDepth = 0;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring submissionQueueDepth must be positive");

  options.submissionQueueDepth = -1;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring submissionQueueDepth must be positive");

  options.submissionQueueDepth = IoUringReader::kDefaultQueueDepth;
  options.completionQueueDepth = 0;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring completionQueueDepth must be positive");

  options.completionQueueDepth = -1;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring completionQueueDepth must be positive");
}

TEST(IoUringReaderOptionsTest, invalidRegisteredFileSlots) {
  IoUringReader::Options options;

  options.registeredFileSlots = 0;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring registeredFileSlots must be positive");

  options.registeredFileSlots = -1;

  VELOX_ASSERT_THROW(
      IoUringReader::setOptions(options),
      "io_uring registeredFileSlots must be positive");
}

} // namespace
