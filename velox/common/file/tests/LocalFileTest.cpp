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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/io/IOBuf.h>
#include <folly/system/HardwareConcurrency.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TempFilePath.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::tests::utils;

namespace {

constexpr int kOneMB = 1 << 20;
constexpr size_t kContentCycleSize = 52;
constexpr size_t kNumTestPages = 1'024;
constexpr size_t kMinConcurrentFilePages = 256;
constexpr size_t kMaxConcurrentFilePages = 2'048;

char testPageContent(size_t pageIndex) {
  return static_cast<char>('a' + (pageIndex % kContentCycleSize));
}

std::string makeTestData(size_t numPages) {
  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  std::string data(numPages * kPageSize, '\0');
  for (size_t i = 0; i < numPages; ++i) {
    std::memset(data.data() + i * kPageSize, testPageContent(i), kPageSize);
  }
  return data;
}

void writeTestData(std::string_view path, std::string_view data) {
  LocalWriteFile writeFile(
      path,
      /*shouldCreateParentDirectories=*/false,
      /*shouldThrowOnFileAlreadyExists=*/false);
  writeFile.append(data);
  writeFile.close();
}

void writeData(WriteFile* writeFile, bool useIOBuf = false) {
  if (useIOBuf) {
    std::unique_ptr<folly::IOBuf> buf = folly::IOBuf::copyBuffer("aaaaa");
    buf->appendToChain(folly::IOBuf::copyBuffer("bbbbb"));
    buf->appendToChain(folly::IOBuf::copyBuffer(std::string(kOneMB, 'c')));
    buf->appendToChain(folly::IOBuf::copyBuffer("ddddd"));
    writeFile->append(std::move(buf));
    ASSERT_EQ(writeFile->size(), 15 + kOneMB);
  } else {
    writeFile->append("aaaaa");
    writeFile->append("bbbbb");
    writeFile->append(std::string(kOneMB, 'c'));
    writeFile->append("ddddd");
    ASSERT_EQ(writeFile->size(), 15 + kOneMB);
  }
}

void writeDataWithOffset(WriteFile* writeFile) {
  ASSERT_EQ(writeFile->size(), 0);
  writeFile->truncate(15 + kOneMB);
  std::vector<iovec> iovecs;
  std::string s1 = "aaaaa";
  std::string s2 = "bbbbb";
  std::string s3 = std::string(kOneMB, 'c');
  std::string s4 = "ddddd";
  iovecs.push_back({s3.data(), s3.length()});
  iovecs.push_back({s4.data(), s4.length()});
  writeFile->write(iovecs, 10, 5 + kOneMB);
  iovecs.clear();
  iovecs.push_back({s1.data(), s1.length()});
  iovecs.push_back({s2.data(), s2.length()});
  writeFile->write(iovecs, 0, 10);
  ASSERT_EQ(writeFile->size(), 15 + kOneMB);
}

void readData(
    ReadFile* readFile,
    bool checkFileSize = true,
    bool testReadAsync = false) {
  if (checkFileSize) {
    ASSERT_EQ(readFile->size(), 15 + kOneMB);
  }
  char buffer1[5];
  ASSERT_EQ(readFile->pread(10 + kOneMB, 5, &buffer1), "ddddd");
  char buffer2[10];
  ASSERT_EQ(readFile->pread(0, 10, &buffer2), "aaaaabbbbb");
  char buffer3[kOneMB];
  ASSERT_EQ(readFile->pread(10, kOneMB, &buffer3), std::string(kOneMB, 'c'));
  if (checkFileSize) {
    ASSERT_EQ(readFile->size(), 15 + kOneMB);
  }
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
  char head[12];
  char middle[4];
  char tail[7];
  std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(head, sizeof(head)),
      folly::Range<char*>(nullptr, (char*)(uint64_t)500000),
      folly::Range<char*>(middle, sizeof(middle)),
      folly::Range<char*>(
          nullptr,
          (char*)(uint64_t)(15 + kOneMB - 500000 - sizeof(head) -
                            sizeof(middle) - sizeof(tail))),
      folly::Range<char*>(tail, sizeof(tail))};
  ASSERT_EQ(15 + kOneMB, readFile->preadv(0, buffers));
  ASSERT_EQ(std::string_view(head, sizeof(head)), "aaaaabbbbbcc");
  ASSERT_EQ(std::string_view(middle, sizeof(middle)), "cccc");
  ASSERT_EQ(std::string_view(tail, sizeof(tail)), "ccddddd");
  if (testReadAsync) {
    std::vector<folly::Range<char*>> buffers1 = {
        folly::Range<char*>(head, sizeof(head)),
        folly::Range<char*>(nullptr, (char*)(uint64_t)500000)};
    auto future1 = readFile->preadvAsync(0, buffers1);
    const auto offset1 = sizeof(head) + 500000;
    std::vector<folly::Range<char*>> buffers2 = {
        folly::Range<char*>(middle, sizeof(middle)),
        folly::Range<char*>(
            nullptr,
            (char*)(uint64_t)(15 + kOneMB - offset1 - sizeof(middle) -
                              sizeof(tail)))};
    auto future2 = readFile->preadvAsync(offset1, buffers2);
    std::vector<folly::Range<char*>> buffers3 = {
        folly::Range<char*>(tail, sizeof(tail))};
    const auto offset2 = 15 + kOneMB - sizeof(tail);
    auto future3 = readFile->preadvAsync(offset2, buffers3);
    ASSERT_EQ(offset1, future1.wait().value());
    ASSERT_EQ(offset2 - offset1, future2.wait().value());
    ASSERT_EQ(sizeof(tail), future3.wait().value());
    ASSERT_EQ(std::string_view(head, sizeof(head)), "aaaaabbbbbcc");
    ASSERT_EQ(std::string_view(middle, sizeof(middle)), "cccc");
    ASSERT_EQ(std::string_view(tail, sizeof(tail)), "ccddddd");
  }
}

class LocalFileTest : public ::testing::TestWithParam<bool> {
 protected:
  LocalFileTest() : useFaultyFs_(GetParam()) {}

  static void SetUpTestCase() {
    filesystems::registerLocalFileSystem();
    tests::utils::registerFaultyFileSystem();
  }

  const bool useFaultyFs_;
  const std::unique_ptr<folly::CPUThreadPoolExecutor> executor_ =
      std::make_unique<folly::CPUThreadPoolExecutor>(
          std::max(1, static_cast<int32_t>(folly::available_concurrency() / 2)),
          std::make_shared<folly::NamedThreadFactory>(
              "LocalFileReadAheadTest"));
};

TEST_P(LocalFileTest, writeAndRead) {
  struct {
    bool useIOBuf;
    bool withOffset;

    std::string debugString() const {
      return fmt::format("useIOBuf {}, withOffset {}", useIOBuf, withOffset);
    }
  } testSettings[] = {{false, false}, {true, false}, {false, true}};
  for (auto testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    auto tempFile = TempFilePath::create(useFaultyFs_);
    const auto& filename = tempFile->getPath();
    auto fs = filesystems::getFileSystem(filename, {});
    fs->remove(filename);
    {
      auto writeFile = fs->openFileForWrite(filename);
      if (testData.withOffset) {
        writeDataWithOffset(writeFile.get());
      } else {
        writeData(writeFile.get(), testData.useIOBuf);
      }
      writeFile->close();
      ASSERT_EQ(writeFile->size(), 15 + kOneMB);
    }
    // Test read async.
    if (!useFaultyFs_) {
      auto readFile =
          std::make_shared<LocalReadFile>(filename, executor_.get());
      readData(readFile.get(), true, true);
      auto readFileWithoutExecutor = std::make_shared<LocalReadFile>(filename);
      readData(readFileWithoutExecutor.get(), true, true);
    }
    auto readFile = fs->openFileForRead(filename);
    readData(readFile.get());
  }
}

TEST_P(LocalFileTest, viaRegistry) {
  auto tempFile = TempFilePath::create(useFaultyFs_);
  const auto& filename = tempFile->getPath();
  auto fs = filesystems::getFileSystem(filename, {});
  fs->remove(filename);
  {
    auto writeFile = fs->openFileForWrite(filename);
    writeFile->append("snarf");
  }
  auto readFile = fs->openFileForRead(filename);
  ASSERT_EQ(readFile->size(), 5);
  char buffer1[5];
  ASSERT_EQ(readFile->pread(0, 5, &buffer1), "snarf");
  fs->remove(filename);
}

TEST_P(LocalFileTest, rename) {
  const auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  const auto a = fmt::format("{}/a", tempFolder->getPath());
  const auto b = fmt::format("{}/b", tempFolder->getPath());
  const auto newA = fmt::format("{}/newA", tempFolder->getPath());
  const std::string data("aaaaa");
  auto localFs = filesystems::getFileSystem(a, nullptr);
  {
    auto writeFile = localFs->openFileForWrite(a);
    writeFile = localFs->openFileForWrite(b);
    writeFile->append(data);
    writeFile->close();
  }
  ASSERT_TRUE(localFs->exists(a));
  ASSERT_TRUE(localFs->exists(b));
  ASSERT_FALSE(localFs->exists(newA));
  VELOX_ASSERT_USER_THROW(localFs->rename(a, b), "");
  localFs->rename(a, newA);
  ASSERT_FALSE(localFs->exists(a));
  ASSERT_TRUE(localFs->exists(b));
  ASSERT_TRUE(localFs->exists(newA));
  localFs->rename(b, newA, true);
  auto readFile = localFs->openFileForRead(newA);
  char buffer[5];
  ASSERT_EQ(readFile->pread(0, 5, &buffer), data);
}

TEST_P(LocalFileTest, exists) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  auto a = fmt::format("{}/a", tempFolder->getPath());
  auto b = fmt::format("{}/b", tempFolder->getPath());
  auto localFs = filesystems::getFileSystem(a, nullptr);
  {
    auto writeFile = localFs->openFileForWrite(a);
    writeFile = localFs->openFileForWrite(b);
  }
  ASSERT_TRUE(localFs->exists(a));
  ASSERT_TRUE(localFs->exists(b));
  localFs->remove(a);
  ASSERT_FALSE(localFs->exists(a));
  ASSERT_TRUE(localFs->exists(b));
  localFs->remove(b);
  ASSERT_FALSE(localFs->exists(a));
  ASSERT_FALSE(localFs->exists(b));
}

TEST_P(LocalFileTest, isDirectory) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  auto a = fmt::format("{}/a", tempFolder->getPath());
  auto localFs = filesystems::getFileSystem(a, nullptr);
  auto writeFile = localFs->openFileForWrite(a);
  ASSERT_TRUE(localFs->isDirectory(tempFolder->getPath()));
  ASSERT_FALSE(localFs->isDirectory(a));
}

TEST_P(LocalFileTest, list) {
  const auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  const auto a = fmt::format("{}/1", tempFolder->getPath());
  const auto b = fmt::format("{}/2", tempFolder->getPath());
  auto localFs = filesystems::getFileSystem(a, nullptr);
  {
    auto writeFile = localFs->openFileForWrite(a);
    writeFile = localFs->openFileForWrite(b);
  }
  auto files = localFs->list(std::string_view(tempFolder->getPath()));
  std::sort(files.begin(), files.end());
  ASSERT_EQ(files, std::vector<std::string>({a, b}));
  localFs->remove(a);
  ASSERT_EQ(
      localFs->list(std::string_view(tempFolder->getPath())),
      std::vector<std::string>({b}));
  localFs->remove(b);
  ASSERT_TRUE(localFs->list(std::string_view(tempFolder->getPath())).empty());
}

TEST_P(LocalFileTest, readFileDestructor) {
  if (useFaultyFs_) {
    return;
  }
  auto tempFile = TempFilePath::create(useFaultyFs_);
  const auto& filename = tempFile->getPath();
  auto fs = filesystems::getFileSystem(filename, {});
  fs->remove(filename);
  {
    auto writeFile = fs->openFileForWrite(filename);
    writeData(writeFile.get());
  }

  {
    auto readFile = fs->openFileForRead(filename);
    readData(readFile.get());
  }

  {
    LocalReadFile readFile(filename);
    readData(&readFile, false);
  }
  {
    LocalReadFile readFile(filename);
    readData(&readFile, false);
  }
}

TEST_P(LocalFileTest, mkdirFailIfPresent) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  std::string path = tempFolder->getPath();
  auto localFs = filesystems::getFileSystem(path, nullptr);

  path += "/level1/level2/level3";
  EXPECT_FALSE(localFs->exists(path));
  EXPECT_NO_THROW(localFs->mkdir(path));
  EXPECT_TRUE(localFs->exists(path));

  // Except that if we try to make the directory again,
  // it will not fail.
  EXPECT_NO_THROW(localFs->mkdir(path));

  // We fail if the directory is already present
  DirectoryOptions options;
  options.failIfExists = true;
  VELOX_ASSERT_THROW(
      localFs->mkdir(path, options),
      fmt::format("Directory: {} already exists", localFs->extractPath(path)));
}

TEST_P(LocalFileTest, mkdir) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);

  std::string path = tempFolder->getPath();
  auto localFs = filesystems::getFileSystem(path, nullptr);

  // Create 3 levels of directories and ensure they exist.
  path += "/level1/level2/level3";
  EXPECT_NO_THROW(localFs->mkdir(path));
  EXPECT_TRUE(localFs->exists(path));

  // Create a completely existing directory - we should not throw.
  EXPECT_NO_THROW(localFs->mkdir(path));

  // Write a file to our directory to double check it exist.
  path += "/a.txt";
  const std::string data("aaaaa");
  {
    auto writeFile = localFs->openFileForWrite(path);
    writeFile->append(data);
    writeFile->close();
  }
  EXPECT_TRUE(localFs->exists(path));
}

TEST_P(LocalFileTest, rmdir) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);

  std::string path = tempFolder->getPath();
  auto localFs = filesystems::getFileSystem(path, nullptr);

  // Create 3 levels of directories and ensure they exist.
  path += "/level1/level2/level3";
  EXPECT_NO_THROW(localFs->mkdir(path));
  EXPECT_TRUE(localFs->exists(path));

  // Write a file to our directory to double check it exist.
  path += "/a.txt";
  const std::string data("aaaaa");
  {
    auto writeFile = localFs->openFileForWrite(path);
    writeFile->append(data);
    writeFile->close();
  }
  EXPECT_TRUE(localFs->exists(path));

  // Now delete the whole temp folder and ensure it is gone.
  EXPECT_NO_THROW(localFs->rmdir(tempFolder->getPath()));
  EXPECT_FALSE(localFs->exists(tempFolder->getPath()));

  // Delete a non-existing directory.
  path += "/does_not_exist/subdir";
  EXPECT_FALSE(localFs->exists(path));
  // The function does not throw, but will return zero files and folders
  // deleted, which is not an error.
  EXPECT_NO_THROW(localFs->rmdir(tempFolder->getPath()));
}

TEST_P(LocalFileTest, fileNotFound) {
  auto tempFolder = TempDirectoryPath::create(useFaultyFs_);
  auto path = fmt::format("{}/file", tempFolder->getPath());
  auto localFs = filesystems::getFileSystem(path, nullptr);
  VELOX_ASSERT_RUNTIME_THROW_CODE(
      localFs->openFileForRead(path),
      error_code::kFileNotFound,
      "No such file or directory");
}

TEST_P(LocalFileTest, attributes) {
  auto tempFile = TempFilePath::create(useFaultyFs_);
  const auto& filename = tempFile->getPath();
  auto fs = filesystems::getFileSystem(filename, {});
  fs->remove(filename);
  auto writeFile = fs->openFileForWrite(filename);
  ASSERT_FALSE(
      LocalWriteFile::Attributes::cowDisabled(writeFile->getAttributes()));
  try {
    writeFile->setAttributes(
        {{std::string(LocalWriteFile::Attributes::kNoCow), "true"}});
  } catch (const std::exception& /*e*/) {
    // Flags like FS_IOC_SETFLAGS might not be supported for certain
    // file systems (e.g., EXT4, XFS).
  }
  ASSERT_TRUE(
      LocalWriteFile::Attributes::cowDisabled(writeFile->getAttributes()));
  writeFile->close();
}

TEST_P(LocalFileTest, directIoAlignment) {
  auto tempFile = TempFilePath::create(useFaultyFs_);
  const auto& filename = tempFile->getPath();
  auto fs = filesystems::getFileSystem(filename, {});
  fs->remove(filename);
  {
    auto writeFile = fs->openFileForWrite(filename);
    writeFile->append("data");
    writeFile->close();
  }

  {
    auto file = fs->openFileForRead(filename);
    uint64_t alignment{17};
    EXPECT_FALSE(file->directIo(alignment));
    EXPECT_EQ(alignment, 1);
  }

  filesystems::FileOptions options;
  options.bufferIo = false;
  auto file = fs->openFileForRead(filename, options);
  uint64_t alignment{0};
  EXPECT_TRUE(file->directIo(alignment));
  EXPECT_GT(alignment, 0);
  EXPECT_EQ(alignment & (alignment - 1), 0);
}

INSTANTIATE_TEST_SUITE_P(
    LocalFileTestSuite,
    LocalFileTest,
    ::testing::Values(false, true),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "FaultyFileSystem" : "LocalFileSystem";
    });

class PageAlignedBuffer {
 public:
  explicit PageAlignedBuffer(size_t size) {
    constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
    size_ = (size + kPageSize - 1) & ~(kPageSize - 1);
    data_ = static_cast<char*>(::aligned_alloc(kPageSize, size_));
    VELOX_CHECK_NOT_NULL(data_, "aligned_alloc failed");
  }

  ~PageAlignedBuffer() {
    ::free(data_);
  }

  char* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  char* data_;
  size_t size_;
};

template <typename Ranges>
folly::Range<const folly::Range<char*>*> asConstRange(Ranges& ranges) {
  return {ranges.data(), ranges.size()};
}

struct IoUringReaderStatsSnapshot {
  uint64_t numReaders{0};
  IoUringReader::Stats stats;
};

IoUringReaderStatsSnapshot ioUringReaderStats() {
  IoUringReaderStatsSnapshot snapshot;
  snapshot.stats = getIoUringReaderStats(snapshot.numReaders);
  return snapshot;
}

void expectIoUringReaderStatsDelta(
    const IoUringReaderStatsSnapshot& prev,
    uint64_t expectedReadCalls,
    uint64_t expectedRegions,
    uint64_t expectedBatches,
    uint64_t expectedNumReaders) {
  const auto after = ioUringReaderStats();
  EXPECT_EQ(prev.numReaders, 0);
  EXPECT_EQ(after.numReaders, expectedNumReaders);
  EXPECT_EQ(after.stats.readCalls, prev.stats.readCalls + expectedReadCalls);
  EXPECT_EQ(after.stats.regions, prev.stats.regions + expectedRegions);
  EXPECT_EQ(after.stats.batches, prev.stats.batches + expectedBatches);
}

void expectIoUringReaderStats(
    const IoUringReaderStatsSnapshot& snapshot,
    uint64_t expectedNumReaders,
    uint64_t expectedReadCalls,
    uint64_t expectedRegions,
    uint64_t expectedBatches) {
  EXPECT_EQ(snapshot.numReaders, expectedNumReaders);
  EXPECT_EQ(snapshot.stats.readCalls, expectedReadCalls);
  EXPECT_EQ(snapshot.stats.regions, expectedRegions);
  EXPECT_EQ(snapshot.stats.batches, expectedBatches);
}

class LocalFileIoUringTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    filesystems::registerLocalFileSystem();
  }

  void SetUp() override {
    ThreadLocalIoUringReader::testingClear();

    IoUringReader::Options options;
    options.submissionQueueDepth = 8;
    options.completionQueueDepth = 8;
    options.registeredFileSlots = 8;
    IoUringReader::setOptions(options);

    tempFile_ = TempFilePath::create();
    writeTestData(tempFile_->getPath(), makeTestData(kNumTestPages));
  }

  void TearDown() override {
    ThreadLocalIoUringReader::testingClear();
    IoUringReader::setOptions(IoUringReader::Options{});
  }

  std::shared_ptr<TempFilePath> tempFile_;
};

struct ConcurrentFile {
  std::string path;
  size_t numPages;
};

struct ConcurrentReaderStats {
  uint64_t readCalls{0};
  uint64_t regions{0};
  uint64_t minRegionsPerRead{std::numeric_limits<uint64_t>::max()};
  uint64_t maxRegionsPerRead{0};

  void recordRead(size_t regionCount) {
    ++readCalls;
    regions += regionCount;
    minRegionsPerRead = std::min<uint64_t>(minRegionsPerRead, regionCount);
    maxRegionsPerRead = std::max<uint64_t>(maxRegionsPerRead, regionCount);
  }

  void merge(const ConcurrentReaderStats& other) {
    if (other.readCalls == 0) {
      return;
    }
    readCalls += other.readCalls;
    regions += other.regions;
    minRegionsPerRead = std::min(minRegionsPerRead, other.minRegionsPerRead);
    maxRegionsPerRead = std::max(maxRegionsPerRead, other.maxRegionsPerRead);
  }
};

size_t randomSize(std::mt19937_64& rng, size_t min, size_t max) {
  return std::uniform_int_distribution<size_t>(min, max)(rng);
}

size_t randomConcurrentFilePages(std::mt19937_64& rng) {
  return randomSize(rng, kMinConcurrentFilePages, kMaxConcurrentFilePages);
}

ConcurrentFile createConcurrentFile(
    std::vector<std::shared_ptr<TempFilePath>>& tempFiles,
    size_t numPages) {
  auto tempFile = TempFilePath::create();
  auto path = tempFile->getPath();
  writeTestData(path, makeTestData(numPages));
  tempFiles.push_back(std::move(tempFile));
  return ConcurrentFile{std::move(path), numPages};
}

void expectPageContent(const char* data, size_t pageIndex) {
  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  const std::string expected(kPageSize, testPageContent(pageIndex));
  VELOX_CHECK_EQ(std::memcmp(data, expected.data(), kPageSize), 0);
}

void runConcurrentPreadvTest(
    const std::vector<ConcurrentFile>& files,
    size_t numReaders) {
  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  constexpr size_t kMaxRegionsPerRead = 8;
  constexpr size_t kMaxPagesPerRegion = 4;
  constexpr auto kRunDuration = std::chrono::seconds(10);
  VELOX_CHECK(!files.empty());
  VELOX_CHECK_GT(numReaders, 0);
  for (const auto& file : files) {
    VELOX_CHECK_GT(file.numPages, 0);
  }

  ThreadLocalIoUringReader::testingClear();
  EXPECT_EQ(ioUringReaderStats().numReaders, 0);

  std::mutex mutex;
  std::condition_variable condition;
  std::exception_ptr error;
  size_t completed{0};
  bool statsChecked{false};
  std::atomic_bool stop{false};
  std::vector<ConcurrentReaderStats> readerStats(numReaders);

  std::vector<std::thread> threads;
  threads.reserve(numReaders);
  for (size_t readerIndex = 0; readerIndex < numReaders; ++readerIndex) {
    threads.emplace_back([&, readerIndex]() {
      try {
        ThreadLocalIoUringReader::testingClear();

        std::mt19937_64 rng(
            0x5eed0000ULL + readerIndex * 0x9e3779b97f4a7c15ULL);
        std::vector<std::unique_ptr<LocalReadFile>> readFiles;
        readFiles.reserve(files.size());
        for (const auto& file : files) {
          readFiles.push_back(
              std::make_unique<LocalReadFile>(
                  file.path,
                  /*executor=*/nullptr,
                  /*bufferIo=*/false,
                  /*useIoUring=*/true));
        }

        ConcurrentReaderStats localStats;
        while (!stop.load(std::memory_order_relaxed)) {
          const auto fileIndex = randomSize(rng, 0, files.size() - 1);
          const auto& file = files[fileIndex];
          const auto regionCount = randomSize(rng, 1, kMaxRegionsPerRead);
          std::vector<common::Region> regions;
          std::vector<std::unique_ptr<PageAlignedBuffer>> storage;
          std::vector<folly::Range<char*>> buffers;
          regions.reserve(regionCount);
          storage.reserve(regionCount);
          buffers.reserve(regionCount);
          for (size_t i = 0; i < regionCount; ++i) {
            const auto pageIndex = randomSize(rng, 0, file.numPages - 1);
            const auto maxPageCount =
                std::min(kMaxPagesPerRegion, file.numPages - pageIndex);
            const auto pageCount = randomSize(rng, 1, maxPageCount);
            const auto length = pageCount * kPageSize;
            regions.push_back(common::Region(pageIndex * kPageSize, length));
            storage.push_back(std::make_unique<PageAlignedBuffer>(length));
            buffers.push_back(
                folly::Range<char*>(storage.back()->data(), length));
          }

          const auto bytesRead = readFiles[fileIndex]->preadv(
              folly::Range<const common::Region*>(
                  regions.data(), regions.size()),
              asConstRange(buffers));

          uint64_t expectedBytes{0};
          for (size_t i = 0; i < regionCount; ++i) {
            const auto pageIndex = regions[i].offset / kPageSize;
            const auto pageCount = regions[i].length / kPageSize;
            expectedBytes += regions[i].length;
            for (size_t page = 0; page < pageCount; ++page) {
              expectPageContent(
                  buffers[i].data() + page * kPageSize, pageIndex + page);
            }
          }
          VELOX_CHECK_EQ(bytesRead, expectedBytes);
          localStats.recordRead(regionCount);
        }

        VELOX_CHECK_GT(localStats.readCalls, 0);
        readerStats[readerIndex] = localStats;
      } catch (...) {
        {
          std::lock_guard<std::mutex> l(mutex);
          if (error == nullptr) {
            error = std::current_exception();
          }
        }
        stop.store(true, std::memory_order_relaxed);
      }

      {
        std::lock_guard<std::mutex> l(mutex);
        ++completed;
      }
      condition.notify_all();
      {
        std::unique_lock<std::mutex> l(mutex);
        condition.wait(l, [&]() { return statsChecked; });
      }
      ThreadLocalIoUringReader::testingClear();
    });
  }

  std::this_thread::sleep_for(kRunDuration);
  stop.store(true, std::memory_order_relaxed);
  {
    std::unique_lock<std::mutex> l(mutex);
    condition.wait(l, [&]() { return completed == numReaders; });
  }

  std::string errorMessage;
  if (error != nullptr) {
    try {
      std::rethrow_exception(error);
    } catch (const std::exception& e) {
      errorMessage = e.what();
    } catch (...) {
      errorMessage = "unknown exception";
    }
  }

  if (errorMessage.empty()) {
    ConcurrentReaderStats expectedStats;
    for (const auto& stats : readerStats) {
      expectedStats.merge(stats);
    }
    const auto snapshot = ioUringReaderStats();
    expectIoUringReaderStats(
        snapshot,
        /*expectedNumReaders=*/numReaders,
        /*expectedReadCalls=*/expectedStats.readCalls,
        /*expectedRegions=*/expectedStats.regions,
        /*expectedBatches=*/expectedStats.readCalls);
    EXPECT_EQ(
        snapshot.stats.minRegionsPerRead, expectedStats.minRegionsPerRead);
    EXPECT_EQ(
        snapshot.stats.maxRegionsPerRead, expectedStats.maxRegionsPerRead);
    EXPECT_EQ(snapshot.stats.minBatchSize, expectedStats.minRegionsPerRead);
    EXPECT_EQ(snapshot.stats.maxBatchSize, expectedStats.maxRegionsPerRead);
  }

  {
    std::lock_guard<std::mutex> l(mutex);
    statsChecked = true;
  }
  condition.notify_all();
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(ioUringReaderStats().numReaders, 0);
  if (!errorMessage.empty()) {
    FAIL() << "Concurrent preadv reader failed: " << errorMessage;
  }
}

TEST_F(LocalFileIoUringTest, preadv) {
  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  const std::string expectedPage2(kPageSize, testPageContent(2));
  const std::string expectedPage1(kPageSize, testPageContent(1));

  struct {
    const char* name;
    bool bufferIo;
    bool useIoUring;
    bool expectedDirectIo;
    uint64_t expectedReadCalls;
    uint64_t expectedRegions;
    uint64_t expectedBatches;
    uint64_t expectedNumReaders;
  } testCases[] = {
      {
          "directIoWithIoUring",
          /*bufferIo=*/false,
          /*useIoUring=*/true,
          /*expectedDirectIo=*/true,
          /*expectedReadCalls=*/1,
          /*expectedRegions=*/2,
          /*expectedBatches=*/1,
          /*expectedNumReaders=*/1,
      },
      {
          "directIoWithoutIoUring",
          /*bufferIo=*/false,
          /*useIoUring=*/false,
          /*expectedDirectIo=*/true,
          /*expectedReadCalls=*/0,
          /*expectedRegions=*/0,
          /*expectedBatches=*/0,
          /*expectedNumReaders=*/0,
      },
      {
          "bufferedFile",
          /*bufferIo=*/true,
          /*useIoUring=*/false,
          /*expectedDirectIo=*/false,
          /*expectedReadCalls=*/0,
          /*expectedRegions=*/0,
          /*expectedBatches=*/0,
          /*expectedNumReaders=*/0,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    ThreadLocalIoUringReader::testingClear();

    LocalReadFile file(
        tempFile_->getPath(),
        /*executor=*/nullptr,
        testCase.bufferIo,
        testCase.useIoUring);
    uint64_t alignment{0};
    EXPECT_EQ(file.directIo(alignment), testCase.expectedDirectIo);
    if (testCase.expectedDirectIo) {
      EXPECT_GT(alignment, 0);
      EXPECT_EQ(alignment & (alignment - 1), 0);
    } else {
      EXPECT_EQ(alignment, 1);
    }

    const auto statsBefore = ioUringReaderStats();
    PageAlignedBuffer first(kPageSize);
    PageAlignedBuffer second(kPageSize);
    std::vector<common::Region> regions = {
        common::Region(2 * kPageSize, kPageSize),
        common::Region(kPageSize, kPageSize),
    };
    std::vector<folly::Range<char*>> buffers = {
        folly::Range<char*>(first.data(), kPageSize),
        folly::Range<char*>(second.data(), kPageSize),
    };

    const auto bytesRead = file.preadv(
        folly::Range<const common::Region*>(regions.data(), regions.size()),
        asConstRange(buffers));

    EXPECT_EQ(bytesRead, 2 * kPageSize);
    EXPECT_EQ(std::memcmp(first.data(), expectedPage2.data(), kPageSize), 0);
    EXPECT_EQ(std::memcmp(second.data(), expectedPage1.data(), kPageSize), 0);
    expectIoUringReaderStatsDelta(
        statsBefore,
        testCase.expectedReadCalls,
        testCase.expectedRegions,
        testCase.expectedBatches,
        testCase.expectedNumReaders);
  }
}

TEST_F(LocalFileIoUringTest, concurrentPreadv) {
  constexpr size_t kNumReaders = 8;

  enum class Mode {
    kSingleFile,
    kMultipleFiles,
  };

  struct {
    const char* name;
    Mode mode;
  } testCases[] = {
      {"singleFile", Mode::kSingleFile},
      {"multipleFiles", Mode::kMultipleFiles},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    std::mt19937_64 rng(
        testCase.mode == Mode::kSingleFile ? 0x51a91eULL : 0x819e5ULL);
    std::vector<std::shared_ptr<TempFilePath>> tempFiles;
    std::vector<ConcurrentFile> files;
    tempFiles.reserve(kNumReaders);
    files.reserve(kNumReaders);

    if (testCase.mode == Mode::kSingleFile) {
      files.push_back(
          createConcurrentFile(tempFiles, randomConcurrentFilePages(rng)));
    } else {
      const auto numFiles = randomSize(rng, 2, kNumReaders);
      for (size_t i = 0; i < numFiles; ++i) {
        files.push_back(
            createConcurrentFile(tempFiles, randomConcurrentFilePages(rng)));
      }
    }

    runConcurrentPreadvTest(files, kNumReaders);
  }
}

TEST_F(LocalFileIoUringTest, useIoUringWithBufferedFileThrows) {
  VELOX_ASSERT_THROW(
      LocalReadFile(
          tempFile_->getPath(),
          /*executor=*/nullptr,
          /*bufferIo=*/true,
          /*useIoUring=*/true),
      "LocalReadFile useIoUring requested but direct IO is disabled");

  filesystems::FileOptions options;
  options.useIoUring = true;
  VELOX_ASSERT_THROW(
      filesystems::getFileSystem(tempFile_->getPath(), nullptr)
          ->openFileForRead(tempFile_->getPath(), options),
      "LocalReadFile useIoUring requested but direct IO is disabled");
}

TEST_F(LocalFileIoUringTest, fileOptionsUseIoUring) {
  ThreadLocalIoUringReader::testingClear();

  filesystems::FileOptions options;
  options.bufferIo = false;
  options.useIoUring = true;
  auto file = filesystems::getFileSystem(tempFile_->getPath(), nullptr)
                  ->openFileForRead(tempFile_->getPath(), options);

  uint64_t alignment{0};
  EXPECT_TRUE(file->directIo(alignment));
  EXPECT_GT(alignment, 0);

  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  PageAlignedBuffer first(kPageSize);
  PageAlignedBuffer second(kPageSize);
  std::vector<common::Region> regions = {
      common::Region(0, kPageSize),
      common::Region(kPageSize, kPageSize),
  };
  std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(first.data(), kPageSize),
      folly::Range<char*>(second.data(), kPageSize),
  };

  const auto statsBefore = ioUringReaderStats();
  const auto bytesRead = file->preadv(
      folly::Range<const common::Region*>(regions.data(), regions.size()),
      asConstRange(buffers));

  const std::string expectedPage0(kPageSize, testPageContent(0));
  const std::string expectedPage1(kPageSize, testPageContent(1));
  EXPECT_EQ(bytesRead, 2 * kPageSize);
  EXPECT_EQ(std::memcmp(first.data(), expectedPage0.data(), kPageSize), 0);
  EXPECT_EQ(std::memcmp(second.data(), expectedPage1.data(), kPageSize), 0);
  expectIoUringReaderStatsDelta(
      statsBefore,
      /*expectedReadCalls=*/1,
      /*expectedRegions=*/2,
      /*expectedBatches=*/1,
      /*expectedNumReaders=*/1);

  ThreadLocalIoUringReader::testingClear();
  EXPECT_EQ(ioUringReaderStats().numReaders, 0);
}

TEST_F(LocalFileIoUringTest, preadAndOffsetPreadvDoNotUseIoUring) {
  LocalReadFile file(
      tempFile_->getPath(),
      /*executor=*/nullptr,
      /*bufferIo=*/false,
      /*useIoUring=*/true);
  uint64_t alignment{0};
  EXPECT_TRUE(file.directIo(/*alignment=*/alignment));

  constexpr size_t kPageSize = memory::AllocationTraits::kPageSize;
  const std::string expectedPage0(kPageSize, testPageContent(0));
  const std::string expectedPage1(kPageSize, testPageContent(1));
  const std::string expectedPage2(kPageSize, testPageContent(2));

  {
    const auto statsBefore = ioUringReaderStats();
    PageAlignedBuffer buffer(kPageSize);
    const auto bytesRead = file.pread(0, kPageSize, buffer.data());
    EXPECT_EQ(bytesRead.size(), kPageSize);
    EXPECT_EQ(std::memcmp(buffer.data(), expectedPage0.data(), kPageSize), 0);
    expectIoUringReaderStatsDelta(
        statsBefore,
        /*expectedReadCalls=*/0,
        /*expectedRegions=*/0,
        /*expectedBatches=*/0,
        /*expectedNumReaders=*/0);
  }

  {
    const auto statsBefore = ioUringReaderStats();
    PageAlignedBuffer first(kPageSize);
    PageAlignedBuffer second(kPageSize);
    std::vector<folly::Range<char*>> buffers = {
        folly::Range<char*>(first.data(), kPageSize),
        folly::Range<char*>(second.data(), kPageSize),
    };

    const auto bytesRead = file.preadv(kPageSize, buffers);
    EXPECT_EQ(bytesRead, 2 * kPageSize);
    EXPECT_EQ(std::memcmp(first.data(), expectedPage1.data(), kPageSize), 0);
    EXPECT_EQ(std::memcmp(second.data(), expectedPage2.data(), kPageSize), 0);
    expectIoUringReaderStatsDelta(
        statsBefore,
        /*expectedReadCalls=*/0,
        /*expectedRegions=*/0,
        /*expectedBatches=*/0,
        /*expectedNumReaders=*/0);
  }
}

} // namespace
