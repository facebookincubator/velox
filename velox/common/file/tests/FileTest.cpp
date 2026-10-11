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

#include "velox/common/file/File.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/testutil/TempDirectoryPath.h"

#include "gtest/gtest.h"

#ifdef _WIN32
#include <folly/portability/Windows.h>
#include <winioctl.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#endif

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using facebook::velox::common::Region;
using namespace facebook::velox::tests::utils;

constexpr int kOneMB = 1 << 20;

#ifdef _WIN32
TEST(LocalFileWindowsTest, binaryAndAppend) {
  const auto dir = TempDirectoryPath::create();
  const auto folder = std::filesystem::path(dir->getPath()) / "nested";
  const auto path = (folder / "data").string();
  std::string data{'\0', '\r', '\n', '\x1a', '\xff'};
  LocalWriteFile file(path, true);
  file.append(data);
  auto chain = folly::IOBuf::copyBuffer(data);
  chain->appendToChain(folly::IOBuf::copyBuffer(data));
  file.append(std::move(chain));
  std::string expected = data + data + data;
  std::string replacement = "xy";
  file.write({{replacement.data(), replacement.size()}}, 1, 2);
  expected.replace(1, 2, replacement);
  file.append("end");
  expected += "end";
  file.write(
      {{replacement.data(), replacement.size()}}, expected.size() + 2, 2);
  expected.append(2, '\0');
  expected += replacement;
  file.append("extended");
  expected += "extended";
  file.flush();
  EXPECT_EQ(file.size(), expected.size());
  file.close();

  std::ifstream input(path, std::ios::binary);
  const std::string diskData(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  EXPECT_EQ(diskData, expected);
  input.close();

  filesystems::registerLocalFileSystem();
  const auto fs = filesystems::getFileSystem(path, nullptr);
  EXPECT_EQ(fs->list(folder.string()), std::vector<std::string>{path});
  const auto read = fs->openFileForRead(path);
  EXPECT_EQ(read->pread(0, expected.size()), expected);
  uint64_t alignment = 0;
  EXPECT_FALSE(read->directIo(alignment));
  EXPECT_EQ(alignment, 1);
}

TEST(LocalFileWindowsTest, largeSparseOffsets) {
  const auto dir = TempDirectoryPath::create();
  const auto path = std::filesystem::path(dir->getPath()) / "sparse";
  LocalWriteFile file(path.string());
  const auto handle = ::CreateFileW(
      path.c_str(),
      GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      nullptr,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  DWORD returned = 0;
  const auto sparse = ::DeviceIoControl(
      handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
  const auto error = ::GetLastError();
  ::CloseHandle(handle);
  ASSERT_TRUE(sparse) << error;

  constexpr int64_t kOffset = (int64_t{1} << 32) + 19;
  std::string data = "tail\r\n\x1a";
  file.truncate(kOffset + 32);
  file.write({{data.data(), data.size()}}, kOffset, data.size());
  file.truncate(kOffset + data.size());
  file.append("end");
  file.close();
  data += "end";
  {
    LocalReadFile read(path.string());
    EXPECT_EQ(read.size(), kOffset + data.size());
    std::string buffer(data.size(), '\0');
    EXPECT_EQ(read.pread(kOffset, buffer.size(), buffer.data()), data);
    buffer.assign(data.size(), '\0');
    EXPECT_EQ(
        read.preadv(
            kOffset,
            {{buffer.data(), 2}, {buffer.data() + 2, buffer.size() - 2}}),
        data.size());
    EXPECT_EQ(buffer, data);
    char zero = 'x';
    EXPECT_EQ(read.pread(kOffset - 1, 1, &zero), std::string_view("\0", 1));
  }
  {
    LocalWriteFile reopened(path.string(), false, false);
    EXPECT_EQ(reopened.size(), kOffset + data.size());
    reopened.append("open");
    EXPECT_EQ(reopened.size(), kOffset + data.size() + 4);
    reopened.truncate(kOffset + 1);
    reopened.append("after");
    EXPECT_EQ(reopened.size(), kOffset + 6);
  }
  LocalReadFile read(path.string());
  char buffer[6];
  EXPECT_EQ(read.pread(kOffset, sizeof(buffer), buffer), "tafter");
}

TEST(LocalFileWindowsTest, concurrentScalarAndVectorReads) {
  const auto dir = TempDirectoryPath::create();
  const auto path =
      (std::filesystem::path(dir->getPath()) / "concurrent").string();
  {
    LocalWriteFile file(path);
    for (int i = 0; i < 32; ++i) {
      file.append(std::string(256, 'a' + i));
    }
  }
  LocalReadFile read(path);
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&, thread] {
      try {
        for (int i = 0; i < 256; ++i) {
          const auto record = (i + thread * 7) % 32;
          const auto offset = record * 256;
          std::string buffer(128, '\0');
          if ((i + thread) % 2 == 0) {
            read.pread(offset, buffer.size(), buffer.data());
          } else {
            const auto bytes = read.preadv(
                offset,
                {{buffer.data(), 64}, {nullptr, 32}, {buffer.data() + 64, 64}});
            if (bytes != 160) {
              ++errors;
            }
          }
          if (buffer != std::string(128, 'a' + record)) {
            ++errors;
          }
        }
      } catch (const std::exception&) {
        ++errors;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(errors, 0);
}

TEST(LocalFileWindowsTest, failuresAndShortReads) {
  const auto dir = TempDirectoryPath::create();
  const auto path = (std::filesystem::path(dir->getPath()) / "data").string();
  EXPECT_THROW((LocalReadFile{path}), VeloxRuntimeError);
  EXPECT_THROW((LocalReadFile{path, nullptr, false}), VeloxRuntimeError);
  EXPECT_THROW((LocalWriteFile{path, false, true, false}), VeloxRuntimeError);
  EXPECT_FALSE(std::filesystem::exists(path));
  LocalWriteFile file(path);
  EXPECT_THROW((LocalWriteFile{path}), VeloxRuntimeError);
  file.append("abcd");
  EXPECT_THROW(file.truncate(-1), VeloxRuntimeError);
  std::string data = "ef";
  const std::vector<iovec> iovecs{{data.data(), data.size()}};
  EXPECT_THROW(file.write(iovecs, -1, 2), VeloxRuntimeError);
  EXPECT_THROW(file.write(iovecs, 0, -1), VeloxRuntimeError);
  EXPECT_THROW(
      file.write(iovecs, std::numeric_limits<int64_t>::max(), 2),
      VeloxRuntimeError);
  EXPECT_THROW(file.write({{nullptr, 1}}, 0, 1), VeloxRuntimeError);
  EXPECT_THROW(
      file.write({{data.data(), data.size()}, {nullptr, 1}}, 0, 3),
      VeloxRuntimeError);
  EXPECT_NO_THROW(file.write({{nullptr, 0}}, 0, 0));
  file.close();
  EXPECT_THROW(file.append("closed"), VeloxRuntimeError);
  EXPECT_THROW(file.write(iovecs, 0, 2), VeloxRuntimeError);
  EXPECT_THROW(file.truncate(0), VeloxRuntimeError);
  EXPECT_THROW(file.flush(), VeloxRuntimeError);
  file.close();

  LocalReadFile read(path);
  char buffer[4]{};
  EXPECT_THROW(read.pread(0, 1, nullptr), VeloxRuntimeError);
  EXPECT_TRUE(read.pread(0, 0, nullptr).empty());
  EXPECT_THROW(read.pread(3, 2, buffer), VeloxRuntimeError);
  EXPECT_THROW(
      read.pread(std::numeric_limits<uint64_t>::max(), 1, buffer),
      VeloxRuntimeError);
  EXPECT_THROW(
      read.pread(0, std::numeric_limits<uint64_t>::max(), buffer),
      VeloxRuntimeError);
  std::fill(std::begin(buffer), std::end(buffer), '?');
  EXPECT_EQ(read.preadv(3, {{buffer, 2}, {buffer + 2, 2}}), 1);
  EXPECT_EQ(std::string_view(buffer, 4), "d???");
  EXPECT_EQ(read.preadv(4, {{buffer, 4}}), 0);
  EXPECT_EQ(read.pread(0, 4, buffer), "abcd");
}
#endif

void writeData(facebook::velox::WriteFile* writeFile, bool useIOBuf = false) {
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

TEST(FileIoContextTest, defaultCacheableIsFalse) {
  FileIoContext defaultContext;
  EXPECT_FALSE(defaultContext.cacheable);

  FileIoContext explicitContext(nullptr);
  EXPECT_FALSE(explicitContext.cacheable);

  FileIoContext cacheableContext(nullptr, {}, nullptr, true);
  EXPECT_TRUE(cacheableContext.cacheable);
}

void readData(
    facebook::velox::ReadFile* readFile,
    bool checkFileSize = true,
    bool testReadAsync = false) {
  if (checkFileSize) {
    ASSERT_EQ(readFile->size(), 15 + kOneMB);
  }
  char buffer1[5];
  ASSERT_EQ(readFile->pread(10 + kOneMB, 5, &buffer1), "ddddd");
  char buffer2[10];
  ASSERT_EQ(readFile->pread(0, 10, &buffer2), "aaaaabbbbb");
  std::vector<char> buffer3(kOneMB);
  ASSERT_EQ(
      readFile->pread(10, kOneMB, buffer3.data()), std::string(kOneMB, 'c'));
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

// We could templated this test, but that's kinda overkill for how simple it is.
TEST(InMemoryFile, writeAndRead) {
  for (bool useIOBuf : {true, false}) {
    std::string buf;
    {
      InMemoryWriteFile writeFile(&buf);
      writeData(&writeFile, useIOBuf);
    }
    InMemoryReadFile readFile(buf);
    readData(&readFile);
  }
}

TEST(InMemoryFile, preadv) {
  std::string buf;
  {
    InMemoryWriteFile writeFile(&buf);
    writeData(&writeFile);
  }
  // aaaaa bbbbb c*1MB ddddd
  InMemoryReadFile readFile(buf);
  std::vector<std::string> expected = {"ab", "a", "cccdd", "ddd"};
  std::vector<Region> readRegions = std::vector<Region>{
      {4, 2UL, {}},
      {0, 1UL, {}},
      {5 + 5 + kOneMB - 3, 5UL, {}},
      {5 + 5 + kOneMB + 2, 3UL, {}}};

  std::vector<folly::IOBuf> iobufs(readRegions.size());
  readFile.preadv(readRegions, {iobufs.data(), iobufs.size()});
  std::vector<std::string> values;
  values.reserve(iobufs.size());
  for (auto& iobuf : iobufs) {
    values.push_back(
        std::string{
            reinterpret_cast<const char*>(iobuf.data()), iobuf.length()});
  }

  EXPECT_EQ(expected, values);
}

TEST(InMemoryFile, ownedBuffer) {
  memory::MemoryManager::Options options;
  memory::MemoryManager manager{options};
  auto pool = manager.addLeafPool();

  const std::string data = "abcdefghij";
  std::shared_ptr<InMemoryReadFile> readFile;
  {
    auto buffer = AlignedBuffer::allocate<char>(data.size(), pool.get());
    memcpy(buffer->asMutable<char>(), data.data(), data.size());
    readFile = std::make_shared<InMemoryReadFile>(std::move(buffer));
  }

  // The file owns the buffer, so reads succeed after the local handle is gone.
  EXPECT_EQ(readFile->size(), data.size());
  EXPECT_EQ(readFile->pread(0, data.size()), data);
}

class FaultyFsTest : public ::testing::Test {
 protected:
  FaultyFsTest() {}

  static void SetUpTestCase() {
    filesystems::registerLocalFileSystem();
    tests::utils::registerFaultyFileSystem();
  }

  void SetUp() {
    dir_ = TempDirectoryPath::create(true);
    fs_ = std::dynamic_pointer_cast<tests::utils::FaultyFileSystem>(
        filesystems::getFileSystem(dir_->getPath(), {}));
    VELOX_CHECK_NOT_NULL(fs_);
    readFilePath_ = fmt::format("{}/faultyTestReadFile", dir_->getPath());
    writeFilePath_ = fmt::format("{}/faultyTestWriteFile", dir_->getPath());
    const int bufSize = 1024;
    buffer_.resize(bufSize);
    for (int i = 0; i < bufSize; ++i) {
      buffer_[i] = i % 256;
    }
    {
      auto writeFile = fs_->openFileForWrite(readFilePath_, {});
      writeData(writeFile.get());
    }
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    readData(readFile.get(), true);
    try {
      VELOX_FAIL("InjectedFaultFileError");
    } catch (VeloxRuntimeError&) {
      fileError_ = std::current_exception();
    }
  }

  void TearDown() {
    fs_->clearFileFaultInjections();
  }

  void writeData(facebook::velox::WriteFile* file) {
    file->append(std::string_view(buffer_));
    file->flush();
  }

  void readData(facebook::velox::ReadFile* file, bool useReadv = false) {
    std::vector<char> readBuf(buffer_.size());
    if (!useReadv) {
      file->pread(0, buffer_.size(), readBuf.data());
    } else {
      std::vector<folly::Range<char*>> buffers;
      buffers.push_back(folly::Range<char*>(readBuf.data(), buffer_.size()));
      file->preadv(0, buffers);
    }
    for (int i = 0; i < buffer_.size(); ++i) {
      if (buffer_[i] != readBuf[i]) {
        VELOX_FAIL("Data Mismatch");
      }
    }
  }

  std::shared_ptr<TempDirectoryPath> dir_;
  std::string readFilePath_;
  std::string writeFilePath_;
  std::shared_ptr<tests::utils::FaultyFileSystem> fs_;
  std::string buffer_;
  std::exception_ptr fileError_;
};

TEST_F(FaultyFsTest, schemCheck) {
  ASSERT_TRUE(
      filesystems::isPathSupportedByRegisteredFileSystems("faulty:/test"));
  ASSERT_FALSE(
      filesystems::isPathSupportedByRegisteredFileSystems("other:/test"));
}

TEST_F(FaultyFsTest, fileReadErrorInjection) {
  // Set read error.
  fs_->setFileInjectionError(fileError_, {FaultFileOperation::Type::kRead});
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    VELOX_ASSERT_THROW(
        readData(readFile.get(), false), "InjectedFaultFileError");
  }
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for pread API so preadv should be fine.
    readData(readFile.get(), true);
  }

  // Set readv error
  fs_->setFileInjectionError(fileError_, {FaultFileOperation::Type::kReadv});
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    VELOX_ASSERT_THROW(
        readData(readFile.get(), true), "InjectedFaultFileError");
  }
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for preadv API so pread should be fine.
    readData(readFile.get(), false);
  }

  // Set error for all kinds of operations.
  fs_->setFileInjectionError(fileError_);
  auto readFile = fs_->openFileForRead(readFilePath_, {});
  VELOX_ASSERT_THROW(readData(readFile.get(), true), "InjectedFaultFileError");
  VELOX_ASSERT_THROW(readData(readFile.get(), false), "InjectedFaultFileError");
  readFile.reset();
  fs_->remove(readFilePath_);
}

TEST_F(FaultyFsTest, fileReadDelayInjection) {
  // Set 2 seconds delay.
  const uint64_t injectDelay{2'000'000};
  fs_->setFileInjectionDelay(injectDelay, {FaultFileOperation::Type::kRead});
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), false);
    }
    ASSERT_GE(readDurationUs, injectDelay);
  }
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for pread API so preadv should be fine.
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), true);
    }
    ASSERT_LT(readDurationUs, injectDelay);
  }

  // Set readv error
  fs_->setFileInjectionDelay(injectDelay, {FaultFileOperation::Type::kReadv});
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), true);
    }
    ASSERT_GE(readDurationUs, injectDelay);
  }
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for pread API so preadv should be fine.
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), false);
    }
    ASSERT_LT(readDurationUs, injectDelay);
  }

  // Set error for all kinds of operations.
  fs_->setFileInjectionDelay(injectDelay);
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for pread API so preadv should be fine.
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), false);
    }
    ASSERT_GE(readDurationUs, injectDelay);
  }
  {
    auto readFile = fs_->openFileForRead(readFilePath_, {});
    // We only inject error for pread API so preadv should be fine.
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      readData(readFile.get(), false);
    }
    ASSERT_GE(readDurationUs, injectDelay);
  }
}

TEST_F(FaultyFsTest, fileReadFaultHookInjection) {
  const std::string path1 = fmt::format("{}/hookFile1", dir_->getPath());
  {
    auto writeFile = fs_->openFileForWrite(path1, {});
    writeData(writeFile.get());
    auto readFile = fs_->openFileForRead(path1, {});
    readData(readFile.get());
  }
  const std::string path2 = fmt::format("{}/hookFile2", dir_->getPath());
  {
    auto writeFile = fs_->openFileForWrite(path2, {});
    writeData(writeFile.get());
    auto readFile = fs_->openFileForRead(path2, {});
    readData(readFile.get());
  }
  // Set read error.
  fs_->setFileInjectionHook([&](FaultFileOperation* op) {
    // Only inject error for readv.
    if (op->type != FaultFileOperation::Type::kReadv) {
      return;
    }
    // Only inject error for path2.
    if (op->path != path2) {
      return;
    }
    VELOX_FAIL("inject hook read failure");
  });
  {
    auto readFile = fs_->openFileForRead(path1, {});
    readData(readFile.get(), false);
    readData(readFile.get(), true);
  }
  {
    auto readFile = fs_->openFileForRead(path2, {});
    // Verify only throw for readv.
    readData(readFile.get(), false);
    VELOX_ASSERT_THROW(
        readData(readFile.get(), true), "inject hook read failure");
  }

  // Set to return fake data.
  fs_->setFileInjectionHook([&](FaultFileOperation* op) {
    // Only inject error for path1.
    if (op->path != path1) {
      return;
    }
    // Only inject error for read.
    if (op->type != FaultFileOperation::Type::kRead) {
      return;
    }
    auto* readOp = static_cast<FaultFileReadOperation*>(op);
    char* readBuf = static_cast<char*>(readOp->buf);
    for (int i = 0; i < readOp->length; ++i) {
      readBuf[i] = 0;
    }
    readOp->delegate = false;
  });

  {
    auto readFile = fs_->openFileForRead(path2, {});
    readData(readFile.get(), false);
    readData(readFile.get(), true);
  }
  {
    auto readFile = fs_->openFileForRead(path1, {});
    // Verify only throw for read.
    readData(readFile.get(), true);
    VELOX_ASSERT_THROW(readData(readFile.get(), false), "Data Mismatch");
  }
}

TEST_F(FaultyFsTest, fileWriteErrorInjection) {
  // Set write error.
  fs_->setFileInjectionError(fileError_, {FaultFileOperation::Type::kAppend});
  {
    auto writeFile = fs_->openFileForWrite(writeFilePath_, {});
    VELOX_ASSERT_THROW(writeFile->append("hello"), "InjectedFaultFileError");
    writeFile->close();
    fs_->remove(writeFilePath_);
  }
  // Set error for all kinds of operations.
  fs_->setFileInjectionError(fileError_);
  {
    auto writeFile = fs_->openFileForWrite(writeFilePath_, {});
    VELOX_ASSERT_THROW(writeFile->append("hello"), "InjectedFaultFileError");
    writeFile->close();
    fs_->remove(writeFilePath_);
  }
}

TEST_F(FaultyFsTest, fileWriteDelayInjection) {
  // Set 2 seconds delay.
  const uint64_t injectDelay{2'000'000};
  fs_->setFileInjectionDelay(injectDelay, {FaultFileOperation::Type::kAppend});
  {
    auto writeFile = fs_->openFileForWrite(writeFilePath_, {});
    uint64_t readDurationUs{0};
    {
      MicrosecondTimer readTimer(&readDurationUs);
      writeFile->append("hello");
    }
    ASSERT_GE(readDurationUs, injectDelay);
    writeFile->close();
    fs_->remove(writeFilePath_);
  }
}

TEST_F(FaultyFsTest, fileWriteFaultHookInjection) {
  const std::string path1 = fmt::format("{}/hookFile1", dir_->getPath());
  const std::string path2 = fmt::format("{}/hookFile2", dir_->getPath());
  // Set to write fake data.
  fs_->setFileInjectionHook([&](FaultFileOperation* op) {
    // Only inject for write.
    if (op->type != FaultFileOperation::Type::kAppend) {
      return;
    }
    // Only inject for path2.
    if (op->path != path2) {
      return;
    }
    auto* writeOp = static_cast<FaultFileAppendOperation*>(op);
    *writeOp->data = "Error data";
  });
  {
    auto writeFile = fs_->openFileForWrite(path1, {});
    writeFile->append("hello");
    writeFile->close();
    auto readFile = fs_->openFileForRead(path1, {});
    char buffer[5];
    ASSERT_EQ(readFile->size(), 5);
    ASSERT_EQ(readFile->pread(0, 5, &buffer), "hello");
    readFile.reset();
    fs_->remove(path1);
  }
  {
    auto writeFile = fs_->openFileForWrite(path2, {});
    writeFile->append("hello");
    writeFile->close();
    auto readFile = fs_->openFileForRead(path2, {});
    char buffer[10];
    ASSERT_EQ(readFile->size(), 10);
    ASSERT_EQ(readFile->pread(0, 10, &buffer), "Error data");
    readFile.reset();
    fs_->remove(path2);
  }

  // Set to not delegate.
  fs_->setFileInjectionHook([&](FaultFileOperation* op) {
    // Only inject for write.
    if (op->type != FaultFileOperation::Type::kAppend) {
      return;
    }
    // Only inject for path2.
    if (op->path != path2) {
      return;
    }
    auto* writeOp = static_cast<FaultFileAppendOperation*>(op);
    writeOp->delegate = false;
  });
  {
    auto writeFile = fs_->openFileForWrite(path1, {});
    writeFile->append("hello");
    writeFile->close();
    auto readFile = fs_->openFileForRead(path1, {});
    char buffer[5];
    ASSERT_EQ(readFile->size(), 5);
    ASSERT_EQ(readFile->pread(0, 5, &buffer), "hello");
    readFile.reset();
    fs_->remove(path1);
  }
  {
    auto writeFile = fs_->openFileForWrite(path2, {});
    writeFile->append("hello");
    writeFile->close();
    auto readFile = fs_->openFileForRead(path2, {});
    ASSERT_EQ(readFile->size(), 0);
    readFile.reset();
    fs_->remove(path2);
  }
}
