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

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using facebook::velox::common::Region;
using namespace facebook::velox::tests::utils;

constexpr int kOneMB = 1 << 20;

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

TEST(FileIoContextTest, defaultCacheableIsFalse) {
  FileIoContext defaultContext;
  EXPECT_FALSE(defaultContext.cacheable);

  FileIoContext explicitContext(nullptr);
  EXPECT_FALSE(explicitContext.cacheable);

  FileIoContext cacheableContext(nullptr, {}, nullptr, true);
  EXPECT_TRUE(cacheableContext.cacheable);
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

  void writeData(WriteFile* file) {
    file->append(std::string_view(buffer_));
    file->flush();
  }

  void readData(ReadFile* file, bool useReadv = false) {
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
    fs_->remove(writeFilePath_);
  }
  // Set error for all kinds of operations.
  fs_->setFileInjectionError(fileError_);
  {
    auto writeFile = fs_->openFileForWrite(writeFilePath_, {});
    VELOX_ASSERT_THROW(writeFile->append("hello"), "InjectedFaultFileError");
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
    fs_->remove(path1);
  }
  {
    auto writeFile = fs_->openFileForWrite(path2, {});
    writeFile->append("hello");
    writeFile->close();
    auto readFile = fs_->openFileForRead(path2, {});
    ASSERT_EQ(readFile->size(), 0);
    fs_->remove(path2);
  }
}
