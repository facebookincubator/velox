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
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/text/tests/writer/FileReaderUtil.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::text {
using namespace facebook::velox::common::testutil;

class BufferedWriterSinkTest : public testing::Test,
                               public velox::test::VectorTestBase {
 public:
  void SetUp() override {
    velox::filesystems::registerLocalFileSystem();
    dwio::common::LocalFileSink::registerFactory();
    rootPool_ = memory::memoryManager()->addRootPool("BufferedWriterSinkTest");
    leafPool_ = rootPool_->addLeafChild("BufferedWriterSinkTest");
    tempPath_ = TempDirectoryPath::create();
  }

 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::shared_ptr<TempDirectoryPath> tempPath_;
};

TEST_F(BufferedWriterSinkTest, write) {
  const auto tempPath = tempPath_->getPath();
  const auto filename = "test_buffered_writer.txt";

  auto filePath = fs::path(fmt::format("{}/{}", tempPath, filename));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});

  auto bufferedWriterSink = std::make_unique<BufferedWriterSink>(
      std::move(sink), rootPool_->addLeafChild("bufferedWriterSinkTest"), 15);

  bufferedWriterSink->write("hello world", 10);
  bufferedWriterSink->write("this is writer", 10);
  bufferedWriterSink->close();

  uint64_t result = readFile(tempPath, filename);
  EXPECT_EQ(result, 20);
}

TEST_F(BufferedWriterSinkTest, abort) {
  const auto tempPath = tempPath_->getPath();
  const auto filename = "test_buffered_abort.txt";

  auto filePath = fs::path(fmt::format("{}/{}", tempPath, filename));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});

  auto bufferedWriterSink = std::make_unique<BufferedWriterSink>(
      std::move(sink), rootPool_->addLeafChild("bufferedWriterSinkTest"), 15);

  bufferedWriterSink->write("hello world", 10);
  bufferedWriterSink->write("this is writer", 10);
  bufferedWriterSink->abort();

  uint64_t result = readFile(tempPath_->getPath(), filename);
  EXPECT_EQ(result, 10);
}

TEST_F(BufferedWriterSinkTest, oversizedWriteBypassesBuffer) {
  // Regression test: a write whose payload exceeds the flush buffer size must
  // drain anything buffered (preserving order) and forward the oversized
  // payload directly to the underlying sink, instead of asserting.
  constexpr uint64_t kFlushBufferSize = 16;
  const std::string oversized(64, 'x');

  const auto tempPath = tempPath_->getPath();
  const auto filename = "test_buffered_oversized.txt";

  auto filePath = fs::path(fmt::format("{}/{}", tempPath, filename));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});

  auto bufferedWriterSink = std::make_unique<BufferedWriterSink>(
      std::move(sink),
      rootPool_->addLeafChild("bufferedWriterSinkTest"),
      kFlushBufferSize);

  // Small write that fits in the buffer, then an oversized write that must
  // first drain the small one and then write directly to the sink.
  const std::string prefix = "head:";
  bufferedWriterSink->write(prefix.data(), prefix.size());
  bufferedWriterSink->write(oversized.data(), oversized.size());
  // Another small write after the bypass to confirm buffering still works.
  const std::string suffix = ":tail";
  bufferedWriterSink->write(suffix.data(), suffix.size());
  bufferedWriterSink->close();

  const auto fs = filesystems::getFileSystem(tempPath, nullptr);
  const auto& file = fs->openFileForRead(filePath.string());
  const auto fileSize = file->size();
  std::string contents(fileSize, '\0');
  file->pread(0, fileSize, contents.data());

  EXPECT_EQ(contents, prefix + oversized + suffix);
}

namespace {
// Reads the file at 'path/name' into a string.
std::string readFileBytes(const std::string& path, const std::string& name) {
  const auto fs = filesystems::getFileSystem(path, nullptr);
  const auto filePath = fs::path(fmt::format("{}/{}", path, name));
  const auto& file = fs->openFileForRead(filePath.string());
  const auto fileSize = file->size();
  std::string out(fileSize, '\0');
  if (fileSize > 0) {
    file->pread(0, fileSize, out.data());
  }
  return out;
}
} // namespace

TEST_F(BufferedWriterSinkTest, writeEqualToFlushBufferUsesBufferedPath) {
  // Boundary: a write of exactly flushBufferSize_ bytes is *not* oversized
  // (the bypass condition is strict `size > flushBufferSize_`); it must go
  // through the buffered path and land on disk intact after close().
  constexpr uint64_t kFlushBufferSize = 16;
  const std::string exact(kFlushBufferSize, 'a');

  const auto tempPath = tempPath_->getPath();
  const auto filename = "test_buffered_equal.txt";
  auto filePath = fs::path(fmt::format("{}/{}", tempPath, filename));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto bufferedWriterSink = std::make_unique<BufferedWriterSink>(
      std::move(sink),
      rootPool_->addLeafChild("bufferedWriterSinkTest"),
      kFlushBufferSize);

  bufferedWriterSink->write(exact.data(), exact.size());
  bufferedWriterSink->close();

  EXPECT_EQ(readFileBytes(tempPath, filename), exact);
}

TEST_F(BufferedWriterSinkTest, writeOneByteOverFlushBufferTriggersBypass) {
  // Boundary: a write of flushBufferSize_ + 1 bytes is the smallest oversized
  // write and must hit the bypass path. The buffer is empty so the flush() is
  // a no-op; verify the payload still lands on disk in full.
  constexpr uint64_t kFlushBufferSize = 16;
  const std::string oneOver(kFlushBufferSize + 1, 'b');

  const auto tempPath = tempPath_->getPath();
  const auto filename = "test_buffered_one_over.txt";
  auto filePath = fs::path(fmt::format("{}/{}", tempPath, filename));
  auto sink = std::make_unique<dwio::common::LocalFileSink>(
      filePath, dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto bufferedWriterSink = std::make_unique<BufferedWriterSink>(
      std::move(sink),
      rootPool_->addLeafChild("bufferedWriterSinkTest"),
      kFlushBufferSize);

  bufferedWriterSink->write(oneOver.data(), oneOver.size());
  bufferedWriterSink->close();

  EXPECT_EQ(readFileBytes(tempPath, filename), oneOver);
}
} // namespace facebook::velox::text
