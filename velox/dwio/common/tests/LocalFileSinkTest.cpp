/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

#include <gtest/gtest.h>

using namespace ::testing;
using namespace facebook::velox::exec::test;

namespace facebook::velox::dwio::common {

class LocalFileSinkTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    velox::filesystems::registerLocalFileSystem();
  }

  void runTest() {
    const auto root = TempDirectoryPath::create();
    const auto filePath =
        fs::path(root->getPath()) / "xxx/yyy/zzz/test_file.ext";

    ASSERT_FALSE(fs::exists(filePath.string()));

    auto localFileSink = FileSink::create(
        fmt::format("file:{}", filePath.string()), {.pool = pool_.get()});
    ASSERT_TRUE(localFileSink->isBuffered());
    localFileSink->close();

    EXPECT_TRUE(fs::exists(filePath.string()));
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

TEST_F(LocalFileSinkTest, missingRegistration) {
  VELOX_ASSERT_THROW(runTest(), "FileSink is not registered for file:");
}

TEST_F(LocalFileSinkTest, create) {
  LocalFileSink::registerFactory();
  runTest();
}

TEST_F(LocalFileSinkTest, existFileCheck) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "xxx/yyy/zzz/test_file.ext";

  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()), {.pool = pool_.get()});
  ASSERT_TRUE(localFileSink->isBuffered());
  localFileSink->close();
  EXPECT_TRUE(fs::exists(filePath.string()));

  VELOX_ASSERT_THROW(
      FileSink::create(
          fmt::format("file:{}", filePath.string()), {.pool = pool_.get()}),
      "File exists");
}

TEST_F(LocalFileSinkTest, getIoStatisticsReturnsNullWhenNotProvided) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "test_stats_null.ext";

  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()), {.pool = pool_.get()});

  EXPECT_EQ(localFileSink->getIoStatistics(), nullptr);
  localFileSink->close();
}

TEST_F(LocalFileSinkTest, getIoStatisticsReturnsProvidedStats) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "test_stats_provided.ext";

  IoStatistics ioStats;
  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()),
      {.pool = pool_.get(), .stats = &ioStats});

  EXPECT_EQ(localFileSink->getIoStatistics(), &ioStats);
  localFileSink->close();
}

TEST_F(LocalFileSinkTest, getFileSystemStatsReturnsNullWhenNotProvided) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "test_fs_stats_null.ext";

  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()), {.pool = pool_.get()});

  EXPECT_EQ(localFileSink->getFileSystemStats(), nullptr);
  localFileSink->close();
}

TEST_F(LocalFileSinkTest, getFileSystemStatsReturnsProvidedStats) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "test_fs_stats_provided.ext";

  velox::IoStats fileSystemStats;
  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()),
      {.pool = pool_.get(), .fileSystemStats = &fileSystemStats});

  EXPECT_EQ(localFileSink->getFileSystemStats(), &fileSystemStats);
  localFileSink->close();
}

TEST_F(LocalFileSinkTest, getFileSystemStatsAndIoStatisticsBothProvided) {
  LocalFileSink::registerFactory();
  auto root = TempDirectoryPath::create();
  auto filePath = fs::path(root->getPath()) / "test_both_stats.ext";

  IoStatistics ioStats;
  velox::IoStats fileSystemStats;
  auto localFileSink = FileSink::create(
      fmt::format("file:{}", filePath.string()),
      {.pool = pool_.get(),
       .stats = &ioStats,
       .fileSystemStats = &fileSystemStats});

  EXPECT_EQ(localFileSink->getIoStatistics(), &ioStats);
  EXPECT_EQ(localFileSink->getFileSystemStats(), &fileSystemStats);
  localFileSink->close();
}

} // namespace facebook::velox::dwio::common
