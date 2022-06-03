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
#include "velox/connectors/hive/storage_adapters/hdfs/HdfsFileSystem.h"
#include <boost/format.hpp>
#include <gmock/gmock-matchers.h>
#include <atomic>
#include <random>
#include "HdfsMiniCluster.h"
#include "gtest/gtest.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/storage_adapters/hdfs/libhdfs3/src/client/hdfs.h"
#include "velox/exec/tests/utils/TempFilePath.h"

using namespace facebook::velox;

constexpr int kOneMB = 1 << 20;
static const std::string destinationPath = "/test_file.txt";
static const std::string harunaDestinationPath = "hdfs://haruna/test_file.txt";

class HdfsFileSystemTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    if (miniCluster == nullptr) {
      miniCluster = std::make_shared<
          facebook::velox::filesystems::test::HdfsMiniCluster>();
      miniCluster->start();
      auto tempFile = createFile();
      miniCluster->addFile(tempFile->path, destinationPath);
    }
  }

  static void TearDownTestSuite() {
    miniCluster->stop();
  }
  static std::atomic<bool> startThreads;
  static std::shared_ptr<facebook::velox::filesystems::test::HdfsMiniCluster>
      miniCluster;

 private:
  static std::shared_ptr<::exec::test::TempFilePath> createFile() {
    auto tempFile = ::exec::test::TempFilePath::create();
    tempFile->append("aaaaa");
    tempFile->append("bbbbb");
    tempFile->append(std::string(kOneMB, 'c'));
    tempFile->append("ddddd");
    return tempFile;
  }
};

std::shared_ptr<facebook::velox::filesystems::test::HdfsMiniCluster>
    HdfsFileSystemTest::miniCluster = nullptr;
std::atomic<bool> HdfsFileSystemTest::startThreads = false;

void readData(ReadFile* readFile) {
  ASSERT_EQ(readFile->size(), 15 + kOneMB);
  char buffer1[5];
  ASSERT_EQ(readFile->pread(10 + kOneMB, 5, &buffer1), "ddddd");
  char buffer2[10];
  ASSERT_EQ(readFile->pread(0, 10, &buffer2), "aaaaabbbbb");
  auto buffer3 = new char[kOneMB];
  ASSERT_EQ(readFile->pread(10, kOneMB, buffer3), std::string(kOneMB, 'c'));
  delete[] buffer3;
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
}

void verifyFailures(ReadFile* readFile) {
  auto startPoint = 10 + kOneMB;
  auto size = 15 + kOneMB;
  auto endpoint = 10 + 2 * kOneMB;
  auto errorMessage =
      (boost::format(
           "(%d vs. %d) Cannot read HDFS file beyond its size: %d, starting point: %d, end point: %d") %
       size % endpoint % size % startPoint % endpoint)
          .str();
  try {
    readFile->pread(10 + kOneMB, kOneMB);
    ;
    FAIL() << "expected VeloxException";
  } catch (facebook::velox::VeloxException const& error) {
    EXPECT_EQ(error.message(), errorMessage);
  }
  try {
    auto buf = std::make_unique<char[]>(8);
    readFile->pread(10 + kOneMB, kOneMB, buf.get());
    ;
    FAIL() << "expected VeloxException";
  } catch (facebook::velox::VeloxException const& error) {
    EXPECT_EQ(error.message(), errorMessage);
  }
}

TEST_F(HdfsFileSystemTest, read) {
  struct hdfsBuilder* builder = hdfsNewBuilder();
  hdfsBuilderSetNameNode(builder, "localhost");
  hdfsBuilderSetNameNodePort(builder, 7878);
  auto hdfs = hdfsBuilderConnect(builder);
  HdfsReadFile readFile(hdfs, destinationPath);
  readData(&readFile);
}

TEST_F(HdfsFileSystemTest, viaFileSystem) {
  facebook::velox::filesystems::registerHdfsFileSystem();
  std::unordered_map<std::string, std::string> configurationValues(
      {{"hdfs_host", "localhost"}, {"hdfs_port", "7878"}});
  auto memConfig =
      std::make_shared<const core::MemConfig>(std::move(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(harunaDestinationPath, memConfig);
  auto readFile = hdfsFileSystem->openFileForRead(harunaDestinationPath);
  readData(readFile.get());
}

TEST_F(HdfsFileSystemTest, missingFileViaFileSystem) {
  try {
    facebook::velox::filesystems::registerHdfsFileSystem();
    std::unordered_map<std::string, std::string> configurationValues(
        {{"hdfs_host", "localhost"}, {"hdfs_port", "7878"}});
    auto memConfig =
        std::make_shared<const core::MemConfig>(std::move(configurationValues));
    auto hdfsFileSystem =
        filesystems::getFileSystem(harunaDestinationPath, memConfig);
    auto readFile =
        hdfsFileSystem->openFileForRead("/path/that/does/not/exist");
    FAIL() << "expected VeloxException";
  } catch (facebook::velox::VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "Unable to open file /path/that/does/not/exist. got error: File does not exist: /path/that/does/not/exist"));
  }
}

TEST_F(HdfsFileSystemTest, missingFileViaReadFile) {
  try {
    struct hdfsBuilder* builder = hdfsNewBuilder();
    hdfsBuilderSetNameNode(builder, "localhost");
    hdfsBuilderSetNameNodePort(builder, 7878);
    auto hdfs = hdfsBuilderConnect(builder);
    HdfsReadFile readFile(hdfs, "/path/that/does/not/exist");
    FAIL() << "expected VeloxException";
  } catch (facebook::velox::VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "Unable to open file /path/that/does/not/exist. got error: File does not exist: /path/that/does/not/exist"));
  }
}

TEST_F(HdfsFileSystemTest, schemeMatching) {
  auto fs =
      std::dynamic_pointer_cast<facebook::velox::filesystems::HdfsFileSystem>(
          filesystems::getFileSystem("/", nullptr));
  ASSERT_TRUE(fs->schemeMatcher());
  fs = std::dynamic_pointer_cast<facebook::velox::filesystems::HdfsFileSystem>(
      filesystems::getFileSystem(harunaDestinationPath, nullptr));
  ASSERT_TRUE(fs->schemeMatcher());
}

TEST_F(HdfsFileSystemTest, writeNotSupported) {
  try {
    facebook::velox::filesystems::registerHdfsFileSystem();
    std::unordered_map<std::string, std::string> configurationValues(
        {{"hdfs_host", "localhost"}, {"hdfs_port", "7878"}});
    auto memConfig =
        std::make_shared<const core::MemConfig>(std::move(configurationValues));
    auto hdfsFileSystem =
        filesystems::getFileSystem(harunaDestinationPath, memConfig);
    hdfsFileSystem->openFileForWrite("/path");
  } catch (facebook::velox::VeloxException const& error) {
    EXPECT_EQ(error.message(), "Does not support write to HDFS");
  }
}

TEST_F(HdfsFileSystemTest, multipleThreadsWithReadFile) {
  startThreads = false;
  struct hdfsBuilder* builder = hdfsNewBuilder();
  hdfsBuilderSetNameNode(builder, "localhost");
  hdfsBuilderSetNameNodePort(builder, 7878);
  auto hdfs = hdfsBuilderConnect(builder);
  HdfsReadFile readFile(hdfs, destinationPath);
  std::vector<std::thread> threads;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> sleepTimesInMicroseconds = {0, 500, 50000};
  std::uniform_int_distribution<std::size_t> distribution(
      0, sleepTimesInMicroseconds.size() - 1);
  for (int i = 0; i < 25; i++) {
    auto thread = std::thread(
        [&readFile, &distribution, &generator, &sleepTimesInMicroseconds] {
          int index = distribution(generator);
          while (!HdfsFileSystemTest::startThreads) {
            std::this_thread::yield();
          }
          std::this_thread::sleep_for(
              std::chrono::microseconds(sleepTimesInMicroseconds[index]));
          readData(&readFile);
        });
    threads.emplace_back(std::move(thread));
  }
  startThreads = true;
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(HdfsFileSystemTest, multipleThreadsWithFileSystem) {
  startThreads = false;
  facebook::velox::filesystems::registerHdfsFileSystem();
  std::unordered_map<std::string, std::string> configurationValues(
      {{"hdfs_host", "localhost"}, {"hdfs_port", "7878"}});
  auto memConfig =
      std::make_shared<const core::MemConfig>(std::move(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(harunaDestinationPath, memConfig);

  std::vector<std::thread> threads;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> sleepTimesInMicroseconds = {0, 500, 50000};
  std::uniform_int_distribution<std::size_t> distribution(
      0, sleepTimesInMicroseconds.size() - 1);
  for (int i = 0; i < 25; i++) {
    auto thread = std::thread([&hdfsFileSystem,
                               &distribution,
                               &generator,
                               &sleepTimesInMicroseconds] {
      int index = distribution(generator);
      while (!HdfsFileSystemTest::startThreads) {
        std::this_thread::yield();
      }
      std::this_thread::sleep_for(
          std::chrono::microseconds(sleepTimesInMicroseconds[index]));
      auto readFile = hdfsFileSystem->openFileForRead(harunaDestinationPath);
      readData(readFile.get());
    });
    threads.emplace_back(std::move(thread));
  }
  startThreads = true;
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(HdfsFileSystemTest, readFailures) {
  struct hdfsBuilder* builder = hdfsNewBuilder();
  hdfsBuilderSetNameNode(builder, "localhost");
  hdfsBuilderSetNameNodePort(builder, 7878);
  auto hdfs = hdfsBuilderConnect(builder);
  HdfsReadFile readFile(hdfs, destinationPath);
  verifyFailures(&readFile);
}
