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
#include "gtest/gtest.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/storage_adapters/hdfs/HdfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/hdfs/RegisterHdfsFileSystem.h"
#include "velox/connectors/hive/storage_adapters/hdfs/tests/HdfsMiniCluster.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/external/hdfs/ArrowHdfsInternal.h"

#include <unistd.h>

using namespace facebook::velox;

using filesystems::arrow::io::internal::LibHdfsShim;

constexpr int kOneMB = 1 << 20;
static const std::string destinationPath = "/test_file.txt";
static const std::string simpleDestinationPath = "hdfs://" + destinationPath;
static const std::string viewfsDestinationPath = "viewfs://" + destinationPath;
std::unordered_map<std::string, std::string> configurationValues;

class HdfsFileSystemTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    filesystems::registerHdfsFileSystem();
    if (miniCluster == nullptr) {
      miniCluster = std::make_shared<filesystems::test::HdfsMiniCluster>();
      miniCluster->start();
      auto tempFile = createFile();
      miniCluster->addFile(tempFile->getPath(), destinationPath);
    }
    configurationValues.insert(
        {"hive.hdfs.host", std::string(miniCluster->host())});
    configurationValues.insert(
        {"hive.hdfs.port", std::string(miniCluster->nameNodePort())});
    fullDestinationPath_ =
        fmt::format("{}{}", miniCluster->url(), destinationPath);
  }

  void SetUp() override {
    if (!miniCluster->isRunning()) {
      miniCluster->start();
    }
    filesystems::registerHdfsFileSystem();
  }

  static void TearDownTestSuite() {
    miniCluster->stop();
  }

  static std::unique_ptr<WriteFile> openFileForWrite(std::string_view path) {
    auto config = std::make_shared<const config::ConfigBase>(
        std::unordered_map<std::string, std::string>(configurationValues));
    auto hdfsFilePath = fmt::format("{}{}", miniCluster->url(), path);
    auto hdfsFileSystem = filesystems::getFileSystem(hdfsFilePath, config);
    return hdfsFileSystem->openFileForWrite(path);
  }

  static std::atomic<bool> startThreads;
  static std::shared_ptr<filesystems::test::HdfsMiniCluster> miniCluster;
  static std::string fullDestinationPath_;

 private:
  static std::shared_ptr<::exec::test::TempFilePath> createFile() {
    auto tempFile = exec::test::TempFilePath::create();
    tempFile->append("aaaaa");
    tempFile->append("bbbbb");
    tempFile->append(std::string(kOneMB, 'c'));
    tempFile->append("ddddd");
    return tempFile;
  }
};

std::shared_ptr<filesystems::test::HdfsMiniCluster>
    HdfsFileSystemTest::miniCluster = nullptr;
std::atomic<bool> HdfsFileSystemTest::startThreads = false;
std::string HdfsFileSystemTest::fullDestinationPath_;

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

void checkReadErrorMessages(
    ReadFile* readFile,
    std::string errorMessage,
    int endpoint) {
  try {
    readFile->pread(10 + kOneMB, endpoint);
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(error.message(), testing::HasSubstr(errorMessage));
  }

  try {
    auto buf = std::make_unique<char[]>(8);
    readFile->pread(10 + kOneMB, endpoint, buf.get());
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(error.message(), testing::HasSubstr(errorMessage));
  }
}

bool checkMiniClusterStop(ReadFile* readFile, const std::string& errorMessage) {
  try {
    readFile->pread(0, 1);
    return false;
  } catch (const VeloxException& error) {
    return error.message().find(errorMessage) != std::string::npos;
  }
}

void verifyFailures(LibHdfsShim* driver, hdfsFS hdfs) {
  HdfsReadFile readFile(driver, hdfs, destinationPath);
  HdfsReadFile readFile2(driver, hdfs, destinationPath);
  auto startPoint = 10 + kOneMB;
  auto size = 15 + kOneMB;
  auto endpoint = 10 + 2 * kOneMB;
  auto offsetErrorMessage =
      (boost::format(
           "(%d vs. %d) Cannot read HDFS file beyond its size: %d, offset: %d, end point: %d") %
       size % endpoint % size % startPoint % endpoint)
          .str();

  auto readFailErrorMessage =
      (boost::format(
           "Unable to open file %s. got error: ConnectException: Connection refused") %
       destinationPath)
          .str();

  checkReadErrorMessages(&readFile, offsetErrorMessage, kOneMB);
  HdfsFileSystemTest::miniCluster->stop();

  constexpr auto kMaxRetries = 10;
  int retries = 0;
  while (true) {
    if (checkMiniClusterStop(&readFile2, readFailErrorMessage)) {
      checkReadErrorMessages(&readFile2, readFailErrorMessage, 1);
      break;
    } else {
      if (retries >= kMaxRetries) {
        FAIL() << "MiniCluster doesn't stop after kMaxRetries try";
      } else {
        sleep(1);
        retries++;
      }
    }
  }
}

hdfsFS connectHdfsDriver(
    filesystems::arrow::io::internal::LibHdfsShim** driver,
    const std::string host,
    const std::string port) {
  filesystems::arrow::io::internal::LibHdfsShim* libhdfs_shim;
  auto status = filesystems::arrow::io::internal::ConnectLibHdfs(&libhdfs_shim);
  VELOX_CHECK(status.ok(), "ConnectLibHdfs failed.");

  // Connect to HDFS with the builder object
  hdfsBuilder* builder = libhdfs_shim->NewBuilder();
  libhdfs_shim->BuilderSetNameNode(builder, host.c_str());
  libhdfs_shim->BuilderSetNameNodePort(builder, std::stoi(port));
  libhdfs_shim->BuilderSetForceNewInstance(builder);

  auto hdfs = libhdfs_shim->BuilderConnect(builder);
  VELOX_CHECK_NOT_NULL(
      hdfs,
      "Unable to connect to HDFS at {}:{}, got error",
      host.c_str(),
      port);
  *driver = libhdfs_shim;
  return hdfs;
}

TEST_F(HdfsFileSystemTest, read) {
  filesystems::arrow::io::internal::LibHdfsShim* driver;
  auto hdfs = connectHdfsDriver(
      &driver,
      std::string(miniCluster->host()),
      std::string(miniCluster->nameNodePort()));
  HdfsReadFile readFile(driver, hdfs, destinationPath);
  readData(&readFile);
}

TEST_F(HdfsFileSystemTest, viaFileSystem) {
  auto config = std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(fullDestinationPath_, config);
  auto readFile = hdfsFileSystem->openFileForRead(fullDestinationPath_);
  readData(readFile.get());
}

TEST_F(HdfsFileSystemTest, initializeFsWithEndpointInfoInFilePath) {
  // Without host/port configured.
  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  auto hdfsFileSystem =
      filesystems::getFileSystem(fullDestinationPath_, config);
  auto readFile = hdfsFileSystem->openFileForRead(fullDestinationPath_);
  readData(readFile.get());

  // Wrong endpoint info specified in hdfs file path.
  const std::string wrongFullDestinationPath =
      "hdfs://not_exist_host:" + std::string(miniCluster->nameNodePort()) +
      destinationPath;
  VELOX_ASSERT_THROW(
      filesystems::getFileSystem(wrongFullDestinationPath, config),
      "Unable to connect to HDFS");
}

TEST_F(HdfsFileSystemTest, fallbackToUseConfig) {
  auto config = std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(fullDestinationPath_, config);
  auto readFile = hdfsFileSystem->openFileForRead(fullDestinationPath_);
  readData(readFile.get());
}

TEST_F(HdfsFileSystemTest, oneFsInstanceForOneEndpoint) {
  auto hdfsFileSystem1 =
      filesystems::getFileSystem(fullDestinationPath_, nullptr);
  auto hdfsFileSystem2 =
      filesystems::getFileSystem(fullDestinationPath_, nullptr);
  ASSERT_TRUE(hdfsFileSystem1 == hdfsFileSystem2);
}

TEST_F(HdfsFileSystemTest, missingFileViaFileSystem) {
  auto config = std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(fullDestinationPath_, config);

  VELOX_ASSERT_RUNTIME_THROW_CODE(
      hdfsFileSystem->openFileForRead(
          "hdfs://localhost:7777/path/that/does/not/exist"),
      error_code::kFileNotFound,
      "Unable to get file path info for file: hdfs://localhost:7777/path/that/does/not/exist. got error: FileNotFoundException: Path hdfs://localhost:7777/path/that/does/not/exist does not exist.");
}

TEST_F(HdfsFileSystemTest, missingHost) {
  try {
    std::unordered_map<std::string, std::string> missingHostConfiguration(
        {{"hive.hdfs.port", std::string(miniCluster->nameNodePort())}});
    auto config = std::make_shared<const config::ConfigBase>(
        std::move(missingHostConfiguration));
    filesystems::HdfsFileSystem hdfsFileSystem(
        config,
        filesystems::HdfsFileSystem::getServiceEndpoint(
            simpleDestinationPath, config.get()));
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "hdfsHost is empty, configuration missing for hdfs host"));
  }
}

TEST_F(HdfsFileSystemTest, missingPort) {
  try {
    std::unordered_map<std::string, std::string> missingPortConfiguration(
        {{"hive.hdfs.host", std::string(miniCluster->host())}});
    auto config = std::make_shared<const config::ConfigBase>(
        std::move(missingPortConfiguration));
    filesystems::HdfsFileSystem hdfsFileSystem(
        config,
        filesystems::HdfsFileSystem::getServiceEndpoint(
            simpleDestinationPath, config.get()));
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "hdfsPort is empty, configuration missing for hdfs port"));
  }
}

TEST_F(HdfsFileSystemTest, missingFileViaReadFile) {
  try {
    filesystems::arrow::io::internal::LibHdfsShim* driver;
    auto hdfs = connectHdfsDriver(
        &driver,
        std::string(miniCluster->host()),
        std::string(miniCluster->nameNodePort()));
    HdfsReadFile readFile(driver, hdfs, "/path/that/does/not/exist");
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "Unable to get file path info for file: /path/that/does/not/exist. got error: FileNotFoundException: Path /path/that/does/not/exist does not exist."));
  }
}

TEST_F(HdfsFileSystemTest, schemeMatching) {
  try {
    auto fs = std::dynamic_pointer_cast<filesystems::HdfsFileSystem>(
        filesystems::getFileSystem("file://", nullptr));
    FAIL() << "expected VeloxException";
  } catch (VeloxException const& error) {
    EXPECT_THAT(
        error.message(),
        testing::HasSubstr(
            "No registered file system matched with file path 'file://'"));
  }
  auto fs = std::dynamic_pointer_cast<filesystems::HdfsFileSystem>(
      filesystems::getFileSystem(fullDestinationPath_, nullptr));
  ASSERT_TRUE(fs->isHdfsFile(fullDestinationPath_));

  fs = std::dynamic_pointer_cast<filesystems::HdfsFileSystem>(
      filesystems::getFileSystem(viewfsDestinationPath, nullptr));
  ASSERT_TRUE(fs->isHdfsFile(viewfsDestinationPath));
}

TEST_F(HdfsFileSystemTest, writeNotSupported) {
  try {
    auto config = std::make_shared<const config::ConfigBase>(
        std::unordered_map<std::string, std::string>(configurationValues));
    auto hdfsFileSystem =
        filesystems::getFileSystem(fullDestinationPath_, config);
    hdfsFileSystem->openFileForWrite("/path");
  } catch (VeloxException const& error) {
    EXPECT_EQ(error.message(), "Write to HDFS is unsupported");
  }
}

TEST_F(HdfsFileSystemTest, removeNotSupported) {
  try {
    auto config = std::make_shared<const config::ConfigBase>(
        std::unordered_map<std::string, std::string>(configurationValues));
    auto hdfsFileSystem =
        filesystems::getFileSystem(fullDestinationPath_, config);
    hdfsFileSystem->remove("/path");
  } catch (VeloxException const& error) {
    EXPECT_EQ(error.message(), "Does not support removing files from hdfs");
  }
}

TEST_F(HdfsFileSystemTest, multipleThreadsWithReadFile) {
  startThreads = false;

  filesystems::arrow::io::internal::LibHdfsShim* driver;
  auto hdfs = connectHdfsDriver(
      &driver,
      std::string(miniCluster->host()),
      std::string(miniCluster->nameNodePort()));
  std::vector<std::thread> threads;
  std::mt19937 generator(std::random_device{}());
  std::vector<int> sleepTimesInMicroseconds = {0, 500, 50000};
  std::uniform_int_distribution<std::size_t> distribution(
      0, sleepTimesInMicroseconds.size() - 1);
  for (int i = 0; i < 25; i++) {
    auto thread = std::thread(
        [&driver, &hdfs, &distribution, &generator, &sleepTimesInMicroseconds] {
          int index = distribution(generator);
          while (!HdfsFileSystemTest::startThreads) {
            std::this_thread::yield();
          }
          std::this_thread::sleep_for(
              std::chrono::microseconds(sleepTimesInMicroseconds[index]));
          HdfsReadFile readFile(driver, hdfs, destinationPath);
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
  auto config = std::make_shared<const config::ConfigBase>(
      std::unordered_map<std::string, std::string>(configurationValues));
  auto hdfsFileSystem =
      filesystems::getFileSystem(fullDestinationPath_, config);

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
      auto readFile = hdfsFileSystem->openFileForRead(fullDestinationPath_);
      readData(readFile.get());
    });
    threads.emplace_back(std::move(thread));
  }
  startThreads = true;
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(HdfsFileSystemTest, write) {
  std::string path = "/a.txt";
  auto writeFile = openFileForWrite(path);
  std::string data = "abcdefghijk";
  writeFile->append(data);
  writeFile->flush();
  ASSERT_EQ(writeFile->size(), 0);
  writeFile->append(data);
  writeFile->append(data);
  writeFile->flush();
  writeFile->close();
  ASSERT_EQ(writeFile->size(), data.size() * 3);
}

TEST_F(HdfsFileSystemTest, missingFileForWrite) {
  const std::string filePath = "hdfs://localhost:7777/path/that/does/not/exist";
  const std::string errorMsg =
      "Failed to open hdfs file: hdfs://localhost:7777/path/that/does/not/exist";
  VELOX_ASSERT_THROW(openFileForWrite(filePath), errorMsg);
}

TEST_F(HdfsFileSystemTest, writeDataFailures) {
  auto writeFile = openFileForWrite("/a.txt");
  writeFile->close();
  VELOX_ASSERT_THROW(
      writeFile->append("abcde"),
      "Cannot append to HDFS file because file handle is null, file path: /a.txt");
}

TEST_F(HdfsFileSystemTest, writeFlushFailures) {
  auto writeFile = openFileForWrite("/a.txt");
  writeFile->close();
  VELOX_ASSERT_THROW(
      writeFile->flush(),
      "Cannot flush HDFS file because file handle is null, file path: /a.txt");
}

TEST_F(HdfsFileSystemTest, writeWithParentDirNotExist) {
  std::string path = "/parent/directory/that/does/not/exist/a.txt";
  auto writeFile = openFileForWrite(path);
  std::string data = "abcdefghijk";
  writeFile->append(data);
  writeFile->flush();
  ASSERT_EQ(writeFile->size(), 0);
  writeFile->append(data);
  writeFile->append(data);
  writeFile->flush();
  writeFile->close();
  ASSERT_EQ(writeFile->size(), data.size() * 3);
}

TEST_F(HdfsFileSystemTest, readFailures) {
  filesystems::arrow::io::internal::LibHdfsShim* driver;
  auto hdfs = connectHdfsDriver(
      &driver,
      std::string(miniCluster->host()),
      std::string(miniCluster->nameNodePort()));
  verifyFailures(driver, hdfs);
}
