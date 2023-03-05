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

#include "connectors/hive/storage_adapters/s3fs/benchmark/S3ReadBenchmark.h"
#include "connectors/hive/storage_adapters/s3fs/S3Util.h"
#include "connectors/hive/storage_adapters/s3fs/util/MinioServer.h"
#include "velox/common/file/File.h"
#include "velox/exec/tests/utils/TempFilePath.h"

#include "gtest/gtest.h"

using namespace facebook::velox;

constexpr int kOneMB = 1 << 20;

class S3ReadBenchmarkTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (minioServer_ == nullptr) {
      minioServer_ = std::make_shared<MinioServer>();
      minioServer_->start();
    }
    // Write S3 Config.
    auto hiveConfig = minioServer_->hiveConfig();
    {
      LocalWriteFile stream(config_->path + "/hive.properties");
      for (const auto& [config, value] : hiveConfig->values()) {
        stream.append(config + "=" + value + '\n');
      }
      stream.close();
    }
    // Write random data.
    {
      const std::string dataFile =
          minioServer_->path() + "/" + bucketName_ + "/" + fileName_;
      minioServer_->addBucket(bucketName_.c_str());
      LocalWriteFile stream(dataFile);
      for (int i = 0; i < 100; i++) {
        stream.append("velox");
      }
      stream.close();
    }
  }

  static void TearDownTestSuite() {
    if (minioServer_ != nullptr) {
      minioServer_->stop();
      minioServer_ = nullptr;
    }
  }

  static std::shared_ptr<exec::test::TempDirectoryPath> config_;
  static const std::string bucketName_;
  static const std::string fileName_;
  static std::shared_ptr<MinioServer> minioServer_;
};

std::shared_ptr<exec::test::TempDirectoryPath> S3ReadBenchmarkTest::config_ =
    exec::test::TempDirectoryPath::create();
const std::string S3ReadBenchmarkTest::bucketName_ = "bucket";
const std::string S3ReadBenchmarkTest::fileName_ = "file";
std::shared_ptr<MinioServer> S3ReadBenchmarkTest::minioServer_ = nullptr;

TEST_F(S3ReadBenchmarkTest, output) {
  FLAGS_config = config_->path + "/hive.properties";
  FLAGS_request_bytes = 100;
  const std::string s3File = s3URI(bucketName_, fileName_);
  FLAGS_path = s3File;
  FLAGS_num_in_run = 4;
  FLAGS_gap = 10;
  FLAGS_measurement_size = 400;
  std::stringstream out;
  S3ReadBenchmark bm(out);
  bm.initialize();
  bm.run();
  std::istringstream result(out.str());
  std::string line;
  std::getline(result, line);
  ASSERT_EQ(line, "Request Size: 100 Gap: 10 Num in Run: 4 Repeats: 3");
  std::getline(result, line);
  // Exclude the actual values since they vary.
  ASSERT_EQ(line.substr(line.size() - 12), "MB/s 1 pread");
  std::getline(result, line);
  ASSERT_EQ(line.substr(line.size() - 13), "MB/s 1 preadv");
  std::getline(result, line);
  ASSERT_EQ(line.substr(line.size() - 19), "MB/s multiple pread");
  std::getline(result, line);
  ASSERT_EQ(line.substr(line.size() - 15), "MB/s 1 pread mt");
  std::getline(result, line);
  ASSERT_EQ(line.substr(line.size() - 16), "MB/s 1 preadv mt");
  std::getline(result, line);
  ASSERT_EQ(line.substr(line.size() - 22), "MB/s multiple pread mt");
}