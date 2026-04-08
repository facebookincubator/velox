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

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/connectors/hive/storage_adapters/test_common/InsertTest.h"

namespace facebook::velox::filesystems {
namespace {

class S3InsertTest : public S3Test, public test::InsertTest {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    filesystems::registerS3FileSystem();
  }

  static void TearDownTestCase() {
    filesystems::finalizeS3FileSystem();
  }

  void SetUp() override {
    S3Test::SetUp();
    InsertTest::SetUp(minioServer_->hiveConfig(), ioExecutor_.get());
  }

  void TearDown() override {
    S3Test::TearDown();
    InsertTest::TearDown();
  }
};
} // namespace

TEST_F(S3InsertTest, s3InsertTest) {
  const int64_t kExpectedRows = 1'000;
  const std::string_view kOutputDirectory{"s3://writedata/"};
  minioServer_->addBucket("writedata");

  runInsertTest(kOutputDirectory, kExpectedRows, pool());
}

// Test with data exceeding the default 5MB minPartSize to trigger multipart
// upload. This test generates enough data to exceed 5MB, which should trigger
// at least one multipart upload part and a remainder.
TEST_F(S3InsertTest, s3MultipartUploadTest) {
  // Generate enough rows to exceed 5MB.
  // Each row has 4 columns: BIGINT (8 bytes), INTEGER (4 bytes),
  // SMALLINT (2 bytes), DOUBLE (8 bytes) = 22 bytes per row minimum.
  // To exceed 5MB (5 * 1024 * 1024 = 5,242,880 bytes), we need at least
  // 5,242,880 / 22 ≈ 238,313 rows. Let's use 300,000 rows to be safe.
  const int64_t kExpectedRows = 300'000;
  const std::string_view kOutputDirectory{"s3://multipartdata/"};
  minioServer_->addBucket("multipartdata");

  runInsertTest(kOutputDirectory, kExpectedRows, pool());
}
} // namespace facebook::velox::filesystems

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
