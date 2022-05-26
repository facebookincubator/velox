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

#include "connectors/hive/storage_adapters/gcs/GCSFileSystem.h"
#include "connectors/hive/storage_adapters/gcs/GCSUtil.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/exec/tests/utils/TempFilePath.h"

#include "gtest/gtest.h"
#include "gtest/internal/custom/gtest.h"
#include <boost/process.hpp>
#include <google/cloud/storage/client.h>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>


namespace bp = boost::process;
namespace gc = google::cloud;
namespace gcs = google::cloud::storage;
constexpr char const* kTestBenchPort{"9000"}; //Use same as s3 port


const std::string kLoremIpsum =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"
  "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
  "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
  "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu"
  "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in"
  "culpa qui officia deserunt mollit anim id est laborum.";

class GcsTestbench : public testing::Environment {
 public:
  GcsTestbench(): port_(kTestBenchPort) {
    //port_ = std::to_string(GetListenPort());
    std::vector<std::string> names{"python3", "python"};
    // If the build script or application developer provides a value in the PYTHON
    // environment variable, then just use that.
    if (const auto* env = std::getenv("PYTHON")) {
      names = {env};
    }
    auto error = std::string(
        "Cloud not start GCS emulator."
        " Used the following list of python interpreter names:");
    for (const auto& interpreter : names) {
      auto exe_path = bp::search_path(interpreter);
      error += " " + interpreter;
      if (exe_path.empty()) {
        error += " (exe not found)";
        continue;
      }

      server_process_ = bp::child(boost::this_process::environment(), exe_path, "-m",
                                  "testbench", "--port", port_, group_);
      if (server_process_.valid() && server_process_.running()) break;
      error += " (failed to start)";
      server_process_.terminate();
      server_process_.wait();
    }
    if (server_process_.valid() && server_process_.valid()) return;
    error_ = std::move(error);
  }

  ~GcsTestbench() override {
    // Brutal shutdown, kill the full process group because the GCS testbench may launch
    // additional children.
    group_.terminate();
    if (server_process_.valid()) {
      server_process_.wait();
    }
  }

  const std::string& port() const { return port_; }
  const std::string& error() const { return error_; }

 private:
  std::string port_;
  bp::child server_process_;
  bp::group group_;
  std::string error_;
};

using namespace facebook::velox;

class GCSFileSystemTest : public testing::Test {
  protected:

    static void SetUpTestSuite() {
      if (testbench_ == nullptr)
      {
        testbench_ = std::make_shared<GcsTestbench>();
      }
      filesystems::registerGCSFileSystem();

      ASSERT_THAT(testbench_, ::testing::NotNull());
      ASSERT_THAT(testbench_->error(), ::testing::IsEmpty());

      // Create a bucket and a small file in the testbench. This makes it easier to
      // bootstrap GcsFileSystem and its tests.
      auto client = gcs::Client(
          google::cloud::Options{}
              .set<gcs::RestEndpointOption>("http://127.0.0.1:" + testbench_->port())
              .set<gc::UnifiedCredentialsOption>(gc::MakeInsecureCredentials()));

      bucket_name_ = "test1-gcs";
      google::cloud::StatusOr<gcs::BucketMetadata> bucket = client.CreateBucketForProject(
          bucket_name_, "ignored-by-testbench", gcs::BucketMetadata{});
      ASSERT_TRUE(bucket.ok()) << "Failed to create bucket <" << bucket_name_
                              << ">, status=" << bucket.status();

      object_name_ = "test-object-name";
      google::cloud::StatusOr<gcs::ObjectMetadata> object = client.InsertObject(
          bucket_name_, object_name_, kLoremIpsum);
      ASSERT_TRUE(object.ok()) << "Failed to create object <" << object_name_
                              << ">, status=" << object.status();
    }


  std::shared_ptr<const Config> TestGcsOptions()const {
    std::unordered_map<std::string, std::string> configOverride = {};
    //TODO
    //configOverride["hive.gcs.credentials"] = google::cloud::MakeGoogleDefaultCredentials();
    configOverride["hive.gcs.scheme"] =  "http";
    configOverride["hive.gcs.endpoint"] =  "127.0.0.1:" + testbench_->port();
    return std::make_shared<const core::MemConfig>(std::move(configOverride));
  }

  std::string PreexistingBucketName()  { return bucket_name_; }

  std::string PreexistingBucketPath()  { return bucket_name_ + '/'; }

  std::string PreexistingObjectName() { return object_name_; }

  std::string PreexistingObjectPath()  {
    return PreexistingBucketPath() + PreexistingObjectName();
  }
  static void TearDownTestSuite() {
    // TODO
  }

  static std::shared_ptr<GcsTestbench> testbench_;
  static std::string bucket_name_;
  static std::string object_name_;
};

std::shared_ptr<GcsTestbench> GCSFileSystemTest::testbench_ = nullptr; //will be destroyed on destructor
std::string GCSFileSystemTest::bucket_name_;
std::string GCSFileSystemTest::object_name_;

TEST_F(GCSFileSystemTest, ReadFile) {
  const std::string gcsFile = gcsURI(PreexistingBucketName(), PreexistingObjectName());

  filesystems::GCSFileSystem gcfs(TestGcsOptions());
  gcfs.initializeClient();
  auto readFile = gcfs.openFileForRead(gcsFile);
  std::int64_t size = readFile->size();
  std::int64_t ref_size = kLoremIpsum.length();
  EXPECT_EQ(size, ref_size);
  EXPECT_EQ(readFile->pread(0, size), kLoremIpsum);

  char buffer1[size];
  ASSERT_EQ(readFile->pread(0, size, &buffer1), kLoremIpsum);
  ASSERT_EQ(readFile->size(), ref_size);

  char buffer2[50];
  ASSERT_EQ(readFile->pread(10, 50, &buffer2), kLoremIpsum.substr(10,50));
  ASSERT_EQ(readFile->size(), ref_size);

  EXPECT_EQ(readFile->pread(10, size-10), kLoremIpsum.substr(10));

  char buff1[10];
  char buff2[20];
  char buff3[30];
  std::vector<folly::Range<char*>> buffers = {
      folly::Range<char*>(buff1, 10),
      folly::Range<char*>(nullptr, 20),
      folly::Range<char*>(buff2, 20),
      folly::Range<char*>(nullptr,30),
      folly::Range<char*>(buff3, 30)};
  ASSERT_EQ(10+20+20+30+30, readFile->preadv(0, buffers));
  ASSERT_EQ(std::string_view(buff1, sizeof(buff1)), kLoremIpsum.substr(0, 10));
  ASSERT_EQ(std::string_view(buff2, sizeof(buff2)), kLoremIpsum.substr(30, 20));
  ASSERT_EQ(std::string_view(buff3, sizeof(buff3)), kLoremIpsum.substr(80, 30));
}

TEST_F(GCSFileSystemTest, WriteAndReadFile) {
  const std::string newFile = "readWriteFile.txt";
  const std::string gcsFile = gcsURI(PreexistingBucketName(), newFile);

  filesystems::GCSFileSystem gcfs(TestGcsOptions());
  gcfs.initializeClient();
  auto writeFile = gcfs.openFileForWrite(gcsFile);
  std::string dataContent = "Dance me to your beauty with a burning violin"
        "Dance me through the panic till I'm gathered safely in"
        "Lift me like an olive branch and be my homeward dove"
        "Dance me to the end of love";

  EXPECT_EQ(writeFile->size(), 0);
  std::int64_t contentSize = dataContent.length();
  writeFile->append(dataContent.substr(0, 10));
  EXPECT_EQ(writeFile->size(), 10);
  writeFile->append(dataContent.substr(10, contentSize - 10));
  EXPECT_EQ(writeFile->size(), contentSize);
  writeFile->flush();
  writeFile->close();

  try {
    writeFile->append(dataContent.substr(0, 10));
    FAIL() << "Expected VeloxException";
  } catch (VeloxException const& err) {
    EXPECT_EQ(
        err.message(),
        std::string(
            "File is closed"));
  }

  auto readFile = gcfs.openFileForRead(gcsFile);
  std::int64_t size = readFile->size();
  EXPECT_EQ(readFile->size(), contentSize);
  EXPECT_EQ(readFile->pread(0, size), dataContent);
}

TEST_F(GCSFileSystemTest, openExistingFileForWrite)
{
  const std::string newFile = "readWriteFile.txt";
  const std::string gcsFile = gcsURI(PreexistingBucketName(), newFile);

  filesystems::GCSFileSystem gcfs(TestGcsOptions());
  gcfs.initializeClient();

  try {
    auto writeFile = gcfs.openFileForWrite(gcsFile);
    FAIL() << "Expected VeloxException";
  } catch (VeloxException const& err) {
    EXPECT_EQ(
        err.message(),
        std::string(
            "File already exists"));
  }
}
TEST_F(GCSFileSystemTest, missingFile) {
  const char* file = "newTest.txt";
  const std::string gcsFile = gcsURI(PreexistingBucketName(), file);
  filesystems::GCSFileSystem gcfs(TestGcsOptions());
  gcfs.initializeClient();
  try {
    gcfs.openFileForRead(gcsFile);
    FAIL() << "Expected VeloxException";
  } catch (VeloxException const& err) {
    EXPECT_EQ(
        err.message(),
        std::string(
            "Failed to get metadata for GCS object due to: Path:'gs://test1-gcs/newTest.txt', "
            "SDK Error Type:, GCS Status Code:Resource not found,  Message:'Permanent error in "
            "GetObjectMetadata: {\"code\":404,\"message\":\"{\\\"error\\\": {\\\"errors\\\": "
            "[{\\\"domain\\\": \\\"global\\\", \\\"message\\\": \\\"Live version of object "
            "test1-gcs/newTest.txt does not exist.\\\"}]}}\"}\n'"));
  }
}

TEST_F(GCSFileSystemTest, missingBucket) {
  filesystems::GCSFileSystem gcfs(TestGcsOptions());
  gcfs.initializeClient();
  try {
    const char* gcsFile = "gs://dummy/foo.txt";
    gcfs.openFileForRead(gcsFile);
    FAIL() << "Expected VeloxException";
  } catch (VeloxException const& err) {
    EXPECT_EQ(
        err.message(),
        std::string(
            "Failed to get metadata for GCS object due to: Path:'gs://dummy/foo.txt', "
            "SDK Error Type:, GCS Status Code:Resource not found,  Message:'Permanent "
            "error in GetObjectMetadata: {\"code\":404,\"message\":\"{\\\"error\\\": "
            "{\\\"errors\\\": [{\\\"domain\\\": \\\"global\\\", \\\"message\\\": "
            "\\\"Bucket dummy does not exist.\\\"}]}}\"}\n'"));
  }
}