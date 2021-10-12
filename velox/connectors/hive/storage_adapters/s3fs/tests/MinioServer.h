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

#include "velox/exec/tests/TempFilePath.h"

#include "boost/process.hpp"

using namespace facebook::velox;

namespace {
// Adapted from the Apache Arrow library
constexpr char const* kMinioExecutableName{"minio"};
constexpr char const* kMinioAccessKey{"minio"};
constexpr char const* kMinioSecretKey{"miniopass"};
constexpr char const* kMinioConnectionString{"127.0.0.1:9000"};
} // namespace

// A minio server, managed as a child process
class MinioServer {
 public:
  MinioServer() {
    tempPath_ = ::exec::test::TempDirectoryPath::create();
  }
  void Start();
  void Stop();

  void addBucket(const char* bucket) {
    std::string path = tempPath_->path + "/" + bucket;
    mkdir(path.c_str(), S_IRWXU | S_IRWXG);
  }

  std::string getConnectionString() const {
    return connectionString_;
  }
  std::string getAccessKey() const {
    return accessKey_;
  }
  std::string getSecretKey() const {
    return secretKey_;
  }
  std::string getPath() const {
    return tempPath_->path;
  }

  std::unordered_map<std::string, std::string> getHiveConfig() const {
    return {
        {"hive.s3.aws-access-key", kMinioAccessKey},
        {"hive.s3.aws-secret-key", kMinioSecretKey},
        {"hive.s3.endpoint", kMinioConnectionString},
        {"hive.s3.ssl.enabled", "false"},
        {"hive.s3.path-style-access", "true"},
    };
  }

 private:
  std::shared_ptr<exec::test::TempDirectoryPath> tempPath_;
  std::string connectionString_ = kMinioConnectionString;
  std::string accessKey_ = kMinioAccessKey;
  std::string secretKey_ = kMinioSecretKey;
  std::shared_ptr<::boost::process::child> serverProcess_;
};

void MinioServer::Start() {
  boost::process::environment env = boost::this_process::environment();
  env["MINIO_ACCESS_KEY"] = getAccessKey();
  env["MINIO_SECRET_KEY"] = getSecretKey();

  auto exePath = boost::process::search_path(kMinioExecutableName);
  if (exePath.empty()) {
    VELOX_FAIL("Failed to find minio executable {}'", kMinioExecutableName);
  }

  try {
    serverProcess_ = std::make_shared<boost::process::child>(
        env,
        exePath,
        "server",
        "--quiet",
        "--compat",
        "--address",
        getConnectionString(),
        tempPath_->path.c_str());
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Minio server: {}", e.what());
  }
}

void MinioServer::Stop() {
  if (serverProcess_ && serverProcess_->valid()) {
    // Brutal shutdown
    serverProcess_->terminate();
    serverProcess_->wait();
  }
}
