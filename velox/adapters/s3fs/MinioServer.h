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

// Adapted from the Apache Arrow library
constexpr char const* kMinioExecutableName{"minio"};
constexpr char const* kMinioAccessKey{"minio"};
constexpr char const* kMinioSecretKey{"miniopass"};
constexpr char const* kMinioConnectionString{"127.0.0.1:9000"};

// A minio server, managed as a child process
class MinioServer {
 public:
  MinioServer() {
    temp_path_ = ::exec::test::TempDirectoryPath::create();
  }
  void Start();
  void Stop();

  void addBucket(const char* bucket) {
    std::string path = temp_path_->path + "/" + bucket;
    mkdir(path.c_str(), S_IRWXU | S_IRWXG);
  }

  std::string getConnectionString() const {
    return connection_string_;
  }
  std::string getAccessKey() const {
    return access_key_;
  }
  std::string getSecretKey() const {
    return secret_key_;
  }
  std::string getPath() const {
    return temp_path_->path;
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
  std::shared_ptr<exec::test::TempDirectoryPath> temp_path_;
  std::string connection_string_ = kMinioConnectionString;
  std::string access_key_ = kMinioAccessKey;
  std::string secret_key_ = kMinioSecretKey;
  std::shared_ptr<::boost::process::child> server_process_;
};

void MinioServer::Start() {
  boost::process::environment env = boost::this_process::environment();
  env["MINIO_ACCESS_KEY"] = getAccessKey();
  env["MINIO_SECRET_KEY"] = getSecretKey();

  auto exe_path = boost::process::search_path(kMinioExecutableName);
  if (exe_path.empty()) {
    VELOX_FAIL("Failed to find minio executable {}'", kMinioExecutableName);
  }

  try {
    server_process_ = std::make_shared<boost::process::child>(
        env,
        exe_path,
        "server",
        "--quiet",
        "--compat",
        "--address",
        getConnectionString(),
        temp_path_->path.c_str());
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Minio server: {}", e.what());
  }
}

void MinioServer::Stop() {
  if (server_process_ && server_process_->valid()) {
    // Brutal shutdown
    server_process_->terminate();
    server_process_->wait();
  }
}
