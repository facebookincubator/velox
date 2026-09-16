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

#pragma once

#include "velox/common/config/Config.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/tests/utils/PortUtil.h"

#include "boost/process.hpp"

using namespace facebook::velox;

using TempDirectoryPath = common::testutil::TempDirectoryPath;

namespace {
constexpr char const* kSiloExecutableName{"silo"};
constexpr char const* kSiloAccessKey{"minio"};
constexpr char const* kSiloSecretKey{"miniopass"};
} // namespace

// A silo server, managed as a child process.
// Adapted from the Apache Arrow library.
class SiloServer {
 public:
  SiloServer() : tempPath_(TempDirectoryPath::create()) {
    constexpr auto kHostAddressTemplate = "127.0.0.1:{}";
    auto ports = facebook::velox::exec::test::getFreePorts(2);
    connectionString_ = fmt::format(kHostAddressTemplate, ports[0]);
    consoleAddress_ = fmt::format(kHostAddressTemplate, ports[1]);
  }

  void start();

  void stop();

  void addBucket(const char* bucket) {
    const std::string path = tempPath_->getPath() + "/" + bucket;
    mkdir(path.c_str(), S_IRWXU | S_IRWXG);
  }

  std::string path() const {
    return tempPath_->getPath();
  }

  std::shared_ptr<const config::ConfigBase> s3Config(
      const std::unordered_map<std::string, std::string> configOverride = {})
      const {
    std::unordered_map<std::string, std::string> config({
        {"s3.aws-access-key", accessKey_},
        {"s3.aws-secret-key", secretKey_},
        {"s3.endpoint", connectionString_},
        {"s3.ssl.enabled", "false"},
        {"s3.path-style-access", "true"},
    });

    // Update the default config map with the supplied configOverride map
    for (const auto& [configName, configValue] : configOverride) {
      config[configName] = configValue;
    }

    return std::make_shared<const config::ConfigBase>(std::move(config));
  }

 private:
  const std::shared_ptr<TempDirectoryPath> tempPath_;
  std::string connectionString_;
  std::string consoleAddress_;
  const std::string accessKey_ = kSiloAccessKey;
  const std::string secretKey_ = kSiloSecretKey;
  std::shared_ptr<::boost::process::child> serverProcess_;
};

void SiloServer::start() {
  boost::process::environment env = boost::this_process::environment();
  // Silo honors the legacy MinIO credential variables; renaming these
  // breaks server authentication.
  env["MINIO_ACCESS_KEY"] = accessKey_;
  env["MINIO_SECRET_KEY"] = secretKey_;

  auto exePath = boost::process::search_path(kSiloExecutableName);
  if (exePath.empty()) {
    VELOX_FAIL("Failed to find silo executable {}'", kSiloExecutableName);
  }

  const auto path = tempPath_->getPath();
  try {
    serverProcess_ = std::make_shared<boost::process::child>(
        env,
        exePath,
        "server",
        "--quiet",
        "--compat",
        "--address",
        connectionString_,
        "--console-address",
        consoleAddress_,
        path.c_str());
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Silo server: {}", e.what());
  }
}

void SiloServer::stop() {
  if (serverProcess_ && serverProcess_->valid()) {
    // Brutal shutdown
    serverProcess_->terminate();
    serverProcess_->wait();
  }
}
