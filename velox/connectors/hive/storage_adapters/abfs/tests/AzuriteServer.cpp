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

#include "velox/connectors/hive/storage_adapters/abfs/tests/AzuriteServer.h"
#include <boost/date_time/posix_time/posix_time.hpp>

namespace facebook::velox::filesystems::test {
void AzuriteServer::start() {
  try {
    serverProcess_ =
        std::make_unique<boost::process::child>(env_, exePath_, CommandOptions);
    serverProcess_->wait_for(std::chrono::duration<int, std::milli>(30000));
    VELOX_CHECK_EQ(
        serverProcess_->exit_code(),
        383,
        "AzuriteServer process exited, code: ",
        serverProcess_->exit_code())
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Azurite server: {}", e.what());
  }
}

void AzuriteServer::stop() {
  if (serverProcess_ && serverProcess_->valid()) {
    serverProcess_->terminate();
    serverProcess_->wait();
    serverProcess_.reset();
  }
}

bool AzuriteServer::isRunning() {
  if (serverProcess_) {
    return true;
  }
  return false;
}

// requires azurite executable to be on the PATH
AzuriteServer::AzuriteServer() {
  env_ = (boost::process::environment)boost::this_process::environment();
  env_["PATH"] = env_["PATH"].to_string() + AzuriteSearchPath;
  env_["AZURITE_ACCOUNTS"] =
      fmt::format("{}:{}", AzuriteAccountName, AzuriteAccountKey);
  auto path = env_["PATH"].to_vector();
  exePath_ = boost::process::search_path(
      AzuriteServerExecutableName,
      std::vector<boost::filesystem::path>(path.begin(), path.end()));
  std::printf("AzuriteServer executable path: %s\n", exePath_.c_str());
  if (exePath_.empty()) {
    VELOX_FAIL(
        "Failed to find azurite executable {}'", AzuriteServerExecutableName);
  }
}

void AzuriteServer::addFile(
    std::string source,
    std::string destination,
    std::string connectionString) {
  auto containerClient = BlobContainerClient::CreateFromConnectionString(
      connectionString, AzuriteContainerName);
  containerClient.CreateIfNotExists();
  auto blobClient = containerClient.GetBlockBlobClient(destination);
  blobClient.UploadFrom(source);
}

AzuriteServer::~AzuriteServer() {
  // stop();
}
} // namespace facebook::velox::filesystems::test
