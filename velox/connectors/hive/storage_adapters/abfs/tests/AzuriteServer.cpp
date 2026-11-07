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
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderImpl.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <thread>

namespace facebook::velox::filesystems {
namespace {

bool isPortReady(int64_t port) {
  const int socketFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socketFd < 0) {
    return false;
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const bool ready =
      ::connect(
          socketFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
      0;
  ::close(socketFd);
  return ready;
}

} // namespace

std::string AzuriteServer::URI() const {
  return fmt::format(
      "abfs://{}@{}.dfs.core.windows.net/", container_, account_);
}

std::string AzuriteServer::fileURI() const {
  return fmt::format(
      "abfs://{}@{}.dfs.core.windows.net/{}", container_, account_, file_);
}

// Return the hiveConfig for the Azurite instance.
// Specify configOverride map to update the default config map.
std::shared_ptr<const config::ConfigBase> AzuriteServer::hiveConfig(
    const std::unordered_map<std::string, std::string> configOverride) const {
  auto endpoint = fmt::format("http://127.0.0.1:{}/{}", port_, account_);
  std::unordered_map<std::string, std::string> config(
      {{"fs.azure.account.key.test.dfs.core.windows.net", key_},
       {kAzureBlobEndpoint, endpoint}});

  for (const auto& [key, value] : configOverride) {
    config[key] = value;
  }

  return std::make_shared<const config::ConfigBase>(std::move(config));
}

void AzuriteServer::start() {
  try {
    serverProcess_ = std::make_unique<boost::process::child>(
        env_, exePath_, commandOptions_);
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Azurite server: {}", e.what());
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!serverProcess_->running()) {
      VELOX_FAIL(
          "AzuriteServer process exited, code: {}",
          serverProcess_->exit_code());
    }
    if (isPortReady(port_)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  VELOX_FAIL("Timed out waiting for AzuriteServer to listen on port {}", port_);
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
AzuriteServer::AzuriteServer(int64_t port) : port_(port) {
  std::string dataLocation = fmt::format("/tmp/azurite_{}", port);
  std::string logFilePath = fmt::format("/tmp/azurite/azurite_{}.log", port);
  std::printf(
      "Launch azurite instance with port - %s, data location - %s, log file path - %s\n",
      std::to_string(port).c_str(),
      dataLocation.c_str(),
      logFilePath.c_str());
  commandOptions_ = {
      "--silent",
      "--blobPort",
      std::to_string(port),
      "--location",
      dataLocation,
      "--debug",
      logFilePath,
  };
  env_ = (boost::process::environment)boost::this_process::environment();
  env_["PATH"] = env_["PATH"].to_string() + std::string(kAzuriteSearchPath);
  env_["AZURITE_ACCOUNTS"] = fmt::format("{}:{}", account_, key_);
  auto path = env_["PATH"].to_vector();
  exePath_ = boost::process::search_path(
      kAzuriteServerExecutableName,
      std::vector<boost::filesystem::path>(path.begin(), path.end()));
  std::printf("AzuriteServer executable path: %s\n", exePath_.c_str());
  if (exePath_.empty()) {
    VELOX_FAIL(
        "Failed to find azurite executable {}'", kAzuriteServerExecutableName);
  }
}

void AzuriteServer::addFile(std::string source) {
  const auto abfsPath = std::make_shared<AbfsPath>(fileURI());
  auto clientProvider = SharedKeyAzureClientProvider();
  auto containerClient = BlobContainerClient::CreateFromConnectionString(
      clientProvider.connectionString(abfsPath, *hiveConfig()), container_);
  containerClient.CreateIfNotExists();
  auto blobClient = containerClient.GetBlockBlobClient(file_);
  blobClient.UploadFrom(source);
}

AzuriteServer::~AzuriteServer() {
  // stop();
}
} // namespace facebook::velox::filesystems
