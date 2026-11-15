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

#include "velox/connectors/hive/storage_adapters/hdfs/tests/HdfsMiniCluster.h"

#include "velox/exec/tests/utils/PortUtil.h"

#include <filesystem>
#include <thread>

namespace facebook::velox::filesystems::test {
void HdfsMiniCluster::start() {
  try {
    serverProcess_ = std::make_unique<boost::process::child>(
        env_,
        exePath_,
        kJarCommand,
        env_["HADOOP_HOME"].to_string() + kMiniclusterJar,
        kMiniclusterCommand,
        kNoMapReduceOption,
        kFormatNameNodeOption,
        kHttpPortOption,
        httpPort_,
        kNameNodePortOption,
        nameNodePort_,
        kConfigurationOption,
        kTurnOffPermissions,
        kWriteDetailsOption,
        detailsPath_);
  } catch (const std::exception& e) {
    VELOX_FAIL("Failed to launch Minicluster server: {}", e.what());
  }
  // The NameNode port opens before the DataNode registers, so a write right
  // after the port opens can fail. The minicluster writes the details file
  // once the cluster is active.
  const auto deadline = std::chrono::steady_clock::now() + kStartTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!serverProcess_->running()) {
      VELOX_FAIL(
          "Minicluster process exited, code: {}", serverProcess_->exit_code());
    }
    std::error_code ec;
    if (std::filesystem::file_size(detailsPath_, ec) > 0 && !ec) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  VELOX_FAIL(
      "Timed out after {} seconds waiting for Minicluster to start",
      kStartTimeout.count());
}

void HdfsMiniCluster::stop() {
  if (serverProcess_ && serverProcess_->valid()) {
    serverProcess_->terminate();
    serverProcess_->wait();
    serverProcess_.reset();
  }
}

bool HdfsMiniCluster::isRunning() {
  if (serverProcess_) {
    return true;
  }
  return false;
}

// requires hadoop executable to be on the PATH
HdfsMiniCluster::HdfsMiniCluster() {
  env_ = (boost::process::environment)boost::this_process::environment();
  env_["PATH"] = env_["PATH"].to_string() + kHadoopSearchPath;
  auto path = env_["PATH"].to_vector();
  exePath_ = boost::process::search_path(
      kMiniClusterExecutableName,
      std::vector<boost::filesystem::path>(path.begin(), path.end()));
  if (exePath_.empty()) {
    VELOX_FAIL(
        "Failed to find minicluster executable {}'",
        kMiniClusterExecutableName);
  }
  constexpr auto kHostAddressTemplate = "hdfs://{}:{}";
  auto ports = facebook::velox::exec::test::PortUtil::getFreePorts(2);
  nameNodePort_ = fmt::format("{}", ports[0]);
  httpPort_ = fmt::format("{}", ports[1]);
  filesystemUrl_ = fmt::format(kHostAddressTemplate, host(), nameNodePort_);
  boost::filesystem::path hadoopHomeDirectory = exePath_;
  hadoopHomeDirectory.remove_leaf().remove_leaf();
  setupEnvironment(hadoopHomeDirectory.string());
}

void HdfsMiniCluster::addFile(std::string source, std::string destination) {
  auto filePutProcess = std::make_shared<boost::process::child>(
      env_,
      exePath_,
      kFilesystemCommand,
      kFilesystemUrlOption,
      filesystemUrl_,
      kFilePutOption,
      source,
      destination);
  bool isExited =
      filePutProcess->wait_for(std::chrono::duration<int, std::milli>(15000));
  if (!isExited) {
    VELOX_FAIL(
        "Failed to add file to hdfs, exit code: {}",
        filePutProcess->exit_code());
  }
}

HdfsMiniCluster::~HdfsMiniCluster() {
  stop();
}

void HdfsMiniCluster::setupEnvironment(const std::string& homeDirectory) {
  env_["HADOOP_HOME"] = homeDirectory;
  env_["HADOOP_INSTALL"] = homeDirectory;
  env_["HADOOP_MAPRED_HOME"] = homeDirectory;
  env_["HADOOP_COMMON_HOME"] = homeDirectory;
  env_["HADOOP_HDFS_HOME"] = homeDirectory;
  env_["YARN_HOME"] = homeDirectory;
  env_["HADOOP_COMMON_LIB_NATIVE_DIR"] = homeDirectory + "/lib/native";
  env_["HADOOP_CONF_DIR"] = homeDirectory;
  env_["HADOOP_PREFIX"] = homeDirectory;
  env_["HADOOP_LIBEXEC_DIR"] = homeDirectory + "/libexec";
  env_["HADOOP_CONF_DIR"] = homeDirectory + "/etc/hadoop";
}
} // namespace facebook::velox::filesystems::test
