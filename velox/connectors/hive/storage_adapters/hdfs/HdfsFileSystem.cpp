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
#include <hdfs/hdfs.h>
#include <mutex>
#include "folly/concurrency/ConcurrentHashMap.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/storage_adapters/hdfs/HdfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/hdfs/HdfsWriteFile.h"
#include "velox/core/Context.h"

namespace facebook::velox::filesystems {
std::string_view HdfsFileSystem::kScheme("hdfs://");
std::mutex mtx;

class HdfsFileSystem::Impl {
 public:
  explicit Impl(const Config* config) {
    auto endpointInfo = getServiceEndpoint(config);
    auto builder = hdfsNewBuilder();
    hdfsBuilderSetNameNode(builder, endpointInfo.host.c_str());
    hdfsBuilderSetNameNodePort(builder, endpointInfo.port);
    hdfsClient_ = hdfsBuilderConnect(builder);
    VELOX_CHECK_NOT_NULL(
        hdfsClient_,
        "Unable to connect to HDFS, got error: {}.",
        hdfsGetLastError())
  }

  explicit Impl(const Config* config, const HdfsServiceEndpoint& endpoint) {
    auto builder = hdfsNewBuilder();
    hdfsBuilderSetNameNode(builder, endpoint.host.c_str());
    hdfsBuilderSetNameNodePort(builder, endpoint.port);
    hdfsClient_ = hdfsBuilderConnect(builder);
    VELOX_CHECK_NOT_NULL(
        hdfsClient_,
        "Unable to connect to HDFS: {}, got error: {}.",
        endpoint.identity,
        hdfsGetLastError())
  }

  ~Impl() {
    LOG(INFO) << "Disconnecting HDFS file system";
    int disconnectResult = hdfsDisconnect(hdfsClient_);
    if (disconnectResult != 0) {
      LOG(WARNING) << "hdfs disconnect failure in HdfsReadFile close: "
                   << errno;
    }
  }

  hdfsFS hdfsClient() {
    return hdfsClient_;
  }

 private:
  hdfsFS hdfsClient_;
};

HdfsFileSystem::HdfsFileSystem(const std::shared_ptr<const Config>& config)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get());
}

HdfsFileSystem::HdfsFileSystem(
    const std::shared_ptr<const Config>& config,
    const HdfsServiceEndpoint& endpoint)
    : FileSystem(config) {
  impl_ = std::make_shared<Impl>(config.get(), endpoint);
}

std::string HdfsFileSystem::name() const {
  return "HDFS";
}

std::unique_ptr<ReadFile> HdfsFileSystem::openFileForRead(
    std::string_view path,
    const FileOptions& /*unused*/) {
  if (path.find(kScheme) == 0) {
    path.remove_prefix(kScheme.length());
  }
  if (auto index = path.find('/')) {
    path.remove_prefix(index);
  }

  return std::make_unique<HdfsReadFile>(impl_->hdfsClient(), path);
}

std::unique_ptr<WriteFile> HdfsFileSystem::openFileForWrite(
    std::string_view path,
    const FileOptions& /*unused*/) {
  return std::make_unique<HdfsWriteFile>(impl_->hdfsClient(), path);
}

bool HdfsFileSystem::isHdfsFile(const std::string_view filePath) {
  return filePath.find(kScheme) == 0;
}

/**
 * Get hdfs endpoint from config. This is applicable to the case that only one
 * hdfs endpoint will be used.
 */
HdfsServiceEndpoint HdfsFileSystem::getServiceEndpoint(const Config* config) {
  auto hdfsHost = config->get("hive.hdfs.host");
  VELOX_CHECK(
      hdfsHost.hasValue(),
      "hdfsHost is empty, configuration missing for hdfs host");
  auto hdfsPort = config->get("hive.hdfs.port");
  VELOX_CHECK(
      hdfsPort.hasValue(),
      "hdfsPort is empty, configuration missing for hdfs port");
  HdfsServiceEndpoint endpoint{*hdfsHost, *hdfsPort};
  return endpoint;
}

/**
 * Get hdfs endpoint from a given file path, instead of getting a fixed one from
 * configuration.
 */
HdfsServiceEndpoint HdfsFileSystem::getServiceEndpoint(
    const std::string_view filePath) {
  auto index1 = filePath.find('/', kScheme.size() + 1);
  std::string hdfsIdentity{
      filePath.data(), kScheme.size(), index1 - kScheme.size()};
  VELOX_CHECK(
      !hdfsIdentity.empty(),
      "hdfsIdentity is empty, expect hdfs endpoint host[:port] is contained in file path");
  auto index2 = hdfsIdentity.find(':', 0);
  // In HDFS HA mode, the hdfsIdentity is a nameservice ID with no port.
  if (index2 == std::string::npos) {
    HdfsServiceEndpoint endpoint{hdfsIdentity, ""};
    return endpoint;
  }
  std::string host{hdfsIdentity.data(), 0, index2};
  std::string port{
      hdfsIdentity.data(), index2 + 1, hdfsIdentity.size() - index2 - 1};
  HdfsServiceEndpoint endpoint{host, port};
  return endpoint;
}

static std::function<std::shared_ptr<FileSystem>(
    std::shared_ptr<const Config>,
    std::string_view)>
    filesystemGenerator = [](std::shared_ptr<const Config> properties,
                             std::string_view filePath) {
      static folly::ConcurrentHashMap<std::string, std::shared_ptr<FileSystem>>
          filesystems;
      static folly::
          ConcurrentHashMap<std::string, std::shared_ptr<folly::once_flag>>
              hdfsInitiationFlags;
      auto endpoint = HdfsFileSystem::getServiceEndpoint(filePath);
      std::string hdfsIdentity = endpoint.identity;
      if (filesystems.find(hdfsIdentity) != filesystems.end()) {
        return filesystems[hdfsIdentity];
      }
      std::unique_lock<std::mutex> lk(mtx, std::defer_lock);
      if (hdfsInitiationFlags.find(hdfsIdentity) == hdfsInitiationFlags.end()) {
        lk.lock();
        if (hdfsInitiationFlags.find(hdfsIdentity) ==
            hdfsInitiationFlags.end()) {
          std::shared_ptr<folly::once_flag> initiationFlagPtr =
              std::make_shared<folly::once_flag>();
          hdfsInitiationFlags.insert(hdfsIdentity, initiationFlagPtr);
        }
        lk.unlock();
      }
      folly::call_once(
          *hdfsInitiationFlags[hdfsIdentity].get(),
          [&properties, endpoint, hdfsIdentity]() {
            auto filesystem =
                std::make_shared<HdfsFileSystem>(properties, endpoint);
            filesystems.insert(hdfsIdentity, filesystem);
          });
      return filesystems[hdfsIdentity];
    };

void HdfsFileSystem::remove(std::string_view path) {
  VELOX_UNSUPPORTED("Does not support removing files from hdfs");
}

void registerHdfsFileSystem() {
  registerFileSystem(HdfsFileSystem::isHdfsFile, filesystemGenerator);
}
} // namespace facebook::velox::filesystems
