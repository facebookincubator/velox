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
#include "HdfsFileSystem.h"
#include <boost/algorithm/string/find.hpp>
#include "libhdfs3/src/client/hdfs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/core/Context.h"

namespace facebook::velox::filesystems {
folly::once_flag hdfsInitiationFlag;
std::string_view HdfsFileSystem::scheme("hdfs:");

HdfsFileSystem::HdfsFileSystem(const std::shared_ptr<const Config>& config)
    : FileSystem(config) {
  auto endpointInfo = getServiceEndpoint(config);
  auto host = endpointInfo.begin()->host;
  auto port = endpointInfo.begin()->port;
  auto builder = hdfsNewBuilder();
  hdfsBuilderConfSetStr(builder, "input.read.timeout", "60000");
  hdfsBuilderConfSetStr(builder, "input.connect.timeout", "60000");
  hdfsBuilderSetNameNode(builder, host.c_str());
  hdfsBuilderSetNameNodePort(builder, port);
  hdfsClient_ = hdfsBuilderConnect(builder);
  VELOX_CHECK_NOT_NULL(
      hdfsClient_,
      "Unable to connect to HDFS, got error:{}.",
      hdfsGetLastError())
}
std::vector<cpputil::consul::ServiceEndpoint>
HdfsFileSystem::getServiceEndpoint(
    const std::shared_ptr<const Config>& config) {
  auto configuration = config.get();
  if (configuration->get("hdfs_host").hasValue() &&
      configuration->get("hdfs_port").hasValue()) {
    auto endpoint = new cpputil::consul::ServiceEndpoint{
        *configuration->get("hdfs_host"),
        atoi(configuration->get("hdfs_port")->data())};
    std::vector<cpputil::consul::ServiceEndpoint> endpoints{*endpoint};
    return endpoints;
  }
  auto consul =
      std::make_unique<cpputil::consul::ServiceDiscovery>("127.0.0.1", 2280);
  return consul->lookup("nnproxy", cpputil::consul::NORMAL);
}

HdfsFileSystem::~HdfsFileSystem() {
  LOG(INFO) << "Disconnecting HDFS file system";
  int disconnectResult = hdfsDisconnect(hdfsClient_);
  if (disconnectResult != 0) {
    LOG(WARNING) << "hdfs disconnect failure in HdfsReadFile close: " << errno;
  }
}

std::string HdfsFileSystem::name() const {
  return "HDFS";
}

std::unique_ptr<ReadFile> HdfsFileSystem::openFileForRead(
    std::string_view path) {
  // Note: our hdfs paths are prefixed with hdfs://haruna*/ depending on region.
  // Since consul already resolves the correct node we do not need haruna prefix
  // so we cut everything upto the 3rd "/" The hdfs client also does not need
  // the protocol: "hdfs://"
  if (path.find("/") == 0) {
    return std::make_unique<HdfsReadFile>(hdfsClient_, path);
  }
  auto result = boost::find_nth(path, "/", 2); // zero indexed
  auto sanitizedPath =
      path.substr(std::distance(path.begin(), result.begin()), path.length());
  return std::make_unique<HdfsReadFile>(hdfsClient_, sanitizedPath);
}

std::unique_ptr<WriteFile> HdfsFileSystem::openFileForWrite(
    std::string_view path) {
  VELOX_UNSUPPORTED("Does not support write to HDFS");
}

std::function<bool(std::string_view)> HdfsFileSystem::schemeMatcher() {
  // Note: presto behavior is to prefix local paths with 'file:'.
  // Check for that prefix and prune to absolute regular paths as needed.
  return [](std::string_view filename) {
    return filename.find("/") == 0 || filename.find(scheme) == 0;
  };
}

std::function<std::shared_ptr<FileSystem>(std::shared_ptr<const Config>)>
HdfsFileSystem::fileSystemGenerator() {
  return [](std::shared_ptr<const Config> properties) {
    // One instance of Hdfs FileSystem is sufficient.
    // Initialize on first access and reuse after that.
    static std::shared_ptr<FileSystem> fs;
    folly::call_once(hdfsInitiationFlag, [&properties]() {
      fs = std::make_shared<HdfsFileSystem>(properties);
    });
    return fs;
  };
}

void registerHdfsFileSystem() {
  registerFileSystem(
      HdfsFileSystem::schemeMatcher(), HdfsFileSystem::fileSystemGenerator());
}
} // namespace facebook::velox::filesystems
