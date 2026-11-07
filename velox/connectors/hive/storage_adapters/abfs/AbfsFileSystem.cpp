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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsFileSystem.h"

#include <fmt/format.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <glog/logging.h>

#include "velox/connectors/hive/storage_adapters/abfs/AbfsAsyncRuntime.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsWriteFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"

namespace facebook::velox::filesystems {

namespace {

constexpr std::string_view kAsyncReadEnabled = "fs.azure.async-read.enabled";
constexpr std::string_view kDisableRetriesForTest =
    "fs.azure.async-read.disable-retries-for-test";
constexpr std::string_view kNumEventThreads =
    "fs.azure.async-read.event-threads";
constexpr std::string_view kMaxActiveRequests =
    "fs.azure.async-read.max-active-requests";
constexpr std::string_view kMaxQueuedRequests =
    "fs.azure.async-read.max-queued-requests";
constexpr std::string_view kNumAuthThreads = "fs.azure.async-read.auth-threads";
constexpr std::string_view kMaxQueuedAuthRefreshes =
    "fs.azure.async-read.max-queued-auth-refreshes";
constexpr std::string_view kMaxConnectionsPerEndpoint =
    "fs.azure.async-read.max-connections-per-endpoint";

} // namespace

AbfsFileSystem::AbfsFileSystem(std::shared_ptr<const config::ConfigBase> config)
    : FileSystem(config) {
  VELOX_CHECK_NOT_NULL(config.get());
  if (!config->get<bool>(std::string{kAsyncReadEnabled}, false)) {
    return;
  }

  VELOX_USER_CHECK(
      config->get<bool>(std::string{kDisableRetriesForTest}, false),
      "Native ABFS async reads require retries to be disabled by the Stage 3 "
      "test-only configuration gate");

  AbfsAsyncRuntimeOptions runtimeOptions;
  runtimeOptions.numEventThreads = config->get<size_t>(
      std::string{kNumEventThreads}, runtimeOptions.numEventThreads);
  runtimeOptions.maxActiveRequests = config->get<size_t>(
      std::string{kMaxActiveRequests}, runtimeOptions.maxActiveRequests);
  runtimeOptions.maxQueuedRequests = config->get<size_t>(
      std::string{kMaxQueuedRequests}, runtimeOptions.maxQueuedRequests);
  runtimeOptions.numAuthThreads = config->get<size_t>(
      std::string{kNumAuthThreads}, runtimeOptions.numAuthThreads);
  runtimeOptions.maxQueuedAuthRefreshes = config->get<size_t>(
      std::string{kMaxQueuedAuthRefreshes},
      runtimeOptions.maxQueuedAuthRefreshes);
  maxAsyncConnectionsPerEndpoint_ = config->get<size_t>(
      std::string{kMaxConnectionsPerEndpoint}, maxAsyncConnectionsPerEndpoint_);
  VELOX_USER_CHECK_GT(
      maxAsyncConnectionsPerEndpoint_,
      0,
      "Native ABFS async endpoint connection limit must be positive");
  asyncRuntime_ = std::make_shared<AbfsAsyncRuntime>(runtimeOptions);
}

std::string AbfsFileSystem::name() const {
  return "ABFS";
}

std::unique_ptr<ReadFile> AbfsFileSystem::openFileForRead(
    std::string_view path,
    const FileOptions& options) {
  auto abfsfile = std::make_unique<AbfsReadFile>(
      path, *config_, asyncRuntime_, maxAsyncConnectionsPerEndpoint_);
  abfsfile->initialize(options);
  return abfsfile;
}

std::unique_ptr<WriteFile> AbfsFileSystem::openFileForWrite(
    std::string_view path,
    const FileOptions& /*unused*/) {
  return std::make_unique<AbfsWriteFile>(path, *config_);
}
} // namespace facebook::velox::filesystems
