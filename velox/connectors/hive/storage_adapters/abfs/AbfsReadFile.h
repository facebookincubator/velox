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

#include "velox/common/file/File.h"

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::filesystems {
class AbfsAsyncRuntime;

class AbfsReadFile final : public ReadFile {
 public:
  explicit AbfsReadFile(
      std::string_view path,
      const config::ConfigBase& config,
      std::shared_ptr<AbfsAsyncRuntime> asyncRuntime = nullptr,
      size_t maxAsyncConnectionsPerEndpoint = 32);

  void initialize(const FileOptions& options);

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const final;

  std::string pread(
      uint64_t offset,
      uint64_t length,
      const FileIoContext& context = {}) const final;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const final;

  uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs,
      const FileIoContext& context = {}) const final;

  /// Returns true when the native ABFS fiber read path is available.
  bool hasPreadvAsync() const final;

  /// Submits one vectored read to the native ABFS fiber runtime.
  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const final;

  uint64_t size() const final;

  uint64_t memoryUsage() const final;

  bool shouldCoalesce() const final;

  std::string getName() const final;

  uint64_t getNaturalReadSize() const final;

 protected:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace facebook::velox::filesystems
