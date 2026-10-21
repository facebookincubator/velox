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

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/container/F14Map.h>
#include <folly/synchronization/CallOnce.h>

#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector::hive::iceberg::test {

/// FileSystem wrapper that records the FileOptions::fileReadOps of every
/// openFileForRead call and otherwise delegates to the underlying real file
/// system. Used to observe which entries reach the FileSystem boundary. Paths
/// are prefixed with scheme() and delegated to the local file system after
/// stripping the prefix.
class RecordingFileSystem : public filesystems::FileSystem {
 public:
  explicit RecordingFileSystem(std::shared_ptr<const config::ConfigBase> config)
      : FileSystem(std::move(config)) {}

  static std::string scheme() {
    return "recording:";
  }

  std::string name() const override {
    return "RecordingFS";
  }

  std::string_view extractPath(std::string_view path) const override {
    return path.find(scheme()) == 0 ? path.substr(scheme().length()) : path;
  }

  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view path,
      const filesystems::FileOptions& options = {}) override {
    {
      std::lock_guard<std::mutex> l(mutex_);
      capturedReadOps_.push_back(options.fileReadOps);
    }
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->openFileForRead(delegated, options);
  }

  std::unique_ptr<WriteFile> openFileForWrite(
      std::string_view path,
      const filesystems::FileOptions& options = {}) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->openFileForWrite(delegated, options);
  }

  void remove(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->remove(delegated);
  }

  void rename(
      std::string_view oldPath,
      std::string_view newPath,
      bool overwrite) override {
    const std::string from{extractPath(oldPath)};
    const std::string to{extractPath(newPath)};
    delegate(from)->rename(from, to, overwrite);
  }

  bool exists(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->exists(delegated);
  }

  bool isDirectory(std::string_view path) const override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->isDirectory(delegated);
  }

  std::vector<std::string> list(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    return delegate(delegated)->list(delegated);
  }

  void mkdir(
      std::string_view path,
      const filesystems::DirectoryOptions& options = {}) override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->mkdir(delegated, options);
  }

  void rmdir(std::string_view path) override {
    const std::string delegated{extractPath(path)};
    delegate(delegated)->rmdir(delegated);
  }

  void clear() {
    std::lock_guard<std::mutex> l(mutex_);
    capturedReadOps_.clear();
  }

  /// Returns the recorded value for 'key' across all openFileForRead calls, or
  /// std::nullopt if no call carried it.
  std::optional<std::string> readOpValue(const std::string& key) {
    std::lock_guard<std::mutex> l(mutex_);
    for (const auto& ops : capturedReadOps_) {
      const auto it = ops.find(key);
      if (it != ops.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  }

 private:
  static std::shared_ptr<FileSystem> delegate(const std::string& path) {
    return filesystems::getFileSystem(path, nullptr);
  }

  mutable std::mutex mutex_;
  std::vector<folly::F14FastMap<std::string, std::string>> capturedReadOps_{};
};

/// Returns the one RecordingFileSystem shared by every test file in the binary.
/// getFileSystem() returns the first registered match for a path, so a second
/// instance registered for the same scheme would never receive a call. Inline
/// functions with external linkage share their static locals across
/// translation units, which keeps this instance and the registration below
/// unique.
inline std::shared_ptr<RecordingFileSystem> recordingFileSystem() {
  static auto fs = std::make_shared<RecordingFileSystem>(nullptr);
  return fs;
}

/// Registers recordingFileSystem() for paths that start with
/// RecordingFileSystem::scheme(). Safe to call more than once.
inline void registerRecordingFileSystem() {
  static folly::once_flag once;
  folly::call_once(once, [] {
    filesystems::registerFileSystem(
        [](std::string_view path) {
          return path.find(RecordingFileSystem::scheme()) == 0;
        },
        [](std::shared_ptr<const config::ConfigBase>, std::string_view) {
          return recordingFileSystem();
        });
  });
}

} // namespace facebook::velox::connector::hive::iceberg::test
