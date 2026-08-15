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

namespace facebook::velox {

/// Current implementation for the local version is quite simple (e.g. no
/// internal arenaing), as local disk writes are expected to be cheap. Local
/// files match against any filepath starting with '/'.
class LocalReadFile final : public ReadFile {
 public:
  /// Opens 'path' for reading. 'executor', when set, backs the asynchronous
  /// preadvAsync(). When 'bufferIo' is false the file is opened with O_DIRECT.
  /// 'useIoUring' enables positioned preadv() batches through a per-thread
  /// io_uring reader. It is valid only when 'bufferIo' is false.
  explicit LocalReadFile(
      std::string_view path,
      folly::Executor* executor = nullptr,
      bool bufferIo = true,
      bool useIoUring = false);

  ~LocalReadFile();

  using ReadFile::preadv;

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const final;

  uint64_t size() const final;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const final;

  uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<const folly::Range<char*>*> buffers,
      const FileIoContext& context = {}) const override;

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override;

  bool hasPreadvAsync() const override {
    return executor_ != nullptr;
  }

  uint64_t memoryUsage() const final;

  bool shouldCoalesce() const final {
    return false;
  }

  std::string getName() const override {
    if (path_.empty()) {
      return "<LocalReadFile>";
    }
    return path_;
  }

  uint64_t getNaturalReadSize() const override {
    return 10 << 20;
  }

  bool directIo(uint64_t& alignment) const override;

 private:
  void preadInternal(uint64_t offset, uint64_t length, char* pos) const;

  folly::Executor* const executor_{nullptr};
  // True when the file is opened with O_DIRECT and reads require alignment.
  const bool directIo_{false};
  // True when io_uring is requested for direct-I/O positioned read batches.
  const bool useIoUring_{false};
  const std::string path_;
  const int32_t fd_;
  const long size_;
  const uint64_t readAlignment_;
};

class LocalWriteFile final : public WriteFile {
 public:
  struct Attributes {
    // If set to true, the file will not be subject to copy-on-write updates.
    // This flag has an effect only on filesystems that support copy-on-write
    // semantics, such as Btrfs.
    static constexpr std::string_view kNoCow{"write-on-copy-disabled"};
    static constexpr bool kDefaultNoCow{false};

    static bool cowDisabled(
        const std::unordered_map<std::string, std::string>& attrs);
  };

  // An error is thrown is a file already exists at |path|,
  // unless flag shouldThrowOnFileAlreadyExists is false
  explicit LocalWriteFile(
      std::string_view path,
      bool shouldCreateParentDirectories = false,
      bool shouldThrowOnFileAlreadyExists = true,
      bool bufferIo = true);

  ~LocalWriteFile();

  void append(std::string_view data) final;

  void append(std::unique_ptr<folly::IOBuf> data) final;

  void write(const std::vector<iovec>& iovecs, int64_t offset, int64_t length)
      final;

  void truncate(int64_t newSize) final;

  void flush() final;

  void setAttributes(
      const std::unordered_map<std::string, std::string>& attributes) final;

  std::unordered_map<std::string, std::string> getAttributes() const final;

  void close() final;

  uint64_t size() const final {
    return size_;
  }

  const std::string getName() const final {
    return path_;
  }

 private:
  // File descriptor.
  int32_t fd_{-1};
  std::string path_;
  uint64_t size_{0};
  std::unordered_map<std::string, std::string> attributes_{};
  bool closed_{false};
};

} // namespace facebook::velox
