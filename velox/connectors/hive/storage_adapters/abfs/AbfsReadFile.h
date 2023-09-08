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

#include <azure/storage/blobs.hpp>
#include <folly/executors/ThreadedExecutor.h>
#include <folly/futures/Future.h>
#include "velox/common/file/File.h"

namespace facebook::velox::filesystems::abfs {
using namespace Azure::Storage::Blobs;
class AbfsReadFile final : public ReadFile {
 public:
  constexpr static uint64_t kNaturalReadSize = 4 << 20; // 4M
  constexpr static uint64_t kReadConcurrency = 8;

  explicit AbfsReadFile(
      const std::string& path,
      const std::string& connectStr,
      const int32_t loadQuantum,
      const std::shared_ptr<folly::Executor> ioExecutor);

  void initialize();

  std::string_view pread(uint64_t offset, uint64_t length, void* buf)
      const final;

  std::string pread(uint64_t offset, uint64_t length) const final;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const final;

  void preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs) const final;

  uint64_t size() const final;

  uint64_t memoryUsage() const final;

  bool shouldCoalesce() const final;

  std::string getName() const final;

  uint64_t getNaturalReadSize() const final;

  static uint64_t calculateSplitQuantum(
      const uint64_t length,
      const uint64_t loadQuantum);

  static void splitRegion(
      const uint64_t length,
      const uint64_t loadQuantum,
      std::vector<std::tuple<uint64_t, uint64_t>>& range);

 private:
  void preadInternal(uint64_t offset, uint64_t length, char* pos) const;

  const std::string path_;
  const std::string connectStr_;
  const int32_t loadQuantum_;
  const std::shared_ptr<folly::Executor> ioExecutor_;
  std::string fileSystem_;
  std::string fileName_;
  std::unique_ptr<BlobClient> fileClient_;

  int64_t length_ = -1;
};
} // namespace facebook::velox::filesystems::abfs
