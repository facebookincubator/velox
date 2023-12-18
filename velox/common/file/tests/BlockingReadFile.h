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
#include <queue>

#include "folly/synchronization/Baton.h"
#include "velox/common/file/File.h"

namespace facebook::velox::tests::utils {

class BlockingReadFile : public velox::ReadFile {
 public:
  explicit BlockingReadFile(velox::ReadFile* file) : file_{file} {}

  // Signal the next preadv in the queue to complete. Returns true if there was
  // a read waiting in the queue.
  bool signalPreadv();

  // Set number of reads to skip blocking on. If this is 0, the
  // next read will block until signalPreadv() is called. If this is > 0, the
  // next read will not block and decrement the counter. Only usable when there
  // are no reads currently awaiting signal.
  void setNumberOfReadNextWithoutBlocking(uint32_t numberOfReads);

  // Resets skip block counter to 0, forcing the next call to preadv to block
  // until signaled.
  void forceBlockOnNextRead();

  void preadv(
      folly::Range<const velox::common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs) const override;

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers) const override;

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* FOLLY_NONNULL buf) const override {
    return file_->pread(offset, length, buf);
  }

  bool shouldCoalesce() const override {
    return false;
  }

  std::string pread(uint64_t offset, uint64_t length) const override {
    return file_->pread(offset, length);
  }

  uint64_t size() const final {
    return file_->size();
  }

  uint64_t memoryUsage() const final {
    return size();
  }

  std::string getName() const override {
    return "<BlockingReadFile>";
  }

  uint64_t getNaturalReadSize() const override {
    return 1024;
  }

 private:
  void waitOnReadCondition() const;

  velox::ReadFile* file_;

  mutable folly::Synchronized<std::queue<std::shared_ptr<folly::Baton<>>>>
      batons_;

  mutable folly::Synchronized<uint32_t> readsToSkipBlocking_;
};

} // namespace facebook::velox::tests::utils
