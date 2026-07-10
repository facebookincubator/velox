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

#include <folly/futures/Future.h>

#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/PositionProvider.h"
#include "velox/dwio/common/wrap/zero-copy-stream-wrapper.h"

namespace facebook::velox::dwio::common {

void printBuffer(std::ostream& out, const char* buffer, uint64_t length);

/**
 * A subclass of Google's ZeroCopyInputStream that supports seek.
 * By extending Google's class, we get the ability to pass it directly
 * to the protobuf readers.
 */
class SeekableInputStream : public google::protobuf::io::ZeroCopyInputStream {
 public:
  ~SeekableInputStream() override = default;

  virtual void seekToPosition(PositionProvider& position) = 0;

  virtual std::string getName() const = 0;

  // Returns the number of position values this input stream uses to identify an
  // ORC/DWRF stream address.
  virtual size_t positionSize() const = 0;

  virtual bool SkipInt64(int64_t count) = 0;

  bool Skip(int32_t count) final override {
    VELOX_FAIL("Use SkipInt64 instead: {}", count);
  }

  void readFully(char* buffer, size_t bufferSize);

  /// Returns true if a subsequent read from this stream will not block
  /// on IO. Default = true (synchronous / already-loaded streams).
  ///
  /// Async-capable backends (CacheInputStream over CoalescedLoad,
  /// DirectInputStream) override to report the underlying load's state.
  /// Used by the parquet reader to reorder per-column / per-row-group
  /// CPU work so that streams whose IO is already complete are
  /// processed first, maximizing CPU/IO overlap on cold-cache reads
  /// with long-tail region latency.
  virtual bool isLoaded() const {
    return true;
  }

  /// Returns a future that completes when this stream is safe to read
  /// without blocking on IO. Default = ready future. Cheap when
  /// already loaded; allocates a folly Promise/Future only when the
  /// load is still in flight.
  ///
  /// Same use case as isLoaded(); preferred when the consumer wants to
  /// chain CPU work as a continuation rather than poll-and-reorder.
  virtual folly::SemiFuture<folly::Unit> loadedFuture() {
    return folly::makeSemiFuture(folly::unit);
  }
};

/**
 * Create a seekable input stream based on a memory range.
 */
class SeekableArrayInputStream : public SeekableInputStream {
 public:
  SeekableArrayInputStream(
      const unsigned char* list,
      uint64_t length,
      uint64_t block_size = 0);
  SeekableArrayInputStream(
      const char* list,
      uint64_t length,
      uint64_t block_size = 0);
  // Same as above, but takes ownership of the underlying data.
  SeekableArrayInputStream(
      std::unique_ptr<char[]> list,
      uint64_t length,
      uint64_t block_size = 0);

  explicit SeekableArrayInputStream(
      std::function<std::tuple<const char*, uint64_t>()> dataRead,
      uint64_t block_size = 0);

  ~SeekableArrayInputStream() override = default;

  virtual bool Next(const void** data, int32_t* size) override;
  virtual void BackUp(int32_t count) override;
  virtual bool SkipInt64(int64_t count) override;
  virtual int64_t ByteCount() const override;
  virtual void seekToPosition(PositionProvider& position) override;
  virtual std::string getName() const override;
  virtual size_t positionSize() const override;

  /// Return the total number of bytes returned from Next() calls.  Intended to
  /// be used for test validation.
  int64_t totalRead() const {
    return totalRead_;
  }

 private:
  void loadIfAvailable();

  // data may optionally be owned by *this via ownedData.
  const std::unique_ptr<char[]> ownedData_;
  const char* data_;
  std::function<std::tuple<const char*, uint64_t>()> dataRead_;
  uint64_t length_;
  uint64_t position_;
  uint64_t blockSize_;
  int64_t totalRead_ = 0;
};

/**
 * Create a seekable input stream based on an io stream.
 */
class SeekableFileInputStream : public SeekableInputStream {
 public:
  SeekableFileInputStream(
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t offset,
      uint64_t byteCount,
      memory::MemoryPool& pool,
      LogType logType,
      uint64_t blockSize = 0);
  ~SeekableFileInputStream() override = default;

  virtual bool Next(const void** data, int32_t* size) override;
  virtual void BackUp(int32_t count) override;
  virtual bool SkipInt64(int64_t count) override;
  virtual int64_t ByteCount() const override;
  virtual void seekToPosition(PositionProvider& position) override;
  virtual std::string getName() const override;
  virtual size_t positionSize() const override;

 private:
  const std::shared_ptr<ReadFileInputStream> input_;
  const LogType logType_;
  const uint64_t start_;
  const uint64_t length_;
  const uint64_t blockSize_;

  DataBuffer<char> buffer_;
  uint64_t position_;
  uint64_t pushback_;
};

} // namespace facebook::velox::dwio::common
