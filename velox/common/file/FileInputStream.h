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

#include <cstdint>

#include "velox/buffer/Buffer.h"
#include "velox/common/base/AsyncSource.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/ByteStream.h"

namespace facebook::velox::common {

/// Readonly byte input stream backed by file.
class FileInputStream : public ByteInputStream {
 public:
  /// Starts read-ahead eagerly on the optional, unowned 'readAheadExecutor' for
  /// files that support asynchronous reads. The executor must outlive the
  /// stream. Requires pending file I/O to make progress independently of calls
  /// that read or destroy the stream.
  FileInputStream(
      std::unique_ptr<ReadFile>&& file,
      uint64_t bufferSize,
      memory::MemoryPool* pool,
      folly::Executor* readAheadExecutor = nullptr);

  ~FileInputStream() override;

  FileInputStream(const FileInputStream&) = delete;
  FileInputStream& operator=(const FileInputStream& other) = delete;
  FileInputStream(FileInputStream&& other) noexcept = delete;
  FileInputStream& operator=(FileInputStream&& other) noexcept = delete;

  size_t size() const override;

  bool atEnd() const override;

  std::streampos tellp() const override;

  void seekp(std::streampos pos) override;

  void skip(int32_t size) override;

  size_t remainingSize() const override;

  uint8_t readByte() override;

  void readBytes(uint8_t* bytes, int32_t size) override;

  std::string_view nextView(int64_t size) override;

  std::string toString() const override;

  /// Records the file read stats.
  struct Stats {
    uint32_t numReads{0};
    uint64_t readBytes{0};
    uint64_t readTimeNs{0};

    bool operator==(const Stats& other) const;

    std::string toString() const;
  };
  Stats stats() const;

 private:
  // Drains pending I/O before the stream releases its read buffers and file.
  struct ReadAheadResult;

  void doSeek(int64_t skipBytes);

  // Invoked to read the next byte range from the file in a buffer.
  void readNextRange();

  // Prepares the read of the next buffer. It runs on the read-ahead executor
  // when there is one, otherwise on the reader at the next refill.
  void maybeIssueReadahead();

  inline uint64_t readSize() const;

  inline uint32_t bufferIndex() const {
    return bufferIndex_;
  }

  inline uint32_t nextBufferIndex() const {
    return (bufferIndex_ + 1) % buffers_.size();
  }

  // Advances buffer index to point to the next buffer for read.
  inline void advanceBuffer() {
    bufferIndex_ = nextBufferIndex();
  }

  inline Buffer* buffer() const {
    return buffers_[bufferIndex()].get();
  }

  inline Buffer* nextBuffer() const {
    return buffers_[nextBufferIndex()].get();
  }

  void updateStats(uint64_t readBytes, uint64_t readTimeNs);

  const std::unique_ptr<ReadFile> file_;
  const uint64_t fileSize_;
  const uint64_t bufferSize_;
  memory::MemoryPool* const pool_;
  // Schedules read-ahead and must outlive this stream.
  folly::Executor* const readAheadExecutor_;
  const bool readAheadEnabled_;

  // Offset of the next byte to read from file.
  uint64_t fileOffset_ = 0;

  std::vector<BufferPtr> buffers_;
  uint32_t bufferIndex_{0};
  // Keeps queued read-ahead and pending native I/O alive until consumed or
  // closed.
  std::shared_ptr<AsyncSource<ReadAheadResult>> readAheadSource_;

  ByteRange range_;

  Stats stats_;
};
} // namespace facebook::velox::common
