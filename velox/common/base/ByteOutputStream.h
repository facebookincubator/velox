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

#include "velox/common/memory/BufferInputStream.h"
#include "velox/common/memory/StreamArena.h"

namespace facebook::velox {

class OutputStream;

/// Stream over a chain of ByteRanges. Provides read, write and
/// comparison for equality between stream contents and memory. Used
/// for streams in repartitioning or for complex variable length data
/// in hash tables. The stream is seekable and supports overwriting of
/// previous content, for example, writing a message body and then
/// seeking back to start to write a length header.
class ByteOutputStream {
 public:
  /// For output.
  ByteOutputStream(
      StreamArena* arena,
      bool isBits = false,
      bool isReverseBitOrder = false)
      : arena_(arena), isBits_(isBits), isReverseBitOrder_(isReverseBitOrder) {}

  ByteOutputStream(const ByteOutputStream& other) = delete;

  void operator=(const ByteOutputStream& other) = delete;

  // Forcing a move constructor to be able to return ByteOutputStream objects
  // from a function.
  ByteOutputStream(ByteOutputStream&&) = default;

  /// Sets 'this' to range over 'range'. If this is for purposes of writing,
  /// lastWrittenPosition specifies the end of any pre-existing content in
  /// 'range'.
  void setRange(ByteRange range, int32_t lastWrittenPosition) {
    ranges_.resize(1);
    ranges_[0] = range;
    current_ = ranges_.data();
    VELOX_CHECK_GE(ranges_.back().size, lastWrittenPosition);
    lastRangeEnd_ = lastWrittenPosition;
  }

  const std::vector<ByteRange>& ranges() const {
    return ranges_;
  }

  /// Prepares 'this' for writing. Can be called several times,
  /// e.g. PrestoSerializer resets these. The memory formerly backing
  /// 'ranges_' is not owned and the caller needs to recycle or free
  /// this independently.
  void startWrite(int32_t initialSize) {
    ranges_.clear();
    isReversed_ = false;
    allocatedBytes_ = 0;
    current_ = nullptr;
    lastRangeEnd_ = 0;
    extend(initialSize);
  }

  void seek(int32_t range, int32_t position) {
    current_ = &ranges_[range];
    current_->position = position;
  }

  std::streampos tellp() const;

  void seekp(std::streampos position);

  /// Returns the size written into ranges_. This is the sum of the capacities
  /// of non-last ranges + the greatest write position of the last range.
  size_t size() const;

  int32_t lastRangeEnd() const {
    updateEnd();
    return lastRangeEnd_;
  }

  template <typename T>
  void append(folly::Range<const T*> values) {
    if (current_->position + sizeof(T) * values.size() > current_->size) {
      appendStringView(std::string_view(
          reinterpret_cast<const char*>(&values[0]),
          values.size() * sizeof(T)));
      return;
    }

    auto* target = reinterpret_cast<T*>(current_->buffer + current_->position);
    const auto* end = target + values.size();
    auto* valuePtr = &values[0];
    while (target != end) {
      *target = *valuePtr;
      ++target;
      ++valuePtr;
    }
    current_->position += sizeof(T) * values.size();
  }

  void appendBool(bool value, int32_t count);

  // A fast path for appending bits into pre-cleared buffers after first extend.
  inline void
  appendBitsFresh(const uint64_t* bits, int32_t begin, int32_t end) {
    const auto position = current_->position;
    if (begin == 0 && end <= 56) {
      const auto available = current_->size - position;
      // There must be 8 bytes writable. If available is 56, there are 7, so >.
      if (available > 56) {
        const auto offset = position & 7;
        uint64_t* buffer =
            reinterpret_cast<uint64_t*>(current_->buffer + (position >> 3));
        const auto mask = bits::lowMask(offset);
        *buffer = (*buffer & mask) | (bits[0] << offset);
        current_->position += end;
        return;
      }
    }
    appendBits(bits, begin, end);
  }

  // Writes 'bits' from bit positions begin..end to the current position of
  // 'this'. Extends 'this' if writing past end.
  void appendBits(const uint64_t* bits, int32_t begin, int32_t end);

  void appendStringView(StringView value);

  void appendStringView(std::string_view value);

  template <typename T>
  void appendOne(const T& value) {
    append(folly::Range(&value, 1));
  }

  void flush(OutputStream* stream);

  /// Returns the next byte that would be written to by a write. This
  /// is used after an append to release the remainder of the reserved
  /// space.
  char* writePosition();

  int32_t testingAllocatedBytes() const {
    return allocatedBytes_;
  }

  /// Returns a ByteInputStream to range over the current content of 'this'. The
  /// result is valid as long as 'this' is live and not changed.
  std::unique_ptr<ByteInputStream> inputStream() const;

  std::string toString() const;

 private:
  // Returns a range of 'size' items of T. If there is no contiguous space in
  // 'this', uses 'scratch' to make a temp block that is appended to 'this' in
  template <typename T>
  T* getAppendWindow(int32_t size, ScratchPtr<T>& scratchPtr) {
    const int32_t bytes = sizeof(T) * size;
    if (!current_) {
      extend(bytes);
    }
    auto available = current_->size - current_->position;
    if (available >= bytes) {
      current_->position += bytes;
      return reinterpret_cast<T*>(
          current_->buffer + current_->position - bytes);
    }
    // If the tail is not large enough, make  temp of the right size
    // in scratch. Extend the stream so that there is guaranteed space to copy
    // the scratch to the stream. This copy takes place in destruction of
    // AppendWindow and must not allocate so that it is noexcept.
    ensureSpace(bytes);
    return scratchPtr.get(size);
  }

  void extend(int32_t bytes);

  // Calls extend() enough times to make sure 'bytes' bytes can be
  // appended without new allocation. Does not change the append
  // position.
  void ensureSpace(int32_t bytes);

  int32_t newRangeSize(int32_t bytes) const;

  void updateEnd() const {
    if (!ranges_.empty() && current_ == &ranges_.back() &&
        current_->position > lastRangeEnd_) {
      lastRangeEnd_ = current_->position;
    }
  }

  StreamArena* const arena_{nullptr};

  // Indicates that position in ranges_ is in bits, not bytes.
  const bool isBits_;

  const bool isReverseBitOrder_;

  // True if the bit order in ranges_ has been inverted. Presto requires
  // reverse bit order.
  bool isReversed_ = false;

  std::vector<ByteRange> ranges_;
  // The total number of bytes allocated from 'arena_' in 'ranges_'.
  int64_t allocatedBytes_{0};

  // Pointer to the current element of 'ranges_'.
  ByteRange* current_{nullptr};

  // Number of bits/bytes that have been written in the last element
  // of 'ranges_'. In a write situation, all non-last ranges are full
  // and the last may be partly full. The position in the last range
  // is not necessarily the the end if there has been a seek.
  mutable int32_t lastRangeEnd_{0};

  template <typename T>
  friend class AppendWindow;
};

/// A scoped wrapper that provides 'size' T's of writable space in 'stream'.
/// Normally gives an address into 'stream's buffer but can use 'scratch' to
/// make a contiguous piece if stream does not have a suitable run.
template <typename T>
class AppendWindow {
 public:
  AppendWindow(ByteOutputStream& stream, Scratch& scratch)
      : stream_(stream), scratchPtr_(scratch) {}

  ~AppendWindow() noexcept {
    if (scratchPtr_.size()) {
      try {
        stream_.appendStringView(std::string_view(
            reinterpret_cast<const char*>(scratchPtr_.get()),
            scratchPtr_.size() * sizeof(T)));
      } catch (const std::exception& e) {
        // This is impossible because construction ensures there is space for
        // the bytes in the stream.
        LOG(FATAL) << "throw from AppendWindo append: " << e.what();
      }
    }
  }

  T* get(int32_t size) {
    return stream_.getAppendWindow(size, scratchPtr_);
  }

 private:
  ByteOutputStream& stream_;
  ScratchPtr<T> scratchPtr_;
};

} // namespace facebook::velox
