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

#include "velox/common/memory/StreamArena.h"
#include "velox/type/Type.h"

#include <folly/io/IOBuf.h>

namespace facebook::velox {

struct ByteRange {
  // Start of buffer. Not owned.
  uint8_t* buffer;

  // Number of bytes or bits starting at 'buffer'.
  int32_t size;

  // Index of next byte/bit to be read/written in 'buffer'.
  int32_t position;
};

class OutputStreamListener {
 public:
  virtual void onWrite(const char* /* s */, std::streamsize /* count */) {}
  virtual ~OutputStreamListener() = default;
};

class OutputStream {
 public:
  explicit OutputStream(OutputStreamListener* listener = nullptr)
      : listener_(listener) {}

  virtual ~OutputStream() = default;

  virtual void write(const char* s, std::streamsize count) = 0;

  virtual std::streampos tellp() const = 0;

  virtual void seekp(std::streampos pos) = 0;

  OutputStreamListener* listener() const {
    return listener_;
  }

 protected:
  OutputStreamListener* listener_;
};

class OStreamOutputStream : public OutputStream {
 public:
  explicit OStreamOutputStream(
      std::ostream* out,
      OutputStreamListener* listener = nullptr)
      : OutputStream(listener), out_(out) {}

  void write(const char* s, std::streamsize count) override {
    out_->write(s, count);
    if (listener_) {
      listener_->onWrite(s, count);
    }
  }

  std::streampos tellp() const override {
    return out_->tellp();
  }

  void seekp(std::streampos pos) override {
    out_->seekp(pos);
  }

 private:
  std::ostream* out_;
};

/// Stream over a chain of ByteRanges. Provides read, write and
/// comparison for equality between stream contents and memory. Used
/// for streams in repartitioning or for complex variable length data
/// in hash tables. The stream is seekable and supports overwriting of
/// previous content, for example, writing a message body and then
/// seeking back to start to write a length header.
class ByteStream {
 public:
  // For input.
  ByteStream() : isBits_(false), isReverseBitOrder_(false) {}
  virtual ~ByteStream() = default;

  // For output.
  ByteStream(
      StreamArena* arena,
      bool isBits = false,
      bool isReverseBitOrder = false)
      : arena_(arena), isBits_(isBits), isReverseBitOrder_(isReverseBitOrder) {}

  ByteStream(const ByteStream& other) = delete;

  void operator=(const ByteStream& other) = delete;

  void resetInput(std::vector<ByteRange>&& ranges) {
    ranges_ = std::move(ranges);
    current_ = &ranges_[0];
    lastRangeEnd_ = ranges_.back().size;
  }

  void setRange(ByteRange range) {
    ranges_.resize(1);
    ranges_[0] = range;
    current_ = ranges_.data();
    lastRangeEnd_ = ranges_[0].size;
  }

  const std::vector<ByteRange>& ranges() const {
    return ranges_;
  }

  void startWrite(int32_t initialSize) {
    extend(initialSize);
  }

  void seek(int32_t range, int32_t position) {
    current_ = &ranges_[range];
    current_->position = position;
  }

  std::streampos tellp() const;

  void seekp(std::streampos position);

  /// Returns the size written into ranges_. This is the sum of the
  /// capacities of non-last ranges + the greatest write position of
  /// the last range.
  size_t size() const;

  /// Returns the remaining size left from current reading position.
  size_t remainingSize() const;

  /// For input. Returns true if all input has been read.
  bool atEnd() const;

  int32_t lastRangeEnd() {
    updateEnd();
    return lastRangeEnd_;
  }

  /// Sets 'current_' to point to the next range of input.  // The
  /// input is consecutive ByteRanges in 'ranges_' for the base class
  /// but any view over external buffers can be made by specialization.
  virtual void next(bool throwIfPastEnd = true);

  uint8_t readByte();

  void readBytes(uint8_t* bytes, int32_t size);

  template <typename T>
  T read() {
    if (current_->position + sizeof(T) <= current_->size) {
      current_->position += sizeof(T);
      return *reinterpret_cast<const T*>(
          current_->buffer + current_->position - sizeof(T));
    }
    // The number straddles two buffers. We read byte by byte and make
    // a little-endian uint64_t. The bytes can be cast to any integer
    // or floating point type since the wire format has the machine byte order.
    static_assert(sizeof(T) <= sizeof(uint64_t));
    uint64_t value = 0;
    for (int32_t i = 0; i < sizeof(T); ++i) {
      value |= static_cast<uint64_t>(readByte()) << (i * 8);
    }
    return *reinterpret_cast<const T*>(&value);
  }

  template <typename Char>
  void readBytes(Char* data, int32_t size) {
    readBytes(reinterpret_cast<uint8_t*>(data), size);
  }

  /// Returns a view over the read buffer for up to 'size' next
  /// bytes. The size of the value may be less if the current byte
  /// range ends within 'size' bytes from the current position.  The
  /// size will be 0 if at end.
  std::string_view nextView(int32_t size);

  void skip(int32_t size);

  template <typename T>
  void append(folly::Range<const T*> values) {
    if (current_->position + sizeof(T) * values.size() > current_->size) {
      appendStringPiece(folly::StringPiece(
          reinterpret_cast<const char*>(&values[0]),
          values.size() * sizeof(T)));
      return;
    }
    auto target = reinterpret_cast<T*>(current_->buffer + current_->position);
    auto end = target + values.size();
    auto valuePtr = &values[0];
    while (target != end) {
      *target = *valuePtr;
      ++target;
      ++valuePtr;
    }
    current_->position += sizeof(T) * values.size();
  }

  void appendBool(bool value, int32_t count);

  void appendStringPiece(folly::StringPiece value);

  template <typename T>
  void appendOne(const T& value) {
    append(folly::Range(&value, 1));
  }

  void flush(OutputStream* stream);

  /// Returns the next byte that would be written to by a write. This
  /// is used after an append to release the remainder of the reserved
  /// space.
  char* writePosition();

  std::string toString() const;

 private:
  void extend(int32_t bytes = memory::AllocationTraits::kPageSize);

  void updateEnd() {
    if (!ranges_.empty() && current_ == &ranges_.back() &&
        current_->position > lastRangeEnd_) {
      lastRangeEnd_ = current_->position;
    }
  }

  StreamArena* arena_{nullptr};

  // Indicates that position in ranges_ is in bits, not bytes.
  const bool isBits_;

  const bool isReverseBitOrder_;

  // True if the bit order in ranges_ has been inverted. Presto requires
  // reverse bit order.
  bool isReversed_ = false;

  std::vector<ByteRange> ranges_;

  // Pointer to the current element of 'ranges_'.
  ByteRange* current_{nullptr};

  // Number of bits/bytes that have been written in the last element
  // of 'ranges_'. In a write situation, all non-last ranges are full
  // and the last may be partly full. The position in the last range
  // is not necessarily the the end if there has been a seek.
  int32_t lastRangeEnd_{0};
};

template <>
inline Timestamp ByteStream::read<Timestamp>() {
  Timestamp value;
  readBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
  return value;
}

template <>
inline int128_t ByteStream::read<int128_t>() {
  int128_t value;
  readBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
  return value;
}

class IOBufOutputStream : public OutputStream {
 public:
  explicit IOBufOutputStream(
      memory::MemoryPool& pool,
      OutputStreamListener* listener = nullptr,
      int32_t initialSize = memory::AllocationTraits::kPageSize)
      : OutputStream(listener),
        arena_(std::make_shared<StreamArena>(&pool)),
        out_(std::make_unique<ByteStream>(arena_.get())) {
    out_->startWrite(initialSize);
  }

  void write(const char* s, std::streamsize count) override {
    out_->appendStringPiece(folly::StringPiece(s, count));
    if (listener_) {
      listener_->onWrite(s, count);
    }
  }

  std::streampos tellp() const override;

  void seekp(std::streampos pos) override;

  /// 'releaseFn' is executed on iobuf destruction if not null.
  std::unique_ptr<folly::IOBuf> getIOBuf(
      const std::function<void()>& releaseFn = nullptr);

 private:
  std::shared_ptr<StreamArena> arena_;
  std::unique_ptr<ByteStream> out_;
};

} // namespace facebook::velox
