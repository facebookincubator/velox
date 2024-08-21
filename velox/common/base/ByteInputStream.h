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

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"

namespace facebook::velox {

struct ByteRange {
  /// Start of buffer. Not owned.
  uint8_t* buffer;

  /// Number of bytes or bits starting at 'buffer'.
  int32_t size;

  /// Index of next byte/bit to be read/written in 'buffer'.
  int32_t position;

  /// Returns the available bytes left in this range.
  uint32_t availableBytes() const;

  std::string toString() const;
};

/// Read-only byte input stream interface.
class ByteInputStream {
 public:
  virtual ~ByteInputStream() = default;

  /// Returns total number of bytes available in the stream.
  virtual size_t size() const = 0;

  /// Returns true if all input has been read.
  virtual bool atEnd() const = 0;

  /// Returns current position (number of bytes from the start) in the stream.
  virtual std::streampos tellp() const = 0;

  /// Moves current position to specified one.
  virtual void seekp(std::streampos pos) = 0;

  /// Returns the remaining size left from current reading position.
  virtual size_t remainingSize() const = 0;

  virtual uint8_t readByte() = 0;

  virtual void readBytes(uint8_t* bytes, int32_t size) = 0;

  template <typename T>
  T read() {
    if (current_->position + sizeof(T) <= current_->size) {
      current_->position += sizeof(T);
      return *reinterpret_cast<const T*>(
          current_->buffer + current_->position - sizeof(T));
    }
    // The number straddles two buffers. We read byte by byte and make a
    // little-endian uint64_t. The bytes can be cast to any integer or floating
    // point type since the wire format has the machine byte order.
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

  /// Returns a view over the read buffer for up to 'size' next bytes. The size
  /// of the value may be less if the current byte range ends within 'size'
  /// bytes from the current position.  The size will be 0 if at end.
  virtual std::string_view nextView(int32_t size) = 0;

  virtual void skip(int32_t size) = 0;

  virtual std::string toString() const = 0;

 protected:
  // Points to the current buffered byte range.
  ByteRange* current_{nullptr};
  std::vector<ByteRange> ranges_;
};

template <>
inline Timestamp ByteInputStream::read<Timestamp>() {
  Timestamp value;
  readBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
  return value;
}

template <>
inline int128_t ByteInputStream::read<int128_t>() {
  int128_t value;
  readBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
  return value;
}

} // namespace facebook::velox
