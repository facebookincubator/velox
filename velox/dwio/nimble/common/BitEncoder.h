/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

namespace facebook::nimble {

// Class to put bits into, and read bits from, a buffer.
class BitEncoder {
 public:
  // It's up to the user to not write/read beyond the provided buffer.
  // Note that you probably want to ensure the memory is zero'd out.
  explicit BitEncoder(char* buffer)
      : buffer_(buffer),
        writeWord_(reinterpret_cast<uint64_t*>(buffer)),
        readWord_(reinterpret_cast<const uint64_t*>(buffer)) {}
  // We don't prevent you from using the putBits call after using this
  // constructor, but you really shouldn't.
  explicit BitEncoder(const char* buffer)
      : BitEncoder(const_cast<char*>(buffer)) {}

  // Places |value| occupying |numBits| into the stream. Behavior
  // is undefined if |value| is >= 2^numBits.
  void putBits(uint64_t value, int numBits) {
    *writeWord_ |= value << writeOffset_;
    const int nextOffset = writeOffset_ + numBits;
    if (nextOffset >= 64) {
      ++writeWord_;
      const int spilloverBits = nextOffset - 64;
      if (numBits == 64 && spilloverBits == 0) {
        return;
      }
      *writeWord_ |= value >> (numBits - spilloverBits);
      writeOffset_ = spilloverBits;
    } else {
      writeOffset_ = nextOffset;
    }
  }

  uint64_t bitsWritten() {
    return ((reinterpret_cast<char*>(writeWord_) - buffer_) << 3) +
        writeOffset_;
  }

  uint64_t getBits(int numBits) {
    const int nextOffset = readOffset_ + numBits;
    if (nextOffset >= 64) {
      const uint64_t lowBits = *readWord_ >> readOffset_;
      ++readWord_;
      const int spilloverBits = nextOffset - 64;
      readOffset_ = spilloverBits;
      if (spilloverBits == 0) {
        return lowBits;
      }
      return lowBits |
          (*readWord_ & ((1ULL << spilloverBits) - 1ULL))
          << (numBits - spilloverBits);
    } else {
      const uint64_t result =
          (*readWord_ >> readOffset_) & ((1ULL << numBits) - 1ULL);
      readOffset_ = nextOffset;
      return result;
    }
  }

 private:
  char* buffer_;
  uint64_t* writeWord_;
  const uint64_t* readWord_;
  int writeOffset_ = 0;
  int readOffset_ = 0;
};

} // namespace facebook::nimble
