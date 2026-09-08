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

#include <cstddef>
#include <cstdint>

#include "velox/common/base/Nulls.h"

namespace facebook::velox::parquet {

/// Decode plain-encoded boolean columns from Parquet.
///
/// Parquet stores booleans as a packed bit stream in LSB-first order.
/// The readWithVisitor fast path processes 8 booleans per byte when
/// reading dense sequential values without nulls.
class BooleanDecoder {
 public:
  BooleanDecoder(const char* start, const char* /*end*/)
      : bufferStart_(start) {}

  /// Reset to decode from a new page buffer without reallocating.
  void reset(const char* start, const char* /*end*/) {
    bufferStart_ = start;
    remainingBits_ = 0;
    reversedLastByte_ = 0;
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    if (remainingBits_ != 0) {
      if (numValues <= remainingBits_) {
        remainingBits_ -= numValues;
        return;
      } else {
        numValues -= remainingBits_;
        remainingBits_ = 0;
      }
    }
    uint64_t numBytes = numValues / 8;
    bufferStart_ += numBytes;
    remainingBits_ = (8 - numValues % 8) % 8;
    if (remainingBits_ != 0) {
      reversedLastByte_ = *reinterpret_cast<const uint8_t*>(bufferStart_);
      bufferStart_++;
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    int32_t toSkip;
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();

    // Fast path: dense reads without nulls. Process 8 booleans per byte
    // by avoiding the per-bit readBoolean() branch overhead.
    if constexpr (!hasNulls && Visitor::dense) {
      for (;;) {
        if (remainingBits_ == 0) {
          // Process a full byte (8 booleans) at once.
          auto byte = *reinterpret_cast<const uint8_t*>(bufferStart_);
          bufferStart_++;
          // Preserve the loaded byte so that, if the visitor reaches
          // 'atEnd' part way through, a later readWithVisitor() call can
          // resume at the correct bit. 'PageReader' keeps the decoder
          // state between calls, so on every early return we must record
          // how many bits of this byte are still unread in
          // 'remainingBits_'.
          reversedLastByte_ = byte;

          // Unrolled: extract each bit and call visitor.process().
          toSkip = visitor.process(static_cast<bool>(byte & 1), atEnd);
          if (atEnd) {
            remainingBits_ = 7;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 2), atEnd);
          if (atEnd) {
            remainingBits_ = 6;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 4), atEnd);
          if (atEnd) {
            remainingBits_ = 5;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 8), atEnd);
          if (atEnd) {
            remainingBits_ = 4;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 16), atEnd);
          if (atEnd) {
            remainingBits_ = 3;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 32), atEnd);
          if (atEnd) {
            remainingBits_ = 2;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 64), atEnd);
          if (atEnd) {
            remainingBits_ = 1;
            return;
          }
          ++current;

          toSkip = visitor.process(static_cast<bool>(byte & 128), atEnd);
          if (atEnd) {
            remainingBits_ = 0;
            return;
          }
          ++current;
        } else {
          // Drain remaining bits from the previously loaded byte.
          toSkip = visitor.process(readBoolean(), atEnd);
          ++current;
          if (atEnd) {
            return;
          }
        }
      }
    }

    // General path: handles nulls, sparse reads, and filters.
    for (;;) {
      if (hasNulls && allowNulls && bits::isBitNull(nulls, current)) {
        toSkip = visitor.processNull(atEnd);
      } else {
        if (hasNulls && !allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) {
            return;
          }
        }

        // We are at a non-null value on a row to visit.
        toSkip = visitor.process(readBoolean(), atEnd);
      }
      ++current;
      if (toSkip) {
        skip<hasNulls>(toSkip, current, nulls);
        current += toSkip;
      }
      if (atEnd) {
        return;
      }
    }
  }

 private:
  bool readBoolean() {
    if (remainingBits_ == 0) {
      remainingBits_ = 7;
      reversedLastByte_ = *reinterpret_cast<const uint8_t*>(bufferStart_);
      bufferStart_++;
      return reversedLastByte_ & 0x1;
    } else {
      return reversedLastByte_ & (1 << (8 - (remainingBits_--)));
    }
  }

  size_t remainingBits_{0};
  uint8_t reversedLastByte_{0};
  const char* bufferStart_{nullptr};
};

} // namespace facebook::velox::parquet
