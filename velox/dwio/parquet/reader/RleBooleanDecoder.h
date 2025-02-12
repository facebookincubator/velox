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

#include "velox/dwio/parquet/common/BitStreamUtilsInternal.h"
#include "velox/dwio/parquet/reader/RleBpDecoder.h"

namespace facebook::velox::parquet {

class RleBooleanDecoder : public RleBpDecoder {
 public:
  static constexpr int32_t kLengthOffset = 4;
  RleBooleanDecoder(const char* start, const char* end, int32_t len)
      : RleBpDecoder{start + kLengthOffset, end, 1} {
    if (len < kLengthOffset) {
      VELOX_FAIL("Received invalid length : {} (corrupt data page?)", len);
    }
    numBytes_ =
        ::arrow::bit_util::FromLittleEndian(::arrow::util::SafeLoadAs<uint32_t>(
            reinterpret_cast<const uint8_t*>(start)));
    if (numBytes_ > static_cast<uint32_t>(len - kLengthOffset)) {
      VELOX_FAIL(
          "Received invalid number of bytes : " + std::to_string(numBytes_) +
          " (corrupt data page?)");
    }
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if constexpr (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }

    RleBpDecoder::skip(numValues);
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    int32_t toSkip = 0;
    bool atEnd = false;
    const bool allowNulls = hasNulls && visitor.allowNulls();
    bool* b = nullptr;
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
        if (!remainingValues_) {
          readHeader();
        }
        if (repeating_) {
          toSkip = visitor.process(value_, atEnd);
        } else {
          value_ = readBitField();
          toSkip = visitor.process(value_, atEnd);
        }
        --remainingValues_;
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
  int64_t readBitField() {
    auto value = dwio::common::safeLoadBits(
                     bufferStart_, bitOffset_, bitWidth_, lastSafeWord_) &
        bitMask_;
    bitOffset_ += bitWidth_;
    bufferStart_ += bitOffset_ >> 3;
    bitOffset_ &= 7;
    return value;
  }

  uint32_t numBytes_ = 0;
};

} // namespace facebook::velox::parquet
