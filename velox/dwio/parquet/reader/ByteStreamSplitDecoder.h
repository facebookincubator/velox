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

#include "velox/common/base/BitUtil.h"

namespace facebook::velox::parquet {

class ByteStreamSplitDecoder {
 public:
  ByteStreamSplitDecoder(const char* start, const char* end, uint32_t numBytes)
      : bufferStart_(start),
        bufferEnd_(end),
        numValues_((end - start) / numBytes) {
    VELOX_CHECK_EQ(
        (end - start) % numBytes,
        0,
        "ByteStreamSplit data size " + std::to_string(end - start) +
            " not aligned with type size " + std::to_string(numBytes))
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    for (int32_t i = 0; i < numValues; ++i) {
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
        toSkip =
            visitor.process(readValue<typename Visitor::DataType>(), atEnd);
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
  template <class T>
  T readValue() {
    uint8_t buf[sizeof(T)];
    for (int i = 0; i < sizeof(T); i++) {
      buf[i] = *(bufferStart_ + i * numValues_);
    }
    bufferStart_++;
    return *reinterpret_cast<T*>(buf);
  }

  const char* bufferStart_;
  const char* bufferEnd_;
  const uint32_t numValues_;
};

} // namespace facebook::velox::parquet
