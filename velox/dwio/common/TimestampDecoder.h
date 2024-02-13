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

#include "velox/dwio/common/DirectDecoder.h"

namespace facebook::velox::dwio::common {

enum TimestampPrecision { kMillis, kMicros };

class TimestampDecoder : public DirectDecoder<false> {
 public:
  TimestampDecoder(
      TimestampPrecision precision,
      std::unique_ptr<dwio::common::SeekableInputStream> input,
      bool useVInts,
      uint32_t numBytes,
      bool bigEndian = false)
      : DirectDecoder<false>{std::move(input), useVInts, numBytes, bigEndian},
        precision_(precision) {}

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(
      const uint64_t* FOLLY_NULLABLE nulls,
      Visitor visitor,
      bool useFastPath = true) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    const bool allowNulls = hasNulls && visitor.allowNulls();
    for (;;) {
      bool atEnd = false;
      int32_t toSkip;
      if (hasNulls) {
        if (!allowNulls) {
          toSkip = visitor.checkAndSkipNulls(nulls, current, atEnd);
          if (!Visitor::dense) {
            skip<false>(toSkip, current, nullptr);
          }
          if (atEnd) {
            return;
          }
        } else {
          if (bits::isBitNull(nulls, current)) {
            toSkip = visitor.processNull(atEnd);
            goto skip;
          }
        }
      }
      if constexpr (std::is_same_v<typename Visitor::DataType, int128_t>) {
        auto units = IntDecoder<false>::template readInt<int64_t>();
        Timestamp timestamp = precision_ == TimestampPrecision::kMillis
            ? util::fromUTCMillis(units)
            : util::fromUTCMicros(units);
        int128_t value;
        memcpy(&value, &timestamp, sizeof(int128_t));
        toSkip = visitor.process(value, atEnd);
      } else {
        toSkip = visitor.process(
            IntDecoder<false>::template readInt<int64_t>(), atEnd);
      }
    skip:
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
  TimestampPrecision precision_;
};
} // namespace facebook::velox::dwio::common
