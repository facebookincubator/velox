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
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"
#include "velox/dwio/parquet/reader/DeltaLengthByteArrayDecoder.h"

namespace facebook::velox::parquet {

using namespace velox::memory;

// DeltaByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L3301-L3545
class DeltaByteArrayDecoder {
 public:
  explicit DeltaByteArrayDecoder(const char* start) {
    prefixLenDecoder_ = std::make_unique<DeltaBpDecoder>(start);
    int64_t numPrefix = prefixLenDecoder_->validValuesCount();
    bufferedPrefixLength_.resize(numPrefix);
    prefixLenDecoder_->readValues<uint32_t>(bufferedPrefixLength_, numPrefix);
    prefixLenOffset_ = 0;
    numValidValues_ = numPrefix;

    suffixDecoder_ = std::make_unique<DeltaLengthByteArrayDecoder>(
        prefixLenDecoder_->bufferStart());
    lastValue_.clear();
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
      readString();
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
        toSkip = visitor.process(readString(), atEnd);
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

  folly::StringPiece readString() {
    auto suffix = suffixDecoder_->readString();
    bool isFirstRun = (prefixLenOffset_ == 0);
    const int64_t prefixLength = bufferedPrefixLength_[prefixLenOffset_++];

    VELOX_CHECK_GE(
        prefixLength, 0, "negative prefix length in DELTA_BYTE_ARRAY");

    std::string_view prefix{lastValue_};
    std::string tempString;
    tempString.resize(prefixLength + suffix.size());
    buildBuffer(isFirstRun, prefixLength, suffix, &prefix, tempString);

    numValidValues_--;
    lastValue_ = std::string{prefix};
    return {lastValue_};
  }

 private:
  void buildBuffer(
      bool isFirstRun,
      const int64_t prefixLength,
      folly::StringPiece& suffix,
      std::string_view* prefix,
      std::string data) {
    VELOX_CHECK_LE(
        prefixLength,
        prefix->length(),
        "prefix length too large in DELTA_BYTE_ARRAY");
    if (prefixLength == 0) {
      // prefix is empty.
      *prefix = std::string_view{suffix.data(), suffix.size()};
      return;
    }

    if (!isFirstRun) {
      if (suffix.empty()) {
        // suffix is empty: suffix can simply point to the prefix.
        // This is not possible for the first run since the prefix
        // would point to the mutable `lastValue_`.
        *prefix = prefix->substr(0, prefixLength);
        suffix = folly::StringPiece(std::string{*prefix});
        return;
      }
    }
    // Both prefix and suffix are non-empty, so we need to decode the string
    // into `data`.
    // 1. Copy the prefix.
    memcpy(data.data(), prefix->data(), prefixLength);
    // 2. Copy the suffix.
    memcpy(data.data() + prefixLength, suffix.data(), suffix.size());
    // 3. Point the suffix to the decoded string.
    suffix = folly::StringPiece(data);
    *prefix = std::string_view{suffix};
  }

  std::unique_ptr<DeltaBpDecoder> prefixLenDecoder_;
  std::unique_ptr<DeltaBpDecoder> suffixLenDecoder_;
  std::unique_ptr<DeltaLengthByteArrayDecoder> suffixDecoder_;

  std::string lastValue_;
  int32_t numValidValues_{0};
  uint32_t prefixLenOffset_{0};
  std::vector<uint32_t> bufferedPrefixLength_;
};

} // namespace facebook::velox::parquet
