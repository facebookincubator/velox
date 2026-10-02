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

#include <cstring>
#include <optional>
#include <string_view>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

namespace facebook::velox::parquet {
namespace detail {

template <typename T>
FOLLY_ALWAYS_INLINE T fromBigEndianBytes(std::string_view bytes) {
  static_assert(
      std::is_same_v<T, int128_t> || std::is_same_v<T, int64_t> ||
          std::is_same_v<T, int32_t>,
      "Only integer types are supported.");

  VELOX_CHECK_LE(
      bytes.size(),
      sizeof(T),
      "Length of byte array passed to fromBigEndianBytes is too large");

  if (bytes.empty()) {
    return {};
  }

  const auto* rawBytes = reinterpret_cast<const uint8_t*>(bytes.data());
  T result = (rawBytes[0] & 0x80) != 0 ? -1 : 0;
  memcpy(
      reinterpret_cast<uint8_t*>(&result) + sizeof(T) - bytes.size(),
      rawBytes,
      bytes.size());

  if constexpr (std::is_same_v<T, int128_t>) {
    return bits::builtin_bswap128(result);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return __builtin_bswap64(result);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return __builtin_bswap32(result);
  } else {
    VELOX_UNREACHABLE();
  }
}

} // namespace detail

// DeltaLengthByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L2758-L2889
class DeltaLengthByteArrayDecoder {
 public:
  DeltaLengthByteArrayDecoder() = default;

  explicit DeltaLengthByteArrayDecoder(const char* start) {
    reset(start);
  }

  void reset(const char* start) {
    lengthDecoder_.emplace(start);
    decodeLengths();
    bufferStart_ = lengthDecoder_->bufferStart();
  }

  FOLLY_ALWAYS_INLINE std::string_view readString() {
    const int64_t length = bufferedLength_[lengthIdx_++];
    VELOX_CHECK_GE(length, 0, "negative string delta length");
    bufferStart_ += length;
    return std::string_view(bufferStart_ - length, length);
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  FOLLY_ALWAYS_INLINE void
  skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
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

 private:
  void decodeLengths() {
    int64_t numLength = lengthDecoder_->validValuesCount();
    bufferedLength_.resize(numLength);
    lengthDecoder_->readValues<uint32_t>(
        bufferedLength_.data(), static_cast<int32_t>(numLength));

    lengthIdx_ = 0;
    numValidValues_ = static_cast<int32_t>(numLength);
  }

  const char* bufferStart_;
  std::optional<DeltaBpDecoder> lengthDecoder_;
  int32_t numValidValues_{0};
  uint32_t lengthIdx_{0};
  std::vector<uint32_t> bufferedLength_;
};

// DeltaByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L3301-L3545
class DeltaByteArrayDecoder {
 public:
  /// Default-constructed instance is uninitialized; call reset() first.
  DeltaByteArrayDecoder() = default;

  explicit DeltaByteArrayDecoder(const char* start) {
    reset(start);
  }

  void reset(const char* start) {
    prefixLenDecoder_.emplace(start);
    int64_t numPrefix = prefixLenDecoder_->validValuesCount();
    bufferedPrefixLength_.resize(numPrefix);
    prefixLenDecoder_->readValues<uint32_t>(
        bufferedPrefixLength_.data(), static_cast<int32_t>(numPrefix));
    prefixLenOffset_ = 0;
    numValidValues_ = static_cast<int32_t>(numPrefix);
    lastValueLen_ = 0;

    if (!suffixDecoder_.has_value()) {
      suffixDecoder_.emplace(prefixLenDecoder_->bufferStart());
    } else {
      suffixDecoder_->reset(prefixLenDecoder_->bufferStart());
    }
  }

  FOLLY_ALWAYS_INLINE std::string_view readString() {
    auto suffix = suffixDecoder_->readString();
    const int64_t prefixLength = bufferedPrefixLength_[prefixLenOffset_++];

    VELOX_CHECK_GE(
        prefixLength, 0, "negative prefix length in DELTA_BYTE_ARRAY");
    VELOX_CHECK_LE(
        prefixLength,
        lastValueLen_,
        "prefix length too large in DELTA_BYTE_ARRAY");

    const size_t need = prefixLength + suffix.size();
    if (need > lastValueBuf_.size()) {
      lastValueBuf_.resize(need);
    }
    if (!suffix.empty()) {
      memcpy(lastValueBuf_.data() + prefixLength, suffix.data(), suffix.size());
    }
    lastValueLen_ = static_cast<int32_t>(need);

    numValidValues_--;
    return {lastValueBuf_.data(), static_cast<size_t>(lastValueLen_)};
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  FOLLY_ALWAYS_INLINE void
  skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
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
        using T = typename Visitor::DataType;
        if constexpr (
            std::is_same_v<T, int128_t> || std::is_same_v<T, int64_t> ||
            std::is_same_v<T, int32_t>) {
          toSkip = visitor.process(
              detail::fromBigEndianBytes<T>(readString()), atEnd);
        } else {
          toSkip = visitor.process(readString(), atEnd);
        }
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
  std::optional<DeltaBpDecoder> prefixLenDecoder_;
  std::optional<DeltaLengthByteArrayDecoder> suffixDecoder_;

  // The string_view returned by readString() aliases this buffer.
  std::vector<char> lastValueBuf_;
  int32_t lastValueLen_{0};
  int32_t numValidValues_{0};
  uint32_t prefixLenOffset_{0};
  std::vector<uint32_t> bufferedPrefixLength_;
};

} // namespace facebook::velox::parquet
