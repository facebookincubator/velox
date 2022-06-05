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

#include "velox/dwio/common/reader/ColumnReader.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

// TODO: Move the IntCodecCommon.h file to velox/common/dwio
#include "velox/dwio/dwrf/common/IntCodecCommon.h"

#include <folly/String.h>

namespace facebook::velox::dwio::common::reader {

using dwio::common::typeutils::CompatChecker;
using memory::MemoryPool;

// Buffer size for reading length stream
constexpr uint64_t BUFFER_SIZE = 1024;

// it's possible stride dictionary only contains zero length string. In that
// case, we still need to make batch point to a valid address
std::array<char, 1> EMPTY_DICT;

namespace detail {

void fillTimestamps(
    Timestamp* timestamps,
    const uint64_t* nullsPtr,
    const int64_t* secondsPtr,
    const uint64_t* nanosPtr,
    vector_size_t numValues) {
  for (vector_size_t i = 0; i < numValues; i++) {
    if (!nullsPtr || !bits::isBitNull(nullsPtr, i)) {
      auto nanos = nanosPtr[i];
      uint64_t zeros = nanos & 0x7;
      nanos >>= 3;
      if (zeros != 0) {
        for (uint64_t j = 0; j <= zeros; ++j) {
          nanos *= 10;
        }
      }
      auto seconds = secondsPtr[i] + dwrf::EPOCH_OFFSET;
      if (seconds < 0 && nanos != 0) {
        seconds -= 1;
      }
      timestamps[i] = Timestamp(seconds, nanos);
    }
  }
};

} // namespace detail

BufferPtr ColumnReader::readNulls(
    vector_size_t numValues,
    VectorPtr& result,
    const uint64_t* incomingNulls) {
  BufferPtr nulls;
  readNulls(numValues, incomingNulls, &result, nulls);
  return nulls;
}

void ColumnReader::readNulls(
    vector_size_t numValues,
    const uint64_t* incomingNulls,
    VectorPtr* result,
    BufferPtr& nulls) {
  if (!notNullDecoder_ && !incomingNulls) {
    nulls = nullptr;
    if (result && *result) {
      (*result)->resetNulls();
    }
    return;
  }
  auto numBytes = bits::nbytes(numValues);
  if (result && *result) {
    nulls = (*result)->mutableNulls(numValues + (simd::kPadding * 8));
    detail::resetIfNotWritable(*result, nulls);
  }
  if (!nulls || nulls->capacity() < numBytes + simd::kPadding) {
    nulls =
        AlignedBuffer::allocate<char>(numBytes + simd::kPadding, &memoryPool_);
  }
  nulls->setSize(numBytes);
  auto* nullsPtr = nulls->asMutable<uint64_t>();
  if (!notNullDecoder_) {
    memcpy(nullsPtr, incomingNulls, numBytes);
    return;
  }
  memset(nullsPtr, bits::kNotNullByte, numBytes);
  notNullDecoder_->next(
      reinterpret_cast<char*>(nullsPtr), numValues, incomingNulls);
}

uint64_t ColumnReader::skip(uint64_t numValues) {
  if (notNullDecoder_) {
    // page through the values that we want to skip
    // and count how many are non-null
    std::array<char, BUFFER_SIZE> buffer;
    constexpr auto bitCount = BUFFER_SIZE * 8;
    uint64_t remaining = numValues;
    while (remaining > 0) {
      uint64_t chunkSize = std::min(remaining, bitCount);
      notNullDecoder_->next(buffer.data(), chunkSize, nullptr);
      remaining -= chunkSize;
      numValues -= bits::countNulls(
          reinterpret_cast<uint64_t*>(buffer.data()), 0, chunkSize);
    }
  }
  return numValues;
}
} // namespace facebook::velox::dwio::common::reader
