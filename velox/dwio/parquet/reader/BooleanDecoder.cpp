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

#include "velox/dwio/parquet/reader/BooleanDecoder.h"

#include <folly/lang/Bits.h>

#include <algorithm>

namespace facebook::velox::parquet {
namespace {

template <typename T>
void unpackBooleans(T packed, int8_t* output) {
  constexpr int32_t kNumBits = sizeof(T) * 8;
  for (int32_t i = 0; i < kNumBits; ++i) {
    output[i] = (packed >> i) & 1;
  }
}

} // namespace

void BooleanDecoder::readDense(int32_t numValues, int8_t* output) {
  constexpr int32_t kValuesPerWord = 32;
  constexpr int32_t kValuesPerByte = 8;
  const auto numBufferedBits =
      std::min(numValues, static_cast<int32_t>(remainingBits_));
  numValues -= numBufferedBits;
  VELOX_USER_CHECK_LE(
      static_cast<int64_t>(bits::nbytes(numValues)),
      bufferEnd_ - bufferStart_,
      "Truncated PLAIN BOOLEAN data");
  for (int32_t i = 0; i < numBufferedBits; ++i) {
    *output++ = readBoolean();
  }
  while (numValues >= kValuesPerWord) {
    const auto word =
        folly::Endian::little(folly::loadUnaligned<uint32_t>(bufferStart_));
    unpackBooleans(word, output);
    bufferStart_ += sizeof(word);
    output += kValuesPerWord;
    numValues -= kValuesPerWord;
  }
  while (numValues >= kValuesPerByte) {
    unpackBooleans(static_cast<uint8_t>(*bufferStart_++), output);
    output += kValuesPerByte;
    numValues -= kValuesPerByte;
  }
  for (int32_t i = 0; i < numValues; ++i) {
    output[i] = readBoolean();
  }
}

} // namespace facebook::velox::parquet
