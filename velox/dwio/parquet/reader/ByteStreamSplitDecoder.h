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

namespace facebook::velox::parquet {

/// Decode BYTE_STREAM_SPLIT encoded page data in-place.
///
/// BYTE_STREAM_SPLIT splits each K-byte value into K separate byte streams
/// concatenated contiguously. This function reassembles the original values
/// into a flat output buffer suitable for DirectDecoder.
///
/// Input layout (N values of K bytes):
///   [byte 0 of val 0..N-1] [byte 1 of val 0..N-1] ... [byte K-1 of val 0..N-1]
///
/// Output layout:
///   [val 0 bytes 0..K-1] [val 1 bytes 0..K-1] ... [val N-1 bytes 0..K-1]
template <int kNumBytes>
inline void decodeByteStreamSplit(
    const uint8_t* input,
    int32_t numValues,
    uint8_t* output) {
  for (int32_t i = 0; i < numValues; ++i) {
    for (int k = 0; k < kNumBytes; ++k) {
      output[i * kNumBytes + k] = input[k * numValues + i];
    }
  }
}

} // namespace facebook::velox::parquet
