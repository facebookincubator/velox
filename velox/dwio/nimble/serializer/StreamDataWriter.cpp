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

#include "velox/dwio/nimble/serializer/StreamDataWriter.h"

#include <limits>

#include "velox/dwio/nimble/serializer/SerializationHeader.h"

#include <folly/io/Cursor.h>

namespace facebook::nimble::serde::detail {

namespace {

// Upper-bound estimators for the encoded payload of one section. `count` is the
// worst-case number of values to emit; `maxValue` is the largest value that
// can appear (used by Delta/FixedBitWidth; Trivial/Varint ignore it).
size_t estimateTrivialSectionSize(size_t count) {
  // count varint (max 5 bytes) + N u32s.
  return 5 + count * sizeof(uint32_t);
}

size_t estimateVarintSectionSize(size_t count) {
  // count varint + N varints (max 5 bytes each).
  return 5 + count * 5;
}

size_t estimateDeltaSectionSize(size_t count) {
  // count varint + first value varint + (N-1) delta varints (max 5 each).
  return 5 + count * 5;
}

size_t estimateFixedBitWidthSectionSize(size_t count) {
  // bitWidth:1B + count varint + bufferSize (32 bits per element max).
  return 1 + 5 + FixedBitArray::bufferSize(count, /*bitWidth=*/32);
}

size_t estimateSectionSize(EncodingType encodingType, size_t count) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (getTrailerEncodingType(encodingType)) {
    case EncodingType::Trivial:
      return estimateTrivialSectionSize(count);
    case EncodingType::Varint:
      return estimateVarintSectionSize(count);
    case EncodingType::Delta:
      return estimateDeltaSectionSize(count);
    case EncodingType::FixedBitWidth:
      return estimateFixedBitWidthSectionSize(count);
    default:
      NIMBLE_FAIL(
          "Unsupported EncodingType for stream sizes trailer section: {}",
          encodingType);
  }
}

} // namespace

size_t estimateTrailerSize(
    size_t numStreams,
    EncodingType indicesEncodingType,
    EncodingType sizesEncodingType) {
  // [indicesEncType:1B][indicesPayload][sizesEncType:1B][sizesPayload]
  // [trailer_size:u32]
  // Worst case: every stream slot is non-zero, so both axes carry numStreams
  // entries.
  return sizeof(uint8_t) +
      estimateSectionSize(indicesEncodingType, numStreams) + sizeof(uint8_t) +
      estimateSectionSize(sizesEncodingType, numStreams) + sizeof(uint32_t);
}

size_t estimateTrailerSize(
    size_t numPresentStreams,
    size_t numUniqueStreams,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType) {
  // [streamIdsEncType:1B][streamIdsPayload]
  // [sizeIndicesEncType:1B][sizeIndicesPayload]
  // [uniqueSizesEncType:1B][uniqueSizesPayload][trailer_size:u32]
  return sizeof(uint8_t) +
      estimateSectionSize(streamIdsEncodingType, numPresentStreams) +
      sizeof(uint8_t) +
      estimateSectionSize(sizeIndicesEncodingType, numPresentStreams) +
      sizeof(uint8_t) +
      estimateSectionSize(uniqueSizesEncodingType, numUniqueStreams) +
      sizeof(uint32_t);
}

} // namespace facebook::nimble::serde::detail
