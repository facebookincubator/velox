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

#include "velox/dwio/dwrf/common/RLEv2.h"
#include "velox/dwio/common/SeekableInputStream.h"

namespace facebook::velox::dwrf {

using memory::MemoryPool;

template <bool isSigned>
RleDecoderV2<isSigned>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool)
    : dwio::common::IntDecoder<isSigned>{std::move(input), false, 0},
      firstByte(0),
      runLength(0),
      runRead(0),
      deltaBase(0),
      byteSize(0),
      firstValue(0),
      prevValue(0),
      bitSize(0),
      bitsLeft(0),
      curByte(0),
      patchBitSize(0),
      unpackedIdx(0),
      patchIdx(0),
      base(0),
      curGap(0),
      curPatch(0),
      patchMask(0),
      actualGap(0),
      unpacked(pool, 0),
      unpackedPatch(pool, 0) {
  // PASS
}

template RleDecoderV2<true>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool);
template RleDecoderV2<false>::RleDecoderV2(
    std::unique_ptr<dwio::common::SeekableInputStream> input,
    MemoryPool& pool);

template <bool isSigned>
void RleDecoderV2<isSigned>::seekToRowGroup(
    dwio::common::PositionProvider& location) {
  // move the input stream
  dwio::common::IntDecoder<isSigned>::inputStream->seekToPosition(location);
  // clear state
  dwio::common::IntDecoder<isSigned>::bufferEnd =
      dwio::common::IntDecoder<isSigned>::bufferStart = 0;
  runRead = runLength = 0;
  // skip ahead the given number of records
  skip(location.next());
}

template void RleDecoderV2<true>::seekToRowGroup(
    dwio::common::PositionProvider& location);
template void RleDecoderV2<false>::seekToRowGroup(
    dwio::common::PositionProvider& location);

template <bool isSigned>
void RleDecoderV2<isSigned>::skip(uint64_t numValues) {
  // simple for now, until perf tests indicate something encoding specific is
  // needed
  const uint64_t N = 64;
  int64_t dummy[N];

  while (numValues) {
    uint64_t nRead = std::min(N, numValues);
    next(dummy, nRead, nullptr);
    numValues -= nRead;
  }
}

template void RleDecoderV2<true>::skip(uint64_t numValues);
template void RleDecoderV2<false>::skip(uint64_t numValues);

template <bool isSigned>
void RleDecoderV2<isSigned>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls) {
  uint64_t nRead = 0;

  while (nRead < numValues) {
    // Skip any nulls before attempting to read first byte.
    while (nulls && bits::isBitNull(nulls, nRead)) {
      if (++nRead == numValues) {
        return; // ended with null values
      }
    }

    if (runRead == runLength) {
      resetRun();
    }

    uint64_t offset = nRead;
    uint64_t length = numValues - nRead;
    uint64_t curRow = offset;

    switch (type) {
      case SHORT_REPEAT:
        nRead += nextShortRepeats(data, offset, length, curRow, nulls);
        break;
      case DIRECT:
        nRead += nextDirect(data, offset, length, curRow, nulls);
        break;
      case PATCHED_BASE:
        nRead += nextPatched(data, offset, length, curRow, nulls);
        break;
      case DELTA:
        nRead += nextDelta(data, offset, length, curRow, nulls);
        break;
      default:
        DWIO_RAISE("unknown encoding");
    }
  }
}

template void RleDecoderV2<true>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls);
template void RleDecoderV2<false>::next(
    int64_t* const data,
    const uint64_t numValues,
    const uint64_t* const nulls);

template <bool isSigned>
int64_t RleDecoderV2<isSigned>::readValue() {
  if (runRead == runLength) {
    resetRun();
  }

  uint64_t curRow = 0;
  uint64_t nRead = 0;
  int64_t value = 0;
  switch (type) {
    case SHORT_REPEAT:
      nRead = nextShortRepeats(&value, 0, 1, curRow);
      break;
    case DIRECT:
      nRead = nextDirect(&value, 0, 1, curRow);
      break;
    case PATCHED_BASE:
      nRead = nextPatched(&value, 0, 1, curRow);
      break;
    case DELTA:
      nRead = nextDelta(&value, 0, 1, curRow);
      break;
    default:
      DWIO_RAISE("unknown encoding");
  }
  VELOX_CHECK(nRead == (uint64_t)1);
  return value;
}

template int64_t RleDecoderV2<true>::readValue();

template int64_t RleDecoderV2<false>::readValue();

} // namespace facebook::velox::dwrf
