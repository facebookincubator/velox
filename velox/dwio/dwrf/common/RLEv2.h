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

#include "velox/common/base/Nulls.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/Adaptor.h"
#include "velox/dwio/common/DataBuffer.h"
#include "velox/dwio/common/IntDecoder.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/dwio/dwrf/common/IntEncoder.h"

#include <vector>

namespace facebook::velox::dwrf {

#define MAX_LITERAL_SIZE 512
#define MAX_SHORT_REPEAT_LENGTH 10
#define MIN_REPEAT 3
#define HIST_LEN 32

enum EncodingType { SHORT_REPEAT = 0, DIRECT = 1, PATCHED_BASE = 2, DELTA = 3 };

struct EncodingOption {
  EncodingType encoding;
  int64_t fixedDelta;
  int64_t gapVsPatchListCount;
  int64_t zigzagLiteralsCount;
  int64_t baseRedLiteralsCount;
  int64_t adjDeltasCount;
  uint32_t zzBits90p;
  uint32_t zzBits100p;
  uint32_t brBits95p;
  uint32_t brBits100p;
  uint32_t bitsDeltaMax;
  uint32_t patchWidth;
  uint32_t patchGapWidth;
  uint32_t patchLength;
  int64_t min;
  bool isFixedDelta;
};

template <bool isSigned>
class RleEncoderV2 : public IntEncoder<isSigned> {
 public:
  RleEncoderV2(
      std::unique_ptr<BufferedOutputStream> outStream,
      bool useVInts,
      uint32_t numBytes)
      : IntEncoder<isSigned>{std::move(outStream), useVInts, numBytes},
        numLiterals(0),
        alignedBitPacking{true},
        fixedRunLength(0),
        variableRunLength(0),
        prevDelta{0} {
    literals = new int64_t[MAX_LITERAL_SIZE];
    gapVsPatchList = new int64_t[MAX_LITERAL_SIZE];
    zigzagLiterals = isSigned ? new int64_t[MAX_LITERAL_SIZE] : nullptr;
    baseRedLiterals = new int64_t[MAX_LITERAL_SIZE];
    adjDeltas = new int64_t[MAX_LITERAL_SIZE];
  }

  ~RleEncoderV2() override {
    delete[] literals;
    delete[] gapVsPatchList;
    delete[] zigzagLiterals;
    delete[] baseRedLiterals;
    delete[] adjDeltas;
  }

  // For 64 bit Integers, only signed type is supported. writeVuLong only
  // supports int64_t and it needs to support uint64_t before this method
  // can support uint64_t overload.
  uint64_t add(
      const int64_t* data,
      const common::Ranges& ranges,
      const uint64_t* nulls) override {
    return addImpl(data, ranges, nulls);
  }

  uint64_t add(
      const int32_t* data,
      const common::Ranges& ranges,
      const uint64_t* nulls) override {
    return addImpl(data, ranges, nulls);
  }

  uint64_t add(
      const uint32_t* data,
      const common::Ranges& ranges,
      const uint64_t* nulls) override {
    return addImpl(data, ranges, nulls);
  }

  uint64_t add(
      const int16_t* data,
      const common::Ranges& ranges,
      const uint64_t* nulls) override {
    return addImpl(data, ranges, nulls);
  }

  uint64_t add(
      const uint16_t* data,
      const common::Ranges& ranges,
      const uint64_t* nulls) override {
    return addImpl(data, ranges, nulls);
  }

  void writeValue(const int64_t value) override {
    write(value);
  }

  uint64_t flush() override {
    if (numLiterals != 0) {
      EncodingOption option = {};
      if (variableRunLength != 0) {
        determineEncoding(option);
        writeValues(option);
      } else if (fixedRunLength != 0) {
        if (fixedRunLength < MIN_REPEAT) {
          variableRunLength = fixedRunLength;
          fixedRunLength = 0;
          determineEncoding(option);
          writeValues(option);
        } else if (
            fixedRunLength >= MIN_REPEAT &&
            fixedRunLength <= MAX_SHORT_REPEAT_LENGTH) {
          option.encoding = SHORT_REPEAT;
          writeValues(option);
        } else {
          option.encoding = DELTA;
          option.isFixedDelta = true;
          writeValues(option);
        }
      }
    }
    return IntEncoder<isSigned>::flush();
  }

  // copied from RLEv1.h
  void recordPosition(PositionRecorder& recorder, int32_t strideIndex = -1)
      const override {
    IntEncoder<isSigned>::recordPosition(recorder, strideIndex);
    recorder.add(static_cast<uint64_t>(numLiterals), strideIndex);
  }

 private:
  int64_t* literals;
  int32_t numLiterals;
  const bool alignedBitPacking;
  uint32_t fixedRunLength;
  uint32_t variableRunLength;
  int64_t prevDelta;
  int32_t histgram[HIST_LEN];

  // The four list below should actually belong to EncodingOption since it only
  // holds temporal values in write(int64_t val), it is move here for
  // performance consideration.
  int64_t* gapVsPatchList;
  int64_t* zigzagLiterals;
  int64_t* baseRedLiterals;
  int64_t* adjDeltas;

  uint32_t getOpCode(EncodingType encoding);
  int64_t* prepareForDirectOrPatchedBase(EncodingOption& option);
  void determineEncoding(EncodingOption& option);
  void computeZigZagLiterals(EncodingOption& option);
  void preparePatchedBlob(EncodingOption& option);
  void writeInts(int64_t* input, uint32_t offset, size_t len, uint32_t bitSize);
  void initializeLiterals(int64_t val);
  void writeValues(EncodingOption& option);
  void writeShortRepeatValues(EncodingOption& option);
  void writeDirectValues(EncodingOption& option);
  void writePatchedBasedValues(EncodingOption& option);
  void writeDeltaValues(EncodingOption& option);
  uint32_t percentileBits(
      int64_t* data,
      size_t offset,
      size_t length,
      double p,
      bool reuseHist = false);

  template <typename T>
  void write(T val) {
    if (numLiterals == 0) {
      initializeLiterals(val);
      return;
    }

    if (numLiterals == 1) {
      prevDelta = val - literals[0];
      literals[numLiterals++] = val;

      if (val == literals[0]) {
        fixedRunLength = 2;
        variableRunLength = 0;
      } else {
        fixedRunLength = 0;
        variableRunLength = 2;
      }
      return;
    }

    int64_t currentDelta = val - literals[numLiterals - 1];
    EncodingOption option = {};
    if (prevDelta == 0 && currentDelta == 0) {
      // case 1: fixed delta run
      literals[numLiterals++] = val;

      if (variableRunLength > 0) {
        // if variable run is non-zero then we are seeing repeating
        // values at the end of variable run in which case fixed Run
        // length is 2
        fixedRunLength = 2;
      }
      fixedRunLength++;

      // if fixed run met the minimum condition and if variable
      // run is non-zero then flush the variable run and shift the
      // tail fixed runs to start of the buffer
      if (fixedRunLength >= MIN_REPEAT && variableRunLength > 0) {
        numLiterals -= MIN_REPEAT;
        variableRunLength -= (MIN_REPEAT - 1);

        determineEncoding(option);
        writeValues(option);

        // shift tail fixed runs to beginning of the buffer
        for (size_t i = 0; i < MIN_REPEAT; ++i) {
          literals[i] = val;
        }
        numLiterals = MIN_REPEAT;
      }

      if (fixedRunLength == MAX_LITERAL_SIZE) {
        option.encoding = DELTA;
        option.isFixedDelta = true;
        writeValues(option);
      }
      return;
    }

    // case 2: variable delta run

    // if fixed run length is non-zero and if it satisfies the
    // short repeat conditions then write the values as short repeats
    // else use delta encoding
    if (fixedRunLength >= MIN_REPEAT) {
      if (fixedRunLength <= MAX_SHORT_REPEAT_LENGTH) {
        option.encoding = SHORT_REPEAT;
      } else {
        option.encoding = DELTA;
        option.isFixedDelta = true;
      }
      writeValues(option);
    }

    // if fixed run length is <MIN_REPEAT and current value is
    // different from previous then treat it as variable run
    if (fixedRunLength > 0 && fixedRunLength < MIN_REPEAT &&
        val != literals[numLiterals - 1]) {
      variableRunLength = fixedRunLength;
      fixedRunLength = 0;
    }

    // after writing values re-initialize the variables
    if (numLiterals == 0) {
      initializeLiterals(val);
    } else {
      prevDelta = val - literals[numLiterals - 1];
      literals[numLiterals++] = val;
      variableRunLength++;

      if (variableRunLength == MAX_LITERAL_SIZE) {
        determineEncoding(option);
        writeValues(option);
      }
    }
  }

  template <typename T>
  uint64_t
  addImpl(const T* data, const common::Ranges& ranges, const uint64_t* nulls);
};

template <bool isSigned>
template <typename T>
uint64_t RleEncoderV2<isSigned>::addImpl(
    const T* data,
    const common::Ranges& ranges,
    const uint64_t* nulls) {
  uint64_t count = 0;
  if (nulls) {
    for (auto& pos : ranges) {
      if (!bits::isBitNull(nulls, pos)) {
        write(data[pos]);
        ++count;
      }
    }
  } else {
    for (auto& pos : ranges) {
      write(data[pos]);
      ++count;
    }
  }
  return count;
}

template <bool isSigned>
class RleDecoderV2 : public dwio::common::IntDecoder<isSigned> {
 public:
  enum EncodingType {
    SHORT_REPEAT = 0,
    DIRECT = 1,
    PATCHED_BASE = 2,
    DELTA = 3
  };

  RleDecoderV2(
      std::unique_ptr<dwio::common::SeekableInputStream> input,
      memory::MemoryPool& pool);

  /**
   * Seek to a specific row group.
   */
  void seekToRowGroup(dwio::common::PositionProvider&) override;

  /**
   * Seek over a given number of values.
   */
  void skip(uint64_t numValues) override;

  /**
   * Read a number of values into the batch.
   */
  void next(int64_t* data, uint64_t numValues, const uint64_t* nulls) override;

  void nextLengths(int32_t* const data, const int32_t numValues) {
    for (int i = 0; i < numValues; ++i) {
      data[i] = readValue();
    }
  }

  int64_t readShortRepeatsValue() {
    int64_t value;
    uint64_t n = nextShortRepeats(&value, 0, 1, nullptr);
    VELOX_CHECK(n == (uint64_t)1);
    return value;
  }

  int64_t readDirectValue() {
    int64_t value;
    uint64_t n = nextDirect(&value, 0, 1, nullptr);
    VELOX_CHECK(n == (uint64_t)1);
    return value;
  }

  int64_t readPatchedBaseValue() {
    int64_t value;
    uint64_t n = nextPatched(&value, 0, 1, nullptr);
    VELOX_CHECK(n == (uint64_t)1);
    return value;
  }

  int64_t readDeltaValue() {
    int64_t value;
    uint64_t n = nextDelta(&value, 0, 1, nullptr);
    VELOX_CHECK(n == (uint64_t)1);
    return value;
  }

  int64_t readValue() {
    if (runRead == runLength) {
      resetRun();
      firstByte = readByte();
    }

    int64_t value = 0;
    auto type = static_cast<EncodingType>((firstByte >> 6) & 0x03);
    if (type == SHORT_REPEAT) {
      value = readShortRepeatsValue();
    } else if (type == DIRECT) {
      value = readDirectValue();
    } else if (type == PATCHED_BASE) {
      value = readPatchedBaseValue();
    } else if (type == DELTA) {
      value = readDeltaValue();
    } else {
      DWIO_RAISE("unknown encoding");
    }

    return value;
  }

  template <bool hasNulls>
  void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if constexpr (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    skip(numValues);
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
        auto value = readValue();
        toSkip = visitor.process(value, atEnd);
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
  // Used by PATCHED_BASE
  void adjustGapAndPatch() {
    curGap = static_cast<uint64_t>(unpackedPatch[patchIdx]) >> patchBitSize;
    curPatch = unpackedPatch[patchIdx] & patchMask;
    actualGap = 0;

    // special case: gap is >255 then patch value will be 0.
    // if gap is <=255 then patch value cannot be 0
    while (curGap == 255 && curPatch == 0) {
      actualGap += 255;
      ++patchIdx;
      curGap = static_cast<uint64_t>(unpackedPatch[patchIdx]) >> patchBitSize;
      curPatch = unpackedPatch[patchIdx] & patchMask;
    }
    // add the left over gap
    actualGap += curGap;
  }

  void resetReadLongs() {
    bitsLeft = 0;
    curByte = 0;
  }

  void resetRun() {
    resetReadLongs();
    bitSize = 0;
  }

  unsigned char readByte() {
    if (dwio::common::IntDecoder<isSigned>::bufferStart ==
        dwio::common::IntDecoder<isSigned>::bufferEnd) {
      int32_t bufferLength;
      const void* bufferPointer;
      DWIO_ENSURE(
          dwio::common::IntDecoder<isSigned>::inputStream->Next(
              &bufferPointer, &bufferLength),
          "bad read in RleDecoderV2::readByte, ",
          dwio::common::IntDecoder<isSigned>::inputStream->getName());
      dwio::common::IntDecoder<isSigned>::bufferStart =
          static_cast<const char*>(bufferPointer);
      dwio::common::IntDecoder<isSigned>::bufferEnd =
          dwio::common::IntDecoder<isSigned>::bufferStart + bufferLength;
    }

    unsigned char result = static_cast<unsigned char>(
        *dwio::common::IntDecoder<isSigned>::bufferStart++);
    return result;
  }

  int64_t readLongBE(uint64_t bsz);
  uint64_t readLongs(
      int64_t* data,
      uint64_t offset,
      uint64_t len,
      uint64_t fb,
      const uint64_t* nulls = nullptr) {
    uint64_t ret = 0;

    // TODO: unroll to improve performance
    for (uint64_t i = offset; i < (offset + len); i++) {
      // skip null positions
      if (nulls && bits::isBitNull(nulls, i)) {
        continue;
      }
      uint64_t result = 0;
      uint64_t bitsLeftToRead = fb;
      while (bitsLeftToRead > bitsLeft) {
        result <<= bitsLeft;
        result |= curByte & ((1 << bitsLeft) - 1);
        bitsLeftToRead -= bitsLeft;
        curByte = readByte();
        bitsLeft = 8;
      }

      // handle the left over bits
      if (bitsLeftToRead > 0) {
        result <<= bitsLeftToRead;
        bitsLeft -= static_cast<uint32_t>(bitsLeftToRead);
        result |= (curByte >> bitsLeft) & ((1 << bitsLeftToRead) - 1);
      }
      data[i] = static_cast<int64_t>(result);
      ++ret;
    }

    return ret;
  }

  uint64_t nextShortRepeats(
      int64_t* data,
      uint64_t offset,
      uint64_t numValues,
      const uint64_t* nulls);
  uint64_t nextDirect(
      int64_t* data,
      uint64_t offset,
      uint64_t numValues,
      const uint64_t* nulls);
  uint64_t nextPatched(
      int64_t* data,
      uint64_t offset,
      uint64_t numValues,
      const uint64_t* nulls);
  uint64_t nextDelta(
      int64_t* data,
      uint64_t offset,
      uint64_t numValues,
      const uint64_t* nulls);

  unsigned char firstByte;
  uint64_t runLength;
  uint64_t runRead;
  int64_t deltaBase; // Used by DELTA
  uint64_t byteSize; // Used by SHORT_REPEAT and PATCHED_BASE
  int64_t firstValue; // Used by SHORT_REPEAT and DELTA
  int64_t prevValue; // Used by DELTA
  uint32_t bitSize; // Used by DIRECT, PATCHED_BASE and DELTA
  uint32_t bitsLeft; // Used by anything that uses readLongs
  uint32_t curByte; // Used by anything that uses readLongs
  uint32_t patchBitSize; // Used by PATCHED_BASE
  uint64_t unpackedIdx; // Used by PATCHED_BASE
  uint64_t patchIdx; // Used by PATCHED_BASE
  int64_t base; // Used by PATCHED_BASE
  uint64_t curGap; // Used by PATCHED_BASE
  int64_t curPatch; // Used by PATCHED_BASE
  int64_t patchMask; // Used by PATCHED_BASE
  int64_t actualGap; // Used by PATCHED_BASE
  dwio::common::DataBuffer<int64_t> unpacked; // Used by PATCHED_BASE
  dwio::common::DataBuffer<int64_t> unpackedPatch; // Used by PATCHED_BASE
};

} // namespace facebook::velox::dwrf
