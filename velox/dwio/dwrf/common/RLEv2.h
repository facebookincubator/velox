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
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/common/IntDecoder.h"
#include "velox/dwio/common/exception/Exception.h"
#include "velox/dwio/dwrf/common/Common.h"

#include <vector>

namespace facebook::velox::dwrf {

template <bool isSigned>
class RleDecoderV2 : public dwio::common::IntDecoder<isSigned> {
 public:
  using super = dwio::common::IntDecoder<isSigned>;

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

  void nextLengths(int32_t* const data, const int32_t numValues) override {
    for (int i = 0; i < numValues; ++i) {
      data[i] = readValue();
    }
  }

  template <bool hasNulls>
  inline void skip(int32_t numValues, int32_t current, const uint64_t* nulls) {
    if constexpr (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    skip(numValues);
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    if (dwio::common::useFastPath<Visitor, hasNulls>(visitor)) {
      return fastPath<hasNulls>(nulls, visitor);
    }
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
  struct FixedBitSizes {
    enum FBS {
      ONE = 0,
      TWO,
      THREE,
      FOUR,
      FIVE,
      SIX,
      SEVEN,
      EIGHT,
      NINE,
      TEN,
      ELEVEN,
      TWELVE,
      THIRTEEN,
      FOURTEEN,
      FIFTEEN,
      SIXTEEN,
      SEVENTEEN,
      EIGHTEEN,
      NINETEEN,
      TWENTY,
      TWENTYONE,
      TWENTYTWO,
      TWENTYTHREE,
      TWENTYFOUR,
      TWENTYSIX,
      TWENTYEIGHT,
      THIRTY,
      THIRTYTWO,
      FORTY,
      FORTYEIGHT,
      FIFTYSIX,
      SIXTYFOUR
    };
  };

  inline uint32_t decodeBitWidth(uint32_t n) {
    if (n <= FixedBitSizes::TWENTYFOUR) {
      return n + 1;
    } else if (n == FixedBitSizes::TWENTYSIX) {
      return 26;
    } else if (n == FixedBitSizes::TWENTYEIGHT) {
      return 28;
    } else if (n == FixedBitSizes::THIRTY) {
      return 30;
    } else if (n == FixedBitSizes::THIRTYTWO) {
      return 32;
    } else if (n == FixedBitSizes::FORTY) {
      return 40;
    } else if (n == FixedBitSizes::FORTYEIGHT) {
      return 48;
    } else if (n == FixedBitSizes::FIFTYSIX) {
      return 56;
    } else {
      return 64;
    }
  }

  inline uint32_t getClosestFixedBits(uint32_t n) {
    if (n == 0) {
      return 1;
    }
    if (n >= 1 && n <= 24) {
      return n;
    } else if (n > 24 && n <= 26) {
      return 26;
    } else if (n > 26 && n <= 28) {
      return 28;
    } else if (n > 28 && n <= 30) {
      return 30;
    } else if (n > 30 && n <= 32) {
      return 32;
    } else if (n > 32 && n <= 40) {
      return 40;
    } else if (n > 40 && n <= 48) {
      return 48;
    } else if (n > 48 && n <= 56) {
      return 56;
    } else {
      return 64;
    }
  }

  template <bool hasNulls, typename Visitor>
  void fastPath(const uint64_t* nulls, Visitor& visitor) {
    constexpr bool hasFilter =
        !std::is_same_v<typename Visitor::FilterType, common::AlwaysTrue>;
    constexpr bool hasHook =
        !std::is_same_v<typename Visitor::HookType, dwio::common::NoHook>;
    auto numRows = visitor.numRows();
    uint64_t lastRow = visitor.rowAt(numRows - 1);
    int32_t* scatterRows = nullptr;
    int32_t tailSkip = -1;
    auto values = visitor.rawValues(numRows);
    auto filterHits = hasFilter ? visitor.outputRows(numRows) : nullptr;

    if constexpr (hasNulls) {
      // skip NULL values first, Filter should't accept NULLs in FastPath
      auto& outerVector = visitor.outerNonNullRows();
      if (Visitor::dense) {
        dwio::common::nonNullRowsFromDense(nulls, numRows, outerVector);
        if (outerVector.empty()) {
          visitor.setAllNull(hasFilter ? 0 : numRows);
          return;
        }
      } else {
        raw_vector<int32_t>& innerVector = visitor.innerNonNullRows();
        auto anyNulls = dwio::common::nonNullRowsFromSparse < hasFilter,
             !hasFilter &&
            !hasHook >
                (nulls,
                 folly::Range<const int32_t*>(visitor.rows(), numRows),
                 innerVector,
                 outerVector,
                 (hasFilter || hasHook) ? nullptr : visitor.rawNulls(numRows),
                 tailSkip);
        if (anyNulls) {
          visitor.setHasNulls();
        }
        if (innerVector.empty()) {
          skip<false>(tailSkip, 0, nullptr);
          visitor.setAllNull(hasFilter ? 0 : numRows);
          return;
        }
      }
      scatterRows = outerVector.data();
      numRows = outerVector.size();
      lastRow = outerVector.back();
    }

    uint64_t curRow = 0;
    int32_t rowIndex = 0;
    for (;;) {
      if (runLength > runRead) {
        uint64_t toRead = lastRow - curRow + 1;
        auto rows = scatterRows
            ? scatterRows + rowIndex
            : (Visitor::dense ? nullptr : visitor.remainingRows());
        int32_t numInRun;
        switch (type) {
          case SHORT_REPEAT:
            numInRun = nextShortRepeats(
                values, rowIndex, toRead, curRow, nulls, rows, true);
            break;
          case DIRECT:
            numInRun =
                nextDirect(values, rowIndex, toRead, curRow, nulls, rows, true);
            break;
          case PATCHED_BASE:
            numInRun = nextPatched(
                values, rowIndex, toRead, curRow, nulls, rows, true);
            break;
          case DELTA:
            numInRun =
                nextDelta(values, rowIndex, toRead, curRow, nulls, rows, true);
            break;
          default:
            DWIO_RAISE("unknown encoding");
        }
        rowIndex += numInRun;
        if (rowIndex >= numRows) { // reach end
          int32_t numValues = 0;
          visitor.template processRun<hasFilter, hasHook, hasNulls>(
              values, rowIndex, scatterRows, filterHits, values, numValues);
          visitor.setNumValues(hasFilter ? numValues : numRows);
          return;
        }
        VELOX_CHECK(runLength == runRead, "should reach RUN's end.");
      }
      resetRun();
    }

    if constexpr (hasNulls && !Visitor::dense) {
      skip<false>(tailSkip, 0, nullptr);
    }
  }

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

  void readPatchedBaseHeader() {
    // extract the number of fixed bits
    unsigned char fbo = (firstByte >> 1) & 0x1f;
    bitSize = decodeBitWidth(fbo);
    // extract the run length
    runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
    runLength |= readByte();
    // runs are one off
    runLength += 1;
    runRead = 0;
    // extract the number of bytes occupied by base
    uint64_t thirdByte = readByte();
    byteSize = (thirdByte >> 5) & 0x07;
    // base width is one off
    byteSize += 1;
    // extract patch width
    uint32_t pwo = thirdByte & 0x1f;
    patchBitSize = decodeBitWidth(pwo);
    // read fourth byte and extract patch gap width
    uint64_t fourthByte = readByte();
    uint32_t pgw = (fourthByte >> 5) & 0x07;
    // patch gap width is one off
    pgw += 1;
    // extract the length of the patch list
    size_t pl = fourthByte & 0x1f;
    DWIO_ENSURE_NE(
        pl,
        0,
        "Corrupt PATCHED_BASE encoded data (pl==0)! ",
        dwio::common::IntDecoder<isSigned>::inputStream->getName());
    // read the next base width number of bytes to extract base value
    base = readLongBE(byteSize);
    int64_t mask = (static_cast<int64_t>(1) << ((byteSize * 8) - 1));
    // if mask of base value is 1 then base is negative value else positive
    if ((base & mask) != 0) {
      base = base & ~mask;
      base = -base;
    }
    // TODO: something more efficient than resize
    unpacked.resize(runLength);
    unpackedIdx = 0;
    readLongs(unpacked.data(), 0, runLength, bitSize);
    // any remaining bits are thrown out
    resetReadLongs();
    // TODO: something more efficient than resize
    unpackedPatch.resize(pl);
    patchIdx = 0;
    // TODO: Skip corrupt?
    //    if ((patchBitSize + pgw) > 64 && !skipCorrupt) {
    DWIO_ENSURE_LE(
        (patchBitSize + pgw),
        64,
        "Corrupt PATCHED_BASE encoded data (patchBitSize + pgw > 64)! ",
        dwio::common::IntDecoder<isSigned>::inputStream->getName());
    uint32_t cfb = getClosestFixedBits(patchBitSize + pgw);
    readLongs(unpackedPatch.data(), 0, pl, cfb);
    // any remaining bits are thrown out
    resetReadLongs();
    // apply the patch directly when decoding the packed data
    patchMask = ((static_cast<int64_t>(1) << patchBitSize) - 1);
    adjustGapAndPatch();
  }

  void resetRun() {
    resetReadLongs();
    bitSize = 0;
    runRead = 0;
    runLength = 0;
    firstByte = readByte();
    type = static_cast<EncodingType>((firstByte >> 6) & 0x03);
    unsigned char fbo;
    switch (type) {
      case SHORT_REPEAT:
        // extract the number of fixed bytes
        byteSize = (firstByte >> 3) & 0x07;
        byteSize += 1;
        runLength = firstByte & 0x07;
        // run lengths values are stored only after MIN_REPEAT value is met
        runLength += RLE_MINIMUM_REPEAT;
        // read the repeated value which is store using fixed bytes
        firstValue = readLongBE(byteSize);
        if (isSigned) {
          firstValue =
              ZigZag::decode<uint64_t>(static_cast<uint64_t>(firstValue));
        }
        break;
      case DIRECT:
        // extract the number of fixed bits
        bitSize = decodeBitWidth((firstByte >> 1) & 0x1f);
        // extract the run length
        runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
        runLength |= readByte();
        // runs are one off
        runLength += 1;
        break;
      case PATCHED_BASE:
        readPatchedBaseHeader();
        break;
      case DELTA:
        // extract the number of fixed bits
        fbo = (firstByte >> 1) & 0x1f;
        if (fbo != 0) {
          bitSize = decodeBitWidth(fbo);
        } else {
          bitSize = 0;
        }
        // extract the run length
        runLength = static_cast<uint64_t>(firstByte & 0x01) << 8;
        runLength |= readByte();
        ++runLength; // account for first value
        runRead = deltaBase = 0;
        // read the first value stored as vint
        if constexpr (isSigned) {
          firstValue = dwio::common::IntDecoder<isSigned>::readVsLong();
        } else {
          firstValue = static_cast<int64_t>(
              dwio::common::IntDecoder<isSigned>::readVuLong());
        }
        prevValue = firstValue;
        // read the fixed delta value stored as vint
        // deltas can be negative even if all number are positive
        deltaBase = dwio::common::IntDecoder<isSigned>::readVsLong();
        break;
      default:
        DWIO_RAISE("unknown encoding");
    }
  }

  unsigned char readByte() {
    if (super::bufferStart == super::bufferEnd) {
      int32_t bufferLength;
      const void* bufferPointer;
      DWIO_ENSURE(
          super::inputStream->Next(&bufferPointer, &bufferLength),
          "bad read in RleDecoderV2::readByte, ",
          super::inputStream->getName());
      super::bufferStart = static_cast<const char*>(bufferPointer);
      super::bufferEnd = super::bufferStart + bufferLength;
    }

    unsigned char result = static_cast<unsigned char>(*super::bufferStart++);
    return result;
  }

  inline int64_t readLongBE(uint64_t bsz) {
    int64_t ret = 0, val;
    uint64_t n = bsz;
    while (n > 0) {
      n--;
      val = readByte();
      ret |= (val << (n * 8));
    }
    return ret;
  }

  template <typename T>
  uint64_t readLongs(
      T* data,
      uint64_t offset,
      uint64_t len,
      uint64_t fb,
      const uint64_t* nulls = nullptr) {
    uint64_t ret = 0;
    uint64_t remaining = runLength - runRead;

    // TODO: unroll to improve performance
    for (uint64_t i = offset; remaining > 0 && i < (offset + len); i++) {
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
      data[i] = static_cast<T>(result);
      ++ret;
      --remaining;
    }

    return ret;
  }

  template <bool hasValue, typename T>
  inline void tryAdvanceRow(
      const int32_t* rows,
      uint64_t& curRow,
      int32_t& rowIndex,
      uint64_t& offset,
      bool skipNulls = false,
      T value = 0,
      T* data = nullptr) {
    if constexpr (hasValue) {
      if (!rows || curRow == rows[rowIndex]) {
        data[offset++] = value;
        rowIndex++;
      }
      runRead++;
    } else {
      // will skip Nulls in fast path
      if (!skipNulls && (!rows || curRow == rows[rowIndex])) {
        offset++;
        rowIndex++;
      }
    }
    curRow++;
  }

  template <typename T>
  inline int32_t nextShortRepeats(
      T* data,
      uint64_t offset,
      uint64_t numRows,
      uint64_t& curRow,
      const uint64_t* nulls = nullptr,
      const int32_t* rows = nullptr,
      bool skipNulls = false) {
    int32_t startIndex = offset;
    int32_t rowIndex = startIndex;
    uint64_t endRow = curRow + numRows;
    if (nulls) {
      while (runRead < runLength && curRow < endRow) {
        if (bits::isBitNull(nulls, curRow)) {
          tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
        } else {
          tryAdvanceRow<true, T>(
              rows,
              curRow,
              rowIndex,
              offset,
              skipNulls,
              static_cast<T>(firstValue),
              data);
        }
      }
    } else {
      while (runRead < runLength && curRow < endRow) {
        tryAdvanceRow<true, T>(
            rows,
            curRow,
            rowIndex,
            offset,
            skipNulls,
            static_cast<T>(firstValue),
            data);
      }
    }
    return rowIndex - startIndex;
  }

  template <typename T>
  inline int32_t nextDirect(
      T* data,
      uint64_t offset,
      uint64_t toRead,
      uint64_t& curRow,
      const uint64_t* nulls = nullptr,
      const int32_t* rows = nullptr,
      bool skipNulls = false) {
    int32_t startIndex = offset;
    int32_t rowIndex = startIndex;
    uint64_t startRow = curRow;
    uint64_t lastRow = curRow + toRead;
    readLongs(data, offset, toRead, bitSize, nulls); // data may contains NULL
    // compact non NULL rows
    if constexpr (isSigned) {
      while (runRead < runLength && curRow < lastRow) {
        if (nulls && bits::isBitNull(nulls, curRow)) {
          tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
        } else {
          T value = data[startIndex + curRow - startRow];
          value = ZigZag::decode(
              static_cast<typename std::make_unsigned<T>::type>(value));
          tryAdvanceRow<true, T>(
              rows, curRow, rowIndex, offset, skipNulls, value, data);
        }
      }
    } else {
      while (runRead < runLength && curRow < lastRow) {
        if (nulls && bits::isBitNull(nulls, curRow)) {
          tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
        } else {
          T value = data[startIndex + curRow - startRow];
          tryAdvanceRow<true, T>(
              rows, curRow, rowIndex, offset, skipNulls, value, data);
        }
      }
    }
    return rowIndex - startIndex;
  }

  template <typename T>
  inline int32_t nextPatched(
      T* data,
      uint64_t offset,
      uint64_t numRows,
      uint64_t& curRow,
      const uint64_t* nulls = nullptr,
      const int32_t* rows = nullptr,
      bool skipNulls = false) {
    int32_t startIndex = offset;
    int32_t rowIndex = startIndex;
    uint64_t endRow = curRow + numRows;
    while (runRead < runLength && curRow < endRow) {
      // skip null positions
      if (nulls && bits::isBitNull(nulls, curRow)) {
        tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
        continue;
      }
      if (static_cast<int64_t>(unpackedIdx) != actualGap) {
        // no patching required. add base to unpacked value to get final value
        T value = static_cast<T>(base + unpacked[unpackedIdx]);
        tryAdvanceRow<true, T>(
            rows, curRow, rowIndex, offset, skipNulls, value, data);
      } else {
        // extract the patch value
        int64_t patchedVal = unpacked[unpackedIdx] | (curPatch << bitSize);
        // add base to patched value
        T value = static_cast<T>(base + patchedVal);
        tryAdvanceRow<true, T>(
            rows, curRow, rowIndex, offset, skipNulls, value, data);
        // increment the patch to point to next entry in patch list
        ++patchIdx;
        if (patchIdx < unpackedPatch.size()) {
          adjustGapAndPatch();
          // next gap is relative to the current gap
          actualGap += unpackedIdx;
        }
      }
      ++unpackedIdx;
    }
    return rowIndex - startIndex;
  }

  template <typename T>
  inline int32_t nextDelta(
      T* data,
      uint64_t offset,
      uint64_t numRows,
      uint64_t& curRow,
      const uint64_t* nulls = nullptr,
      const int32_t* rows = nullptr,
      bool skipNulls = false) {
    int32_t startIndex = offset;
    int32_t rowIndex = startIndex;
    uint64_t endRow = curRow + numRows;
    // skip null positions
    while (runRead < runLength && curRow < endRow) {
      if (!nulls || !bits::isBitNull(nulls, curRow)) {
        break;
      }
      // target value is NULL, advance to next
      tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
    }
    // deal with Run's first value
    if (runRead == 0 && curRow < endRow) {
      tryAdvanceRow<true, T>(
          rows,
          curRow,
          rowIndex,
          offset,
          skipNulls,
          static_cast<T>(firstValue),
          data);
    }

    if (bitSize == 0) {
      // add fixed deltas to adjacent values
      while (runRead < runLength && curRow < endRow) {
        if (nulls && bits::isBitNull(nulls, curRow)) {
          tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
        } else {
          prevValue += deltaBase;
          tryAdvanceRow<true, T>(
              rows,
              curRow,
              rowIndex,
              offset,
              skipNulls,
              static_cast<T>(prevValue),
              data);
        }
      }
    } else {
      while (runRead < runLength && curRow < endRow) {
        // skip null positions
        if (!nulls || !bits::isBitNull(nulls, curRow)) {
          break;
        }
        tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
      }
      if (runRead < 2 && curRow < endRow) {
        // add delta base and first value
        prevValue = firstValue + deltaBase;
        tryAdvanceRow<true, T>(
            rows,
            curRow,
            rowIndex,
            offset,
            skipNulls,
            static_cast<T>(prevValue),
            data);
      }

      // write the unpacked values, add it to previous value and store final
      // value to result buffer. if the delta base value is negative then it
      // is a decreasing sequence else an increasing sequence
      uint64_t remaining = endRow - curRow;
      readLongs(data, offset, remaining, bitSize, nulls);
      uint64_t startOff = offset;
      uint64_t startRow = curRow;
      if (deltaBase < 0) {
        while (runRead < runLength && curRow < endRow) {
          if (nulls && bits::isBitNull(nulls, curRow)) {
            tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
          } else {
            prevValue = prevValue - data[startOff + curRow - startRow];
            tryAdvanceRow<true, T>(
                rows,
                curRow,
                rowIndex,
                offset,
                skipNulls,
                static_cast<T>(prevValue),
                data);
          }
        }
      } else {
        while (runRead < runLength && curRow < endRow) {
          if (nulls && bits::isBitNull(nulls, curRow)) {
            tryAdvanceRow<false, T>(rows, curRow, rowIndex, offset, skipNulls);
          } else {
            prevValue = prevValue + data[startOff + curRow - startRow];
            tryAdvanceRow<true, T>(
                rows,
                curRow,
                rowIndex,
                offset,
                skipNulls,
                static_cast<T>(prevValue),
                data);
          }
        }
      }
    }
    return rowIndex - startIndex;
  }

  int64_t readValue();

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
  EncodingType type;
  dwio::common::DataBuffer<int64_t> unpacked; // Used by PATCHED_BASE
  dwio::common::DataBuffer<int64_t> unpackedPatch; // Used by PATCHED_BASE
};

} // namespace facebook::velox::dwrf
