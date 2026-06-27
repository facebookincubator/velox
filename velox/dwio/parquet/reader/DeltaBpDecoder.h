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

#include <array>

#include <folly/Varint.h>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Nulls.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {

// DeltaBpDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-12.0.0/cpp/src/parquet/encoding.cc#LL2357C18-L2586C3
class DeltaBpDecoder {
 public:
  /// Trailing readable bytes the SIMD kernel needs past the last
  /// page byte; PageReader::kPageReadPadding must be >= this.
  static constexpr int kRequiredTrailingPadding = 8;

  explicit DeltaBpDecoder(const char* start) : bufferStart_(start) {
    initHeader();
  }

  void reset(const char* start) {
    bufferStart_ = start;
    initHeader();
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
    skipValues(numValues);
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* nulls, Visitor visitor) {
    int32_t current = visitor.start();
    skip<hasNulls>(current, 0, nulls);
    if constexpr (
        Visitor::dense && !hasNulls && Visitor::FilterType::deterministic &&
        std::is_same_v<typename Visitor::HookType, dwio::common::NoHook> &&
        std::is_integral_v<typename Visitor::DataType>) {
      readWithVisitorDenseBatched(visitor);
      return;
    }
    if constexpr (
        !Visitor::dense && !hasNulls && Visitor::FilterType::deterministic &&
        std::is_same_v<typename Visitor::HookType, dwio::common::NoHook> &&
        std::is_integral_v<typename Visitor::DataType>) {
      readWithVisitorSparseBuffered(visitor);
      return;
    }
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
        toSkip = visitor.process(readLong(), atEnd);
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

  const char* bufferStart() {
    return bufferStart_;
  }

  int64_t validValuesCount() {
    return static_cast<int64_t>(totalValuesRemaining_);
  }

  template <typename T>
  FOLLY_ALWAYS_INLINE void readValues(T* values, int32_t numValues) {
    VELOX_DCHECK_LE(numValues, totalValuesRemaining_);
    if constexpr (std::is_integral_v<T>) {
      decodeLongs(values, numValues);
    } else {
      for (auto i = 0; i < numValues; i++) {
        values[i] = T(readLong());
      }
    }
  }

 private:
  static constexpr int32_t kBatch = 1024;

  /// Skip 'numValues' values efficiently. For constant-delta miniblocks
  /// (bitWidth==0), compute the final value in O(1). For other cases, use
  /// the batched decode path which is faster than per-value readLong().
  void skipValues(int32_t numValues) {
    if (numValues <= 0) {
      return;
    }

    // Handle the first value (block header) if block is uninitialized.
    if (!firstBlockInitialized_ && numValues > 0) {
      readLong();
      --numValues;
      if (numValues <= 0) {
        return;
      }
    }

    while (numValues > 0) {
      if (valuesRemainingCurrentMiniBlock_ == 0) {
        advanceMiniBlock();
      }

      auto toSkipInBlock = std::min<int32_t>(
          numValues,
          std::min<int32_t>(
              valuesRemainingCurrentMiniBlock_, totalValuesRemaining_));

      if (deltaBitWidth_ == 0) {
        // Constant-delta miniblock: compute lastValue in O(1).
        lastValue_ = static_cast<int64_t>(
            static_cast<uint64_t>(lastValue_) +
            static_cast<uint64_t>(minDelta_) *
                static_cast<uint64_t>(toSkipInBlock));
        valuesRemainingCurrentMiniBlock_ -= toSkipInBlock;
        totalValuesRemaining_ -= toSkipInBlock;
        if (valuesRemainingCurrentMiniBlock_ == 0 ||
            totalValuesRemaining_ == 0) {
          bufferStart_ += bits::nbytes(deltaBitWidth_ * valuesPerMiniBlock_);
        }
      } else {
        // Non-constant delta: for skip we only need the final lastValue,
        // not intermediate prefix-sums. When skipping full miniblocks from
        // their start, compute sum(deltas) directly (no dependency chain).
        if (toSkipInBlock == static_cast<int32_t>(valuesRemainingCurrentMiniBlock_) &&
            valuesRemainingCurrentMiniBlock_ == valuesPerMiniBlock_) {
          // Full miniblock from start: sum deltas without prefix-sum.
          uint64_t deltaSum = sumMiniBlockDeltas(
              bufferStart_, valuesPerMiniBlock_, deltaBitWidth_);
          lastValue_ = static_cast<int64_t>(
              static_cast<uint64_t>(lastValue_) +
              static_cast<uint64_t>(minDelta_) * valuesPerMiniBlock_ +
              deltaSum);
          valuesRemainingCurrentMiniBlock_ = 0;
          totalValuesRemaining_ -= toSkipInBlock;
          bufferStart_ += bits::nbytes(deltaBitWidth_ * valuesPerMiniBlock_);
        } else {
          // Partial miniblock: must decode (prefix-sum needed for lastValue).
          int64_t scratch[kBatch];
          int32_t skipped = 0;
          while (skipped < toSkipInBlock) {
            auto batch =
                std::min<int32_t>(kBatch, toSkipInBlock - skipped);
            decodeLongs(scratch, batch);
            skipped += batch;
          }
        }
      }
      numValues -= toSkipInBlock;
    }
  }

  template <typename Visitor>
  void readWithVisitorDenseBatched(Visitor& visitor) {
    using DataType = typename Visitor::DataType;
    constexpr bool kHasFilter =
        !std::
            is_same_v<typename Visitor::FilterType, velox::common::AlwaysTrue>;
    const int32_t total = visitor.numRows();
    DataType* output = visitor.rawValues(total);
    int32_t* filterHits = kHasFilter ? visitor.outputRows(total) : nullptr;
    int32_t numValues = 0;
    int32_t consumed = 0;
    while (consumed < total) {
      const int32_t n = std::min<int32_t>(kBatch, total - consumed);
      DataType* dst = output + numValues;
      decodeLongs(dst, n);
      visitor.template processRun<
          kHasFilter,
          /*hasHook=*/false,
          /*scatter=*/false>(
          dst,
          n,
          /*scatterRows=*/nullptr,
          filterHits,
          output,
          numValues);
      consumed += n;
    }
    visitor.setNumValues(numValues);
  }

  // DELTA's chain forces every physical row to be decoded; batch the
  // decode side and run the scalar visitor.process() over a buffer.
  template <typename Visitor>
  void readWithVisitorSparseBuffered(Visitor& visitor) {
    using DataType = typename Visitor::DataType;
    DataType buf[kBatch];

    const auto* rows = visitor.rows();
    const int32_t numRows = visitor.numRows();
    int32_t rowIdx = 0;
    int32_t currentPhys = (numRows > 0) ? rows[0] : 0;
    bool atEnd = false;
    while (!atEnd && rowIdx < numRows) {
      const int32_t remaining = static_cast<int32_t>(totalValuesRemaining_);
      if (remaining <= 0) {
        return;
      }
      // Cap n to the visitor's residual span; over-decoding misaligns
      // the next readWithVisitor call.
      const int32_t lastRow = rows[numRows - 1];
      const int32_t maxSpan = lastRow - currentPhys + 1;
      const int32_t n = std::min<int32_t>({kBatch, remaining, maxSpan});
      decodeLongs(buf, n);
      const int32_t batchPhysStart = currentPhys;
      const int32_t batchPhysEnd = currentPhys + n; // exclusive
      currentPhys = batchPhysEnd;
      while (rowIdx < numRows && rows[rowIdx] < batchPhysEnd) {
        const int32_t i = rows[rowIdx] - batchPhysStart;
        visitor.process(buf[i], atEnd);
        ++rowIdx;
        if (atEnd) {
          return;
        }
      }
    }
  }

  template <typename DataType>
  void decodeLongs(DataType* out, int32_t n) {
    const char* bufStart = bufferStart_;
    uint64_t valuesPerMiniBlock = valuesPerMiniBlock_;
    uint64_t miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
    uint64_t totalRemaining = totalValuesRemaining_;
    int64_t lastValue = lastValue_;
    int64_t minDelta = minDelta_;
    uint64_t deltaBitWidth = deltaBitWidth_;
    uint64_t bitOffset =
        (valuesPerMiniBlock - miniBlockRemaining) * deltaBitWidth;

    int32_t i = 0;
    while (i < n) {
      if (miniBlockRemaining == 0) {
        if (!firstBlockInitialized_) {
          bufferStart_ = bufStart;
          valuesRemainingCurrentMiniBlock_ = 0;
          totalValuesRemaining_ = totalRemaining;
          lastValue_ = lastValue;
          int64_t v = readLong();
          out[i] = static_cast<DataType>(v);
          bufStart = bufferStart_;
          miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
          totalRemaining = totalValuesRemaining_;
          lastValue = lastValue_;
          minDelta = minDelta_;
          deltaBitWidth = deltaBitWidth_;
          bitOffset = (valuesPerMiniBlock - miniBlockRemaining) * deltaBitWidth;
          ++i;
          continue;
        }
        bufferStart_ = bufStart;
        valuesRemainingCurrentMiniBlock_ = 0;
        totalValuesRemaining_ = totalRemaining;
        lastValue_ = lastValue;
        advanceMiniBlock();
        bufStart = bufferStart_;
        miniBlockRemaining = valuesRemainingCurrentMiniBlock_;
        minDelta = minDelta_;
        deltaBitWidth = deltaBitWidth_;
        bitOffset = 0;
      }

      if (miniBlockRemaining == valuesPerMiniBlock &&
          static_cast<uint64_t>(n - i) >= valuesPerMiniBlock &&
          totalRemaining >= valuesPerMiniBlock && deltaBitWidth <= 32) {
        const int32_t miniBlockValues =
            static_cast<int32_t>(valuesPerMiniBlock);
        DataType* dst = out + i;
        const bool dispatched = dispatchSimdMiniBlock<DataType>(
            deltaBitWidth, bufStart, miniBlockValues, minDelta, lastValue, dst);
        if (dispatched) {
          bufStart += bits::nbytes(deltaBitWidth * valuesPerMiniBlock);
          const uint64_t consumed = valuesPerMiniBlock;
          miniBlockRemaining = 0;
          totalRemaining -= consumed;
          i += static_cast<int32_t>(consumed);
          bitOffset = 0;
          continue;
        }
      }

      uint64_t value = 0;
      if (deltaBitWidth) {
        value = bits::detail::loadBits<uint64_t>(
            reinterpret_cast<const uint64_t*>(bufStart),
            bitOffset,
            deltaBitWidth);
        value &= (~0ULL >> (64 - deltaBitWidth));
      }
      uint64_t result = static_cast<uint64_t>(minDelta) + value +
          static_cast<uint64_t>(lastValue);
      lastValue = static_cast<int64_t>(result);
      out[i] = static_cast<DataType>(result);
      bitOffset += deltaBitWidth;
      --miniBlockRemaining;
      --totalRemaining;
      if (miniBlockRemaining == 0 || totalRemaining == 0) {
        bufStart += bits::nbytes(deltaBitWidth * valuesPerMiniBlock);
        bitOffset = 0;
      }
      ++i;
    }

    bufferStart_ = bufStart;
    valuesRemainingCurrentMiniBlock_ = miniBlockRemaining;
    totalValuesRemaining_ = totalRemaining;
    lastValue_ = lastValue;
  }

  /// Compute the sum of packed deltas in a miniblock without a prefix-sum
  /// dependency chain. Used by skipValues() when only the final lastValue
  /// is needed (not intermediate values). This is a simple horizontal
  /// reduction that the compiler can auto-vectorize.
  uint64_t sumMiniBlockDeltas(
      const char* src,
      uint64_t numValues,
      uint64_t bitWidth) {
    if (bitWidth == 0) {
      return 0;
    }
    const uint64_t mask =
        (bitWidth >= 64) ? ~0ULL : ((1ULL << bitWidth) - 1);
    uint64_t sum = 0;
    for (uint64_t i = 0; i < numValues; ++i) {
      uint64_t bitPos = i * bitWidth;
      uint64_t byteOff = bitPos >> 3;
      uint64_t bitInByte = bitPos & 7;
      uint64_t word = *reinterpret_cast<const uint64_t*>(src + byteOff);
      sum += (word >> bitInByte) & mask;
    }
    return sum;
  }

  /// Decode one whole miniblock with prefix-sum fused. Unsigned mod-2^64
  /// per Parquet spec. Reads up to 7 bytes past the miniblock end;
  /// safe via kPageReadPadding at page end and adjacent miniblocks
  /// intra-page.
  template <typename DataType, uint8_t bitWidth>
  FOLLY_ALWAYS_INLINE void decodeMiniBlockSimd(
      const char* src,
      int32_t numValues,
      int64_t minDelta,
      int64_t& lastValue,
      DataType* out) {
    static_assert(bitWidth >= 1 && bitWidth <= 32);
    constexpr uint64_t mask =
        (bitWidth == 32) ? 0xFFFFFFFFULL : ((1ULL << bitWidth) - 1);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
    // Use narrower accumulator for INT32 to improve throughput.
    // Parquet spec: arithmetic is unsigned modular, so 32-bit wraparound
    // is correct for INT32 columns.
    using AccumType = std::conditional_t<sizeof(DataType) <= 4, uint32_t, uint64_t>;
    AccumType cumulative = static_cast<AccumType>(lastValue);
    const AccumType step = static_cast<AccumType>(minDelta);
    if constexpr (bitWidth <= 16) {
      // 4*bw <= 64; one u64 load per iter.
      for (int32_t i = 0; i < numValues; i += 4) {
        const int32_t bitPos = i * bitWidth;
        const int32_t byteOff = bitPos >> 3;
        const int32_t bitInByte = bitPos & 7;
        const uint64_t word =
            *reinterpret_cast<const uint64_t*>(p + byteOff) >> bitInByte;
        cumulative += step + static_cast<AccumType>(word & mask);
        out[i + 0] = static_cast<DataType>(cumulative);
        cumulative += step + static_cast<AccumType>((word >> bitWidth) & mask);
        out[i + 1] = static_cast<DataType>(cumulative);
        cumulative += step + static_cast<AccumType>((word >> (2 * bitWidth)) & mask);
        out[i + 2] = static_cast<DataType>(cumulative);
        cumulative += step + static_cast<AccumType>((word >> (3 * bitWidth)) & mask);
        out[i + 3] = static_cast<DataType>(cumulative);
      }
    } else {
      // 2*bw + bitInByte > 64; load via __uint128_t (two u64 + SHRD).
      for (int32_t i = 0; i < numValues; i += 2) {
        const int32_t bitPos = i * bitWidth;
        const int32_t byteOff = bitPos >> 3;
        const int32_t bitInByte = bitPos & 7;
        const __uint128_t window =
            static_cast<__uint128_t>(
                *reinterpret_cast<const uint64_t*>(p + byteOff)) |
            (static_cast<__uint128_t>(
                 *reinterpret_cast<const uint64_t*>(p + byteOff + 8))
             << 64);
        const uint64_t word = static_cast<uint64_t>(window >> bitInByte);
        cumulative += step + static_cast<AccumType>(word & mask);
        out[i + 0] = static_cast<DataType>(cumulative);
        cumulative += step + static_cast<AccumType>((word >> bitWidth) & mask);
        out[i + 1] = static_cast<DataType>(cumulative);
      }
    }
    lastValue = static_cast<int64_t>(cumulative);
  }

  template <typename DataType>
  FOLLY_ALWAYS_INLINE void decodeMiniBlockConstantDelta(
      int32_t numValues,
      int64_t minDelta,
      int64_t& lastValue,
      DataType* out) {
    uint64_t cumulative = static_cast<uint64_t>(lastValue);
    const uint64_t step = static_cast<uint64_t>(minDelta);
    for (int32_t i = 0; i < numValues; ++i) {
      cumulative += step;
      out[i] = static_cast<DataType>(cumulative);
    }
    lastValue = static_cast<int64_t>(cumulative);
  }

  /// Dispatch to the compile-time-specialized miniblock decoder for the given
  /// bitWidth. Uses a switch for a proper jump table (better I-cache behavior
  /// than the fold-expression linear comparison chain).
  template <typename DataType>
  bool dispatchSimdMiniBlock(
      uint64_t bitWidth,
      const char* src,
      int32_t numValues,
      int64_t minDelta,
      int64_t& lastValue,
      DataType* out) {
    switch (bitWidth) {
      case 0: decodeMiniBlockConstantDelta(numValues, minDelta, lastValue, out); return true;
      case 1: decodeMiniBlockSimd<DataType, 1>(src, numValues, minDelta, lastValue, out); return true;
      case 2: decodeMiniBlockSimd<DataType, 2>(src, numValues, minDelta, lastValue, out); return true;
      case 3: decodeMiniBlockSimd<DataType, 3>(src, numValues, minDelta, lastValue, out); return true;
      case 4: decodeMiniBlockSimd<DataType, 4>(src, numValues, minDelta, lastValue, out); return true;
      case 5: decodeMiniBlockSimd<DataType, 5>(src, numValues, minDelta, lastValue, out); return true;
      case 6: decodeMiniBlockSimd<DataType, 6>(src, numValues, minDelta, lastValue, out); return true;
      case 7: decodeMiniBlockSimd<DataType, 7>(src, numValues, minDelta, lastValue, out); return true;
      case 8: decodeMiniBlockSimd<DataType, 8>(src, numValues, minDelta, lastValue, out); return true;
      case 9: decodeMiniBlockSimd<DataType, 9>(src, numValues, minDelta, lastValue, out); return true;
      case 10: decodeMiniBlockSimd<DataType, 10>(src, numValues, minDelta, lastValue, out); return true;
      case 11: decodeMiniBlockSimd<DataType, 11>(src, numValues, minDelta, lastValue, out); return true;
      case 12: decodeMiniBlockSimd<DataType, 12>(src, numValues, minDelta, lastValue, out); return true;
      case 13: decodeMiniBlockSimd<DataType, 13>(src, numValues, minDelta, lastValue, out); return true;
      case 14: decodeMiniBlockSimd<DataType, 14>(src, numValues, minDelta, lastValue, out); return true;
      case 15: decodeMiniBlockSimd<DataType, 15>(src, numValues, minDelta, lastValue, out); return true;
      case 16: decodeMiniBlockSimd<DataType, 16>(src, numValues, minDelta, lastValue, out); return true;
      case 17: decodeMiniBlockSimd<DataType, 17>(src, numValues, minDelta, lastValue, out); return true;
      case 18: decodeMiniBlockSimd<DataType, 18>(src, numValues, minDelta, lastValue, out); return true;
      case 19: decodeMiniBlockSimd<DataType, 19>(src, numValues, minDelta, lastValue, out); return true;
      case 20: decodeMiniBlockSimd<DataType, 20>(src, numValues, minDelta, lastValue, out); return true;
      case 21: decodeMiniBlockSimd<DataType, 21>(src, numValues, minDelta, lastValue, out); return true;
      case 22: decodeMiniBlockSimd<DataType, 22>(src, numValues, minDelta, lastValue, out); return true;
      case 23: decodeMiniBlockSimd<DataType, 23>(src, numValues, minDelta, lastValue, out); return true;
      case 24: decodeMiniBlockSimd<DataType, 24>(src, numValues, minDelta, lastValue, out); return true;
      case 25: decodeMiniBlockSimd<DataType, 25>(src, numValues, minDelta, lastValue, out); return true;
      case 26: decodeMiniBlockSimd<DataType, 26>(src, numValues, minDelta, lastValue, out); return true;
      case 27: decodeMiniBlockSimd<DataType, 27>(src, numValues, minDelta, lastValue, out); return true;
      case 28: decodeMiniBlockSimd<DataType, 28>(src, numValues, minDelta, lastValue, out); return true;
      case 29: decodeMiniBlockSimd<DataType, 29>(src, numValues, minDelta, lastValue, out); return true;
      case 30: decodeMiniBlockSimd<DataType, 30>(src, numValues, minDelta, lastValue, out); return true;
      case 31: decodeMiniBlockSimd<DataType, 31>(src, numValues, minDelta, lastValue, out); return true;
      case 32: decodeMiniBlockSimd<DataType, 32>(src, numValues, minDelta, lastValue, out); return true;
      default: return false;
    }
  }

  bool getVlqInt(uint64_t& v) {
    uint64_t tmp = 0;
    for (int i = 0; i < folly::kMaxVarintLength64; i++) {
      uint8_t byte = *(bufferStart_++);
      tmp |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);
      if ((byte & 0x80) == 0) {
        v = tmp;
        return true;
      }
    }
    return false;
  }

  bool getZigZagVlqInt(int64_t& v) {
    uint64_t u;
    if (!getVlqInt(u)) {
      return false;
    }
    v = (u >> 1) ^ (~(u & 1) + 1);
    return true;
  }

  void initHeader() {
    if (!getVlqInt(valuesPerBlock_) || !getVlqInt(miniBlocksPerBlock_) ||
        !getVlqInt(totalValueCount_) || !getZigZagVlqInt(lastValue_)) {
      VELOX_FAIL("initHeader EOF");
    }

    VELOX_CHECK_GT(valuesPerBlock_, 0, "cannot have zero value per block");
    VELOX_CHECK_EQ(
        valuesPerBlock_ % 128,
        0,
        "the number of values in a block must be multiple of 128, but it's {}",
        valuesPerBlock_);
    VELOX_CHECK_GT(
        miniBlocksPerBlock_, 0, "cannot have zero miniblock per block");
    valuesPerMiniBlock_ = valuesPerBlock_ / miniBlocksPerBlock_;
    VELOX_CHECK_GT(
        valuesPerMiniBlock_, 0, "cannot have zero value per miniblock");
    VELOX_CHECK_EQ(
        valuesPerMiniBlock_ % 32,
        0,
        "the number of values in a miniblock must be multiple of 32, but it's {}",
        valuesPerMiniBlock_);

    totalValuesRemaining_ = totalValueCount_;
    VELOX_CHECK_LE(
        miniBlocksPerBlock_,
        kMaxMiniBlocksPerBlock,
        "miniBlocksPerBlock exceeds supported maximum");
    firstBlockInitialized_ = false;
    valuesRemainingCurrentMiniBlock_ = 0;
  }

  void initBlock() {
    VELOX_DCHECK_GT(totalValuesRemaining_, 0, "initBlock called at EOF");

    if (!getZigZagVlqInt(minDelta_)) {
      VELOX_FAIL("initBlock EOF");
    }

    // read the bitwidth of each miniblock
    for (uint32_t i = 0; i < miniBlocksPerBlock_; ++i) {
      deltaBitWidths_[i] = *(bufferStart_++);
      // Note that non-conformant bitwidth entries are allowed by the Parquet
      // spec for extraneous miniblocks in the last block (GH-14923), so we
      // check the bitwidths when actually using them (see initMiniBlock()).
    }

    miniBlockIdx_ = 0;
    firstBlockInitialized_ = true;
    initMiniBlock(deltaBitWidths_[0]);
  }

  void initMiniBlock(int32_t bitWidth) {
    VELOX_DCHECK_LE(
        bitWidth,
        kMaxDeltaBitWidth,
        "delta bit width larger than integer bit width");
    deltaBitWidth_ = bitWidth;
    valuesRemainingCurrentMiniBlock_ = valuesPerMiniBlock_;
  }

  // Advance to the next miniblock without decoding any value.
  void advanceMiniBlock() {
    VELOX_DCHECK(firstBlockInitialized_);
    VELOX_DCHECK_EQ(valuesRemainingCurrentMiniBlock_, 0);
    ++miniBlockIdx_;
    if (miniBlockIdx_ < miniBlocksPerBlock_) {
      initMiniBlock(deltaBitWidths_[miniBlockIdx_]);
    } else {
      initBlock();
    }
  }

  FOLLY_ALWAYS_INLINE int64_t readLong() {
    int64_t value = 0;
    if (valuesRemainingCurrentMiniBlock_ == 0) {
      if (!firstBlockInitialized_) {
        value = lastValue_;
        // When block is uninitialized we have two different possibilities:
        // 1. totalValueCount_ == 1, which means that the page may have only
        // one value (encoded in the header), and we should not initialize
        // any block.
        // 2. totalValueCount_ != 1, which means we should initialize the
        // incoming block for subsequent reads.
        if (totalValueCount_ != 1) {
          initBlock();
        }
        totalValuesRemaining_--;
        return value;
      } else {
        ++miniBlockIdx_;
        if (miniBlockIdx_ < miniBlocksPerBlock_) {
          initMiniBlock(deltaBitWidths_[miniBlockIdx_]);
        } else {
          initBlock();
        }
      }
    }

    uint64_t consumedBits =
        (valuesPerMiniBlock_ - valuesRemainingCurrentMiniBlock_) *
        deltaBitWidth_;
    if (deltaBitWidth_) {
      value = bits::detail::loadBits<uint64_t>(
          reinterpret_cast<const uint64_t*>(bufferStart_),
          consumedBits,
          deltaBitWidth_);
      value &= (~0ULL >> (64 - deltaBitWidth_));
    }
    // Addition between minDelta_, packed int and lastValue_ should be treated
    // as unsigned addition. Overflow is as expected.
    value = static_cast<uint64_t>(minDelta_) + static_cast<uint64_t>(value) +
        static_cast<uint64_t>(lastValue_);
    lastValue_ = value;
    valuesRemainingCurrentMiniBlock_--;
    totalValuesRemaining_--;

    if (valuesRemainingCurrentMiniBlock_ == 0 || totalValuesRemaining_ == 0) {
      bufferStart_ += bits::nbytes(deltaBitWidth_ * valuesPerMiniBlock_);
    }
    return value;
  }

  static constexpr int kMaxDeltaBitWidth =
      static_cast<int>(sizeof(int64_t) * 8);

  const char* bufferStart_;

  uint64_t valuesPerBlock_;
  uint64_t miniBlocksPerBlock_;
  uint64_t valuesPerMiniBlock_;
  uint64_t totalValueCount_;

  uint64_t totalValuesRemaining_;
  // Remaining values in current mini block. If the current block is the last
  // mini block, valuesRemainingCurrentMiniBlock_ may greater than
  // totalValuesRemaining_.
  uint64_t valuesRemainingCurrentMiniBlock_;

  // If the page doesn't contain any block, `firstBlockInitialized_` will
  // always be false. Otherwise, it will be true when first block initialized.
  bool firstBlockInitialized_;
  int64_t minDelta_;
  uint64_t miniBlockIdx_;
  // Parquet default: 4 miniblocks per block. Use a fixed-size array to
  // avoid heap allocation. The spec allows arbitrary values but real-world
  // writers use <= 8.
  static constexpr uint64_t kMaxMiniBlocksPerBlock = 64;
  std::array<uint8_t, kMaxMiniBlocksPerBlock> deltaBitWidths_;
  uint64_t deltaBitWidth_;

  int64_t lastValue_;
};

} // namespace facebook::velox::parquet
