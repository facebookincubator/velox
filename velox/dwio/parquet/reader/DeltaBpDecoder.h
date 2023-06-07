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

#include "velox/dwio/common/BitPackDecoder.h"

#include <folly/Varint.h>

namespace facebook::velox::parquet {

class DeltaBpDecoder {
 public:
  DeltaBpDecoder(const char* FOLLY_NONNULL start, const char* FOLLY_NONNULL end)
      : bufferStart_(start), bufferEnd_(end) {
    readHeader();
  }

  void skip(uint64_t numValues) {
    skip<false>(numValues, 0, nullptr);
  }

  template <bool hasNulls>
  inline void skip(
      int32_t numValues,
      int32_t current,
      const uint64_t* FOLLY_NULLABLE nulls) {
    if (hasNulls) {
      numValues = bits::countNonNulls(nulls, current, current + numValues);
    }
    for (int32_t i = 0; i < numValues; ++i) {
      readInt64();
    }
  }

  template <bool hasNulls, typename Visitor>
  void readWithVisitor(const uint64_t* FOLLY_NULLABLE nulls, Visitor visitor) {
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
        toSkip = visitor.process(readInt64(), atEnd);
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

 protected:
  uint64_t readULEB128() {
    uint64_t tmp = 0;
    for (int i = 0; i < kMaxULEB128ByteLengthForInt64; i++) {
      uint8_t byte = *(bufferStart_++);
      tmp |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);
      if ((byte & 0x80) == 0) {
        return tmp;
      }
    }
    VELOX_FAIL("Invalid ULEB128 int");
  }

  int64_t readZigZagULEB128() {
    uint64_t u = readULEB128();
    return (u >> 1) ^ (~(u & 1) + 1);
  }

  void readHeader() {
    valuesPerBlock_ = readULEB128();
    miniBlocksPerBlock_ = readULEB128();
    totalValueCount_ = readULEB128();
    lastValue_ = readZigZagULEB128();

    VELOX_CHECK_GT(valuesPerBlock_, 0, "cannot have zero value per block");
    VELOX_CHECK_EQ(
        valuesPerBlock_ % 128,
        0,
        "the number of values in a block must be multiple of 128, but it's " +
            std::to_string(valuesPerBlock_));
    VELOX_CHECK_GT(
        miniBlocksPerBlock_, 0, "cannot have zero miniblock per block");
    valuesPerMiniBlock_ = valuesPerBlock_ / miniBlocksPerBlock_;
    VELOX_CHECK_GT(
        valuesPerMiniBlock_, 0, "cannot have zero value per miniblock");
    VELOX_CHECK_EQ(
        valuesPerMiniBlock_ % 32,
        0,
        "the number of values in a miniblock must be multiple of 32, but it's " +
            std::to_string(valuesPerMiniBlock_));

    totalValuesRemaining_ = totalValueCount_;
    deltaBitWidths_.resize(miniBlocksPerBlock_);
    firstBlockInitialized_ = false;
    valuesRemainingCurrentMiniBlock_ = 0;
  }

  void readBlock() {
    VELOX_CHECK_GT(totalValuesRemaining_, 0, "readBlock called at EOF");

    minDelta_ = readZigZagULEB128();

    // read the bitwidth of each miniblock
    for (uint32_t i = 0; i < miniBlocksPerBlock_; ++i) {
      deltaBitWidths_[i] = *(bufferStart_++);
      // Note that non-conformant bitwidth entries are allowed by the Parquet
      // spec for extraneous miniblocks in the last block (GH-14923), so we
      // check the bitwidths when actually using them (see InitMiniBlock()).
    }

    miniBlockIdx_ = 0;
    firstBlockInitialized_ = true;
    readMiniBlock(deltaBitWidths_[0]);
  }

  void readMiniBlock(int32_t bitWidth) {
    VELOX_CHECK_LE(
        bitWidth,
        kMaxDeltaBitWidth,
        "delta bit width larger than integer bit width");
    deltaBitWidth_ = bitWidth;
    valuesRemainingCurrentMiniBlock_ = valuesPerMiniBlock_;
  }

  int64_t readInt64() {
    int64_t value = 0;
    if (valuesRemainingCurrentMiniBlock_ == 0) {
      if (!firstBlockInitialized_) {
        value = lastValue_;
        readBlock();
        return value;
      } else {
        ++miniBlockIdx_;
        if (miniBlockIdx_ < miniBlocksPerBlock_) {
          readMiniBlock(deltaBitWidths_[miniBlockIdx_]);
        } else {
          readBlock();
        }
      }
    }

    uint64_t consumedBits =
        (valuesPerMiniBlock_ - valuesRemainingCurrentMiniBlock_) *
        deltaBitWidth_;
    bits::copyBits(
        reinterpret_cast<const uint64_t*>(bufferStart_),
        consumedBits,
        reinterpret_cast<uint64_t*>(&value),
        0,
        deltaBitWidth_);
    value += minDelta_ + lastValue_;
    lastValue_ = value;

    valuesRemainingCurrentMiniBlock_--;
    totalValuesRemaining_--;

    if (valuesRemainingCurrentMiniBlock_ == 0) {
      bufferStart_ += bits::nbytes(deltaBitWidth_ * valuesPerMiniBlock_);
    }
    return value;
  }

  static constexpr int kMaxDeltaBitWidth =
      static_cast<int>(sizeof(int64_t) * 8);

  // Maximum byte length of a ULEB128 encoded int32
  static constexpr int kMaxULEB128ByteLengthForInt32 = 5;

  // Maximum byte length of a ULEB128 encoded int64
  static constexpr int kMaxULEB128ByteLengthForInt64 = 10;

  const char* FOLLY_NULLABLE bufferStart_;
  const char* FOLLY_NULLABLE bufferEnd_;

  uint64_t valuesPerBlock_;
  uint64_t miniBlocksPerBlock_;
  uint64_t totalValueCount_;
  uint64_t lastValue_;
  uint64_t valuesPerMiniBlock_;
  uint64_t totalValuesRemaining_;
  std::vector<uint8_t> deltaBitWidths_;
  bool firstBlockInitialized_;
  uint64_t valuesRemainingCurrentMiniBlock_;
  uint64_t minDelta_;
  uint64_t miniBlockIdx_;
  uint64_t deltaBitWidth_;
};

} // namespace facebook::velox::parquet
