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
#include "velox/dwio/nimble/encodings/RLEEncoding.h"

namespace facebook::nimble {
namespace {

// Counts true rows before rowLimit for alternating bool RLE runs that all have
// the same run length.
uint64_t countTrueWithConstantRunLength(
    bool value,
    uint32_t runLength,
    uint64_t rowLimit) {
  NIMBLE_CHECK_GT(runLength, 0, "RLE run length must be positive.");
  const uint64_t numFullRuns = rowLimit / runLength;
  const uint64_t tail = rowLimit % runLength;
  const uint64_t trueFullRuns = value ? (numFullRuns + 1) / 2 : numFullRuns / 2;
  const bool tailIsTrue = (numFullRuns % 2 == 0) ? value : !value;
  return trueFullRuns * runLength + (tailIsTrue ? tail : 0);
}

} // namespace

RLEEncoding<bool>::RLEEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : internal::RLEEncodingBase<bool, RLEEncoding<bool>>(
          pool,
          data,
          stringBufferFactory,
          options) {
  initialValue_ = *reinterpret_cast<const bool*>(
      internal::RLEEncodingBase<bool, RLEEncoding<bool>>::getValuesStart());
  NIMBLE_CHECK(
      (internal::RLEEncodingBase<bool, RLEEncoding<bool>>::getValuesStart() +
       1) == data.end(),
      "Unexpected run length encoding end");
  internal::RLEEncodingBase<bool, RLEEncoding<bool>>::reset();
}

bool RLEEncoding<bool>::nextValue() {
  value_ = !value_;
  return !value_;
}

void RLEEncoding<bool>::resetValues() {
  value_ = initialValue_;
}

void RLEEncoding<bool>::materializeBoolsAsBits(
    uint32_t rowCount,
    uint64_t* buffer,
    int begin) {
  auto rowsLeft = rowCount;
  while (rowsLeft > 0) {
    if (copiesRemaining_ == 0) {
      advanceRun();
    }
    if (rowsLeft < copiesRemaining_) {
      velox::bits::fillBits(buffer, begin, begin + rowsLeft, currentValue_);
      copiesRemaining_ -= rowsLeft;
      return;
    }
    velox::bits::fillBits(
        buffer, begin, begin + copiesRemaining_, currentValue_);
    begin += copiesRemaining_;
    rowsLeft -= copiesRemaining_;
    copiesRemaining_ = 0;
  }
}

void RLEEncoding<bool>::countTrue(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    RangeCounts& counts,
    const Encoding::Options& options) {
  counts = {};
  NIMBLE_CHECK_EQ(
      EncodingPrefix::encodingType(encoded),
      EncodingType::RLE,
      "Expected RLE encoding.");
  NIMBLE_CHECK_EQ(
      EncodingPrefix::dataType(encoded),
      DataType::Bool,
      "Expected boolean RLE encoding.");
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot count zero rows.");

  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const uint32_t runLengthsSize = encoding::readUint32(pos);
  const std::string_view runLengthsData{pos, runLengthsSize};
  pos += runLengthsSize;
  NIMBLE_CHECK_EQ(
      pos + sizeof(bool), encoded.end(), "Unexpected boolean RLE encoding end");
  bool value = encoding::read<bool>(pos);

  NIMBLE_CHECK_EQ(
      EncodingPrefix::dataType(runLengthsData),
      DataType::Uint32,
      "RLE run-length child must be uint32.");
  const auto runCount =
      EncodingPrefix::readRowCount(runLengthsData, options.useVarintRowCount);
  NIMBLE_CHECK_GT(runCount, 0, "Boolean RLE must have at least one run.");

  const uint64_t rangeEnd = static_cast<uint64_t>(offset) + length;
  if (EncodingPrefix::encodingType(runLengthsData) == EncodingType::Constant) {
    const char* runLengthPos = runLengthsData.data() +
        EncodingPrefix::prefixSize(runLengthsData, options.useVarintRowCount);
    const uint32_t runLength = encoding::readUint32(runLengthPos);
    NIMBLE_CHECK_GE(
        static_cast<uint64_t>(runLength) * runCount,
        rangeEnd,
        "Boolean RLE run lengths are too short.");
    const auto numTrueBeforeRange =
        countTrueWithConstantRunLength(value, runLength, offset);
    const auto numTrueAtRangeEnd =
        countTrueWithConstantRunLength(value, runLength, rangeEnd);
    counts = {
        .numTrueBeforeRange = static_cast<uint32_t>(numTrueBeforeRange),
        .numTrueInRange =
            static_cast<uint32_t>(numTrueAtRangeEnd - numTrueBeforeRange)};
    return;
  }

  auto* pool = &buffer.getMemoryPool();
  auto runLengthsEncoding = EncodingFactory{options}.create(
      *pool, runLengthsData, [](uint32_t /*totalLength*/) -> void* {
        return nullptr;
      });
  constexpr uint32_t kRunLengthChunkSize{256};
  ScopedVector<uint32_t> runLengths{
      std::min<uint32_t>(runCount, kRunLengthChunkSize),
      pool,
      options.bufferPool};
  uint64_t row{0};
  for (uint32_t run = 0; run < runCount;) {
    const auto chunkSize =
        std::min<uint32_t>(runLengths->size(), runCount - run);
    runLengthsEncoding->materialize(chunkSize, runLengths->data());
    for (uint32_t i = 0; i < chunkSize; ++i, ++run) {
      const uint64_t runStart = row;
      const uint64_t runEnd = row + (*runLengths)[i];
      NIMBLE_CHECK_GT((*runLengths)[i], 0, "RLE run length must be positive.");
      if (value) {
        if (runStart < offset) {
          counts.numTrueBeforeRange += static_cast<uint32_t>(
              std::min<uint64_t>(runEnd, offset) - runStart);
        }
        const auto rangeStart = std::max<uint64_t>(runStart, offset);
        const auto rangeLimit = std::min<uint64_t>(runEnd, rangeEnd);
        if (rangeStart < rangeLimit) {
          counts.numTrueInRange +=
              static_cast<uint32_t>(rangeLimit - rangeStart);
        }
      }
      row = runEnd;
      if (row >= rangeEnd) {
        return;
      }
      value = !value;
    }
  }
  NIMBLE_CHECK_GE(row, rangeEnd, "Boolean RLE run lengths are too short.");
}

uint32_t RLEEncoding<bool>::countTrue(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  RangeCounts counts;
  countTrue(encoded, offset, length, buffer, counts, options);
  return counts.numTrueInRange;
}

} // namespace facebook::nimble
