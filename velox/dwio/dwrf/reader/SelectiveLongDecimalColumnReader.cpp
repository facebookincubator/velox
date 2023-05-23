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

#include "velox/dwio/dwrf/reader/SelectiveLongDecimalColumnReader.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/SelectiveShortDecimalColumnReader.h"

namespace facebook::velox::dwrf {

using namespace dwio::common;

void SelectiveLongDecimalColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  // because scale's type is int64_t
  prepareRead<int64_t>(offset, rows, incomingNulls);

  bool isDense = rows.back() == rows.size() - 1;
  velox::common::Filter* filter =
      scanSpec_->filter() ? scanSpec_->filter() : &alwaysTrue();

  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        processValueHook<false>(rows, scanSpec_->valueHook());
      }
      return;
    }

    if (isDense) {
      processFilter<true>(filter, ExtractToReader(this), rows);
    } else {
      processFilter<false>(filter, ExtractToReader(this), rows);
    }
  } else {
    if (isDense) {
      processFilter<true>(filter, DropValues(), rows);
    } else {
      processFilter<false>(filter, DropValues(), rows);
    }
  }
}

namespace {
void scaleInt128(int128_t& value, uint32_t scale, uint32_t currentScale) {
  if (scale > currentScale) {
    while (scale > currentScale) {
      uint32_t scaleAdjust = std::min(
          SelectiveShortDecimalColumnReader::MAX_PRECISION_64,
          scale - currentScale);
      value *= SelectiveShortDecimalColumnReader::POWERS_OF_TEN[scaleAdjust];
      currentScale += scaleAdjust;
    }
  } else if (scale < currentScale) {
    while (currentScale > scale) {
      uint32_t scaleAdjust = std::min(
          SelectiveShortDecimalColumnReader::MAX_PRECISION_64,
          currentScale - scale);
      value /= SelectiveShortDecimalColumnReader::POWERS_OF_TEN[scaleAdjust];
      currentScale -= scaleAdjust;
    }
  }
}
} // namespace

void SelectiveLongDecimalColumnReader::getValues(
    RowSet rows,
    VectorPtr* result) {
  auto nullsPtr = nullsInReadRange_
      ? (returnReaderNulls_ ? nullsInReadRange_->as<uint64_t>()
                            : rawResultNulls_)
      : nullptr;

  auto decimalValues =
      AlignedBuffer::allocate<UnscaledLongDecimal>(numValues_, &memoryPool_);
  auto rawDecimalValues = decimalValues->asMutable<UnscaledLongDecimal>();

  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->as<int128_t>();

  // transfer to UnscaledLongDecimal
  for (vector_size_t i = 0; i < numValues_; i++) {
    if (!nullsPtr || !bits::isBitNull(nullsPtr, i)) {
      int32_t currentScale = scales[i];
      int128_t value = values[i];

      scaleInt128(value, scale_, currentScale);

      rawDecimalValues[i] = UnscaledLongDecimal(value);
    }
  }

  values_ = decimalValues;
  rawValues_ = values_->asMutable<char>();
  getFlatValues<UnscaledLongDecimal, UnscaledLongDecimal>(
      rows, result, type_, true);
}

} // namespace facebook::velox::dwrf
