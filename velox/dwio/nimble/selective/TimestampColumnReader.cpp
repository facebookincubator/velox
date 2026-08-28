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

#include "velox/dwio/nimble/selective/TimestampColumnReader.h"

#include "velox/dwio/common/SelectiveColumnReaderInternal.h"

namespace facebook::nimble {

using namespace facebook::velox;

uint64_t TimestampColumnReader::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);

  if (numValues == 0) {
    return 0;
  }

  // Read micros to temp buffer to count non-null values.
  // We need this count because the nanos stream only contains values for
  // non-null micros positions.
  auto microsBuffer = velox::AlignedBuffer::allocate<int64_t>(numValues, pool_);
  auto nullsBuffer = velox::allocateNulls(numValues, pool_);

  microsDecoder_.decodeNullable<int64_t>(
      nullsBuffer->asMutable<uint64_t>(),
      microsBuffer->asMutable<int64_t>(),
      numValues,
      nullptr);

  auto nonNullCount =
      velox::bits::countNonNulls(nullsBuffer->as<uint64_t>(), 0, numValues);

  if (nonNullCount > 0) {
    nanosDecoder_.skip(nonNullCount);
  }

  return numValues;
}

void TimestampColumnReader::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  prepareRead<int64_t>(offset, rows, incomingNulls);

  VELOX_CHECK(
      !scanSpec_->valueHook(),
      "Selective reader for TIMESTAMP doesn't support aggregation pushdown");

  readHelper(scanSpec_->filter(), rows);
  readOffset_ += rows.back() + 1;
}

void TimestampColumnReader::readHelper(
    const velox::common::Filter* filter,
    const velox::RowSet& rows) {
  auto scannedPositions = rows.back() + 1;

  // Get incoming nulls from parent. Nested shapes could have null elements.
  const uint64_t* incomingNulls =
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;

  if (!microsBuffer_ || !microsBuffer_->unique() ||
      microsBuffer_->capacity() < scannedPositions * sizeof(int64_t)) {
    microsBuffer_ = AlignedBuffer::allocate<int64_t>(scannedPositions, pool_);
  }
  auto* microsData = microsBuffer_->asMutable<int64_t>();

  velox::BufferPtr nullsBuffer;
  if (!incomingNulls && nullsInReadRange_ && nullsInReadRange_->unique() &&
      nullsInReadRange_->capacity() >= velox::bits::nbytes(scannedPositions)) {
    // No incoming nulls and buffer is reusable.
    nullsBuffer = nullsInReadRange_;
  } else {
    nullsBuffer = velox::allocateNulls(scannedPositions, pool_);
  }
  auto* nullsData = nullsBuffer->asMutable<uint64_t>();

  // Read micros with proper position tracking.
  // Pass incomingNulls so decodeNullable knows which positions were
  // already determined to be null by prepareRead.
  microsDecoder_.decodeNullable<int64_t>(
      nullsData, microsData, scannedPositions, incomingNulls);

  // Count non-nulls from the nulls buffer.
  auto nonNullCount =
      velox::bits::countNonNulls(nullsData, 0, scannedPositions);

  // Update nullsInReadRange_ with the combined nulls.
  nullsInReadRange_ = nullsBuffer;
  const uint64_t* rawNulls = nullsData;

  if (!nanosBuffer_ || !nanosBuffer_->unique() ||
      nanosBuffer_->capacity() < nonNullCount * sizeof(uint16_t)) {
    nanosBuffer_ = AlignedBuffer::allocate<uint16_t>(nonNullCount, pool_);
  }
  auto* nanosData = nanosBuffer_->asMutable<uint16_t>();
  if (nonNullCount > 0) {
    nanosDecoder_.decodeNullable<uint16_t>(
        nullptr, nanosData, nonNullCount, nullptr);
  }

  // Calculate numValues_ based on the rows we're extracting.
  numValues_ = rows.size();

  if (!values_ || !values_->unique() ||
      values_->capacity() < numValues_ * sizeof(Timestamp)) {
    values_ = AlignedBuffer::allocate<Timestamp>(numValues_, pool_);
  }
  rawValues_ = values_->asMutable<char>();
  auto rawTs = values_->asMutable<Timestamp>();

  // Create nulls buffer for the rows we are actually returning.
  if (!resultNulls_ || !resultNulls_->unique() ||
      resultNulls_->capacity() < velox::bits::nbytes(numValues_)) {
    resultNulls_ = velox::allocateNulls(numValues_, pool_);
  }
  rawResultNulls_ = resultNulls_->asMutable<uint64_t>();
  // Forces resultNulls() to use resultNulls_/rawResultNulls_
  returnReaderNulls_ = false;

  // Tracks nanos position by incrementing during non-null encounters.
  velox::vector_size_t nanosIndex = 0;
  // Tracks the index in our RowSet to properly map micros data
  velox::vector_size_t rowIdx = 0;
  velox::vector_size_t outputNonNullCount = 0;

  // Build timestamps for requested rows by combining scattered micros with
  // packed nanos. nanosIndex must track all non-nulls, not just requested rows,
  // to ensure correct stream alignment
  for (velox::vector_size_t pos = 0;
       pos < scannedPositions && rowIdx < numValues_;
       pos++) {
    bool isNull = velox::bits::isBitNull(rawNulls, pos);

    if (pos == rows[rowIdx]) {
      // This position is in our sparse rowset.
      velox::bits::setNull(rawResultNulls_, rowIdx, isNull);

      if (!isNull) {
        // microsData is scattered (values at their original positions)
        // nanosData is packed (values at sequential indices for non-nulls)
        rawTs[rowIdx] =
            convertToVeloxTimestamp(microsData[pos], nanosData[nanosIndex]);
        outputNonNullCount++;
      }
      rowIdx++;
    }

    if (!isNull) {
      nanosIndex++;
    }
  }

  // Reader uses these for nulls optimizations
  anyNulls_ = (outputNonNullCount < numValues_);
  allNull_ = (outputNonNullCount == 0 && numValues_ > 0);

  // Apply filter.
  auto effectiveKind =
      (!filter ||
       (filter->kind() == common::FilterKind::kIsNotNull && !anyNulls_))
      ? common::FilterKind::kAlwaysTrue
      : filter->kind();

  switch (effectiveKind) {
    case common::FilterKind::kAlwaysTrue:
      setOutputRows(rows);
      break;
    case common::FilterKind::kIsNull:
      processNulls(true, rows, rawResultNulls_);
      break;
    case common::FilterKind::kIsNotNull:
      processNulls(false, rows, rawResultNulls_);
      break;
    case common::FilterKind::kTimestampRange:
    case common::FilterKind::kMultiRange:
      processFilter(filter, rows, rawResultNulls_);
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported filter for timestamp.");
  }
}

// kIsNull and kIsNotNull filter for Timestamps.
void TimestampColumnReader::processNulls(
    bool isNull,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  if (!rawNulls) {
    return;
  }
  auto rawTs = values_->asMutable<Timestamp>();

  anyNulls_ = isNull;
  allNull_ = isNull;
  vector_size_t idx = 0;
  for (vector_size_t i = 0; i < numValues_; ++i) {
    if (isNull) {
      if (velox::bits::isBitNull(rawNulls, i)) {
        velox::bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        idx++;
      }
    } else {
      if (!velox::bits::isBitNull(rawNulls, i)) {
        velox::bits::setNull(rawResultNulls_, idx, false);
        rawTs[idx] = rawTs[i];
        addOutputRow(rows[i]);
        idx++;
      }
    }
  }
}

// Current support limited to kTimestampRange and kMultiRange.
void TimestampColumnReader::processFilter(
    const common::Filter* filter,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  auto rawTs = values_->asMutable<Timestamp>();

  anyNulls_ = false;
  allNull_ = true;
  vector_size_t idx = 0;
  for (vector_size_t i = 0; i < numValues_; ++i) {
    if (rawNulls && velox::bits::isBitNull(rawNulls, i)) {
      if (filter->testNull()) {
        velox::bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        anyNulls_ = true;
        idx++;
      }
    } else {
      if (filter->testTimestamp(rawTs[i])) {
        if (rawNulls) {
          velox::bits::setNull(rawResultNulls_, idx, false);
        }
        rawTs[idx] = rawTs[i];
        addOutputRow(rows[i]);
        allNull_ = false;
        idx++;
      }
    }
  }
}

void TimestampColumnReader::getValues(const RowSet& rows, VectorPtr* result) {
  getFlatValues<Timestamp, Timestamp>(rows, result, fileType_->type(), true);
}

bool TimestampColumnReader::estimateMaterializedSize(
    size_t& byteSize,
    size_t& rowCount) const {
  auto decoderRowCount = microsDecoder_.estimateRowCount();
  if (!decoderRowCount.has_value()) {
    return false;
  }
  rowCount = *decoderRowCount;
  byteSize = sizeof(Timestamp) * rowCount;
  return true;
}

// TODO: Unify this helper method with batch reader.
// - Nimble stores time as:
//     micros         -> whole microseconds since epoch
//     subMicrosNanos -> extra nanoseconds inside the microsecond [0, 999]
// - Velox stores time as:
//     seconds + nanos (0 <= nanos < 1_000_000_000)
//
// The math below splits 'micros' into whole seconds and the remainder, then
// converts the remainder to nanoseconds and adds the sub-microsecond nanos.
// For negative remainders, we use a branchless correction to ensure nanos
// is always positive.
Timestamp TimestampColumnReader::convertToVeloxTimestamp(
    int64_t micros,
    uint16_t subMicrosNanos) {
  int64_t seconds = micros / 1000000;
  int64_t remainder = micros % 1000000;
  // Branchless Sign Correction
  // If micros was negative (e.g., -100us), remainder will be -100.
  // We need to borrow 1 second to make the remainder positive.
  // mask will be -1 (0xFF...FF) if remainder < 0, else 0.
  int64_t mask = remainder >> 63;
  // If negative: seconds -= 1; remainder += 1000000;
  seconds += mask;
  remainder += (1000000 & mask);
  // remainder is now guaranteed [0, 999999].
  // Convert remainder micros to nanos (* 1000) and add the fractional nanos.
  uint64_t nanos = static_cast<uint64_t>(remainder) * 1000 + subMicrosNanos;
  return velox::Timestamp(seconds, nanos);
}

} // namespace facebook::nimble
