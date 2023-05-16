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

#include "velox/dwio/parquet/reader/IntegerColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"

namespace facebook::velox::parquet {

class TimestampColumnReader : public IntegerColumnReader {
 public:
  TimestampColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& nodeType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(nodeType, nodeType, params, scanSpec) {}

  static constexpr int64_t JULIAN_TO_UNIX_EPOCH_DAYS = 2440588LL;
  static constexpr int64_t SECONDS_PER_DAY = 86400LL;

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* /*incomingNulls*/) override {
    auto& data = formatData_->as<ParquetData>();
    // Use int128_t instead because of the lack of int96 implementation.
    prepareRead<int128_t>(offset, rows, nullptr);
    readCommon<IntegerColumnReader>(rows);
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    auto type = nodeType_->type;
    VELOX_CHECK(type->kind() == TypeKind::TIMESTAMP, "Timestamp expected.");
    VELOX_CHECK_NE(valueSize_, kNoValueSize);
    VELOX_CHECK(mayGetValues_);
    if (allNull_) {
      *result = std::make_shared<ConstantVector<Timestamp>>(
          &memoryPool_,
          rows.size(),
          true,
          type,
          Timestamp(),
          SimpleVectorStats<Timestamp>{},
          sizeof(Timestamp) * rows.size());
      return;
    }
    VELOX_CHECK_LE(rows.size(), numValues_);
    VELOX_CHECK(!rows.empty());
    if (!values_) {
      return;
    }

    auto tsValues = AlignedBuffer::allocate<Timestamp>(
        rows.size(), &memoryPool_, Timestamp());
    auto* valuesPtr = tsValues->asMutable<Timestamp>();
    char* rawValues = reinterpret_cast<char*>(rawValues_);

    vector_size_t rowIndex = 0;
    auto nextRow = rows[rowIndex];
    bool moveNulls = shouldMoveNulls(rows);
    bool emptyOutputRows = outputRows_.size() == 0;
    for (size_t i = 0; i < numValues_; i++) {
      if (!emptyOutputRows && outputRows_[i] < nextRow) {
        continue;
      }
      VELOX_DCHECK(emptyOutputRows || (outputRows_[i] == nextRow));

      // Convert the timestamp into seconds and nanos since the Unix epoch,
      // 00:00:00.000000 on 1 January 1970.
      uint64_t nanos;
      memcpy(&nanos, rawValues + nextRow * sizeof(int96_t), sizeof(uint64_t));
      int32_t days;
      memcpy(
          &days,
          rawValues + nextRow * sizeof(int96_t) + sizeof(uint64_t),
          sizeof(int32_t));
      valuesPtr[rowIndex] = Timestamp(
          (days - JULIAN_TO_UNIX_EPOCH_DAYS) * SECONDS_PER_DAY, nanos);

      if (moveNulls && rowIndex != i) {
        bits::setBit(
            rawResultNulls_, rowIndex, bits::isBitSet(rawResultNulls_, i));
      }
      if (!emptyOutputRows) {
        outputRows_[rowIndex] = nextRow;
      }
      rowIndex++;
      if (rowIndex >= rows.size()) {
        break;
      }
      nextRow = rows[rowIndex];
    }

    BufferPtr nulls = anyNulls_
        ? (returnReaderNulls_ ? nullsInReadRange_ : resultNulls_)
        : nullptr;

    *result = std::make_shared<FlatVector<Timestamp>>(
        &memoryPool_,
        type,
        nulls,
        rows.size(),
        tsValues,
        std::move(stringBuffers_));
  }
};

} // namespace facebook::velox::parquet
