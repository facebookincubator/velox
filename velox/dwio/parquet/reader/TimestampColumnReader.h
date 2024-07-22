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
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec),
        timestampPrecision_(params.timestampPrecision()) {}

  bool hasBulkPath() const override {
    return false;
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    getFlatValues<Timestamp, Timestamp>(rows, result, requestedType_);
    if (allNull_) {
      return;
    }

    // Adjust timestamp nanos to the requested precision.
    VectorPtr resultVector = *result;
    auto rawValues =
        resultVector->asUnchecked<FlatVector<Timestamp>>()->mutableRawValues();
    for (auto i = 0; i < numValues_; ++i) {
      if (resultVector->isNullAt(i)) {
        continue;
      }
      const auto timestamp = rawValues[i];
      uint64_t nanos = timestamp.getNanos();
      switch (timestampPrecision_) {
        case TimestampPrecision::kMilliseconds:
          nanos = nanos / 1'000'000 * 1'000'000;
          break;
        case TimestampPrecision::kMicroseconds:
          nanos = nanos / 1'000 * 1'000;
          break;
        case TimestampPrecision::kNanoseconds:
          break;
      }
      rawValues[i] = Timestamp(timestamp.getSeconds(), nanos);
    }
  }

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* /*incomingNulls*/) override {
    auto& data = formatData_->as<ParquetData>();
    // Use int128_t as a workaroud. Timestamp in Velox is of 16-byte length.
    prepareRead<int128_t>(offset, rows, nullptr);
    readCommon<IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // The requested precision can be specified from HiveConfig to read timestamp
  // from Parquet.
  TimestampPrecision timestampPrecision_;
};

class TimestampINT64ColumnReader : public IntegerColumnReader {
 public:
  TimestampINT64ColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(BIGINT(), fileType, params, scanSpec) {
    auto& parquetFileType = static_cast<const ParquetTypeWithId&>(*fileType_);
    auto logicalTypeOpt = parquetFileType.logicalType_;
    VELOX_CHECK(logicalTypeOpt.has_value());

    auto logicalType = logicalTypeOpt.value();
    VELOX_CHECK(logicalType.__isset.TIMESTAMP);

    if (!logicalType.TIMESTAMP.isAdjustedToUTC) {
      VELOX_NYI("Only UTC adjusted Timestamp is supported.");
    }

    if (logicalType.TIMESTAMP.unit.__isset.MICROS) {
      sourcePrecision_ = TimestampPrecision::kMicroseconds;
    } else if (logicalType.TIMESTAMP.unit.__isset.MILLIS) {
      sourcePrecision_ = TimestampPrecision::kMilliseconds;
    } else {
      VELOX_NYI("Nano Timestamp unit is not supported.");
    }
  }

  bool hasBulkPath() const override {
    return false;
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    // Upcast to int128_t here so we have enough memory already in vector to
    // hold Timestamp (16bit) vs int64_t (8bit)
    getFlatValues<int64_t, int128_t>(rows, result, requestedType_);

    VectorPtr resultVector = *result;
    auto intValues = resultVector->asUnchecked<FlatVector<int128_t>>();

    auto rawValues =
        resultVector->asUnchecked<FlatVector<int128_t>>()->mutableRawValues();

    Timestamp timestamp;
    for (vector_size_t i = 0; i < numValues_; ++i) {
      if (intValues->isNullAt(i))
        continue;

      const auto timestampInt = intValues->valueAt(i);
      std::cout << static_cast<int64_t>(timestampInt) << std::endl;
      Timestamp timestamp;
      if (sourcePrecision_ == TimestampPrecision::kMicroseconds) {
        timestamp = Timestamp::fromMicros(timestampInt);
      } else {
        timestamp = Timestamp::fromMillis(timestampInt);
      }

      memcpy(&rawValues[i], &timestamp, sizeof(int128_t));
    }

    *result = std::make_shared<FlatVector<Timestamp>>(
        &memoryPool_,
        TIMESTAMP(),
        resultNulls(),
        numValues_,
        intValues->values(),
        std::move(stringBuffers_));
  }

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* /*incomingNulls*/) override {
    auto& data = formatData_->as<ParquetData>();
    // Use int128_t as a workaroud. Timestamp in Velox is of 16-byte length.
    prepareRead<int64_t>(offset, rows, nullptr);
    readCommon<IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  TimestampPrecision sourcePrecision_;
};
} // namespace facebook::velox::parquet
