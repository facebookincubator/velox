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

class TimestampINT96ColumnReader : public IntegerColumnReader {
 public:
  TimestampINT96ColumnReader(
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
      : IntegerColumnReader(requestedType, fileType, params, scanSpec),
        timestampPrecision_(params.timestampPrecision()) {
    auto& parquetFileType = static_cast<const ParquetTypeWithId&>(*fileType_);
    auto logicalTypeOpt = parquetFileType.logicalType_;
    VELOX_CHECK(logicalTypeOpt.has_value());

    auto logicalType = logicalTypeOpt.value();
    VELOX_CHECK(logicalType.__isset.TIMESTAMP);

    if (!logicalType.TIMESTAMP.isAdjustedToUTC) {
      VELOX_NYI("Only UTC adjusted Timestamp is supported.");
    }

    if (logicalType.TIMESTAMP.unit.__isset.MICROS) {
      parquetTimestampPrecision_ = TimestampPrecision::kMicroseconds;
    } else if (logicalType.TIMESTAMP.unit.__isset.MILLIS) {
      parquetTimestampPrecision_ = TimestampPrecision::kMilliseconds;
    } else {
      VELOX_NYI("Nano Timestamp unit is not supported.");
    }
  }

  bool hasBulkPath() const override {
    return false;
  }

  void
  processNulls(const bool isNull, const RowSet rows, const uint64_t* rawNulls) {
    if (!rawNulls) {
      return;
    }
    auto rawTs = values_->asMutable<Timestamp>();

    returnReaderNulls_ = false;
    anyNulls_ = !isNull;
    allNull_ = isNull;
    vector_size_t idx = 0;
    for (vector_size_t i = 0; i < numValues_; i++) {
      if (isNull) {
        if (bits::isBitNull(rawNulls, i)) {
          bits::setNull(rawResultNulls_, idx);
          addOutputRow(rows[i]);
          idx++;
        }
      } else {
        if (!bits::isBitNull(rawNulls, i)) {
          bits::setNull(rawResultNulls_, idx, false);
          rawTs[idx] = rawTs[i];
          addOutputRow(rows[i]);
          idx++;
        }
      }
    }
  }

  void processFilter(
      const common::Filter* filter,
      const RowSet rows,
      const uint64_t* rawNulls) {
    auto rawTs = values_->asMutable<Timestamp>();

    returnReaderNulls_ = false;
    anyNulls_ = false;
    allNull_ = true;
    vector_size_t idx = 0;
    for (vector_size_t i = 0; i < numValues_; i++) {
      if (rawNulls && bits::isBitNull(rawNulls, i)) {
        if (filter->testNull()) {
          bits::setNull(rawResultNulls_, idx);
          addOutputRow(rows[i]);
          anyNulls_ = true;
          idx++;
        }
      } else {
        if (filter->testTimestamp(rawTs[i])) {
          if (rawNulls) {
            bits::setNull(rawResultNulls_, idx, false);
          }
          rawTs[idx] = rawTs[i];
          addOutputRow(rows[i]);
          allNull_ = false;
          idx++;
        }
      }
    }
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    getFlatValues<Timestamp, Timestamp>(rows, result, requestedType_);
  }

  Timestamp adjustToPrecision(
      const Timestamp& timestamp,
      TimestampPrecision precision) {
    auto nano = timestamp.getNanos();
    switch (precision) {
      case TimestampPrecision::kMilliseconds:
        nano = nano / 1'000'000 * 1'000'000;
        break;
      case TimestampPrecision::kMicroseconds:
        nano = nano / 1'000 * 1'000;
        break;
      case TimestampPrecision::kNanoseconds:
        break;
    }

    return Timestamp(timestamp.getSeconds(), nano);
  }

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* /*incomingNulls*/) override {
    auto& data = formatData_->as<ParquetData>();
    prepareRead<int64_t>(offset, rows, nullptr);

    // Remove filter so that we can do filtering here once data is represented
    // as timestamp type
    const auto filter = scanSpec_->filter()->clone();
    scanSpec_->setFilter(nullptr);

    readCommon<IntegerColumnReader, true>(rows);

    auto tsValues =
        AlignedBuffer::allocate<Timestamp>(numValues_, &memoryPool_);
    auto rawTs = tsValues->asMutable<Timestamp>();
    auto rawTsInt64 = values_->asMutable<int64_t>();
    const auto rawNulls =
        resultNulls() ? resultNulls()->as<uint64_t>() : nullptr;

    Timestamp timestamp;
    for (vector_size_t i = 0; i < numValues_; i++) {
      if (!rawNulls || !bits::isBitNull(rawNulls, i)) {
        if (parquetTimestampPrecision_ == TimestampPrecision::kMicroseconds) {
          timestamp = Timestamp::fromMicros(rawTsInt64[i]);
        } else {
          timestamp = Timestamp::fromMillis(rawTsInt64[i]);
        }

        rawTs[i] = adjustToPrecision(timestamp, timestampPrecision_);
      }
    }
    values_ = tsValues;
    rawValues_ = values_->asMutable<char>();

    switch (
        !filter ||
                (filter->kind() == common::FilterKind::kIsNotNull && !rawNulls)
            ? common::FilterKind::kAlwaysTrue
            : filter->kind()) {
      case common::FilterKind::kAlwaysTrue:
        // Simply add all rows to output.
        for (vector_size_t i = 0; i < numValues_; i++) {
          addOutputRow(rows[i]);
        }
        break;
      case common::FilterKind::kIsNull:
        processNulls(true, rows, rawNulls);
        break;
      case common::FilterKind::kIsNotNull:
        processNulls(false, rows, rawNulls);
        break;
      case common::FilterKind::kTimestampRange:
      case common::FilterKind::kMultiRange:
        processFilter(filter.get(), rows, rawNulls);
        break;
      default:
        VELOX_UNSUPPORTED("Unsupported filter.");
    }

    readOffset_ += rows.back() + 1;
  }

 private:
  TimestampPrecision parquetTimestampPrecision_;
  TimestampPrecision timestampPrecision_;
};
} // namespace facebook::velox::parquet
