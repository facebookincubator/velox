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
#include "velox/type/Timestamp.h"

namespace facebook::velox::functions::sparksql {

class TimestampNTZColumnReader : public parquet::IntegerColumnReader {
 public:
  TimestampNTZColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      parquet::ParquetParams& params,
      common::ScanSpec& scanSpec)
      : parquet::IntegerColumnReader(
            requestedType,
            fileType,
            params,
            scanSpec) {
    const auto typeWithId =
        std::static_pointer_cast<const parquet::ParquetTypeWithId>(fileType_);
    if (auto logicalType = typeWithId->logicalType_) {
      VELOX_CHECK(logicalType->__isset.TIMESTAMP);
      const auto unit = logicalType->TIMESTAMP.unit;
      if (unit.__isset.MILLIS) {
        filePrecision_ = TimestampPrecision::kMilliseconds;
      } else if (unit.__isset.MICROS) {
        filePrecision_ = TimestampPrecision::kMicroseconds;
      } else {
        VELOX_UNREACHABLE();
      }
    } else if (auto convertedType = typeWithId->convertedType_) {
      if (convertedType ==
          ::facebook::velox::parquet::thrift::ConvertedType::type::
              TIMESTAMP_MILLIS) {
        filePrecision_ = TimestampPrecision::kMilliseconds;
      } else if (
          convertedType ==
          ::facebook::velox::parquet::thrift::ConvertedType::type::
              TIMESTAMP_MICROS) {
        filePrecision_ = TimestampPrecision::kMicroseconds;
      } else {
        VELOX_UNREACHABLE();
      }
    } else {
      VELOX_NYI("Logical type and converted type are not provided.");
    }
  }

  bool hasBulkPath() const override {
    return false;
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    getIntValues(rows, BIGINT(), result);

    VectorPtr resultVector = *result;
    auto rawValues =
        resultVector->asUnchecked<FlatVector<int64_t>>()->mutableRawValues();
    for (auto i = 0; i < numValues_; ++i) {
      if (resultVector->isNullAt(i)) {
        continue;
      }
      if (filePrecision_ == TimestampPrecision::kMilliseconds) {
        rawValues[i] = rawValues[i] * 1'000;
      }
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    // The 'fileType_' is 'TIMESTAMP' and needs to be read as 'BIGINT' and
    // converted to 'TIMESTAMP_NTZ'.
    VELOX_WIDTH_DISPATCH(sizeof(int64_t), prepareRead, offset, rows, nullptr);
    readCommon<parquet::IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // The precision of int64_t timestamp in Parquet.
  TimestampPrecision filePrecision_;
};

} // namespace facebook::velox::functions::sparksql
