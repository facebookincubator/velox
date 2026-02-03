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

#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/parquet/reader/IntegerColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

/// Column reader for Parquet TIME type.
/// Handles conversion from Parquet TIME_MILLIS (INT32, milliseconds) and
/// TIME_MICROS (INT64, microseconds) to Velox TIME type (BIGINT, milliseconds).
class TimeColumnReader : public IntegerColumnReader {
 public:
  TimeColumnReader(
      const TypePtr& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec) {
    const auto typeWithId =
        std::static_pointer_cast<const ParquetTypeWithId>(fileType_);
    if (auto logicalType = typeWithId->logicalType_) {
      VELOX_CHECK(logicalType->__isset.TIME);
      const auto unit = logicalType->TIME.unit;
      if (unit.__isset.MILLIS) {
        isMicros_ = false;
      } else if (unit.__isset.MICROS) {
        isMicros_ = true;
      } else if (unit.__isset.NANOS) {
        VELOX_NYI("TIME with nanosecond precision is not supported");
      } else {
        VELOX_UNREACHABLE("Unknown TIME unit");
      }
    } else if (auto convertedType = typeWithId->convertedType_) {
      if (convertedType == thrift::ConvertedType::type::TIME_MILLIS) {
        isMicros_ = false;
      } else if (convertedType == thrift::ConvertedType::type::TIME_MICROS) {
        isMicros_ = true;
      } else {
        VELOX_UNREACHABLE("Unexpected converted type for TIME column");
      }
    } else {
      VELOX_NYI("Logical type and converted type are not provided for TIME.");
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    // Parquet stores TIME as INT32 (TIME_MILLIS) or INT64 (TIME_MICROS).
    // Velox represents TIME as BIGINT (8-byte).
    // Use the physical width: TIME_MILLIS is INT32 (4 bytes),
    // TIME_MICROS is INT64 (8 bytes).
    VELOX_WIDTH_DISPATCH(isMicros_ ? 8 : 4, prepareRead, offset, rows, nullptr);
    readCommon<IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // Whether the Parquet file stores TIME in microseconds (TIME_MICROS).
  // If false, it's in milliseconds (TIME_MILLIS).
  bool isMicros_{false};
};

} // namespace facebook::velox::parquet
