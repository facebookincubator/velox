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
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

/// Column reader for Parquet TIME type.
/// Handles conversion from:
/// - Parquet TIME_MILLIS (INT32, milliseconds) to Velox TIME (BIGINT,
///   milliseconds).
/// - Parquet TIME_MICROS (INT64, microseconds) to Velox TIME MICRO UTC
///   (BIGINT, microseconds).
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
      VELOX_CHECK(
          unit.__isset.MILLIS || unit.__isset.MICROS,
          "TIME precision other than milliseconds or microseconds is not supported");
      isMicros_ = unit.__isset.MICROS;
    } else if (auto convertedType = typeWithId->convertedType_) {
      VELOX_CHECK(
          convertedType == thrift::ConvertedType::type::TIME_MILLIS ||
              convertedType == thrift::ConvertedType::type::TIME_MICROS,
          "TIME converted type other than TIME_MILLIS or TIME_MICROS is not supported");
      isMicros_ = convertedType == thrift::ConvertedType::type::TIME_MICROS;
    } else {
      VELOX_NYI("Logical type and converted type are not provided for TIME.");
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    // Velox represents TIME as BIGINT (8 bytes) for both precisions.
    // Parquet stores TIME_MILLIS as INT32 (4 bytes) and TIME_MICROS as
    // INT64 (8 bytes). Dispatch on the physical width.
    const int32_t physicalWidth = isMicros_ ? 8 : 4;
    VELOX_WIDTH_DISPATCH(physicalWidth, prepareRead, offset, rows, nullptr);
    readCommon<IntegerColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // True for TIME_MICROS (INT64), false for TIME_MILLIS (INT32).
  bool isMicros_{false};
};

} // namespace facebook::velox::parquet
