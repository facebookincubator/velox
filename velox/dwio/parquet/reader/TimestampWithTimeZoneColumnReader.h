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
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {
namespace detail {

// UTC time zone key used when packing values read from Parquet. Parquet INT64
// TIMESTAMP_MILLIS / TIMESTAMP_MICROS (or TIMESTAMP logical types with
// isAdjustedToUTC=true) store instants in UTC, so the time zone component of
// the packed Velox representation is always UTC.
constexpr TimeZoneKey kUtcTimeZoneKey = 0;

inline int64_t toPackedTimestampWithTimeZone(
    int64_t value,
    TimestampPrecision filePrecision) {
  Timestamp ts;
  switch (filePrecision) {
    case TimestampPrecision::kMilliseconds:
      ts = Timestamp::fromMillis(value);
      break;
    case TimestampPrecision::kMicroseconds:
      ts = Timestamp::fromMicros(value);
      break;
    case TimestampPrecision::kNanoseconds:
      ts = Timestamp::fromNanos(value);
      break;
    default:
      VELOX_USER_FAIL(
          "Unsupported parquet timestamp unit/converted type: {}",
          filePrecision);
  }
  return pack(ts.toMillis(), kUtcTimeZoneKey);
}

class PackedTimestampWithTimeZoneFilter : public common::Filter {
 public:
  PackedTimestampWithTimeZoneFilter(
      const common::Filter* delegate,
      TimestampPrecision filePrecision)
      : Filter(
            delegate->isDeterministic(),
            delegate->nullAllowed(),
            delegate->kind()),
        delegate_(delegate),
        filePrecision_(filePrecision) {}

  bool testNull() const {
    return delegate_->testNull();
  }

  bool testNonNull() const final {
    return delegate_->testNonNull();
  }

  bool testInt64(int64_t value) const final {
    return delegate_->testInt64(
        toPackedTimestampWithTimeZone(value, filePrecision_));
  }

  bool testInt64Range(int64_t min, int64_t max, bool hasNull) const final {
    if (hasNull && nullAllowed_) {
      return true;
    }

    // Packing is monotonic with respect to UTC millis, so a raw Parquet range
    // maps to a packed TIMESTAMP WITH TIME ZONE range.
    return delegate_->testInt64Range(
        toPackedTimestampWithTimeZone(min, filePrecision_),
        toPackedTimestampWithTimeZone(max, filePrecision_),
        hasNull);
  }

  std::unique_ptr<common::Filter> clone(
      std::optional<bool> /*nullAllowed*/ = std::nullopt) const final {
    VELOX_UNSUPPORTED(
        "PackedTimestampWithTimeZoneFilter::clone is not supported");
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED(
        "PackedTimestampWithTimeZoneFilter::serialize is not supported");
  }

  bool testingEquals(const common::Filter& /*other*/) const final {
    return false;
  }

  std::string toString() const override {
    return fmt::format(
        "PackedTimestampWithTimeZoneFilter({})", delegate_->toString());
  }

 private:
  const common::Filter* const delegate_;
  const TimestampPrecision filePrecision_;
};
} // namespace detail

template <typename T>
class TimestampWithTimeZoneColumnReader : public IntegerColumnReader {
 public:
  // Only INT64 Parquet timestamps are supported
  static_assert(std::is_same_v<T, int64_t>);

  TimestampWithTimeZoneColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec) {
    const auto typeWithId =
        std::static_pointer_cast<const ParquetTypeWithId>(fileType_);
    if (auto logicalType = typeWithId->logicalType_) {
      VELOX_CHECK(
          logicalType->__isset.TIMESTAMP &&
          logicalType->TIMESTAMP.isAdjustedToUTC);
      const auto unit = logicalType->TIMESTAMP.unit;
      if (unit.__isset.MILLIS) {
        filePrecision_ = TimestampPrecision::kMilliseconds;
      } else if (unit.__isset.MICROS) {
        filePrecision_ = TimestampPrecision::kMicroseconds;
      } else if (unit.__isset.NANOS) {
        filePrecision_ = TimestampPrecision::kNanoseconds;
      } else {
        VELOX_UNREACHABLE();
      }
    } else if (auto convertedType = typeWithId->convertedType_) {
      if (convertedType == thrift::ConvertedType::type::TIMESTAMP_MILLIS) {
        filePrecision_ = TimestampPrecision::kMilliseconds;
      } else if (
          convertedType == thrift::ConvertedType::type::TIMESTAMP_MICROS) {
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
    getFlatValues<int64_t, int64_t>(rows, result, requestedType_);
    if (allNull_) {
      return;
    }

    // Replace the raw INT64 values in the result with packed
    // TIMESTAMP WITH TIME ZONE values.
    VectorPtr resultVector = *result;
    auto rawValues =
        resultVector->asUnchecked<FlatVector<int64_t>>()->mutableRawValues();
    for (auto i = 0; i < numValues_; ++i) {
      if (resultVector->isNullAt(i)) {
        continue;
      }
      rawValues[i] =
          detail::toPackedTimestampWithTimeZone(rawValues[i], filePrecision_);
    }
  }

  template <
      typename Reader,
      typename TFilter,
      bool isDense,
      typename ExtractValues>
  void readHelper(
      const velox::common::Filter* filter,
      const RowSet& rows,
      ExtractValues extractValues) {
    detail::PackedTimestampWithTimeZoneFilter packedFilter{
        filter, filePrecision_};

    this->readWithVisitor(
        rows,
        dwio::common::ColumnVisitor<
            int64_t,
            detail::PackedTimestampWithTimeZoneFilter,
            ExtractValues,
            isDense>(packedFilter, this, rows, extractValues));
  }
  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    prepareRead<int64_t>(offset, rows, nullptr);
    readCommon<TimestampWithTimeZoneColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // The precision of int64_t timestamp in Parquet.
  TimestampPrecision filePrecision_;
};

} // namespace facebook::velox::parquet
