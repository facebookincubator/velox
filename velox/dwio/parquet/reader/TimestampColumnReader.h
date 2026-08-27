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
#include "velox/dwio/parquet/thrift/ParquetThrift.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::parquet {
namespace {

Timestamp toInt64Timestamp(int64_t value, TimestampPrecision filePrecision) {
  switch (filePrecision) {
    case TimestampPrecision::kMilliseconds:
      return Timestamp::fromMillis(value);
    case TimestampPrecision::kMicroseconds:
      return Timestamp::fromMicros(value);
    case TimestampPrecision::kNanoseconds:
      return Timestamp::fromNanos(value);
    default:
      VELOX_UNREACHABLE();
  }
}

// Zone key that every Parquet instant is stamped with, resolved once instead of
// hardcoding its numeric value.
TimeZoneKey utcTimeZoneKey() {
  static const TimeZoneKey kUtc = tz::getTimeZoneID("UTC");
  return kUtc;
}

Timestamp toInt96Timestamp(const int128_t& value) {
  // Convert int128_t to Int96 Timestamp by extracting days and nanos.
  const int32_t days = static_cast<int32_t>(value >> 64);
  const uint64_t nanos = value & ((((1ULL << 63) - 1ULL) << 1) + 1);
  return Timestamp::fromDaysAndNanos(days, nanos);
}

// Range filter for Parquet Timestamp.
template <typename T>
class ParquetTimestampRange final : public common::TimestampRange {
 public:
  // Use int128_t for Int96
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>);

  // @param lower Lower end of the range, inclusive.
  // @param upper Upper end of the range, inclusive.
  // @param nullAllowed Null values are passing the filter if true.
  // @param timestampUnit Unit of the Int64 Timestamp.
  ParquetTimestampRange(
      const Timestamp& lower,
      const Timestamp& upper,
      bool nullAllowed,
      TimestampPrecision filePrecision)
      : TimestampRange(lower, upper, nullAllowed),
        filePrecision_(filePrecision) {}

  bool testInt128(const int128_t& value) const final {
    Timestamp ts;
    if constexpr (std::is_same_v<T, int64_t>) {
      ts = toInt64Timestamp(value, filePrecision_);
    } else if constexpr (std::is_same_v<T, int128_t>) {
      ts = toInt96Timestamp(value);
    }
    return ts >= this->lower() && ts <= this->upper();
  }

 private:
  // Only used when T is int64_t.
  const TimestampPrecision filePrecision_;
};

} // namespace

template <typename T>
class TimestampColumnReader : public IntegerColumnReader {
 public:
  // Use int128_t for Int96
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>);

  TimestampColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec),
        requestedPrecision_(params.timestampPrecision()) {
    // Whether the file declares the values normalized to UTC. Int96 carries no
    // such flag, and a converted type without a logical type is UTC normalized
    // by definition, so both default to true.
    bool isAdjustedToUtc = true;

    if constexpr (std::is_same_v<T, int64_t>) {
      const auto typeWithId =
          std::static_pointer_cast<const ParquetTypeWithId>(fileType_);
      if (auto logicalType = typeWithId->logicalType_) {
        VELOX_CHECK(
            logicalType->getType() == thrift::LogicalType::Type::TIMESTAMP);
        isAdjustedToUtc = *logicalType->get_TIMESTAMP().isAdjustedToUTC();
        auto unit = logicalType->get_TIMESTAMP().unit();
        const auto unittype = unit->gettype();
        if (unittype == thrift::timeunit::type::millis) {
          filePrecision_ = TimestampPrecision::kMilliseconds;
        } else if (unitType == thrift::TimeUnit::Type::MICROS) {
          filePrecision_ = TimestampPrecision::kMicroseconds;
        } else if (unitType == thrift::TimeUnit::Type::NANOS) {
          filePrecision_ = TimestampPrecision::kNanoseconds;
        } else {
          VELOX_UNREACHABLE();
        }
      } else if (auto convertedType = typeWithId->convertedType_) {
        if (convertedType == thrift::ConvertedType::TIMESTAMP_MILLIS) {
          filePrecision_ = TimestampPrecision::kMilliseconds;
        } else if (convertedType == thrift::ConvertedType::TIMESTAMP_MICROS) {
          filePrecision_ = TimestampPrecision::kMicroseconds;
        } else {
          VELOX_UNREACHABLE();
        }
      } else {
        VELOX_NYI("Logical type and converted type are not provided.");
      }
      if (filePrecision_ != requestedPrecision_) {
        needsConversion_ = true;
      }
    }

    if (isTimestampWithTimeZoneType(requestedType_)) {
      // A column that is not UTC normalized holds local wall clock readings in
      // a zone the file does not record, so there is no instant to pack. Fail
      // rather than stamp UTC and silently shift every value.
      VELOX_USER_CHECK(
          isAdjustedToUtc,
          "Cannot read a Parquet timestamp that is not UTC-normalized as TIMESTAMP WITH TIME ZONE: {}",
          fileType_->fullName());

      // Checked here as well as in read() because the reader is built before
      // row group pruning, which would otherwise be the first to see the
      // filter and would report a less clear error from the statistics path.
      checkNoValueFilter(scanSpec.filter());
    }
  }

  bool hasBulkPath() const override {
    return false;
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    if (isTimestampWithTimeZoneType(requestedType_)) {
      getTimestampWithTimeZoneValues(rows, result);
      return;
    }

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

      const int128_t encoded = reinterpret_cast<int128_t&>(rawValues[i]);
      if constexpr (std::is_same_v<T, int64_t>) {
        rawValues[i] = toInt64Timestamp(encoded, filePrecision_);
        if (needsConversion_) {
          rawValues[i] = rawValues[i].toPrecision(requestedPrecision_);
        }
      } else if constexpr (std::is_same_v<T, int128_t>) {
        rawValues[i] =
            toInt96Timestamp(encoded).toPrecision(requestedPrecision_);
      }
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
    if (auto* range = dynamic_cast<const common::TimestampRange*>(filter)) {
      ParquetTimestampRange<T> newRange{
          range->lower(), range->upper(), range->nullAllowed(), filePrecision_};
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<
              int128_t,
              common::TimestampRange,
              ExtractValues,
              isDense>(newRange, this, rows, extractValues));
    } else if (
        auto* multiRange = dynamic_cast<const common::MultiRange*>(filter)) {
      std::vector<std::unique_ptr<common::Filter>> filters;
      filters.reserve(multiRange->filters().size());
      for (const auto& filter : multiRange->filters()) {
        if (auto* range = dynamic_cast<common::TimestampRange*>(filter.get())) {
          filters.emplace_back(
              std::make_unique<ParquetTimestampRange<T>>(
                  range->lower(),
                  range->upper(),
                  range->nullAllowed(),
                  filePrecision_));
        } else {
          filters.emplace_back(filter->clone(filter->nullAllowed()));
        }
      }
      auto newMultiRange =
          common::MultiRange(std::move(filters), multiRange->nullAllowed());
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<
              int128_t,
              common::MultiRange,
              ExtractValues,
              isDense>(newMultiRange, this, rows, extractValues));
    } else {
      this->readWithVisitor(
          rows,
          dwio::common::
              ColumnVisitor<int128_t, TFilter, ExtractValues, isDense>(
                  *static_cast<const TFilter*>(filter),
                  this,
                  rows,
                  extractValues));
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    if (isTimestampWithTimeZoneType(requestedType_)) {
      checkNoValueFilter(scanSpec_->filter());
    }

    // Use int128_t as a workaround. Timestamp in Velox is of 16-byte length.
    prepareRead<int128_t>(offset, rows, nullptr);
    readCommon<TimestampColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

 private:
  // Rejects a pushed down filter that inspects values of a TIMESTAMP WITH TIME
  // ZONE column.
  //
  // A filter on such a column is built from its physical BIGINT kind, so its
  // bounds are packed millis-plus-zone-key values while the column reader sees
  // raw file values. Rewriting the bounds is not possible: the packed domain is
  // neither order- nor equality-isomorphic to the instant domain, because two
  // values denoting the same instant in different zones pack differently yet
  // compare equal under TimestampWithTimeZoneType. Failing loudly is the only
  // honest option; the alternative is silently dropping rows.
  //
  // Called from read() as well as from the constructor, because
  // FileDataSource::addDynamicFilter() installs join-pushed filters on the
  // ScanSpec after the reader is built.
  static void checkNoValueFilter(const velox::common::Filter* filter) {
    if (filter == nullptr) {
      return;
    }
    switch (filter->kind()) {
      // These never inspect a value, so they are safe to push down.
      case velox::common::FilterKind::kAlwaysTrue:
      case velox::common::FilterKind::kAlwaysFalse:
      case velox::common::FilterKind::kIsNull:
      case velox::common::FilterKind::kIsNotNull:
        return;
      default:
        VELOX_NYI(
            "Filter pushdown on TIMESTAMP WITH TIME ZONE is not supported by the Parquet reader: {}",
            filter->toString());
    }
  }

  // Converts the raw Parquet value at 'source[index]' to UTC milliseconds.
  // 'kDivisor' is the number of file units in one millisecond, and is unused
  // for Int96, which carries days and nanos instead of a single scaled
  // integer.
  template <int64_t kDivisor>
  static int64_t toMillis(const Timestamp* source, vector_size_t index) {
    if constexpr (std::is_same_v<T, int64_t>) {
      const auto value = static_cast<int64_t>(
          reinterpret_cast<const int128_t&>(source[index]));
      if constexpr (kDivisor == 1) {
        return value;
      } else {
        // Timestamp floors when splitting into seconds and nanos, while integer
        // division truncates toward zero. The two differ for pre-epoch values,
        // hence the correction term.
        return value / kDivisor - ((value % kDivisor) < 0);
      }
    } else {
      return toInt96Timestamp(reinterpret_cast<const int128_t&>(source[index]))
          .toMillisAllowOverflow();
    }
  }

  // True if 'millis' is outside the range TimestampWithTimeZone can represent.
  static bool millisOutOfRange(int64_t millis) {
    return millis > kMaxMillisUtc || millis < kMinMillisUtc;
  }

  // Converts the raw Parquet values in 'source' to UTC milliseconds, packs each
  // with 'timeZoneBits' and writes the result to 'packed'. 'kHasNulls' skips
  // converting null rows; instantiating it as false leaves the loop branch-free
  // so the compiler can vectorize it.
  //
  // Returns a non-zero value if any converted timestamp falls outside the range
  // TimestampWithTimeZone can represent. Reporting is deferred to the caller so
  // that the loop stays free of anything that can throw.
  template <int64_t kDivisor, bool kHasNulls>
  static int64_t packMillisLoop(
      const Timestamp* __restrict source,
      int64_t* __restrict packed,
      vector_size_t size,
      const uint64_t* nulls,
      int64_t timeZoneBits) {
    int64_t outOfRange = 0;
    for (vector_size_t i = 0; i < size; ++i) {
      if constexpr (kHasNulls) {
        if (!bits::isBitSet(nulls, i)) {
          // Write a defined value rather than skipping the row. The reader may
          // be handed a recycled buffer, and leaving the slot alone would
          // surface the previous batch's value to anything that copies whole
          // ranges without consulting the null flags.
          packed[i] = 0;
          continue;
        }
      }

      const auto millis = toMillis<kDivisor>(source, i);
      outOfRange |= millisOutOfRange(millis);
      // Shift as unsigned. An out-of-range value overflows here, and signed
      // overflow would be undefined behavior before 'outOfRange' is consulted.
      packed[i] =
          static_cast<int64_t>(static_cast<uint64_t>(millis) << kMillisShift) |
          timeZoneBits;
    }
    return outOfRange;
  }

  // Selects the null-handling variant of packMillisLoop for 'source'.
  template <int64_t kDivisor>
  static int64_t packMillis(
      const FlatVector<Timestamp>& source,
      int64_t* packed,
      vector_size_t size,
      int64_t timeZoneBits) {
    if (source.mayHaveNulls()) {
      return packMillisLoop<kDivisor, true>(
          source.rawValues(), packed, size, source.rawNulls(), timeZoneBits);
    }
    return packMillisLoop<kDivisor, false>(
        source.rawValues(), packed, size, nullptr, timeZoneBits);
  }

  // Reports the first value that does not fit TimestampWithTimeZone. Only
  // reached on the error path, so it can afford to rescan.
  template <int64_t kDivisor>
  void failOnOutOfRange(const FlatVector<Timestamp>& source) const {
    for (vector_size_t i = 0; i < source.size(); ++i) {
      if (source.isNullAt(i)) {
        continue;
      }
      const auto millis = toMillis<kDivisor>(source.rawValues(), i);
      VELOX_USER_CHECK(
          !millisOutOfRange(millis),
          "Timestamp is out of the range TIMESTAMP WITH TIME ZONE can represent: {} ms in column {}",
          millis,
          fileType_->fullName());
    }
    VELOX_UNREACHABLE();
  }

  // Reads the column as TIMESTAMP WITH TIME ZONE, whose physical
  // representation is an int64 packing UTC milliseconds with a time zone key.
  void getTimestampWithTimeZoneValues(const RowSet& rows, VectorPtr* result) {
    // Materialize as TIMESTAMP first. prepareRead() lays values out in 16-byte
    // slots, and getFlatValues() compacts values and nulls down to the
    // selected rows.
    VectorPtr timestamps;
    getFlatValues<Timestamp, Timestamp>(rows, &timestamps, TIMESTAMP());
    const auto size = timestamps->size();
    if (allNull_) {
      *result = BaseVector::createNullConstant(requestedType_, size, pool_);
      return;
    }

    const auto& source = *timestamps->asUnchecked<FlatVector<Timestamp>>();
    if (*result != nullptr &&
        (*result)->encoding() == VectorEncoding::Simple::FLAT &&
        (*result)->type()->equivalent(*requestedType_)) {
      BaseVector::prepareForReuse(*result, size);
    } else {
      *result = BaseVector::create(requestedType_, size, pool_);
    }
    auto* target = (*result)->asUnchecked<FlatVector<int64_t>>();
    target->setNulls(source.nulls());

    // Parquet stores instants in UTC, so every value carries the UTC key.
    const int64_t timeZoneBits = utcTimeZoneKey() & kTimezoneMask;

    auto* packed = target->mutableRawValues();
    if constexpr (std::is_same_v<T, int128_t>) {
      if (packMillis<1>(source, packed, size, timeZoneBits) != 0) {
        failOnOutOfRange<1>(source);
      }
    } else {
      switch (filePrecision_) {
        case TimestampPrecision::kMilliseconds:
          if (packMillis<1>(source, packed, size, timeZoneBits) != 0) {
            failOnOutOfRange<1>(source);
          }
          break;
        case TimestampPrecision::kMicroseconds:
          if (packMillis<1'000>(source, packed, size, timeZoneBits) != 0) {
            failOnOutOfRange<1'000>(source);
          }
          break;
        case TimestampPrecision::kNanoseconds:
          if (packMillis<1'000'000>(source, packed, size, timeZoneBits) != 0) {
            failOnOutOfRange<1'000'000>(source);
          }
          break;
        default:
          VELOX_UNREACHABLE();
      }
    }
  }

  // The requested precision can be specified from HiveConfig to read timestamp
  // from Parquet.
  const TimestampPrecision requestedPrecision_;

  // The precision of int64_t timestamp in Parquet. Only set when T is int64_t.
  TimestampPrecision filePrecision_;

  // Whether Int64 Timestamp needs to be converted to the requested precision.
  bool needsConversion_ = false;
};

} // namespace facebook::velox::parquet
