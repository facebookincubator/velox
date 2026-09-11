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
namespace detail {

/// Narrows one raw Parquet timestamp value to milliseconds since the epoch.
///
/// Truncates toward zero, deliberately. Presto's Java reader for the same
/// column (presto-parquet LongTimestampMicrosColumnReader) computes
///     long utcMillis = MICROSECONDS.toMillis(value);
///     packDateTimeWithZone(utcMillis, UTC_KEY);
/// and TimeUnit.toMillis truncates toward zero, so -499'999us becomes -499ms
/// there. A native worker has to return the same value as a Java worker reading
/// the same file, so this follows Java rather than Velox's Timestamp, which
/// keeps a non-negative nanos field and would floor to -500ms. The two differ
/// only for pre-epoch instants with sub-millisecond precision.
inline int64_t toMillisUtc(int64_t value, TimestampPrecision filePrecision) {
  switch (filePrecision) {
    case TimestampPrecision::kMilliseconds:
      return value;
    case TimestampPrecision::kMicroseconds:
      return value / 1'000;
    case TimestampPrecision::kNanoseconds:
      return value / 1'000'000;
    default:
      VELOX_USER_FAIL(
          "Unsupported Parquet timestamp precision for TIMESTAMP WITH TIME "
          "ZONE: {}",
          static_cast<int32_t>(filePrecision));
  }
}

/// Applies a filter that is expressed over PACKED TIMESTAMP WITH TIME ZONE
/// values to the raw Parquet values this reader actually sees, by converting
/// each value before delegating.
///
/// Range predicates do not need this: translatePackedRange() rewrites them into
/// a real BigintRange over file values, which lets the decoder keep its
/// specialised paths and costs no per-value work. This adapter exists for the
/// kinds that have no same-kind rewrite -- IN lists map one packed value to a
/// whole tick window each, so a values set would have to become a union of
/// ranges -- and for OR-ed and negated combinations of them.
///
/// The adapter never escapes readHelper(): it is a stack local handed to one
/// ColumnVisitor, never stored in a ScanSpec and never used to test statistics.
/// That matters because it reports its delegate's kind() while being a
/// different class, which is safe only as long as nothing downstream dispatches
/// on kind() and casts to the corresponding concrete filter. Nothing in the
/// decode path does today: ColumnVisitor applies filters through the templated
/// common::applyFilter, on the static type.
class PackedFilterAdapter final : public velox::common::Filter {
 public:
  PackedFilterAdapter(
      const velox::common::Filter* delegate,
      TimestampPrecision filePrecision,
      int16_t utcZoneKey)
      : Filter(
            delegate->isDeterministic(),
            delegate->nullAllowed(),
            delegate->kind()),
        delegate_{delegate},
        filePrecision_{filePrecision},
        utcZoneKey_{utcZoneKey} {}

  // Filter::testNull() is not virtual in this class hierarchy -- it is reached
  // through the visitor's static type -- so this hides rather than overrides.
  bool testNull() const {
    return delegate_->testNull();
  }

  bool testNonNull() const final {
    return delegate_->testNonNull();
  }

  bool testInt64(int64_t value) const final {
    return delegate_->testInt64(toPacked(value));
  }

  bool testInt64Range(int64_t min, int64_t max, bool hasNull) const final {
    // Narrowing truncates toward zero and packing shifts left, so the mapping
    // from a raw file value to a packed value is monotone non-decreasing.
    // Mapping the endpoints therefore yields the enclosing packed range.
    return delegate_->testInt64Range(toPacked(min), toPacked(max), hasNull);
  }

  std::unique_ptr<velox::common::Filter> clone(
      std::optional<bool> nullAllowed = std::nullopt) const final {
    // Clone the delegate too and own it, so the copy does not outlive a
    // borrowed pointer.
    auto cloned = delegate_->clone(nullAllowed);
    return std::unique_ptr<velox::common::Filter>(new PackedFilterAdapter(
        std::move(cloned), filePrecision_, utcZoneKey_));
  }

  folly::dynamic serialize() const final {
    // Deliberately unsupported rather than delegating. Serializing the delegate
    // would produce a filter that looks like a packed-domain filter but would
    // be applied to raw Parquet values after a round trip, which is exactly the
    // silent mismatch this class exists to prevent. The adapter is transient:
    // it is constructed inside readHelper() and never placed in a ScanSpec or a
    // serialized plan, so this is unreachable in practice.
    VELOX_UNSUPPORTED(
        "PackedFilterAdapter is a transient reader-local filter and cannot be "
        "serialized");
  }

  bool testingEquals(const velox::common::Filter& other) const final {
    const auto* casted = dynamic_cast<const PackedFilterAdapter*>(&other);
    return casted != nullptr && filePrecision_ == casted->filePrecision_ &&
        utcZoneKey_ == casted->utcZoneKey_ &&
        delegate_->testingEquals(*casted->delegate_);
  }

 private:
  PackedFilterAdapter(
      std::unique_ptr<velox::common::Filter> owned,
      TimestampPrecision filePrecision,
      int16_t utcZoneKey)
      : Filter(owned->isDeterministic(), owned->nullAllowed(), owned->kind()),
        ownedDelegate_{std::move(owned)},
        delegate_{ownedDelegate_.get()},
        filePrecision_{filePrecision},
        utcZoneKey_{utcZoneKey} {}

  int64_t toPacked(int64_t value) const {
    return pack(toMillisUtc(value, filePrecision_), utcZoneKey_);
  }

  // Set only for clones, which own their delegate.
  std::unique_ptr<velox::common::Filter> ownedDelegate_;
  const velox::common::Filter* delegate_;
  const TimestampPrecision filePrecision_;
  const int16_t utcZoneKey_;
};

} // namespace detail

/// Reads an INT64 Parquet timestamp column annotated
/// TIMESTAMP(isAdjustedToUTC=true) into Velox's TIMESTAMP WITH TIME ZONE, whose
/// physical representation is a packed BIGINT: (millisUtc << 12) | zoneKey.
///
/// This is the shape Iceberg writes for a `timestamptz` column, and the type
/// Presto reports for it. IntegerColumnReader alone cannot serve it: it would
/// hand back the raw file integer (micros, typically), which is neither the
/// right unit nor packed with a zone key.
///
/// The stored zone is UTC because that is all the file carries. Iceberg does
/// not retain the writer's zone -- `2017-11-16 17:10:34 PST` is stored and read
/// back as `2017-11-17 01:10:34 UTC`, and the two are the same value -- so UTC
/// is the only honest label, not a default.
class TimestampWithTimeZoneColumnReader : public IntegerColumnReader {
 public:
  TimestampWithTimeZoneColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec),
        utcZoneKey_(tz::getTimeZoneID("UTC")) {
    const auto typeWithId =
        std::static_pointer_cast<const ParquetTypeWithId>(fileType_);
    const auto logicalType = typeWithId->logicalType_;
    VELOX_CHECK(
        logicalType &&
            logicalType->getType() == thrift::LogicalType::Type::TIMESTAMP,
        "TIMESTAMP WITH TIME ZONE requires a TIMESTAMP logical type");
    // ParquetReader::convertType has already refused anything else; this keeps
    // the invariant local so the packing below cannot be read as an assumption.
    VELOX_CHECK(
        *logicalType->get_TIMESTAMP().isAdjustedToUTC(),
        "TIMESTAMP WITH TIME ZONE requires isAdjustedToUTC=true");

    // By value, not const ref: see the note in ParquetReader.cpp -- a
    // const-qualified thrift field_ref uses a deprecated operator->.
    auto unit = logicalType->get_TIMESTAMP().unit();
    switch (unit->getType()) {
      case thrift::TimeUnit::Type::MILLIS:
        filePrecision_ = TimestampPrecision::kMilliseconds;
        break;
      case thrift::TimeUnit::Type::MICROS:
        filePrecision_ = TimestampPrecision::kMicroseconds;
        break;
      case thrift::TimeUnit::Type::NANOS:
        filePrecision_ = TimestampPrecision::kNanoseconds;
        break;
      default:
        VELOX_USER_FAIL(
            "Unsupported Parquet TIMESTAMP unit for TIMESTAMP WITH TIME ZONE");
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    // A value hook (aggregation pushdown) would receive the raw file integer
    // before getValues() narrows and packs it, so refuse rather than hand back
    // unpacked micros as if they were TIMESTAMP WITH TIME ZONE values.
    VELOX_CHECK_NULL(
        scanSpec_->valueHook(),
        "Aggregation pushdown into a TIMESTAMP WITH TIME ZONE Parquet column is "
        "not supported yet: the hook would see raw Parquet timestamps rather "
        "than packed (millisUtc << 12 | zoneKey) values");
    prepareRead<int64_t>(offset, rows, nullptr);
    // Reader = this class, so readHelper below translates pushed-down filters.
    readCommon<TimestampWithTimeZoneColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }

  /// Number of file ticks in one millisecond.
  int64_t ticksPerMilli() const {
    switch (filePrecision_) {
      case TimestampPrecision::kMilliseconds:
        return 1;
      case TimestampPrecision::kMicroseconds:
        return 1'000;
      case TimestampPrecision::kNanoseconds:
        return 1'000'000;
      default:
        VELOX_UNREACHABLE();
    }
  }

  /// Rewrites a range expressed over PACKED TIMESTAMP WITH TIME ZONE values
  /// into the equivalent range over the raw file values this reader actually
  /// sees.
  ///
  /// Presto sends the predicate as a BigintRange over packed values, because
  /// TIMESTAMP WITH TIME ZONE is BIGINT-backed (see
  /// PrestoToVeloxConnectorUtils.cpp, which routes it through
  /// bigintRangeToFilter). Applied to raw Parquet timestamps such a filter
  /// silently drops matching rows, so it has to be translated.
  ///
  /// Every value this reader emits is packed as (m << 12) | utcZoneKey_, so
  ///     lower <= (m << 12 | k) <= upper
  /// holds exactly for
  ///     m in [ceil((lower - k) / 4096), floor((upper - k) / 4096)]
  /// and a file value x belongs to millisecond m exactly when
  ///     x in [m * ticks, m * ticks + ticks - 1]
  /// because the narrowing floors. The rewritten filter therefore selects
  /// exactly the rows the original would select if applied to this reader's
  /// output: pushdown stays equivalent to filtering above the scan rather than
  /// becoming an approximation of it.
  velox::common::BigintRange translatePackedRange(
      const velox::common::BigintRange& filter) const {
    constexpr __int128_t kPackUnit = static_cast<__int128_t>(1) << kMillisShift;
    const auto floorDiv = [](__int128_t a, __int128_t b) {
      __int128_t q = a / b;
      if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
      }
      return q;
    };
    const auto ceilDiv = [&](__int128_t a, __int128_t b) {
      return -floorDiv(-a, b);
    };
    const __int128_t key = utcZoneKey_;
    const __int128_t millisLow =
        ceilDiv(static_cast<__int128_t>(filter.lower()) - key, kPackUnit);
    const __int128_t millisHigh =
        floorDiv(static_cast<__int128_t>(filter.upper()) - key, kPackUnit);
    if (millisLow > millisHigh) {
      // No instant can satisfy the range: lower > upper matches nothing.
      return velox::common::BigintRange(1, 0, filter.nullAllowed());
    }
    const __int128_t ticks = ticksPerMilli();
    // toMillisUtc truncates toward zero, so the tick window mapping to
    // millisecond m is
    //     [m*ticks,             m*ticks + ticks - 1]  for m > 0
    //     [m*ticks - ticks + 1, m*ticks + ticks - 1]  for m == 0
    //     [m*ticks - ticks + 1, m*ticks]              for m < 0
    // Millisecond 0 straddles zero because both -999us and +999us truncate to
    // 0ms, which is why the boundary tests below are <= 0 and >= 0 rather than
    // < 0. The union over a contiguous millisecond range is itself contiguous,
    // so the low edge comes from millisLow and the high edge from millisHigh.
    const __int128_t lowEdge =
        millisLow <= 0 ? millisLow * ticks - ticks + 1 : millisLow * ticks;
    const __int128_t highEdge =
        millisHigh >= 0 ? millisHigh * ticks + ticks - 1 : millisHigh * ticks;
    // Clamping is safe in both directions: if a true bound falls outside int64,
    // every representable file value is on the passing side of it.
    const auto clamp = [](__int128_t v) -> int64_t {
      if (v < static_cast<__int128_t>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
      }
      if (v > static_cast<__int128_t>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
      }
      return static_cast<int64_t>(v);
    };
    return velox::common::BigintRange(
        clamp(lowEdge), clamp(highEdge), filter.nullAllowed());
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
    if constexpr (std::is_same_v<TFilter, velox::common::BigintRange>) {
      const auto translated = translatePackedRange(
          *static_cast<const velox::common::BigintRange*>(filter));
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<
              int64_t,
              velox::common::BigintRange,
              ExtractValues,
              isDense>(translated, this, rows, extractValues));
    } else if constexpr (
        std::is_same_v<TFilter, velox::common::AlwaysTrue> ||
        std::is_same_v<TFilter, velox::common::IsNull> ||
        std::is_same_v<TFilter, velox::common::IsNotNull>) {
      // These never compare a value, so they need no translation.
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<int64_t, TFilter, ExtractValues, isDense>(
              *static_cast<const TFilter*>(filter), this, rows, extractValues));
    } else {
      // Everything else -- IN lists, negated ranges and value sets, OR-ed
      // combinations -- goes through the adapter, which converts each raw value
      // to its packed form before delegating. Slower than the same-kind rewrite
      // above, but it needs no per-kind translation and cannot silently compare
      // packed bounds against raw Parquet timestamps.
      const detail::PackedFilterAdapter adapter{
          filter, filePrecision_, utcZoneKey_};
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<
              int64_t,
              detail::PackedFilterAdapter,
              ExtractValues,
              isDense>(adapter, this, rows, extractValues));
    }
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    // Same helper IntegerColumnReader uses: it sizes the output to the
    // requested type's width and carries the null flags across.
    getIntValues(rows, requestedType_, result);
    if (allNull_) {
      return;
    }
    auto* rawValues =
        (*result)->asUnchecked<FlatVector<int64_t>>()->mutableRawValues();
    for (auto i = 0; i < numValues_; ++i) {
      if ((*result)->isNullAt(i)) {
        continue;
      }
      rawValues[i] =
          pack(detail::toMillisUtc(rawValues[i], filePrecision_), utcZoneKey_);
    }
  }

 private:
  const int16_t utcZoneKey_;

  // Unit of the INT64 timestamps in the file.
  TimestampPrecision filePrecision_;
};

} // namespace facebook::velox::parquet
