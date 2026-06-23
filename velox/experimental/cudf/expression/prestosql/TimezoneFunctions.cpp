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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/TimezoneConversion.h"
#include "velox/experimental/cudf/expression/prestosql/TimezoneFunctions.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneRegistration.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>

#include <cctype>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

using velox::exec::ConstantExpr;

constexpr cudf::type_id kInt64 = cudf::type_id::INT64;
constexpr cudf::type_id kBool8 = cudf::type_id::BOOL8;

cudf::data_type int64Type() {
  return cudf::data_type{kInt64};
}

// Reads a required constant string argument (e.g. a timezone name or format).
std::string constStringArg(
    const std::shared_ptr<velox::exec::Expr>& expr,
    int32_t index) {
  auto constant =
      std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[index]);
  VELOX_CHECK_NOT_NULL(
      constant, "Expected a constant argument at index {}", index);
  return constant->value()->toString(0);
}

// Reads a required constant integer argument (e.g. an hour/minute offset).
int64_t constIntArg(
    const std::shared_ptr<velox::exec::Expr>& expr,
    int32_t index) {
  auto constant =
      std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[index]);
  VELOX_CHECK_NOT_NULL(
      constant, "Expected a constant argument at index {}", index);
  return std::stoll(constant->value()->toString(0));
}

// Reinterprets an 8-byte-wide column (timestamp/duration/int64) as another
// 8-byte type without copying. Used to move between the packed int64
// representation and timestamp/duration columns.
cudf::column_view bitcastColumn(
    const cudf::column_view& view,
    cudf::type_id id) {
  return cudf::column_view{
      cudf::data_type{id},
      view.size(),
      view.head<int64_t>(),
      view.null_mask(),
      view.null_count(),
      view.offset()};
}

cudf::numeric_scalar<int64_t> i64(int64_t value, rmm::cuda_stream_view stream) {
  return cudf::numeric_scalar<int64_t>(value, true, stream);
}

std::unique_ptr<cudf::column> binOp(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::binary_operator op,
    cudf::data_type outType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(lhs, rhs, op, outType, stream, mr);
}

// Unpacks the UTC millis (arithmetic >> 12) from a packed TIMESTAMP WITH TIME
// ZONE column.
std::unique_ptr<cudf::column> unpackMillis(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return binOp(
      packed,
      i64(kMillisShift, stream),
      cudf::binary_operator::SHIFT_RIGHT,
      int64Type(),
      stream,
      mr);
}

// Returns the single zone-key shared by every row of a packed column, throwing
// if the column mixes zones (the GPU offset/render paths build one transition
// table per zone). Empty columns default to GMT.
int16_t uniformZoneKey(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (packed.size() == 0) {
    return 0;
  }
  // An all-null column has no zone to read; cudf::reduce excludes nulls, so its
  // min/max scalars come back invalid and value() would be a meaningless device
  // read. Default to GMT (key 0), as the empty-column path above does.
  if (packed.null_count() == packed.size()) {
    return 0;
  }
  auto keys = binOp(
      packed,
      i64(kTimezoneMask, stream),
      cudf::binary_operator::BITWISE_AND,
      int64Type(),
      stream,
      mr);
  auto minScalar = cudf::reduce(
      keys->view(),
      *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  auto maxScalar = cudf::reduce(
      keys->view(),
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  auto lo = static_cast<cudf::numeric_scalar<int64_t>*>(minScalar.get())
                ->value(stream);
  auto hi = static_cast<cudf::numeric_scalar<int64_t>*>(maxScalar.get())
                ->value(stream);
  VELOX_USER_CHECK_EQ(
      lo, hi, "cuDF timezone functions require a single time zone per column");
  return static_cast<int16_t>(lo);
}

// Per-row UT offset in whole seconds (INT64) for a packed column, using the
// uniform zone's transition table.
std::unique_ptr<cudf::column> offsetSecondsForPacked(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto zoneKey = uniformZoneKey(packed, stream, mr);
  auto millis = unpackMillis(packed, stream, mr);
  auto millisTs =
      bitcastColumn(millis->view(), cudf::type_id::TIMESTAMP_MILLISECONDS);
  auto offsetDuration =
      utcOffsetSeconds(millisTs, tz::getTimeZoneName(zoneKey), stream, mr);
  return std::make_unique<cudf::column>(
      bitcastColumn(offsetDuration->view(), kInt64), stream, mr);
}

// Renders a column of UT offsets (INT64 seconds) as a time-zone token. With
// includeColon the form is "+HH:MM" (Joda 'ZZ'); otherwise "+HHMM" (Joda 'Z').
// When zeroOffsetText is set, rows with a zero offset render that text instead
// (e.g. "Z" for to_iso8601's ISO8601 output).
std::unique_ptr<cudf::column> formatOffsetStrings(
    const cudf::column_view& offsetSeconds,
    bool includeColon,
    const std::optional<std::string>& zeroOffsetText,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto isNegative = binOp(
      offsetSeconds,
      i64(0, stream),
      cudf::binary_operator::LESS,
      cudf::data_type{kBool8},
      stream,
      mr);
  // abs(offset) = isNegative ? -offset : offset.
  auto negated = cudf::binary_operation(
      i64(0, stream),
      offsetSeconds,
      cudf::binary_operator::SUB,
      int64Type(),
      stream,
      mr);
  auto absolute = cudf::copy_if_else(
      negated->view(), offsetSeconds, isNegative->view(), stream, mr);
  auto hours = binOp(
      absolute->view(),
      i64(3600, stream),
      cudf::binary_operator::DIV,
      int64Type(),
      stream,
      mr);
  auto totalMinutes = binOp(
      absolute->view(),
      i64(60, stream),
      cudf::binary_operator::DIV,
      int64Type(),
      stream,
      mr);
  auto minutes = binOp(
      totalMinutes->view(),
      i64(60, stream),
      cudf::binary_operator::MOD,
      int64Type(),
      stream,
      mr);

  auto hoursStr = cudf::strings::from_integers(hours->view(), stream, mr);
  auto hoursPadded = cudf::strings::zfill(
      cudf::strings_column_view(hoursStr->view()), 2, stream, mr);
  auto minutesStr = cudf::strings::from_integers(minutes->view(), stream, mr);
  auto minutesPadded = cudf::strings::zfill(
      cudf::strings_column_view(minutesStr->view()), 2, stream, mr);

  auto sign = cudf::copy_if_else(
      cudf::string_scalar("-", true, stream),
      cudf::string_scalar("+", true, stream),
      isNegative->view(),
      stream,
      mr);

  // "+/-" + "HH", then join with ":" before "MM".
  auto signHour = cudf::strings::concatenate(
      cudf::table_view{{sign->view(), hoursPadded->view()}},
      cudf::string_scalar("", true, stream),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::YES,
      stream,
      mr);
  auto offsetStr = cudf::strings::concatenate(
      cudf::table_view{{signHour->view(), minutesPadded->view()}},
      cudf::string_scalar(includeColon ? ":" : "", true, stream),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::YES,
      stream,
      mr);
  if (!zeroOffsetText.has_value()) {
    return offsetStr;
  }
  // Render the zero-offset rows as the supplied text (e.g. "Z").
  auto isZero = binOp(
      offsetSeconds,
      i64(0, stream),
      cudf::binary_operator::EQUAL,
      cudf::data_type{kBool8},
      stream,
      mr);
  return cudf::copy_if_else(
      cudf::string_scalar(*zeroOffsetText, true, stream),
      offsetStr->view(),
      isZero->view(),
      stream,
      mr);
}

// Computes the local wall-clock timestamp (TIMESTAMP_MILLISECONDS) and the UT
// offset (INT64 seconds) for a packed column with a uniform zone.
struct LocalAndOffset {
  std::unique_ptr<cudf::column> localMillis;
  std::unique_ptr<cudf::column> offsetSeconds;
};

LocalAndOffset localAndOffset(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto zoneKey = uniformZoneKey(packed, stream, mr);
  auto millis = unpackMillis(packed, stream, mr);
  auto millisTs =
      bitcastColumn(millis->view(), cudf::type_id::TIMESTAMP_MILLISECONDS);
  auto offsetDuration =
      utcOffsetSeconds(millisTs, tz::getTimeZoneName(zoneKey), stream, mr);
  auto offsetMillis = cudf::cast(
      offsetDuration->view(),
      cudf::data_type{cudf::type_id::DURATION_MILLISECONDS},
      stream,
      mr);
  auto localMillis = cudf::binary_operation(
      millisTs,
      offsetMillis->view(),
      cudf::binary_operator::ADD,
      cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
      stream,
      mr);
  auto offsetSeconds = std::make_unique<cudf::column>(
      bitcastColumn(offsetDuration->view(), kInt64), stream, mr);
  return {std::move(localMillis), std::move(offsetSeconds)};
}

// Classifies the trailing Joda time-zone token so the caller can render it: an
// offset (with or without a colon), the zone id, or the zone name. Matches CPU
// DateTimeFormatter: single 'Z' has no colon, 'ZZ' has a colon, 'ZZZ' or more
// is the zone id, and lowercase 'z' is the zone abbreviation/name.
enum class TrailingZone {
  kNone,
  kOffsetNoColon,
  kOffsetColon,
  kZoneId,
  kZoneName,
};

// Translates the subset of Joda DateTimeFormat pattern letters used by the
// covered functions to cuDF strftime/strptime specifiers. A trailing time-zone
// token (Z/z) is classified via trailing and rendered separately.
std::string jodaToStrftime(const std::string& joda, TrailingZone& trailing) {
  trailing = TrailingZone::kNone;
  std::string out;
  size_t i = 0;
  while (i < joda.size()) {
    const char c = joda[i];
    if (c == '\'') {
      ++i;
      while (i < joda.size() && joda[i] != '\'') {
        out += joda[i++];
      }
      if (i < joda.size()) {
        ++i;
      }
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c))) {
      size_t j = i;
      while (j < joda.size() && joda[j] == c) {
        ++j;
      }
      const size_t runLength = j - i;
      switch (c) {
        case 'y':
        case 'Y':
          out += runLength >= 3 ? "%Y" : "%y";
          break;
        case 'M':
          out += runLength >= 4 ? "%B" : (runLength == 3 ? "%b" : "%m");
          break;
        case 'd':
          out += "%d";
          break;
        case 'H':
          out += "%H";
          break;
        case 'h':
          out += "%I";
          break;
        case 'm':
          out += "%M";
          break;
        case 's':
          out += "%S";
          break;
        case 'S':
          out += "%3f";
          break;
        case 'a':
          out += "%p";
          break;
        case 'E':
          out += runLength >= 4 ? "%A" : "%a";
          break;
        case 'Z':
        case 'z':
          VELOX_CHECK_EQ(
              j,
              joda.size(),
              "cuDF datetime format supports a time zone token only at the end");
          if (c == 'z') {
            trailing = TrailingZone::kZoneName;
          } else if (runLength == 1) {
            trailing = TrailingZone::kOffsetNoColon;
          } else if (runLength == 2) {
            trailing = TrailingZone::kOffsetColon;
          } else {
            trailing = TrailingZone::kZoneId;
          }
          break;
        default:
          VELOX_NYI(
              "Unsupported datetime format letter on GPU: {}",
              std::string(1, c));
      }
      i = j;
      continue;
    }
    out += c;
    ++i;
  }
  return out;
}

// to_unixtime(timestamp with time zone) -> double.
class ToUnixtimeFunction : public CudfFunction {
 public:
  explicit ToUnixtimeFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "to_unixtime expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    auto millis = unpackMillis(packed, stream, mr);
    auto millisDouble = cudf::cast(
        millis->view(), cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
    auto thousand = cudf::numeric_scalar<double>(1000.0, true, stream);
    return cudf::binary_operation(
        millisDouble->view(),
        thousand,
        cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::FLOAT64},
        stream,
        mr);
  }
};

// at_timezone(timestamp with time zone, varchar) -> timestamp with time zone.
class AtTimezoneFunction : public CudfFunction {
 public:
  explicit AtTimezoneFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "at_timezone expects exactly 2 inputs");
    targetZoneId_ = tz::getTimeZoneID(constStringArg(expr, 1));
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    // Keep the UTC millis bits, replace the low 12 zone bits with the new key.
    auto cleared = binOp(
        packed,
        i64(~static_cast<int64_t>(kTimezoneMask), stream),
        cudf::binary_operator::BITWISE_AND,
        int64Type(),
        stream,
        mr);
    return binOp(
        cleared->view(),
        i64(targetZoneId_ & kTimezoneMask, stream),
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }

 private:
  int16_t targetZoneId_;
};

// timezone_hour / timezone_minute (timestamp with time zone) -> bigint.
class TimezoneFieldFunction : public CudfFunction {
 public:
  TimezoneFieldFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      bool minuteField)
      : minuteField_(minuteField) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    auto offsetSeconds = offsetSecondsForPacked(packed, stream, mr);
    if (minuteField_) {
      auto perMinute = binOp(
          offsetSeconds->view(),
          i64(60, stream),
          cudf::binary_operator::DIV,
          int64Type(),
          stream,
          mr);
      return binOp(
          perMinute->view(),
          i64(60, stream),
          cudf::binary_operator::MOD,
          int64Type(),
          stream,
          mr);
    }
    return binOp(
        offsetSeconds->view(),
        i64(3600, stream),
        cudf::binary_operator::DIV,
        int64Type(),
        stream,
        mr);
  }

 private:
  bool minuteField_;
};

// to_iso8601(timestamp with time zone) -> varchar.
class ToIso8601Function : public CudfFunction {
 public:
  explicit ToIso8601Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "to_iso8601 expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    auto parts = localAndOffset(packed, stream, mr);
    auto dateStr = cudf::strings::from_timestamps(
        parts.localMillis->view(),
        "%Y-%m-%dT%H:%M:%S.%3f",
        cudf::strings_column_view{},
        stream,
        mr);
    // ISO8601 uses "+HH:MM" but renders a zero offset as "Z".
    auto offsetStr = formatOffsetStrings(
        parts.offsetSeconds->view(),
        /*includeColon=*/true,
        std::string("Z"),
        stream,
        mr);
    return cudf::strings::concatenate(
        cudf::table_view{{dateStr->view(), offsetStr->view()}},
        cudf::string_scalar("", true, stream),
        cudf::string_scalar("", false),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
  }
};

// format_datetime(timestamp with time zone, varchar) -> varchar.
class FormatDatetimeFunction : public CudfFunction {
 public:
  explicit FormatDatetimeFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "format_datetime expects exactly 2 inputs");
    strftime_ = jodaToStrftime(constStringArg(expr, 1), trailing_);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    auto parts = localAndOffset(packed, stream, mr);
    auto dateStr = cudf::strings::from_timestamps(
        parts.localMillis->view(),
        strftime_,
        cudf::strings_column_view{},
        stream,
        mr);
    if (trailing_ == TrailingZone::kNone) {
      return dateStr;
    }

    std::unique_ptr<cudf::column> zoneStr;
    switch (trailing_) {
      case TrailingZone::kOffsetNoColon:
        zoneStr = formatOffsetStrings(
            parts.offsetSeconds->view(),
            /*includeColon=*/false,
            std::nullopt,
            stream,
            mr);
        break;
      case TrailingZone::kOffsetColon:
        zoneStr = formatOffsetStrings(
            parts.offsetSeconds->view(),
            /*includeColon=*/true,
            std::nullopt,
            stream,
            mr);
        break;
      case TrailingZone::kZoneId: {
        // The zone id is constant for the column (one zone per column).
        const std::string zoneName =
            tz::getTimeZoneName(uniformZoneKey(packed, stream, mr));
        zoneStr = cudf::make_column_from_scalar(
            cudf::string_scalar(zoneName, true, stream),
            dateStr->size(),
            stream,
            mr);
        break;
      }
      case TrailingZone::kZoneName:
        // The zone abbreviation/name ('z') is DST- and instant-dependent; cuDF
        // cannot render it on device.
        VELOX_NYI(
            "format_datetime zone-name token 'z' is not supported on GPU");
      case TrailingZone::kNone:
        VELOX_UNREACHABLE();
    }
    return cudf::strings::concatenate(
        cudf::table_view{{dateStr->view(), zoneStr->view()}},
        cudf::string_scalar("", true, stream),
        cudf::string_scalar("", false),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
  }

 private:
  std::string strftime_;
  TrailingZone trailing_{TrailingZone::kNone};
};

// Mirrors the CPU pack() range check: throws if any non-null millis value falls
// outside [kMinMillisUtc, kMaxMillisUtc]. Without this, from_unixtime would
// shift an out-of-range instant into the zone-key bits and silently corrupt the
// packed value instead of rejecting it as CPU does.
void checkMillisInRange(
    const cudf::column_view& millis,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (millis.size() == 0 || millis.null_count() == millis.size()) {
    return;
  }
  auto minScalar = cudf::reduce(
      millis,
      *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  auto maxScalar = cudf::reduce(
      millis,
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  const auto lo = static_cast<cudf::numeric_scalar<int64_t>*>(minScalar.get())
                      ->value(stream);
  const auto hi = static_cast<cudf::numeric_scalar<int64_t>*>(maxScalar.get())
                      ->value(stream);
  VELOX_USER_CHECK(
      lo >= kMinMillisUtc && hi <= kMaxMillisUtc,
      "TimestampWithTimeZone overflow: [{}, {}] ms",
      lo,
      hi);
}

// Mirrors the CPU offset bound: from_iso8601_timestamp normalizes its parsed
// offset through tz::getTimeZoneID, which rejects magnitudes beyond +/-14h (840
// minutes). magnitudeMinutes holds the absolute offset minutes (always
// non-negative), so an upper bound covers both signs. Without this, an offset
// like "+99:00" (5940 minutes) maps to a zone key that overflows the 12-bit
// zone field and corrupts the packed millis.
void checkOffsetMagnitudeInRange(
    const cudf::column_view& magnitudeMinutes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (magnitudeMinutes.size() == 0 ||
      magnitudeMinutes.null_count() == magnitudeMinutes.size()) {
    return;
  }
  auto maxScalar = cudf::reduce(
      magnitudeMinutes,
      *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
      int64Type(),
      stream,
      mr);
  const auto hi = static_cast<cudf::numeric_scalar<int64_t>*>(maxScalar.get())
                      ->value(stream);
  VELOX_USER_CHECK_LE(
      hi, 840, "Invalid timezone offset in from_iso8601_timestamp (minutes)");
}

// from_unixtime(double, ...) -> timestamp with time zone. The zone id is fixed
// at construction (from a zone name or an hour/minute offset).
class FromUnixtimeWithZoneFunction : public CudfFunction {
 public:
  explicit FromUnixtimeWithZoneFunction(int16_t zoneId) : zoneId_(zoneId) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto seconds = asView(inputColumns[0]);
    const auto doubleType = cudf::data_type{cudf::type_id::FLOAT64};
    auto thousand = cudf::numeric_scalar<double>(1000.0, true, stream);
    auto millisDouble = cudf::binary_operation(
        seconds, thousand, cudf::binary_operator::MUL, doubleType, stream, mr);
    // Round half away from zero (matching std::llround on the CPU path) by
    // adding +/-0.5 and truncating: a FLOAT64->INT64 cast truncates toward
    // zero.
    auto isNegative = cudf::binary_operation(
        millisDouble->view(),
        cudf::numeric_scalar<double>(0.0, true, stream),
        cudf::binary_operator::LESS,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto half = cudf::copy_if_else(
        cudf::numeric_scalar<double>(-0.5, true, stream),
        cudf::numeric_scalar<double>(0.5, true, stream),
        isNegative->view(),
        stream,
        mr);
    auto adjusted = cudf::binary_operation(
        millisDouble->view(),
        half->view(),
        cudf::binary_operator::ADD,
        doubleType,
        stream,
        mr);
    auto millis = cudf::cast(adjusted->view(), int64Type(), stream, mr);
    checkMillisInRange(millis->view(), stream, mr);
    auto shifted = binOp(
        millis->view(),
        i64(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
    return binOp(
        shifted->view(),
        i64(zoneId_ & kTimezoneMask, stream),
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }

 private:
  int16_t zoneId_;
};

// now() / current_timestamp -> timestamp with time zone. Emits a constant
// column from the session start time and session zone; the value is not
// compared against the CPU (now() is non-deterministic).
class NowFunction : public CudfFunction {
 public:
  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(), "now() needs an input column for sizing");
    const auto size = asView(inputColumns[0]).size();
    const int16_t zoneId = context_.sessionTimezone.empty()
        ? 0
        : tz::getTimeZoneID(context_.sessionTimezone);
    const int64_t packed = (context_.sessionStartTimeMs << kMillisShift) |
        (zoneId & kTimezoneMask);
    auto scalar = i64(packed, stream);
    return cudf::make_column_from_scalar(scalar, size, stream, mr);
  }
};

// parse_datetime(varchar, varchar) -> timestamp with time zone.
class ParseDatetimeFunction : public CudfFunction {
 public:
  explicit ParseDatetimeFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "parse_datetime expects exactly 2 inputs");
    TrailingZone trailing = TrailingZone::kNone;
    strptime_ = jodaToStrftime(constStringArg(expr, 1), trailing);
    if (trailing == TrailingZone::kOffsetNoColon ||
        trailing == TrailingZone::kOffsetColon) {
      strptime_ += "%z";
    } else if (trailing != TrailingZone::kNone) {
      VELOX_NYI("parse_datetime zone-name token is not supported on GPU");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    // cuDF parses the wall clock as UTC. With no embedded zone the result is
    // interpreted in the session timezone (GMT when unset), so the parsed
    // value equals the UTC instant in the GMT case the tests exercise.
    VELOX_CHECK(
        context_.sessionTimezone.empty(),
        "parse_datetime on GPU with a non-UTC session timezone is not yet "
        "supported");
    auto parsed = cudf::strings::to_timestamps(
        cudf::strings_column_view(input),
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
        strptime_,
        stream,
        mr);
    auto millis = bitcastColumn(parsed->view(), kInt64);
    // pack(millis, GMT) == millis << 12.
    return binOp(
        millis,
        i64(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
  }

 private:
  std::string strptime_;
};

// from_iso8601_timestamp(varchar) -> timestamp with time zone.
class FromIso8601Function : public CudfFunction {
 public:
  explicit FromIso8601Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        1,
        "from_iso8601_timestamp expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    // Permissive ISO8601: the year is required; month, day, the time fields,
    // the fractional seconds and the zone suffix are all optional. Missing
    // date/time components default to the start of the period (matching CPU); a
    // missing or "Z" suffix is GMT. The offset sign is captured on its own so a
    // sub-hour negative offset like "-00:30" keeps its sign.
    auto prog = cudf::strings::regex_program::create(
        "^([0-9]{4})(?:-([0-9]{2}))?(?:-([0-9]{2}))?"
        "(?:[T ]([0-9]{2}))?(?::([0-9]{2}))?(?::([0-9]{2}))?"
        "(?:[.,]([0-9]+))?"
        "(?:Z|([+-])([0-9]{2})(?::?([0-9]{2}))?)?$");
    auto groups = cudf::strings::extract(
        cudf::strings_column_view(input), *prog, stream, mr);
    auto g = groups->view();
    // Columns: 0 year, 1 month, 2 day, 3 hour, 4 minute, 5 second,
    //          6 fraction, 7 sign, 8 offset hours, 9 offset minutes.

    auto orDefault = [&](int index, const char* value) {
      return cudf::replace_nulls(
          g.column(index),
          cudf::string_scalar(value, true, stream),
          stream,
          mr);
    };
    auto month = orDefault(1, "01");
    auto day = orDefault(2, "01");
    auto hour = orDefault(3, "00");
    auto minute = orDefault(4, "00");
    auto second = orDefault(5, "00");

    // Build "YYYY-MM-DDTHH:MM:SS". A non-matching row leaves the year null, so
    // separator_on_nulls yields a null that parses to null.
    auto ymd = cudf::strings::concatenate(
        cudf::table_view{{g.column(0), month->view(), day->view()}},
        cudf::string_scalar("-", true, stream),
        cudf::string_scalar("", false),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    auto hms = cudf::strings::concatenate(
        cudf::table_view{{hour->view(), minute->view(), second->view()}},
        cudf::string_scalar(":", true, stream),
        cudf::string_scalar("", false),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    auto canonical = cudf::strings::concatenate(
        cudf::table_view{{ymd->view(), hms->view()}},
        cudf::string_scalar("T", true, stream),
        cudf::string_scalar("", false),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    auto wallTs = cudf::strings::to_timestamps(
        cudf::strings_column_view(canonical->view()),
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
        "%Y-%m-%dT%H:%M:%S",
        stream,
        mr);
    auto wallMillisBase = bitcastColumn(wallTs->view(), kInt64);

    // Fractional seconds -> milliseconds: first 3 digits, right-padded to 3
    // (".1" -> 100, ".12" -> 120, ".123456" -> 123). Missing -> 0.
    auto frac3 = cudf::strings::slice_strings(
        cudf::strings_column_view(g.column(6)),
        cudf::numeric_scalar<cudf::size_type>(0, true, stream),
        cudf::numeric_scalar<cudf::size_type>(3, true, stream),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream),
        stream,
        mr);
    auto fracPadded = cudf::strings::pad(
        cudf::strings_column_view(frac3->view()),
        3,
        cudf::strings::side_type::RIGHT,
        "0",
        stream,
        mr);
    auto fracInts = cudf::strings::to_integers(
        cudf::strings_column_view(fracPadded->view()), int64Type(), stream, mr);
    auto fracMillis =
        cudf::replace_nulls(fracInts->view(), i64(0, stream), stream, mr);
    auto wallMillis = cudf::binary_operation(
        wallMillisBase,
        fracMillis->view(),
        cudf::binary_operator::ADD,
        int64Type(),
        stream,
        mr);

    // Signed offset minutes from the captured sign + HH(:MM); a missing offset
    // (Z or no suffix) yields 0 (GMT). The sign is read from the sign character
    // so "-00:30" stays negative.
    auto offsetHours = cudf::replace_nulls(
        cudf::strings::to_integers(
            cudf::strings_column_view(g.column(8)), int64Type(), stream, mr)
            ->view(),
        i64(0, stream),
        stream,
        mr);
    auto offsetMins = cudf::replace_nulls(
        cudf::strings::to_integers(
            cudf::strings_column_view(g.column(9)), int64Type(), stream, mr)
            ->view(),
        i64(0, stream),
        stream,
        mr);
    auto signStr = cudf::replace_nulls(
        g.column(7), cudf::string_scalar("+", true, stream), stream, mr);
    auto isNegativeSign = cudf::strings::starts_with(
        cudf::strings_column_view(signStr->view()),
        cudf::string_scalar("-", true, stream),
        stream,
        mr);
    auto hourMinutes = binOp(
        offsetHours->view(),
        i64(60, stream),
        cudf::binary_operator::MUL,
        int64Type(),
        stream,
        mr);
    auto magnitude = cudf::binary_operation(
        hourMinutes->view(),
        offsetMins->view(),
        cudf::binary_operator::ADD,
        int64Type(),
        stream,
        mr);
    // Reject offsets beyond +/-14h before they pack into a zone key that
    // overflows the 12-bit zone field, matching CPU's tz::getTimeZoneID bound.
    checkOffsetMagnitudeInRange(magnitude->view(), stream, mr);
    auto negativeMagnitude = cudf::binary_operation(
        i64(0, stream),
        magnitude->view(),
        cudf::binary_operator::SUB,
        int64Type(),
        stream,
        mr);
    auto offsetMinutes = cudf::copy_if_else(
        negativeMagnitude->view(),
        magnitude->view(),
        isNegativeSign->view(),
        stream,
        mr);

    // utcMillis = wallMillis - offsetMinutes * 60000.
    auto offsetMillis = binOp(
        offsetMinutes->view(),
        i64(60000, stream),
        cudf::binary_operator::MUL,
        int64Type(),
        stream,
        mr);
    auto utcMillis = cudf::binary_operation(
        wallMillis->view(),
        offsetMillis->view(),
        cudf::binary_operator::SUB,
        int64Type(),
        stream,
        mr);

    // zoneId from offset minutes: 0 -> 0; <0 -> off+841; >0 -> off+840.
    auto idPositive = binOp(
        offsetMinutes->view(),
        i64(840, stream),
        cudf::binary_operator::ADD,
        int64Type(),
        stream,
        mr);
    auto idNegative = binOp(
        offsetMinutes->view(),
        i64(841, stream),
        cudf::binary_operator::ADD,
        int64Type(),
        stream,
        mr);
    auto isNegativeOffset = binOp(
        offsetMinutes->view(),
        i64(0, stream),
        cudf::binary_operator::LESS,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto idNonZero = cudf::copy_if_else(
        idNegative->view(),
        idPositive->view(),
        isNegativeOffset->view(),
        stream,
        mr);
    auto isZeroOffset = binOp(
        offsetMinutes->view(),
        i64(0, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto zoneId = cudf::copy_if_else(
        i64(0, stream), idNonZero->view(), isZeroOffset->view(), stream, mr);

    // pack(utcMillis, zoneId).
    auto shifted = binOp(
        utcMillis->view(),
        i64(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
    return cudf::binary_operation(
        shifted->view(),
        zoneId->view(),
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }
};

exec::FunctionSignaturePtr twtzArgSignature(const std::string& returnType) {
  return exec::FunctionSignatureBuilder()
      .returnType(returnType)
      .argumentType("timestamp with time zone")
      .build();
}

} // namespace

void registerTimezoneFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // The signatures below reference the TIMESTAMP WITH TIME ZONE custom type,
  // which the worker has not registered yet when it registers cuDF (cuDF is
  // registered before the CPU prestosql functions). Register it here instead of
  // depending on registration order; registerCustomType is idempotent.
  registerTimestampWithTimeZoneType();

  registerCudfFunction(
      prefix + "to_unixtime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ToUnixtimeFunction>(expr);
      },
      {twtzArgSignature("double")});

  registerCudfFunction(
      prefix + "at_timezone",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<AtTimezoneFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("timestamp with time zone")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "timezone_hour",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<TimezoneFieldFunction>(expr, /*minute=*/false);
      },
      {twtzArgSignature("bigint")});

  registerCudfFunction(
      prefix + "timezone_minute",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<TimezoneFieldFunction>(expr, /*minute=*/true);
      },
      {twtzArgSignature("bigint")});

  registerCudfFunction(
      prefix + "to_iso8601",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ToIso8601Function>(expr);
      },
      {twtzArgSignature("varchar")});

  registerCudfFunction(
      prefix + "format_datetime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FormatDatetimeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("timestamp with time zone")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "from_unixtime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FromUnixtimeWithZoneFunction>(
            tz::getTimeZoneID(constStringArg(expr, 1)));
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("double")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "from_unixtime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        const auto offsetMinutes = static_cast<int32_t>(
            constIntArg(expr, 1) * 60 + constIntArg(expr, 2));
        return std::make_shared<FromUnixtimeWithZoneFunction>(
            tz::getTimeZoneID(offsetMinutes));
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("double")
           .constantArgumentType("bigint")
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "parse_datetime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ParseDatetimeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "from_iso8601_timestamp",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FromIso8601Function>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("varchar")
           .build()});

  // now() / current_timestamp take no arguments; an empty signature list always
  // matches by name.
  registerCudfFunctions(
      {prefix + "now", prefix + "current_timestamp"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>&) {
        return std::make_shared<NowFunction>();
      },
      {});
}

} // namespace facebook::velox::cudf_velox
