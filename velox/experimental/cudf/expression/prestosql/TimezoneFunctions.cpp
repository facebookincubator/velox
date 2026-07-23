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

#include "velox/common/base/CheckedArithmetic.h"
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
#include <cudf/datetime.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
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
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/durations.hpp>

#include <cuda_runtime_api.h>

#include <cctype>
#include <limits>
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

cudf::numeric_scalar<int64_t> int64Scalar(
    int64_t value,
    rmm::cuda_stream_view stream) {
  return cudf::numeric_scalar<int64_t>(value, true, stream);
}

std::unique_ptr<cudf::column> binaryOp(
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
  return binaryOp(
      packed,
      int64Scalar(kMillisShift, stream),
      cudf::binary_operator::SHIFT_RIGHT,
      int64Type(),
      stream,
      mr);
}

// The per-row zone key of a packed column plus the distinct non-null keys
// present, computed once and shared by the numeric-offset and zone-name paths.
struct DistinctZones {
  // packed & kTimezoneMask (INT64); nulls preserved (a null packed row yields a
  // null key, which matches no real key in the per-zone selects below).
  std::unique_ptr<cudf::column> perRowKey;
  // Distinct valid zone keys present in the column (the null key excluded).
  std::vector<int16_t> keys;
};

// Extracts the per-row zone key, finds the distinct set on device
// (cudf::distinct), and copies the (small) distinct keys plus their validity to
// host so each zone's name/transition lookup runs once. One device->host sync.
DistinctZones distinctZones(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto perRowKey = binaryOp(
      packed,
      int64Scalar(kTimezoneMask, stream),
      cudf::binary_operator::BITWISE_AND,
      int64Type(),
      stream,
      mr);

  auto unique = cudf::distinct(
      cudf::table_view{{perRowKey->view()}},
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  auto uniqueKeys = unique->view().column(0);
  auto uniqueValid = cudf::is_valid(uniqueKeys, stream, mr);
  std::vector<int64_t> hostKeys(uniqueKeys.size());
  std::vector<int8_t> hostValid(uniqueKeys.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostKeys.data(),
      uniqueKeys.data<int64_t>(),
      hostKeys.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostValid.data(),
      uniqueValid->view().data<int8_t>(),
      hostValid.size() * sizeof(int8_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  std::vector<int16_t> keys;
  keys.reserve(uniqueKeys.size());
  for (cudf::size_type i = 0; i < uniqueKeys.size(); ++i) {
    if (hostValid[i]) { // Skip the null zone key.
      keys.push_back(static_cast<int16_t>(hostKeys[i]));
    }
  }
  return {std::move(perRowKey), std::move(keys)};
}

// Per-row UT offset in whole seconds (INT64) for a packed column that may mix
// zone keys. For each distinct key, computes utcOffsetSeconds over the whole
// column and selects the rows carrying that key. Null rows stay null (their
// null key matches no real key, so copy_if_else keeps the null default).
// O(#distinct zones) device passes.
std::unique_ptr<cudf::column> perRowOffsetSeconds(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto millis = unpackMillis(packed, stream, mr);
  auto millisTs =
      bitcastColumn(millis->view(), cudf::type_id::TIMESTAMP_MILLISECONDS);
  auto zones = distinctZones(packed, stream, mr);

  // Start all-null; fill each zone's rows. A null key matches no real key, so
  // its rows keep the null default (CPU propagates null).
  auto result = cudf::make_numeric_column(
      int64Type(), packed.size(), cudf::mask_state::ALL_NULL, stream, mr);
  for (const auto zoneKey : zones.keys) {
    auto offsetDuration =
        utcOffsetSeconds(millisTs, tz::getTimeZoneName(zoneKey), stream, mr);
    auto offsetSeconds = std::make_unique<cudf::column>(
        bitcastColumn(offsetDuration->view(), kInt64), stream, mr);
    auto isThisZone = binaryOp(
        zones.perRowKey->view(),
        int64Scalar(zoneKey, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    result = cudf::copy_if_else(
        offsetSeconds->view(), result->view(), isThisZone->view(), stream, mr);
  }
  return result;
}

// Per-row zone *name* (STRING) for a packed column that may mix zone keys, for
// the format_datetime 'ZZZ' zone-id token. Starts all-null and fills each
// distinct zone's rows with tz::getTimeZoneName(key) via a string-scalar
// copy_if_else. Null rows stay null. O(#distinct zones) device passes.
std::unique_ptr<cudf::column> perRowZoneName(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto zones = distinctZones(packed, stream, mr);

  // Start all-null strings; fill each zone's rows with its name. An invalid
  // string_scalar builds an all-null strings column; a null key's rows are
  // never selected, so they keep that null (CPU propagates null).
  auto result = cudf::make_column_from_scalar(
      cudf::string_scalar("", false, stream), packed.size(), stream, mr);
  for (const auto zoneKey : zones.keys) {
    auto isThisZone = binaryOp(
        zones.perRowKey->view(),
        int64Scalar(zoneKey, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    // string_scalar-lhs / column-rhs overload: true -> the zone-name scalar,
    // false or null-mask -> the accumulated result.
    result = cudf::copy_if_else(
        cudf::string_scalar(tz::getTimeZoneName(zoneKey), true, stream),
        result->view(),
        isThisZone->view(),
        stream,
        mr);
  }
  return result;
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
  auto isNegative = binaryOp(
      offsetSeconds,
      int64Scalar(0, stream),
      cudf::binary_operator::LESS,
      cudf::data_type{kBool8},
      stream,
      mr);
  // abs(offset) = isNegative ? -offset : offset.
  auto negated = cudf::binary_operation(
      int64Scalar(0, stream),
      offsetSeconds,
      cudf::binary_operator::SUB,
      int64Type(),
      stream,
      mr);
  auto absolute = cudf::copy_if_else(
      negated->view(), offsetSeconds, isNegative->view(), stream, mr);
  auto hours = binaryOp(
      absolute->view(),
      int64Scalar(3'600, stream),
      cudf::binary_operator::DIV,
      int64Type(),
      stream,
      mr);
  auto totalMinutes = binaryOp(
      absolute->view(),
      int64Scalar(60, stream),
      cudf::binary_operator::DIV,
      int64Type(),
      stream,
      mr);
  auto minutes = binaryOp(
      totalMinutes->view(),
      int64Scalar(60, stream),
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
      cudf::string_scalar("", false, stream),
      cudf::strings::separator_on_nulls::YES,
      stream,
      mr);
  auto offsetStr = cudf::strings::concatenate(
      cudf::table_view{{signHour->view(), minutesPadded->view()}},
      cudf::string_scalar(includeColon ? ":" : "", true, stream),
      cudf::string_scalar("", false, stream),
      cudf::strings::separator_on_nulls::YES,
      stream,
      mr);
  if (!zeroOffsetText.has_value()) {
    return offsetStr;
  }
  // Render the zero-offset rows as the supplied text (e.g. "Z").
  auto isZero = binaryOp(
      offsetSeconds,
      int64Scalar(0, stream),
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
// offset (INT64 seconds) for a packed column that may mix zone keys, applying
// each row's own offset.
struct LocalAndOffset {
  std::unique_ptr<cudf::column> localMillis;
  std::unique_ptr<cudf::column> offsetSeconds;
};

LocalAndOffset localAndOffset(
    const cudf::column_view& packed,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto millis = unpackMillis(packed, stream, mr);
  auto offsetSeconds = perRowOffsetSeconds(packed, stream, mr);
  auto offsetMillis = binaryOp(
      offsetSeconds->view(),
      int64Scalar(1'000, stream),
      cudf::binary_operator::MUL,
      int64Type(),
      stream,
      mr);
  auto localMillis = cudf::binary_operation(
      millis->view(),
      offsetMillis->view(),
      cudf::binary_operator::ADD,
      int64Type(),
      stream,
      mr);
  auto localTs =
      bitcastColumn(localMillis->view(), cudf::type_id::TIMESTAMP_MILLISECONDS);
  return {
      std::make_unique<cudf::column>(localTs, stream, mr),
      std::move(offsetSeconds)};
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
          // The 'S' run length is the fractional-second digit count ('S' -> 1
          // digit, 'SSSSSS' -> 6), matching CPU's formatFractionOfSecond. cuDF
          // renders "%<n>f" for n in 1..9; nothing finer than nanoseconds is
          // representable.
          if (runLength > 9) {
            VELOX_NYI(
                "format_datetime supports at most 9 fractional-second digits "
                "on GPU, got {}",
                runLength);
          }
          out += "%" + std::to_string(runLength) + "f";
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
      [[maybe_unused]] cudf::size_type numRows,
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
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    // Keep the UTC millis bits, replace the low 12 zone bits with the new key.
    auto cleared = binaryOp(
        packed,
        int64Scalar(~static_cast<int64_t>(kTimezoneMask), stream),
        cudf::binary_operator::BITWISE_AND,
        int64Type(),
        stream,
        mr);
    return binaryOp(
        cleared->view(),
        int64Scalar(targetZoneId_ & kTimezoneMask, stream),
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
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        1,
        "timezone_hour/timezone_minute expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto packed = asView(inputColumns[0]);
    auto offsetSeconds = perRowOffsetSeconds(packed, stream, mr);
    if (minuteField_) {
      auto perMinute = binaryOp(
          offsetSeconds->view(),
          int64Scalar(60, stream),
          cudf::binary_operator::DIV,
          int64Type(),
          stream,
          mr);
      return binaryOp(
          perMinute->view(),
          int64Scalar(60, stream),
          cudf::binary_operator::MOD,
          int64Type(),
          stream,
          mr);
    }
    return binaryOp(
        offsetSeconds->view(),
        int64Scalar(3'600, stream),
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
      [[maybe_unused]] cudf::size_type numRows,
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
        cudf::string_scalar("", false, stream),
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
      [[maybe_unused]] cudf::size_type numRows,
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
        // Each row renders its own zone name; the column may mix zones.
        zoneStr = perRowZoneName(packed, stream, mr);
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
        cudf::string_scalar("", false, stream),
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

// True if any row of the boolean mask is set. An empty or all-null mask ->
// false (so a batch of only SQL-NULL rows raises no error).
bool anyRowTrue(
    const cudf::column_view& mask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (mask.size() == 0) {
    return false;
  }
  auto reduced = cudf::reduce(
      mask,
      *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{kBool8},
      stream,
      mr);
  auto& scalar = static_cast<cudf::numeric_scalar<bool>&>(*reduced);
  return scalar.is_valid(stream) && scalar.value(stream);
}

// Maps captured offset groups (sign character, hours digits, minutes digits) to
// a signed offset in whole minutes as an INT64 column. A null in the hours or
// minutes group is treated as absent (contributes 0), and a null sign defaults
// to '+', so a missing offset (Z or no suffix) yields 0 (GMT). The sign is read
// from the sign character so "-00:30" stays negative. Rejects magnitudes beyond
// +/-840 minutes via checkOffsetMagnitudeInRange, matching CPU's
// tz::getTimeZoneID bound.
std::unique_ptr<cudf::column> signedOffsetMinutes(
    const cudf::column_view& signChar,
    const cudf::column_view& hoursDigits,
    const cudf::column_view& minutesDigits,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto offsetHours = cudf::replace_nulls(
      cudf::strings::to_integers(
          cudf::strings_column_view(hoursDigits), int64Type(), stream, mr)
          ->view(),
      int64Scalar(0, stream),
      stream,
      mr);
  auto offsetMins = cudf::replace_nulls(
      cudf::strings::to_integers(
          cudf::strings_column_view(minutesDigits), int64Type(), stream, mr)
          ->view(),
      int64Scalar(0, stream),
      stream,
      mr);
  auto signStr = cudf::replace_nulls(
      signChar, cudf::string_scalar("+", true, stream), stream, mr);
  auto isNegativeSign = cudf::strings::starts_with(
      cudf::strings_column_view(signStr->view()),
      cudf::string_scalar("-", true, stream),
      stream,
      mr);
  auto hourMinutes = binaryOp(
      offsetHours->view(),
      int64Scalar(60, stream),
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
      int64Scalar(0, stream),
      magnitude->view(),
      cudf::binary_operator::SUB,
      int64Type(),
      stream,
      mr);
  return cudf::copy_if_else(
      negativeMagnitude->view(),
      magnitude->view(),
      isNegativeSign->view(),
      stream,
      mr);
}

// Maps a signed offset-minutes INT64 column to packed fixed-offset zone keys,
// mirroring Velox's TimeZoneMap ordering: 0 -> 0 (GMT); <0 -> offset+841;
// >0 -> offset+840.
std::unique_ptr<cudf::column> zoneKeyFromOffsetMinutes(
    const cudf::column_view& offsetMinutes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto idPositive = binaryOp(
      offsetMinutes,
      int64Scalar(840, stream),
      cudf::binary_operator::ADD,
      int64Type(),
      stream,
      mr);
  auto idNegative = binaryOp(
      offsetMinutes,
      int64Scalar(841, stream),
      cudf::binary_operator::ADD,
      int64Type(),
      stream,
      mr);
  auto isNegativeOffset = binaryOp(
      offsetMinutes,
      int64Scalar(0, stream),
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
  auto isZeroOffset = binaryOp(
      offsetMinutes,
      int64Scalar(0, stream),
      cudf::binary_operator::EQUAL,
      cudf::data_type{kBool8},
      stream,
      mr);
  return cudf::copy_if_else(
      int64Scalar(0, stream),
      idNonZero->view(),
      isZeroOffset->view(),
      stream,
      mr);
}

// Selects the millisecond rounding for from_unixtime, which differs between the
// two CPU overloads. from_unixtime(double, varchar) rounds the whole value with
// llround(x*1000); from_unixtime(double, hours, minutes) floors the seconds and
// rounds the fractional millisecond separately. The two agree except on
// negative-fractional input, where they can differ by 1 ms (e.g. -0.0005 s ->
// -1 ms for kWhole, 0 ms for kFloorThenFraction).
enum class FromUnixtimeRounding {
  kWhole,
  kFloorThenFraction,
};

// from_unixtime(double, ...) -> timestamp with time zone. The zone id is fixed
// at construction (from a zone name or an hour/minute offset).
class FromUnixtimeWithZoneFunction : public CudfFunction {
 public:
  FromUnixtimeWithZoneFunction(int16_t zoneId, FromUnixtimeRounding rounding)
      : zoneId_(zoneId), rounding_(rounding) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto seconds = asView(inputColumns[0]);
    const auto doubleType = cudf::data_type{cudf::type_id::FLOAT64};

    // Emulate std::llround on a FLOAT64 column: add +/-0.5 then truncate toward
    // zero via the FLOAT64->INT64 cast.
    auto llroundEmu = [&](const cudf::column_view& value) {
      auto isNegative = cudf::binary_operation(
          value,
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
          value,
          half->view(),
          cudf::binary_operator::ADD,
          doubleType,
          stream,
          mr);
      return cudf::cast(adjusted->view(), int64Type(), stream, mr);
    };

    std::unique_ptr<cudf::column> millis;
    if (rounding_ == FromUnixtimeRounding::kWhole) {
      auto millisDouble = cudf::binary_operation(
          seconds,
          cudf::numeric_scalar<double>(1000.0, true, stream),
          cudf::binary_operator::MUL,
          doubleType,
          stream,
          mr);
      millis = llroundEmu(millisDouble->view());
    } else {
      // floor(x) whole seconds, plus the fractional second rounded on its own
      // (matching CPU's no-zone fromUnixtime). The fraction is in [0, 1), so
      // its round is the non-negative case; a fraction that rounds up to 1000
      // ms carries naturally through the addition.
      auto secondsFloor = cudf::unary_operation(
          seconds, cudf::unary_operator::FLOOR, stream, mr);
      auto fraction = cudf::binary_operation(
          seconds,
          secondsFloor->view(),
          cudf::binary_operator::SUB,
          doubleType,
          stream,
          mr);
      auto fractionMillisDouble = cudf::binary_operation(
          fraction->view(),
          cudf::numeric_scalar<double>(1000.0, true, stream),
          cudf::binary_operator::MUL,
          doubleType,
          stream,
          mr);
      auto fractionMillis = llroundEmu(fractionMillisDouble->view());
      auto secondsInt =
          cudf::cast(secondsFloor->view(), int64Type(), stream, mr);
      auto secondsMillis = binaryOp(
          secondsInt->view(),
          int64Scalar(1000, stream),
          cudf::binary_operator::MUL,
          int64Type(),
          stream,
          mr);
      millis = cudf::binary_operation(
          secondsMillis->view(),
          fractionMillis->view(),
          cudf::binary_operator::ADD,
          int64Type(),
          stream,
          mr);
    }

    // Match CPU's non-finite handling, which a FLOAT64->INT64 cast does not
    // give on its own: NaN maps to pack(0), and +/-Inf saturates out of range
    // so pack (here checkMillisInRange) rejects it. A null input stays null
    // because a null comparison yields a null mask element, which copy_if_else
    // treats as false and so keeps the (null) computed millis.
    const auto infinity = std::numeric_limits<double>::infinity();
    auto isNan = cudf::binary_operation(
        seconds,
        seconds,
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto isPositiveInf = cudf::binary_operation(
        seconds,
        cudf::numeric_scalar<double>(infinity, true, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto isNegativeInf = cudf::binary_operation(
        seconds,
        cudf::numeric_scalar<double>(-infinity, true, stream),
        cudf::binary_operator::EQUAL,
        cudf::data_type{kBool8},
        stream,
        mr);
    auto isInf = cudf::binary_operation(
        isPositiveInf->view(),
        isNegativeInf->view(),
        cudf::binary_operator::LOGICAL_OR,
        cudf::data_type{kBool8},
        stream,
        mr);
    millis = cudf::copy_if_else(
        int64Scalar(0, stream), millis->view(), isNan->view(), stream, mr);
    millis = cudf::copy_if_else(
        int64Scalar(kMaxMillisUtc + 1, stream),
        millis->view(),
        isInf->view(),
        stream,
        mr);

    checkMillisInRange(millis->view(), stream, mr);
    auto shifted = binaryOp(
        millis->view(),
        int64Scalar(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
    return binaryOp(
        shifted->view(),
        int64Scalar(zoneId_ & kTimezoneMask, stream),
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }

 private:
  int16_t zoneId_;
  FromUnixtimeRounding rounding_;
};

// now() / current_timestamp -> timestamp with time zone. Emits a constant
// column packing the session start time with the session zone, matching CPU's
// CurrentTimestampFunction (the value is not compared against a live CPU now(),
// which is non-deterministic). Rejects like CPU when the session zone is
// unusable: getTimeZoneFromConfig returns null when
// adjust_timestamp_to_session_timezone is off or the session timezone is empty,
// and CPU then throws "Timezone cannot be null".
class NowFunction : public CudfFunction {
 public:
  ColumnOrView eval(
      [[maybe_unused]] std::vector<ColumnOrView>& inputColumns,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_USER_CHECK(
        context_.adjustTimestampToTimezone && !context_.sessionTimezone.empty(),
        "Timezone cannot be null");
    const auto zoneId = tz::getTimeZoneID(context_.sessionTimezone);
    const int64_t packed = pack(context_.sessionStartTimeMs, zoneId);
    auto scalar = int64Scalar(packed, stream);
    return cudf::make_column_from_scalar(scalar, numRows, stream, mr);
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
      // to_timestamps folds the %z offset into the UTC instant. The parsed
      // offset is recovered per-row in eval so the packed zone key reflects it
      // instead of GMT.
      strptime_ += "%z";
      hasOffset_ = true;
      // Trailing signed offset with an optional colon, matching both "-09:00"
      // and "-0900". Groups: 0 sign, 1 hours, 2 minutes.
      offsetProgram_ =
          cudf::strings::regex_program::create("([+-])([0-9]{2}):?([0-9]{2})$");
    } else if (trailing != TrailingZone::kNone) {
      VELOX_NYI("parse_datetime zone-name token is not supported on GPU");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    // cuDF parses the wall clock as UTC. With no embedded zone the result is
    // interpreted in the session timezone (GMT when unset), so the parsed
    // value equals the UTC instant in the GMT case the tests exercise.
    if (!context_.sessionTimezone.empty()) {
      VELOX_NYI(
          "parse_datetime on GPU with a non-UTC session timezone is not yet "
          "supported");
    }
    auto parsed = cudf::strings::to_timestamps(
        cudf::strings_column_view(input),
        cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
        strptime_,
        stream,
        mr);
    auto millis = bitcastColumn(parsed->view(), kInt64);
    auto shifted = binaryOp(
        millis,
        int64Scalar(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
    if (!hasOffset_) {
      // pack(millis, GMT) == millis << 12.
      return shifted;
    }
    // Recover the per-row offset that to_timestamps folded into the instant and
    // pack the matching fixed-offset zone key, so timezone_hour/to_iso8601
    // reflect the parsed offset instead of GMT.
    auto groups = cudf::strings::extract(
        cudf::strings_column_view(input), *offsetProgram_, stream, mr);
    auto g = groups->view();
    auto offsetMinutes =
        signedOffsetMinutes(g.column(0), g.column(1), g.column(2), stream, mr);
    auto zoneId = zoneKeyFromOffsetMinutes(offsetMinutes->view(), stream, mr);
    return cudf::binary_operation(
        shifted->view(),
        zoneId->view(),
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }

 private:
  std::string strptime_;
  // True when the Joda format carries a numeric offset token (%z appended);
  // gates the per-row offset recovery in eval.
  bool hasOffset_{false};
  // Compiled trailing-offset extraction program, built once when hasOffset_.
  std::unique_ptr<cudf::strings::regex_program> offsetProgram_;
};

// from_iso8601_timestamp(varchar) -> timestamp with time zone.
class FromIso8601Function : public CudfFunction {
 public:
  explicit FromIso8601Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        1,
        "from_iso8601_timestamp expects exactly 1 input");
    // Permissive ISO8601 (date-anchored). The year is required; month, day, the
    // time fields, the fractional seconds and the zone suffix are all optional,
    // and missing components default to the start of the period (matching CPU).
    // A 'T' may appear with no time after it ("2021-01-01T", "2021T+14:00");
    // the date/time separator is a literal 'T' only, since CPU rejects a space.
    // Time-only inputs ("T11:38") carry no date; eval prefixes the epoch date
    // "1970-01-01" to them before this program runs, so the single
    // date-anchored program still covers them. The whole zone suffix is
    // captured (group 7) to tell an absent suffix from an explicit "Z"; the
    // sign is captured on its own (group 8) so a sub-hour offset like "-00:30"
    // keeps it. Groups: 0 year, 1 month, 2 day, 3 hour, 4 minute, 5 second, 6
    // fraction, 7 zone suffix, 8 sign, 9 offset hours, 10 offset minutes.
    // Batch-independent, so build once.
    isoProgram_ = cudf::strings::regex_program::create(
        "^([0-9]{4})(?:-([0-9]{2}))?(?:-([0-9]{2}))?"
        "(?:T([0-9]{2})?(?::([0-9]{2}))?(?::([0-9]{2}))?)?"
        "(?:[.,]([0-9]+))?"
        "(Z|([+-])([0-9]{2})(?::?([0-9]{2}))?)?$");
    // Identifies a leading time-only form ("Thh...") so eval can prefix the
    // epoch date "1970-01-01" and reuse the date-anchored program. A bare "T"
    // (no digits) does not match, so it stays unprefixed and is later rejected
    // as malformed, like CPU.
    timeOnlyProgram_ = cudf::strings::regex_program::create("^T[0-9]{2}");
    // Matches an otherwise-valid ISO8601 string whose year is signed or has 5+
    // digits -- the CPU-valid extreme years cudf::strings::to_timestamps (int16
    // %Y) cannot represent. Same tail as isoProgram_ so only the year token
    // differs; used only as a match test (captures are ignored).
    extremeProgram_ = cudf::strings::regex_program::create(
        "^(?:[+-][0-9]{4,}|[0-9]{5,})(?:-([0-9]{2}))?(?:-([0-9]{2}))?"
        "(?:T([0-9]{2})?(?::([0-9]{2}))?(?::([0-9]{2}))?)?"
        "(?:[.,]([0-9]+))?"
        "(Z|([+-])([0-9]{2})(?::?([0-9]{2}))?)?$");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto input = asView(inputColumns[0]);
    // Time-only inputs carry no date; CPU defaults them to 1970-01-01. Prefix
    // that date to leading-'T' rows so the single date-anchored program (built
    // in the ctor) handles them; every other row is passed through unchanged.
    auto isTimeOnly = cudf::replace_nulls(
        cudf::strings::matches_re(
            cudf::strings_column_view(input), *timeOnlyProgram_, stream, mr)
            ->view(),
        cudf::numeric_scalar<bool>(false, true, stream),
        stream,
        mr);
    auto epochDate = cudf::make_column_from_scalar(
        cudf::string_scalar("1970-01-01", true, stream),
        input.size(),
        stream,
        mr);
    auto prefixed = cudf::strings::concatenate(
        cudf::table_view{{epochDate->view(), input}},
        cudf::string_scalar("", true, stream),
        cudf::string_scalar("", false, stream),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    auto work = cudf::copy_if_else(
        prefixed->view(), input, isTimeOnly->view(), stream, mr);
    auto workView = cudf::strings_column_view(work->view());
    // Extract the ISO8601 fields with the program built in the constructor; see
    // there for the field layout, and the group-column map just below.
    auto groups = cudf::strings::extract(workView, *isoProgram_, stream, mr);
    auto g = groups->view();
    // Columns: 0 year, 1 month, 2 day, 3 hour, 4 minute, 5 second, 6 fraction,
    //          7 zone suffix, 8 sign, 9 offset hours, 10 offset minutes.

    auto orDefault = [&](int index, const char* value) {
      return cudf::replace_nulls(
          g.column(index),
          cudf::string_scalar(value, true, stream),
          stream,
          mr);
    };
    // cudf's extract yields an empty string (not a null) for an optional group
    // that did not participate in an otherwise-matching row, so replace_nulls
    // alone leaves an absent month or day empty. to_timestamps then reads the
    // empty numeric field as 0 and underflows, e.g. "2021" (no month/day)
    // parses as 2020-11-30 instead of 2021-01-01. Month and day must default to
    // "01", so replace an empty (or null) capture explicitly. The time fields
    // default to 0, which an empty string already yields, so they keep
    // replace_nulls.
    auto orFirstOfPeriod = [&](int index) {
      auto filled = cudf::replace_nulls(
          g.column(index), cudf::string_scalar("01", true, stream), stream, mr);
      auto length = cudf::strings::count_characters(
          cudf::strings_column_view(filled->view()), stream, mr);
      auto isEmpty = cudf::binary_operation(
          length->view(),
          cudf::numeric_scalar<cudf::size_type>(0, true, stream),
          cudf::binary_operator::EQUAL,
          cudf::data_type{kBool8},
          stream,
          mr);
      return cudf::copy_if_else(
          cudf::string_scalar("01", true, stream),
          filled->view(),
          isEmpty->view(),
          stream,
          mr);
    };
    auto month = orFirstOfPeriod(1);
    auto day = orFirstOfPeriod(2);
    auto hour = orDefault(3, "00");
    auto minute = orDefault(4, "00");
    auto second = orDefault(5, "00");

    // Build "YYYY-MM-DD" first; the throw block below and the canonical
    // timestamp string both reuse it. A non-matching row leaves the year null,
    // so separator_on_nulls yields a null that parses to null.
    auto ymd = cudf::strings::concatenate(
        cudf::table_view{{g.column(0), month->view(), day->view()}},
        cudf::string_scalar("-", true, stream),
        cudf::string_scalar("", false, stream),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);

    // Match CPU exactly for every non-null row: parse it, or throw. Genuine
    // SQL-NULL rows are excluded via is_valid, so they keep propagating as
    // NULL. A non-null row falls into one of three buckets:
    //   - matches the in-range program (isoProgram_) and names a real calendar
    //     date -> parsed normally below;
    //   - matches neither program, or matches isoProgram_ but names a
    //     nonexistent date (month/day out of range) -> malformed, exactly as
    //     CPU's fromTimestampWithTimezoneString / isValidDate ->
    //     VELOX_USER_FAIL;
    //   - matches only the extreme-year program -> CPU-valid but beyond what
    //     to_timestamps (int16 %Y) can represent -> VELOX_NYI.
    // Malformed is checked first so a batch mixing malformed + extreme rows
    // reports the parse error, as CPU would.
    {
      const auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream);
      const auto trueScalar = cudf::numeric_scalar<bool>(true, true, stream);
      auto nonNull = cudf::is_valid(input, stream, mr);
      auto tier1 = cudf::replace_nulls(
          cudf::strings::matches_re(workView, *isoProgram_, stream, mr)->view(),
          falseScalar,
          stream,
          mr);
      auto extreme = cudf::replace_nulls(
          cudf::strings::matches_re(workView, *extremeProgram_, stream, mr)
              ->view(),
          falseScalar,
          stream,
          mr);
      auto known = cudf::binary_operation(
          tier1->view(),
          extreme->view(),
          cudf::binary_operator::LOGICAL_OR,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto unknown = cudf::unary_operation(
          known->view(), cudf::unary_operator::NOT, stream, mr);
      auto malformedShape = cudf::binary_operation(
          nonNull->view(),
          unknown->view(),
          cudf::binary_operator::LOGICAL_AND,
          cudf::data_type{kBool8},
          stream,
          mr);

      // cudf::strings::to_timestamps normalizes an out-of-range month or day
      // (month 13 -> next year, day 30 in Feb -> March) instead of failing, so
      // "2021-13-45" matches isoProgram_ yet is not a real date. Parse the
      // date, read month and day back, and require they equal the parsed input.
      // Any normalization moves the value into a different month (a day
      // underflow or overflow always crosses a month boundary), so comparing
      // month and day catches every invalid combination, including a non-leap
      // Feb 29. The 4-digit year is regex-bounded to [0000, 9999], so it always
      // round-trips.
      const auto int16Type = cudf::data_type{cudf::type_id::INT16};
      auto dateTs = cudf::strings::to_timestamps(
          cudf::strings_column_view(ymd->view()),
          cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
          "%Y-%m-%d",
          stream,
          mr);
      auto backMonth = cudf::datetime::extract_datetime_component(
          dateTs->view(),
          cudf::datetime::datetime_component::MONTH,
          stream,
          mr);
      auto backDay = cudf::datetime::extract_datetime_component(
          dateTs->view(), cudf::datetime::datetime_component::DAY, stream, mr);
      auto inMonth = cudf::strings::to_integers(
          cudf::strings_column_view(month->view()), int16Type, stream, mr);
      auto inDay = cudf::strings::to_integers(
          cudf::strings_column_view(day->view()), int16Type, stream, mr);
      auto monthOk = cudf::binary_operation(
          backMonth->view(),
          inMonth->view(),
          cudf::binary_operator::EQUAL,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto dayOk = cudf::binary_operation(
          backDay->view(),
          inDay->view(),
          cudf::binary_operator::EQUAL,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto dateOk = cudf::binary_operation(
          monthOk->view(),
          dayOk->view(),
          cudf::binary_operator::LOGICAL_AND,
          cudf::data_type{kBool8},
          stream,
          mr);
      // A non-tier1 row has a null date, hence a null dateOk; treat null as ok
      // so only tier1 rows can be flagged here (also AND'd with tier1 below).
      auto dateOkFilled =
          cudf::replace_nulls(dateOk->view(), trueScalar, stream, mr);
      auto dateInvalid = cudf::unary_operation(
          dateOkFilled->view(), cudf::unary_operator::NOT, stream, mr);
      auto tier1NonNull = cudf::binary_operation(
          nonNull->view(),
          tier1->view(),
          cudf::binary_operator::LOGICAL_AND,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto calendarBad = cudf::binary_operation(
          tier1NonNull->view(),
          dateInvalid->view(),
          cudf::binary_operator::LOGICAL_AND,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto malformed = cudf::binary_operation(
          malformedShape->view(),
          calendarBad->view(),
          cudf::binary_operator::LOGICAL_OR,
          cudf::data_type{kBool8},
          stream,
          mr);
      VELOX_USER_CHECK(
          !anyRowTrue(malformed->view(), stream, mr),
          "Unable to parse timestamp value in from_iso8601_timestamp");
      auto extremeNonNull = cudf::binary_operation(
          nonNull->view(),
          extreme->view(),
          cudf::binary_operator::LOGICAL_AND,
          cudf::data_type{kBool8},
          stream,
          mr);
      if (anyRowTrue(extremeNonNull->view(), stream, mr)) {
        VELOX_NYI(
            "from_iso8601_timestamp does not support years outside [0000, 9999] on GPU");
      }
    }

    auto hms = cudf::strings::concatenate(
        cudf::table_view{{hour->view(), minute->view(), second->view()}},
        cudf::string_scalar(":", true, stream),
        cudf::string_scalar("", false, stream),
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    auto canonical = cudf::strings::concatenate(
        cudf::table_view{{ymd->view(), hms->view()}},
        cudf::string_scalar("T", true, stream),
        cudf::string_scalar("", false, stream),
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
    auto fracMillis = cudf::replace_nulls(
        fracInts->view(), int64Scalar(0, stream), stream, mr);
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
    auto offsetMinutes =
        signedOffsetMinutes(g.column(8), g.column(9), g.column(10), stream, mr);

    // utcMillis = wallMillis - offsetMinutes * 60'000.
    auto offsetMillis = binaryOp(
        offsetMinutes->view(),
        int64Scalar(60'000, stream),
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
    auto zoneId = zoneKeyFromOffsetMinutes(offsetMinutes->view(), stream, mr);

    // An offset-less input is interpreted in the session timezone, not GMT,
    // when one is set -- matching CPU's FromIso8601Timestamp, which reads the
    // wall clock as that zone's local time (via Timestamp::toGMT) and packs the
    // session zone key. Rows carrying an explicit "Z" or numeric offset keep
    // the result computed above; the captured zone suffix (group 7)
    // distinguishes them from an absent suffix. toUtcTimestamp does the exact
    // DST-aware local->UTC conversion: it fails on a nonexistent local time
    // (spring-forward gap) and resolves an ambiguous one (fall-back overlap)
    // to the earliest instant, like CPU. Only the offset-less rows are
    // converted; the rest are nulled out so their wall clock is never flagged
    // as a gap.
    std::unique_ptr<cudf::column> selectedMillis;
    std::unique_ptr<cudf::column> selectedZone;
    cudf::column_view finalMillis = utcMillis->view();
    cudf::column_view finalZone = zoneId->view();
    if (!context_.sessionTimezone.empty()) {
      auto zoneSuffix = cudf::replace_nulls(
          g.column(7), cudf::string_scalar("", true, stream), stream, mr);
      auto suffixLength = cudf::strings::count_characters(
          cudf::strings_column_view(zoneSuffix->view()), stream, mr);
      auto hasExplicitZone = cudf::binary_operation(
          suffixLength->view(),
          cudf::numeric_scalar<cudf::size_type>(0, true, stream),
          cudf::binary_operator::GREATER,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto offsetless = cudf::binary_operation(
          suffixLength->view(),
          cudf::numeric_scalar<cudf::size_type>(0, true, stream),
          cudf::binary_operator::EQUAL,
          cudf::data_type{kBool8},
          stream,
          mr);
      auto wallTimestamp = bitcastColumn(
          wallMillis->view(), cudf::type_id::TIMESTAMP_MILLISECONDS);
      // Null the explicit-zone rows so only the offset-less rows reach the gap
      // check inside toUtcTimestamp; their result is discarded below anyway.
      auto nullWall = cudf::make_default_constructed_scalar(
          cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, stream, mr);
      auto sessionWall = cudf::copy_if_else(
          wallTimestamp, *nullWall, offsetless->view(), stream, mr);
      auto sessionUtcTimestamp = toUtcTimestamp(
          sessionWall->view(), context_.sessionTimezone, stream, mr);
      auto sessionUtcMillis =
          bitcastColumn(sessionUtcTimestamp->view(), kInt64);
      const auto sessionZoneKey = tz::getTimeZoneID(context_.sessionTimezone);
      selectedMillis = cudf::copy_if_else(
          utcMillis->view(),
          sessionUtcMillis,
          hasExplicitZone->view(),
          stream,
          mr);
      selectedZone = cudf::copy_if_else(
          zoneId->view(),
          int64Scalar(sessionZoneKey & kTimezoneMask, stream),
          hasExplicitZone->view(),
          stream,
          mr);
      finalMillis = selectedMillis->view();
      finalZone = selectedZone->view();
    }

    // pack(finalMillis, finalZone).
    auto shifted = binaryOp(
        finalMillis,
        int64Scalar(kMillisShift, stream),
        cudf::binary_operator::SHIFT_LEFT,
        int64Type(),
        stream,
        mr);
    return cudf::binary_operation(
        shifted->view(),
        finalZone,
        cudf::binary_operator::BITWISE_OR,
        int64Type(),
        stream,
        mr);
  }

 private:
  // Compiled ISO8601 field-extraction program. Batch-independent, so it is
  // built once in the constructor and reused across eval calls.
  std::unique_ptr<cudf::strings::regex_program> isoProgram_;
  // Recognizes a leading time-only form ("Thh...") so eval can prefix the epoch
  // date and reuse isoProgram_.
  std::unique_ptr<cudf::strings::regex_program> timeOnlyProgram_;
  // Matches an ISO8601 string whose year is signed or 5+ digits -- CPU-valid
  // but unrepresentable by to_timestamps; eval raises VELOX_NYI for these.
  std::unique_ptr<cudf::strings::regex_program> extremeProgram_;
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
            tz::getTimeZoneID(constStringArg(expr, 1)),
            FromUnixtimeRounding::kWhole);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp with time zone")
           .argumentType("double")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "from_unixtime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        // Compute hours*60 + minutes in int64 with overflow checks, mirroring
        // CPU FromUnixtimeFunction; tz::getTimeZoneID then bounds the result to
        // +/-840 minutes. Guards against a large hours value overflowing the
        // product and truncating into a bogus in-range offset.
        const auto offsetMinutes = checkedPlus(
            checkedMultiply<int64_t>(constIntArg(expr, 1), 60),
            constIntArg(expr, 2));
        return std::make_shared<FromUnixtimeWithZoneFunction>(
            tz::getTimeZoneID(static_cast<int32_t>(offsetMinutes)),
            FromUnixtimeRounding::kFloorThenFraction);
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
