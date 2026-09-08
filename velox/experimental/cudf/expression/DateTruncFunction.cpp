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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/TimestampWithTimeZoneColumn.h"
#include "velox/experimental/cudf/expression/TimezoneConversion.h"

#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

using functions::DateTimeUnit;

namespace {

// Maps a timestamp data type to the duration type of the same resolution, used
// to subtract the hour-truncation remainder from the UTC instant.
cudf::data_type durationTypeForTimestamp(cudf::data_type timestampType) {
  switch (timestampType.id()) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return cudf::data_type(cudf::type_id::DURATION_SECONDS);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return cudf::data_type(cudf::type_id::DURATION_MILLISECONDS);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return cudf::data_type(cudf::type_id::DURATION_MICROSECONDS);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return cudf::data_type(cudf::type_id::DURATION_NANOSECONDS);
    default:
      VELOX_FAIL(
          "date_trunc hour requires a timestamp column, got cudf type id: {}",
          static_cast<int32_t>(timestampType.id()));
  }
}

} // namespace

bool DateTruncFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 2) {
    return false;
  }
  if (expr->inputs()[1]->isConstantKind()) {
    return false;
  }
  auto unitString = constantVarcharValue(expr->inputs()[0]);
  if (!unitString.has_value()) {
    return false;
  }
  auto unit = functions::fromDateTimeUnitString(*unitString, false);
  if (!unit.has_value()) {
    return false;
  }
  const auto& inputType = expr->inputs()[1]->type();
  const bool isTimestamp = inputType->isTimestamp();
  const bool isDate = inputType->isDate();
  const bool isTswtz = isTimestampWithTimeZoneType(inputType);
  if (!isTimestamp && !isDate && !isTswtz) {
    return false;
  }
  if (*unit == DateTimeUnit::kSecond || *unit == DateTimeUnit::kMinute ||
      *unit == DateTimeUnit::kHour) {
    return isTimestamp || isTswtz;
  }
  if (*unit == DateTimeUnit::kDay || *unit == DateTimeUnit::kWeek ||
      *unit == DateTimeUnit::kMonth || *unit == DateTimeUnit::kQuarter ||
      *unit == DateTimeUnit::kYear) {
    return true;
  }
  return false;
}

DateTruncFunction::DateTruncFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_trunc expects exactly 2 inputs");
  auto unitString = constantVarcharValue(expr->inputs()[0]);
  VELOX_CHECK(
      unitString.has_value(), "date_trunc unit must be a non-null constant");
  auto inputType = expr->inputs()[1]->type();
  const bool isTimestamp = inputType->isTimestamp();
  const bool isDate = inputType->isDate();
  isTimestampWithTimeZone_ = isTimestampWithTimeZoneType(inputType);
  VELOX_CHECK(
      isTimestamp || isDate || isTimestampWithTimeZone_,
      "date_trunc only supports date, timestamp, or timestamp with time zone inputs");
  auto parsed = functions::fromDateTimeUnitString(*unitString, true);
  VELOX_CHECK(parsed.has_value(), "Invalid date_trunc unit: {}", *unitString);
  unit_ = *parsed;

  // Validate time-only units require an instant (timestamp or TSWTZ) input.
  if (unit_ == DateTimeUnit::kSecond || unit_ == DateTimeUnit::kMinute ||
      unit_ == DateTimeUnit::kHour) {
    VELOX_CHECK(
        isTimestamp || isTimestampWithTimeZone_,
        "date_trunc {} requires timestamp input",
        *unitString);
  }

  auto stream = cudf::get_default_stream(cudf::allow_default_stream);
  auto mr = get_temp_mr();
  oneScalar_ =
      std::make_unique<cudf::numeric_scalar<int32_t>>(1, true, stream, mr);
  threeScalar_ =
      std::make_unique<cudf::numeric_scalar<int32_t>>(3, true, stream, mr);
  negOneScalar_ =
      std::make_unique<cudf::numeric_scalar<int32_t>>(-1, true, stream, mr);
  stream.synchronize();
}

ColumnOrView DateTruncFunction::truncateOnColumn(
    cudf::column_view inputCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto outputType = inputCol.type();
  auto dayType = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
  auto intType = cudf::data_type(cudf::type_id::INT32);
  auto durationDayType = cudf::data_type(cudf::type_id::DURATION_DAYS);

  auto floorToDay = [&](cudf::column_view col) {
    if (col.type() == dayType) {
      return std::make_unique<cudf::column>(col, stream, mr);
    }
    auto floored = cudf::datetime::floor_datetimes(
        col, cudf::datetime::rounding_frequency::DAY, stream, mr);
    return cudf::cast(floored->view(), dayType, stream, mr);
  };
  auto castToInt32 = [&](cudf::column_view col) {
    return cudf::cast(col, intType, stream, mr);
  };
  auto castToDurationDays = [&](cudf::column_view col) {
    return cudf::cast(col, durationDayType, stream, mr);
  };
  auto castDaysToOutput =
      [&](std::unique_ptr<cudf::column> daysCol) -> ColumnOrView {
    if (daysCol->type() == outputType) {
      return daysCol;
    }
    return cudf::cast(daysCol->view(), outputType, stream, mr);
  };

  switch (unit_) {
    case DateTimeUnit::kSecond:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::SECOND, stream, mr);
    case DateTimeUnit::kMinute:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::MINUTE, stream, mr);
    case DateTimeUnit::kHour:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::HOUR, stream, mr);
    case DateTimeUnit::kDay:
      return castDaysToOutput(floorToDay(inputCol));
    case DateTimeUnit::kWeek: {
      auto dayCol = floorToDay(inputCol);
      auto dowCol = cudf::datetime::extract_datetime_component(
          dayCol->view(),
          cudf::datetime::datetime_component::WEEKDAY,
          stream,
          mr);
      auto dowInt = castToInt32(dowCol->view());
      auto offset = cudf::binary_operation(
          dowInt->view(),
          *oneScalar_,
          cudf::binary_operator::SUB,
          intType,
          stream,
          mr);
      auto offsetDur = castToDurationDays(offset->view());
      auto weekStartDay = cudf::binary_operation(
          dayCol->view(),
          offsetDur->view(),
          cudf::binary_operator::SUB,
          dayType,
          stream,
          mr);
      return castDaysToOutput(std::move(weekStartDay));
    }
    case DateTimeUnit::kMonth:
    case DateTimeUnit::kQuarter:
    case DateTimeUnit::kYear: {
      auto dayCol = floorToDay(inputCol);
      auto dayOfMonth = cudf::datetime::extract_datetime_component(
          dayCol->view(), cudf::datetime::datetime_component::DAY, stream, mr);
      auto dayOfMonthInt = castToInt32(dayOfMonth->view());
      auto dayOffset = cudf::binary_operation(
          dayOfMonthInt->view(),
          *oneScalar_,
          cudf::binary_operator::SUB,
          intType,
          stream,
          mr);
      auto dayOffsetDur = castToDurationDays(dayOffset->view());
      auto monthStartDay = cudf::binary_operation(
          dayCol->view(),
          dayOffsetDur->view(),
          cudf::binary_operator::SUB,
          dayType,
          stream,
          mr);

      if (unit_ == DateTimeUnit::kMonth) {
        return castDaysToOutput(std::move(monthStartDay));
      }

      auto monthCol = cudf::datetime::extract_datetime_component(
          dayCol->view(),
          cudf::datetime::datetime_component::MONTH,
          stream,
          mr);
      auto monthInt = castToInt32(monthCol->view());
      auto monthIndex = cudf::binary_operation(
          monthInt->view(),
          *oneScalar_,
          cudf::binary_operator::SUB,
          intType,
          stream,
          mr);

      std::unique_ptr<cudf::column> monthsToSubtract;
      if (unit_ == DateTimeUnit::kYear) {
        monthsToSubtract = std::move(monthIndex);
      } else {
        monthsToSubtract = cudf::binary_operation(
            monthIndex->view(),
            *threeScalar_,
            cudf::binary_operator::MOD,
            intType,
            stream,
            mr);
      }

      auto negMonths = cudf::binary_operation(
          monthsToSubtract->view(),
          *negOneScalar_,
          cudf::binary_operator::MUL,
          intType,
          stream,
          mr);
      auto truncated = cudf::datetime::add_calendrical_months(
          monthStartDay->view(), negMonths->view(), stream, mr);
      return castDaysToOutput(std::move(truncated));
    }
    default:
      break;
  }
  VELOX_UNREACHABLE();
}

ColumnOrView DateTruncFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_EQ(inputColumns.size(), 1, "date_trunc expects one column input");

  if (isTimestampWithTimeZone_) {
    // TIMESTAMP WITH TIME ZONE carries its own zone per row; truncate on each
    // row's embedded wall clock (per-row multi-zone), independent of the
    // session zone. Matches CPU DateTruncFunction::call(TSWTZ).
    auto packed = asView(inputColumns[0]);
    auto zoneKey = tswtzZoneKey(packed, stream, mr);
    auto distinct = tswtzDistinctZoneKeys(zoneKey->view(), stream, mr);
    auto local = tswtzLocalWallClock(packed, stream, mr);

    std::unique_ptr<cudf::column> truncatedUtcMillis;
    if (unit_ == DateTimeUnit::kSecond || unit_ == DateTimeUnit::kMinute ||
        unit_ == DateTimeUnit::kHour) {
      // unit < day: take the local-to-truncated delta and subtract it from the
      // UTC instant. Whole-minute offsets make this exact for second/minute,
      // and it is the DST-safe form for the hour branch.
      ColumnOrView flooredLocal = truncateOnColumn(local->view(), stream, mr);
      auto delta = cudf::binary_operation(
          local->view(),
          asView(flooredLocal),
          cudf::binary_operator::SUB,
          cudf::data_type(cudf::type_id::DURATION_MILLISECONDS),
          stream,
          mr);
      auto utcInstant = tswtzUtcInstant(packed, stream, mr);
      truncatedUtcMillis = cudf::binary_operation(
          utcInstant->view(),
          delta->view(),
          cudf::binary_operator::SUB,
          cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS),
          stream,
          mr);
    } else {
      // day and above: truncate the local wall clock, then convert back to UTC
      // per row's zone (a spring-forward gap throws, matching toGMT).
      ColumnOrView truncatedLocal = truncateOnColumn(local->view(), stream, mr);
      truncatedUtcMillis = tswtzLocalToUtc(
          asView(truncatedLocal),
          zoneKey->view(),
          distinct,
          /*correctForward=*/false,
          stream,
          mr);
    }
    return tswtzPack(truncatedUtcMillis->view(), zoneKey->view(), stream, mr);
  }

  auto inputCol = asView(inputColumns[0]);
  const auto outputType = inputCol.type();

  // DATE (TIMESTAMP_DAYS) is zone-free, and under a UTC session no conversion
  // is needed; both use the raw truncation directly.
  const bool applyTimezone = outputType.id() != cudf::type_id::TIMESTAMP_DAYS &&
      context_.appliesSessionTimezone();
  if (!applyTimezone || unit_ == DateTimeUnit::kSecond ||
      unit_ == DateTimeUnit::kMinute) {
    // second/minute truncate the UTC epoch directly (every zone offset is a
    // whole number of minutes), matching CPU truncateTimestamp.
    return truncateOnColumn(inputCol, stream, mr);
  }

  const std::string& zone = context_.sessionTimezone;

  if (unit_ == DateTimeUnit::kHour) {
    // Compute the local-to-truncated-hour delta and subtract it from the UTC
    // instant. This reproduces CPU truncateTimestamp's DST-safe hour branch
    // (which avoids the ambiguous local->UTC roundtrip) and handles
    // fractional-offset zones such as Asia/Kolkata (+05:30).
    auto local = toLocalTimestamp(inputCol, zone, stream, mr);
    auto flooredLocal = cudf::datetime::floor_datetimes(
        local->view(), cudf::datetime::rounding_frequency::HOUR, stream, mr);
    auto delta = cudf::binary_operation(
        local->view(),
        flooredLocal->view(),
        cudf::binary_operator::SUB,
        durationTypeForTimestamp(outputType),
        stream,
        mr);
    return cudf::binary_operation(
        inputCol,
        delta->view(),
        cudf::binary_operator::SUB,
        outputType,
        stream,
        mr);
  }

  // day and above: truncate on the local wall clock, then convert back to UTC.
  // Matches CPU truncateTimestamp (truncate local, then toGMT); a local time in
  // a spring-forward gap raises in toUtcTimestamp, which is the correct parity.
  auto local = toLocalTimestamp(inputCol, zone, stream, mr);
  ColumnOrView truncatedLocal = truncateOnColumn(local->view(), stream, mr);
  return toUtcTimestamp(asView(truncatedLocal), zone, stream, mr);
}

} // namespace facebook::velox::cudf_velox
