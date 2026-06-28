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
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/functions/lib/TimeUtils.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

using functions::DateTimeUnit;

bool DateTruncFunction::canEvaluate(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 2) {
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
  if (!isTimestamp && !isDate) {
    return false;
  }
  if (*unit == DateTimeUnit::kSecond || *unit == DateTimeUnit::kMinute ||
      *unit == DateTimeUnit::kHour) {
    return isTimestamp;
  }
  if (*unit == DateTimeUnit::kDay || *unit == DateTimeUnit::kWeek ||
      *unit == DateTimeUnit::kMonth || *unit == DateTimeUnit::kQuarter ||
      *unit == DateTimeUnit::kYear) {
    return true;
  }
  return false;
}

DateTruncFunction::DateTruncFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_trunc expects exactly 2 inputs");
  auto unitString = constantVarcharValue(expr->inputs()[0]);
  VELOX_CHECK(
      unitString.has_value(), "date_trunc unit must be a non-null constant");
  auto inputType = expr->inputs()[1]->type();
  isTimestamp_ = inputType->isTimestamp();
  isDate_ = inputType->isDate();
  VELOX_CHECK(
      isTimestamp_ || isDate_,
      "date_trunc only supports date or timestamp inputs");
  auto parsed = functions::fromDateTimeUnitString(*unitString, true);
  VELOX_CHECK(parsed.has_value(), "Invalid date_trunc unit: {}", *unitString);
  unit_ = *parsed;

  // Validate time-only units require timestamp input.
  if (unit_ == DateTimeUnit::kSecond || unit_ == DateTimeUnit::kMinute ||
      unit_ == DateTimeUnit::kHour) {
    VELOX_CHECK(
        isTimestamp_, "date_trunc {} requires timestamp input", *unitString);
  }
}

ColumnOrView DateTruncFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto inputCol = asView(inputColumns[0]);
  auto outputType = inputCol.type();
  auto dayType = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
  auto intType = cudf::data_type(cudf::type_id::INT32);
  auto durationDayType = cudf::data_type(cudf::type_id::DURATION_DAYS);

  auto castToDay = [&](cudf::column_view col) {
    return cudf::cast(col, dayType, stream, mr);
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

  auto makeScalar = [&](int32_t value) {
    return cudf::numeric_scalar<int32_t>(value, true, stream, mr);
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
    case DateTimeUnit::kDay: {
      auto dayCol = castToDay(inputCol);
      return castDaysToOutput(std::move(dayCol));
    }
    case DateTimeUnit::kWeek: {
      auto dayCol = castToDay(inputCol);
      auto dowCol = cudf::datetime::extract_datetime_component(
          dayCol->view(),
          cudf::datetime::datetime_component::WEEKDAY,
          stream,
          mr);
      auto dowInt = castToInt32(dowCol->view());
      auto oneScalar = makeScalar(1);
      auto offset = cudf::binary_operation(
          dowInt->view(),
          oneScalar,
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
      auto dayCol = castToDay(inputCol);
      auto dayOfMonth = cudf::datetime::extract_datetime_component(
          dayCol->view(), cudf::datetime::datetime_component::DAY, stream, mr);
      auto dayOfMonthInt = castToInt32(dayOfMonth->view());
      auto oneScalar = makeScalar(1);
      auto dayOffset = cudf::binary_operation(
          dayOfMonthInt->view(),
          oneScalar,
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
          oneScalar,
          cudf::binary_operator::SUB,
          intType,
          stream,
          mr);

      std::unique_ptr<cudf::column> monthsToSubtract;
      if (unit_ == DateTimeUnit::kYear) {
        monthsToSubtract = std::move(monthIndex);
      } else {
        auto threeScalar = makeScalar(3);
        auto quarterIndex = cudf::binary_operation(
            monthIndex->view(),
            threeScalar,
            cudf::binary_operator::FLOOR_DIV,
            intType,
            stream,
            mr);
        auto quarterStart = cudf::binary_operation(
            quarterIndex->view(),
            threeScalar,
            cudf::binary_operator::MUL,
            intType,
            stream,
            mr);
        monthsToSubtract = cudf::binary_operation(
            monthIndex->view(),
            quarterStart->view(),
            cudf::binary_operator::SUB,
            intType,
            stream,
            mr);
      }

      auto negOneScalar = makeScalar(-1);
      auto negMonths = cudf::binary_operation(
          monthsToSubtract->view(),
          negOneScalar,
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

} // namespace facebook::velox::cudf_velox
