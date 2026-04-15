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
#include "velox/experimental/cudf/expression/DateTruncFunction.h"

#include "velox/expression/ConstantExpr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/unary.hpp>

#include <folly/String.h>

namespace facebook::velox::cudf_velox {

std::string DateTruncFunction::normalizeDateTruncUnit(std::string unit) {
  if (unit.size() >= 2 && unit.front() == '\'' && unit.back() == '\'') {
    unit = unit.substr(1, unit.size() - 2);
  }
  folly::toLowerAscii(unit);
  return unit;
}

DateTruncFunction::DateTruncUnit DateTruncFunction::parseDateTruncUnit(
    const std::string& unit,
    bool isTimestamp,
    bool isDate) {
  if (unit == "second") {
    VELOX_CHECK(isTimestamp, "date_trunc second requires timestamp input");
    return DateTruncUnit::kSecond;
  }
  if (unit == "minute") {
    VELOX_CHECK(isTimestamp, "date_trunc minute requires timestamp input");
    return DateTruncUnit::kMinute;
  }
  if (unit == "hour") {
    VELOX_CHECK(isTimestamp, "date_trunc hour requires timestamp input");
    return DateTruncUnit::kHour;
  }
  if (unit == "day") {
    return DateTruncUnit::kDay;
  }
  if (unit == "week") {
    return DateTruncUnit::kWeek;
  }
  if (unit == "month") {
    return DateTruncUnit::kMonth;
  }
  if (unit == "quarter") {
    return DateTruncUnit::kQuarter;
  }
  if (unit == "year") {
    return DateTruncUnit::kYear;
  }
  VELOX_FAIL(
      "date_trunc does not support unit '{}' for {} input",
      unit,
      isTimestamp ? "timestamp" : (isDate ? "date" : "unknown"));
}

bool DateTruncFunction::canEvaluate(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 2) {
    return false;
  }
  auto unitExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]);
  if (!unitExpr || unitExpr->value()->isNullAt(0)) {
    return false;
  }
  auto unit = normalizeDateTruncUnit(unitExpr->value()->toString(0));
  const auto& inputType = expr->inputs()[1]->type();
  const bool isTimestamp = inputType->isTimestamp();
  const bool isDate = inputType->isDate();
  if (!isTimestamp && !isDate) {
    return false;
  }
  if (unit == "second" || unit == "minute" || unit == "hour") {
    return isTimestamp;
  }
  if (unit == "day" || unit == "week" || unit == "month" || unit == "quarter" ||
      unit == "year") {
    return true;
  }
  return false;
}

DateTruncFunction::DateTruncFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::ConstantExpr;
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_trunc expects exactly 2 inputs");
  auto unitExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
  VELOX_CHECK_NOT_NULL(unitExpr, "date_trunc unit must be a constant");
  auto inputType = expr->inputs()[1]->type();
  isTimestamp_ = inputType->isTimestamp();
  isDate_ = inputType->isDate();
  VELOX_CHECK(
      isTimestamp_ || isDate_,
      "date_trunc only supports date or timestamp inputs");
  unit_ = parseDateTruncUnit(
      normalizeDateTruncUnit(unitExpr->value()->toString(0)),
      isTimestamp_,
      isDate_);
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
    case DateTruncUnit::kSecond:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::SECOND, stream, mr);
    case DateTruncUnit::kMinute:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::MINUTE, stream, mr);
    case DateTruncUnit::kHour:
      return cudf::datetime::floor_datetimes(
          inputCol, cudf::datetime::rounding_frequency::HOUR, stream, mr);
    case DateTruncUnit::kDay: {
      auto dayCol = castToDay(inputCol);
      return castDaysToOutput(std::move(dayCol));
    }
    case DateTruncUnit::kWeek: {
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
    case DateTruncUnit::kMonth:
    case DateTruncUnit::kQuarter:
    case DateTruncUnit::kYear: {
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

      if (unit_ == DateTruncUnit::kMonth) {
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
      if (unit_ == DateTruncUnit::kYear) {
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
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::cudf_velox
