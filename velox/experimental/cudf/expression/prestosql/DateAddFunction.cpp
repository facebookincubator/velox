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
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/functions/prestosql/DateTimeFunctions.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/unary.hpp>

#include <limits>
#include <optional>

namespace facebook::velox::cudf_velox::prestosql {

using functions::DateTimeUnit;

namespace {

// Day and week units add a duration_D to the date directly. Month, quarter,
// and year units must go through add_calendrical_months because calendar
// months are not a fixed number of days.
bool isDayBasedUnit(DateTimeUnit unit) {
  return unit == DateTimeUnit::kDay || unit == DateTimeUnit::kWeek;
}

// Multiplier from the user-supplied value to the underlying duration_D or
// month count. Sub-day units are rejected upstream by getDateUnit so they
// are unreachable here.
int32_t unitScale(DateTimeUnit unit) {
  switch (unit) {
    case DateTimeUnit::kDay:
    case DateTimeUnit::kMonth:
      return 1;
    case DateTimeUnit::kWeek:
      return 7;
    case DateTimeUnit::kQuarter:
      return 3;
    case DateTimeUnit::kYear:
      return 12;
    default:
      VELOX_UNREACHABLE();
  }
}

// Multiplies value by scale and returns the result as int32_t, throwing if
// either value or value*scale overflow int32_t.
int32_t checkedScaleValue(int64_t value, int32_t scale) {
  functions::checkValueInInt32Range(value);
  const auto scaledValue = value * scale;
  functions::checkValueInInt32Range(scaledValue);
  return static_cast<int32_t>(scaledValue);
}

// Throws if any non-null entry of valueCol would overflow int32_t when
// multiplied by scale. The bound is computed in int64_t, so the predicate
// itself cannot overflow.
void checkScaledValueRange(
    cudf::column_view valueCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  constexpr auto kMin = std::numeric_limits<int32_t>::min();
  constexpr auto kMax = std::numeric_limits<int32_t>::max();
  const auto minValue = static_cast<int64_t>(kMin / scale);
  const auto maxValue = static_cast<int64_t>(kMax / scale);
  const auto boolType = cudf::data_type(cudf::type_id::BOOL8);

  cudf::numeric_scalar<int64_t> minScalar(minValue, true, stream, mr);
  auto geMin = cudf::binary_operation(
      valueCol,
      minScalar,
      cudf::binary_operator::GREATER_EQUAL,
      boolType,
      stream,
      mr);

  cudf::numeric_scalar<int64_t> maxScalar(maxValue, true, stream, mr);
  auto leMax = cudf::binary_operation(
      valueCol,
      maxScalar,
      cudf::binary_operator::LESS_EQUAL,
      boolType,
      stream,
      mr);

  auto inRange = cudf::binary_operation(
      geMin->view(),
      leMax->view(),
      cudf::binary_operator::LOGICAL_AND,
      boolType,
      stream,
      mr);
  checkAllTrue(inRange->view(), "date_add value is out of range", stream, mr);
}

// Casts an int64 column to int32 and multiplies by scale, asserting the
// scaled values fit int32 first. The range check runs on the int64 input
// before the cast, so the cast itself cannot truncate.
std::unique_ptr<cudf::column> scaleToInt32(
    cudf::column_view valueCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  checkScaledValueRange(valueCol, scale, stream, mr);

  auto int32Type = cudf::data_type(cudf::type_id::INT32);
  auto int32Value = cudf::cast(valueCol, int32Type, stream, mr);
  if (scale == 1) {
    return int32Value;
  }

  cudf::numeric_scalar<int32_t> scaleScalar(scale, true, stream, mr);
  return cudf::binary_operation(
      int32Value->view(),
      scaleScalar,
      cudf::binary_operator::MUL,
      int32Type,
      stream,
      mr);
}

} // namespace

bool DateAddFunction::canEvaluate(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 3 || !expr->type()->isDate() ||
      !expr->inputs()[2]->type()->isDate()) {
    return false;
  }

  auto valueExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[1]);
  auto dateExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[2]);
  if (valueExpr && dateExpr) {
    return false;
  }

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  if (!unitString.has_value()) {
    return false;
  }

  return functions::getDateUnit(*unitString, false).has_value();
}

DateAddFunction::DateAddFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::ConstantExpr;
  VELOX_CHECK(
      canEvaluate(expr),
      "date_add expression cannot be evaluated by prestosql::DateAddFunction");

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  unit_ = *functions::getDateUnit(*unitString, true);

  auto valueExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
  valueIsLiteral_ = valueExpr != nullptr;
  dateIsLiteral_ =
      std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]) != nullptr;

  if (valueIsLiteral_) {
    literalValueIsValid_ = !valueExpr->value()->isNullAt(0);
    if (literalValueIsValid_) {
      literalValue_ =
          valueExpr->value()->as<ConstantVector<int64_t>>()->value();
    }
  }
  if (dateIsLiteral_) {
    literalDate_ = makeScalarFromConstantExpr(expr->inputs()[2]);
  }
}

ColumnOrView DateAddFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  // Walk the non-literal inputs in argument order. Constants were captured at
  // construction time and never appear in inputColumns, so the first slot
  // holds value (if value is a column), and the next slot holds date (if
  // date is a column).
  size_t idx = 0;

  std::optional<cudf::column_view> valueCol;
  if (!valueIsLiteral_) {
    valueCol = asView(inputColumns[idx++]);
  }

  std::unique_ptr<cudf::column> literalDateColumn;
  cudf::column_view dateCol;
  if (!dateIsLiteral_) {
    dateCol = asView(inputColumns[idx++]);
  } else {
    // Expand the literal date scalar to a column matching the value column's
    // size. literalDateColumn is kept alive for the duration of this call so
    // the view stays valid.
    VELOX_CHECK_NOT_NULL(literalDate_);
    VELOX_CHECK(
        valueCol.has_value(),
        "date_add with only literal inputs is not supported");
    literalDateColumn = cudf::make_column_from_scalar(
        *literalDate_, valueCol->size(), stream, mr);
    dateCol = literalDateColumn->view();
  }

  return isDayBasedUnit(unit_) ? evalDayBased(dateCol, valueCol, stream, mr)
                               : evalMonthBased(dateCol, valueCol, stream, mr);
}

ColumnOrView DateAddFunction::evalDayBased(
    cudf::column_view dateCol,
    std::optional<cudf::column_view> valueCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto outType = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
  const auto scale = unitScale(unit_);

  if (!valueCol.has_value()) {
    cudf::duration_scalar<cudf::duration_D> days(
        checkedScaleValue(literalValue_, scale),
        literalValueIsValid_,
        stream,
        mr);
    return cudf::binary_operation(
        dateCol, days, cudf::binary_operator::ADD, outType, stream, mr);
  }

  auto daysInt = scaleToInt32(*valueCol, scale, stream, mr);
  auto days = cudf::cast(
      daysInt->view(),
      cudf::data_type(cudf::type_id::DURATION_DAYS),
      stream,
      mr);
  return cudf::binary_operation(
      dateCol, days->view(), cudf::binary_operator::ADD, outType, stream, mr);
}

ColumnOrView DateAddFunction::evalMonthBased(
    cudf::column_view dateCol,
    std::optional<cudf::column_view> valueCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto scale = unitScale(unit_);

  if (!valueCol.has_value()) {
    cudf::numeric_scalar<int32_t> months(
        checkedScaleValue(literalValue_, scale),
        literalValueIsValid_,
        stream,
        mr);
    return cudf::datetime::add_calendrical_months(dateCol, months, stream, mr);
  }

  auto months = scaleToInt32(*valueCol, scale, stream, mr);
  return cudf::datetime::add_calendrical_months(
      dateCol, months->view(), stream, mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
