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
#include "velox/experimental/cudf/expression/TimezoneConversion.h"
#include "velox/experimental/cudf/expression/prestosql/DateAddFunction.h"

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
// month count. Callers invoke this only for calendar units.
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

// Matches the CPU date_add policy: validate the raw value before unit scaling.
void checkDateAddValueInInt32Range(int64_t value) {
  VELOX_USER_CHECK(
      value == static_cast<int32_t>(value), "date_add value is out of range");
}

int32_t checkedScaleValue(int64_t value, int32_t scale) {
  checkDateAddValueInInt32Range(value);
  return static_cast<int32_t>(value * scale);
}

// Throws if a non-null value outside int32_t corresponds to a non-null input.
// Unit scaling happens after the cast to match the CPU date_add policy.
void checkValueRange(
    cudf::column_view valueCol,
    cudf::column_view inputCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  constexpr auto kMin = std::numeric_limits<int32_t>::min();
  constexpr auto kMax = std::numeric_limits<int32_t>::max();
  const auto boolType = cudf::data_type(cudf::type_id::BOOL8);

  cudf::numeric_scalar<int64_t> minScalar(kMin, true, stream, mr);
  auto geMin = cudf::binary_operation(
      valueCol,
      minScalar,
      cudf::binary_operator::GREATER_EQUAL,
      boolType,
      stream,
      mr);

  cudf::numeric_scalar<int64_t> maxScalar(kMax, true, stream, mr);
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
  auto inputIsNull = cudf::is_null(inputCol, stream, mr);
  auto validOrInputNull = cudf::binary_operation(
      inRange->view(),
      inputIsNull->view(),
      cudf::binary_operator::BITWISE_OR,
      boolType,
      stream,
      mr);
  checkAllTrue(
      validOrInputNull->view(), "date_add value is out of range", stream, mr);
}

// Casts an int64 column to int32 after validating the raw values, then applies
// unit scaling in int32 to match the CPU date_add implementation.
std::unique_ptr<cudf::column> scaleToInt32(
    cudf::column_view valueCol,
    cudf::column_view inputCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  checkValueRange(valueCol, inputCol, stream, mr);

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
          "date_add requires a timestamp column, got cudf type id: {}",
          static_cast<int32_t>(timestampType.id()));
  }
}

bool isMonthBasedUnit(DateTimeUnit unit) {
  return unit == DateTimeUnit::kMonth || unit == DateTimeUnit::kQuarter ||
      unit == DateTimeUnit::kYear;
}

// Adds to a timestamp without changing its resolution. All values stay on the
// device; no microsecond- or nanosecond-specific representation is assumed.
std::unique_ptr<cudf::column> addUnitToTimestamp(
    cudf::column_view timestamp,
    std::optional<cudf::column_view> valueCol,
    int64_t literalValue,
    bool literalValid,
    DateTimeUnit unit,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (isMonthBasedUnit(unit)) {
    const int32_t scale = unitScale(unit);
    if (valueCol.has_value()) {
      auto months = scaleToInt32(*valueCol, timestamp, scale, stream, mr);
      return cudf::datetime::add_calendrical_months(
          timestamp, months->view(), stream, mr);
    }
    cudf::numeric_scalar<int32_t> months(
        checkedScaleValue(literalValue, scale), literalValid, stream, mr);
    return cudf::datetime::add_calendrical_months(
        timestamp, months, stream, mr);
  }

  const auto nativeDuration = durationTypeForTimestamp(timestamp.type());
  std::unique_ptr<cudf::column> duration;
  if (!functions::isTimeUnit(unit)) {
    const int32_t scale = unitScale(unit);
    std::unique_ptr<cudf::column> daysInt;
    if (valueCol.has_value()) {
      daysInt = scaleToInt32(*valueCol, timestamp, scale, stream, mr);
    } else {
      cudf::numeric_scalar<int32_t> days(
          checkedScaleValue(literalValue, scale), literalValid, stream, mr);
      daysInt =
          cudf::make_column_from_scalar(days, timestamp.size(), stream, mr);
    }
    auto days = cudf::cast(
        daysInt->view(),
        cudf::data_type(cudf::type_id::DURATION_DAYS),
        stream,
        mr);
    duration = cudf::cast(days->view(), nativeDuration, stream, mr);
  } else {
    const bool milliseconds = unit == DateTimeUnit::kMillisecond;
    const auto sourceDuration = cudf::data_type(
        milliseconds ? cudf::type_id::DURATION_MILLISECONDS
                     : cudf::type_id::DURATION_SECONDS);
    const int64_t multiplier = unit == DateTimeUnit::kHour
        ? 3600
        : (unit == DateTimeUnit::kMinute ? 60 : 1);

    std::unique_ptr<cudf::column> count;
    if (valueCol.has_value()) {
      checkValueRange(*valueCol, timestamp, stream, mr);
      if (multiplier == 1) {
        count = cudf::cast(
            *valueCol, cudf::data_type(cudf::type_id::INT64), stream, mr);
      } else {
        cudf::numeric_scalar<int64_t> scale(multiplier, true, stream, mr);
        count = cudf::binary_operation(
            *valueCol,
            scale,
            cudf::binary_operator::MUL,
            cudf::data_type(cudf::type_id::INT64),
            stream,
            mr);
      }
    } else {
      checkDateAddValueInInt32Range(literalValue);
      cudf::numeric_scalar<int64_t> countScalar(
          literalValue * multiplier, literalValid, stream, mr);
      count = cudf::make_column_from_scalar(
          countScalar, timestamp.size(), stream, mr);
    }
    auto source = cudf::cast(count->view(), sourceDuration, stream, mr);
    duration = cudf::cast(source->view(), nativeDuration, stream, mr);
  }

  return cudf::binary_operation(
      timestamp,
      duration->view(),
      cudf::binary_operator::ADD,
      timestamp.type(),
      stream,
      mr);
}

} // namespace

bool DateAddFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 3 || !expr->type()->isDate() ||
      !expr->inputs()[2]->type()->isDate()) {
    return false;
  }

  const bool valueIsConstant = expr->inputs()[1]->isConstantKind();
  const bool dateIsConstant = expr->inputs()[2]->isConstantKind();
  if (valueIsConstant && dateIsConstant) {
    return false;
  }

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  if (!unitString.has_value()) {
    return false;
  }

  return functions::getDateUnit(*unitString, false).has_value();
}

DateAddFunction::DateAddFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK(
      canEvaluate(expr),
      "date_add expression cannot be evaluated by prestosql::DateAddFunction");

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  unit_ = *functions::getDateUnit(*unitString, true);

  valueIsLiteral_ = expr->inputs()[1]->isConstantKind();
  dateIsLiteral_ = expr->inputs()[2]->isConstantKind();

  if (valueIsLiteral_) {
    const auto* valueExpr =
        expr->inputs()[1]->asUnchecked<core::ConstantTypedExpr>();
    const auto valueVector = valueExpr->hasValueVector()
        ? valueExpr->valueVector()
        : valueExpr->toConstantVector(pool);
    literalValueIsValid_ = !valueVector->isNullAt(0);
    if (literalValueIsValid_) {
      literalValue_ = valueVector->as<ConstantVector<int64_t>>()->value();
    }
  }
  if (dateIsLiteral_) {
    literalDate_ = makeScalarFromConstantExpr(expr->inputs()[2], pool);
  }
}

ColumnOrView DateAddFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
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
    cuda::stream_ref stream,
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

  auto daysInt = scaleToInt32(*valueCol, dateCol, scale, stream, mr);
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
    cuda::stream_ref stream,
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

  auto months = scaleToInt32(*valueCol, dateCol, scale, stream, mr);
  return cudf::datetime::add_calendrical_months(
      dateCol, months->view(), stream, mr);
}

bool DateAddTimestampFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 3 || !expr->type()->isTimestamp() ||
      !expr->inputs()[2]->type()->isTimestamp()) {
    return false;
  }

  const bool valueIsConstant = expr->inputs()[1]->isConstantKind();
  const bool timestampIsConstant = expr->inputs()[2]->isConstantKind();
  if (valueIsConstant && timestampIsConstant) {
    return false;
  }

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  if (!unitString.has_value()) {
    return false;
  }
  return functions::fromDateTimeUnitString(*unitString, false).has_value();
}

DateAddTimestampFunction::DateAddTimestampFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK(
      canEvaluate(expr),
      "date_add expression cannot be evaluated by "
      "prestosql::DateAddTimestampFunction");

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  unit_ = *functions::fromDateTimeUnitString(*unitString, true);

  valueIsLiteral_ = expr->inputs()[1]->isConstantKind();
  timestampIsLiteral_ = expr->inputs()[2]->isConstantKind();

  if (valueIsLiteral_) {
    const auto* valueExpr =
        expr->inputs()[1]->asUnchecked<core::ConstantTypedExpr>();
    const auto valueVector = valueExpr->hasValueVector()
        ? valueExpr->valueVector()
        : valueExpr->toConstantVector(pool);
    literalValueIsValid_ = !valueVector->isNullAt(0);
    if (literalValueIsValid_) {
      literalValue_ = valueVector->as<ConstantVector<int64_t>>()->value();
    }
  }
  if (timestampIsLiteral_) {
    literalTimestamp_ = makeScalarFromConstantExpr(expr->inputs()[2], pool);
  }
}

ColumnOrView DateAddTimestampFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  size_t idx = 0;

  std::optional<cudf::column_view> valueCol;
  if (!valueIsLiteral_) {
    valueCol = asView(inputColumns[idx++]);
  }

  std::unique_ptr<cudf::column> literalTimestampColumn;
  cudf::column_view timestampCol;
  if (!timestampIsLiteral_) {
    timestampCol = asView(inputColumns[idx++]);
  } else {
    VELOX_CHECK_NOT_NULL(literalTimestamp_);
    VELOX_CHECK(
        valueCol.has_value(),
        "date_add with only literal inputs is not supported");
    literalTimestampColumn = cudf::make_column_from_scalar(
        *literalTimestamp_, valueCol->size(), stream, mr);
    timestampCol = literalTimestampColumn->view();
  }

  // Presto treats sub-day units as fixed durations even when a session zone
  // applies. Calendar units are applied to the local wall clock; nonexistent
  // results move forward across the DST gap and overlaps pick the earliest
  // instant.
  if (!context_.appliesSessionTimezone() || functions::isTimeUnit(unit_)) {
    return addUnitToTimestamp(
        timestampCol,
        valueCol,
        literalValue_,
        literalValueIsValid_,
        unit_,
        stream,
        mr);
  }

  const auto& zone = context_.sessionTimezone;
  auto local = toLocalTimestamp(timestampCol, zone, stream, mr);
  auto added = addUnitToTimestamp(
      local->view(),
      valueCol,
      literalValue_,
      literalValueIsValid_,
      unit_,
      stream,
      mr);
  return toUtcTimestampCorrecting(added->view(), zone, stream, mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
