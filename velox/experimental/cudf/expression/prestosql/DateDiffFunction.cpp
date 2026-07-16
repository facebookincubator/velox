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
#include "velox/experimental/cudf/expression/prestosql/DateDiffFunction.h"

#include "velox/expression/ConstantExpr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace facebook::velox::cudf_velox::prestosql {

DateDiffFunction::DateDiffFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 3, "date_diff expects exactly 3 inputs");

  auto unitExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]);
  VELOX_CHECK_NOT_NULL(unitExpr, "date_diff unit must be a constant");
  unit_ = unitExpr->value()->toString(0);
  std::transform(
      unit_.begin(), unit_.end(), unit_.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

  isDate_ = expr->inputs()[1]->type()->isDate();

  static const std::unordered_set<std::string> kDateUnits = {
      "day", "week", "month", "quarter", "year"};
  static const std::unordered_set<std::string> kTimestampUnits = {
      "millisecond",
      "second",
      "minute",
      "hour",
      "day",
      "week",
      "month",
      "quarter",
      "year"};
  const auto& supportedUnits = isDate_ ? kDateUnits : kTimestampUnits;
  VELOX_USER_CHECK(
      supportedUnits.find(unit_) != supportedUnits.end(),
      "Unsupported date_diff unit for {}: {}",
      isDate_ ? "DATE" : "TIMESTAMP",
      unit_);

  // Either date argument may be a constant (e.g. DATE '2025-03-01' or
  // CURRENT_DATE). Literals are excluded from inputColumns by the framework,
  // so we capture them here as cuDF scalars and pass them directly to
  // cudf::binary_operation's scalar overloads to avoid materializing
  // full columns on every eval() call.
  if (auto c = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
          expr->inputs()[1])) {
    leftScalar_ = makeScalarFromConstantExpr(c);
    leftIsConst_ = true;
  }
  if (auto c = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
          expr->inputs()[2])) {
    rightScalar_ = makeScalarFromConstantExpr(c);
    rightIsConst_ = true;
  }
}

ColumnOrView DateDiffFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  // Resolve the two date/timestamp operands. Constants were captured at
  // construction as scalars; column refs arrive via inputColumns in
  // left-to-right order (skipping literals).
  size_t colIdx = 0;
  Operand left, right;

  if (leftIsConst_) {
    left.sc = leftScalar_.get();
  } else {
    left.col = asView(inputColumns[colIdx++]);
  }

  if (rightIsConst_) {
    right.sc = rightScalar_.get();
  } else {
    right.col = asView(inputColumns[colIdx++]);
  }

  if (unit_ == "day") {
    return diffBySubtraction(left, right, 1, stream, mr);
  } else if (unit_ == "week") {
    return diffBySubtraction(left, right, 7, stream, mr);
  } else if (unit_ == "month") {
    return diffByComponent(left, right, /*isYear=*/false, stream, mr);
  } else if (unit_ == "quarter") {
    auto months = diffByComponent(left, right, /*isYear=*/false, stream, mr);
    auto monthsView = asView(months);
    auto three = cudf::numeric_scalar<int64_t>(3, true, stream, mr);
    return cudf::binary_operation(
        monthsView,
        three,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
  } else if (unit_ == "year") {
    return diffByComponent(left, right, /*isYear=*/true, stream, mr);
  } else if (!isDate_) {
    static constexpr int64_t kUsPerMs = 1000LL;
    static constexpr int64_t kUsPerSecond = 1000LL * 1000;
    static constexpr int64_t kUsPerMinute = 60LL * kUsPerSecond;
    static constexpr int64_t kUsPerHour = 60LL * kUsPerMinute;
    if (unit_ == "second") {
      return diffTimestamp(left, right, kUsPerSecond, stream, mr);
    } else if (unit_ == "millisecond") {
      return diffTimestamp(left, right, kUsPerMs, stream, mr);
    } else if (unit_ == "minute") {
      return diffTimestamp(left, right, kUsPerMinute, stream, mr);
    } else if (unit_ == "hour") {
      return diffTimestamp(left, right, kUsPerHour, stream, mr);
    }
  }
  VELOX_USER_FAIL("Unsupported date_diff unit: {}", unit_);
}

std::unique_ptr<cudf::column> DateDiffFunction::binaryOp(
    const Operand& lhs,
    const Operand& rhs,
    cudf::binary_operator op,
    cudf::data_type out,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (lhs.col && rhs.col) {
    return cudf::binary_operation(*lhs.col, *rhs.col, op, out, stream, mr);
  } else if (lhs.sc && rhs.col) {
    return cudf::binary_operation(*lhs.sc, *rhs.col, op, out, stream, mr);
  } else if (lhs.col && rhs.sc) {
    return cudf::binary_operation(*lhs.col, *rhs.sc, op, out, stream, mr);
  }
  VELOX_FAIL("Both date_diff operands are scalar");
}

cudf::size_type DateDiffFunction::getSize(const Operand& a, const Operand& b) {
  if (a.col) {
    return a.col->size();
  }
  VELOX_CHECK(b.col.has_value(), "At least one operand must be a column");
  return b.col->size();
}

cudf::column_view DateDiffFunction::ensureColumn(
    const Operand& op,
    cudf::size_type size,
    std::unique_ptr<cudf::column>& owned,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (op.col) {
    return *op.col;
  }
  owned = cudf::make_column_from_scalar(*op.sc, size, stream, mr);
  return owned->view();
}

ColumnOrView DateDiffFunction::diffBySubtraction(
    const Operand& left,
    const Operand& right,
    int64_t divisor,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (isDate_) {
    // DATE columns are TIMESTAMP_DAYS. cuDF can't cast timestamps to int
    // directly - subtract to get DURATION_DAYS, then cast duration to INT.
    auto duration = binaryOp(
        right,
        left,
        cudf::binary_operator::SUB,
        cudf::data_type(cudf::type_id::DURATION_DAYS),
        stream,
        mr);
    auto diff = cudf::cast(
        duration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
    if (divisor == 1) {
      return diff;
    }
    auto div = cudf::numeric_scalar<int64_t>(divisor, true, stream, mr);
    // Use C-style truncating DIV (not FLOOR_DIV) to match Velox CPU's
    // sign-symmetric behavior: the unsigned magnitude is divided and the
    // sign is re-applied, which is equivalent to truncation toward zero
    // rather than flooring toward -infinity.
    return cudf::binary_operation(
        diff->view(),
        div,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
  }
  static constexpr int64_t kUsPerDay = 86400LL * 1000000LL;
  return diffTimestamp(left, right, divisor * kUsPerDay, stream, mr);
}

ColumnOrView DateDiffFunction::diffTimestamp(
    const Operand& left,
    const Operand& right,
    int64_t usPerUnit,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  // Determine the matching duration type for the timestamp resolution.
  // Velox TIMESTAMP maps to TIMESTAMP_NANOSECONDS in cudf.
  auto durationTypeId = cudf::type_id::DURATION_MICROSECONDS;
  int64_t scaleFactor = usPerUnit;
  if (left.col.has_value() &&
      left.col->type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS) {
    durationTypeId = cudf::type_id::DURATION_NANOSECONDS;
    scaleFactor = usPerUnit * 1000;
  } else if (
      right.col.has_value() &&
      right.col->type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS) {
    durationTypeId = cudf::type_id::DURATION_NANOSECONDS;
    scaleFactor = usPerUnit * 1000;
  }
  auto duration = binaryOp(
      right,
      left,
      cudf::binary_operator::SUB,
      cudf::data_type(durationTypeId),
      stream,
      mr);
  auto diff = cudf::cast(
      duration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
  if (scaleFactor > 1) {
    auto div = cudf::numeric_scalar<int64_t>(scaleFactor, true, stream, mr);
    // See diffBySubtraction() for why DIV (not FLOOR_DIV) is required.
    return cudf::binary_operation(
        diff->view(),
        div,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
  }
  return diff;
}

std::unique_ptr<cudf::column> DateDiffFunction::extractComponentAsInt64(
    cudf::column_view col,
    cudf::datetime::datetime_component component,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto extracted =
      cudf::datetime::extract_datetime_component(col, component, stream, mr);
  return cudf::cast(
      extracted->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
}

std::unique_ptr<cudf::column> DateDiffFunction::timeOfDayMicros(
    cudf::column_view ts,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const bool isNanos = ts.type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS;
  auto dayFloor = cudf::cast(
      ts, cudf::data_type(cudf::type_id::TIMESTAMP_DAYS), stream, mr);
  auto dayFloorSameRes = cudf::cast(dayFloor->view(), ts.type(), stream, mr);
  auto durationType = cudf::data_type(
      isNanos ? cudf::type_id::DURATION_NANOSECONDS
              : cudf::type_id::DURATION_MICROSECONDS);
  auto duration = cudf::binary_operation(
      ts,
      dayFloorSameRes->view(),
      cudf::binary_operator::SUB,
      durationType,
      stream,
      mr);
  auto asInt64 = cudf::cast(
      duration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
  if (!isNanos) {
    return asInt64;
  }
  auto thousand = cudf::numeric_scalar<int64_t>(1000, true, stream, mr);
  return cudf::binary_operation(
      asInt64->view(),
      thousand,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
}

ColumnOrView DateDiffFunction::diffByComponent(
    const Operand& left,
    const Operand& right,
    bool isYear,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  using cudf::datetime::datetime_component;
  auto n = getSize(left, right);
  std::unique_ptr<cudf::column> leftOwned, rightOwned;
  auto leftCol = ensureColumn(left, n, leftOwned, stream, mr);
  auto rightCol = ensureColumn(right, n, rightOwned, stream, mr);

  auto int64Type = cudf::data_type(cudf::type_id::INT64);
  auto bool8Type = cudf::data_type(cudf::type_id::BOOL8);

  // Sort into (lo, hi) so the day/time correction below always compares
  // "from" (lo) against "to" (hi), matching Velox CPU's min/max swap.
  auto leftLessEqual = cudf::binary_operation(
      leftCol,
      rightCol,
      cudf::binary_operator::LESS_EQUAL,
      bool8Type,
      stream,
      mr);
  auto loCol =
      cudf::copy_if_else(leftCol, rightCol, leftLessEqual->view(), stream, mr);
  auto hiCol =
      cudf::copy_if_else(rightCol, leftCol, leftLessEqual->view(), stream, mr);
  // sign = (left <= right) ? +1 : -1. When left == right the magnitude
  // below is always 0, so the sign choice for ties is immaterial.
  auto plusOne = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
  auto minusOne = cudf::numeric_scalar<int64_t>(-1, true, stream, mr);
  auto sign =
      cudf::copy_if_else(plusOne, minusOne, leftLessEqual->view(), stream, mr);

  auto y1 = extractComponentAsInt64(
      loCol->view(), datetime_component::YEAR, stream, mr);
  auto y2 = extractComponentAsInt64(
      hiCol->view(), datetime_component::YEAR, stream, mr);
  auto m1 = extractComponentAsInt64(
      loCol->view(), datetime_component::MONTH, stream, mr);
  auto m2 = extractComponentAsInt64(
      hiCol->view(), datetime_component::MONTH, stream, mr);
  auto d1 = extractComponentAsInt64(
      loCol->view(), datetime_component::DAY, stream, mr);
  auto d2 = extractComponentAsInt64(
      hiCol->view(), datetime_component::DAY, stream, mr);

  // toLastDayOfMonth: day-of-month of the last day of the "to" (hi) month,
  // for the respectLastDay exception.
  auto hiLastDay = cudf::datetime::last_day_of_month(hiCol->view(), stream, mr);
  auto toLastDayOfMonth = extractComponentAsInt64(
      hiLastDay->view(), datetime_component::DAY, stream, mr);

  // dayDecrement = (d1 > d2) && (d2 != toLastDayOfMonth)
  auto dayGreater = cudf::binary_operation(
      d1->view(),
      d2->view(),
      cudf::binary_operator::GREATER,
      bool8Type,
      stream,
      mr);
  auto notLastDay = cudf::binary_operation(
      d2->view(),
      toLastDayOfMonth->view(),
      cudf::binary_operator::NOT_EQUAL,
      bool8Type,
      stream,
      mr);
  auto dayDecrement = cudf::binary_operation(
      dayGreater->view(),
      notLastDay->view(),
      cudf::binary_operator::LOGICAL_AND,
      bool8Type,
      stream,
      mr);

  // timeDecrement = (d1 == d2) && (timeOfDay1 > timeOfDay2). DATE inputs
  // have no time component (always equal), so this term is always false.
  auto dayEqual = cudf::binary_operation(
      d1->view(),
      d2->view(),
      cudf::binary_operator::EQUAL,
      bool8Type,
      stream,
      mr);
  std::unique_ptr<cudf::column> timeDecrement;
  if (isDate_) {
    timeDecrement = cudf::make_column_from_scalar(
        cudf::numeric_scalar<bool>(false, true, stream, mr), n, stream, mr);
  } else {
    auto t1 = timeOfDayMicros(loCol->view(), stream, mr);
    auto t2 = timeOfDayMicros(hiCol->view(), stream, mr);
    auto timeGreater = cudf::binary_operation(
        t1->view(),
        t2->view(),
        cudf::binary_operator::GREATER,
        bool8Type,
        stream,
        mr);
    timeDecrement = cudf::binary_operation(
        dayEqual->view(),
        timeGreater->view(),
        cudf::binary_operator::LOGICAL_AND,
        bool8Type,
        stream,
        mr);
  }

  auto decrementBool = cudf::binary_operation(
      dayDecrement->view(),
      timeDecrement->view(),
      cudf::binary_operator::LOGICAL_OR,
      bool8Type,
      stream,
      mr);

  if (isYear) {
    // YEAR additionally decrements unconditionally when fromMonth >
    // toMonth, and only applies the day/time correction when
    // fromMonth == toMonth.
    auto monthGreater = cudf::binary_operation(
        m1->view(),
        m2->view(),
        cudf::binary_operator::GREATER,
        bool8Type,
        stream,
        mr);
    auto monthEqual = cudf::binary_operation(
        m1->view(),
        m2->view(),
        cudf::binary_operator::EQUAL,
        bool8Type,
        stream,
        mr);
    auto monthEqualDecrement = cudf::binary_operation(
        monthEqual->view(),
        decrementBool->view(),
        cudf::binary_operator::LOGICAL_AND,
        bool8Type,
        stream,
        mr);
    auto finalDecrementBool = cudf::binary_operation(
        monthGreater->view(),
        monthEqualDecrement->view(),
        cudf::binary_operator::LOGICAL_OR,
        bool8Type,
        stream,
        mr);
    auto decrement =
        cudf::cast(finalDecrementBool->view(), int64Type, stream, mr);
    auto yearDiff = cudf::binary_operation(
        y2->view(),
        y1->view(),
        cudf::binary_operator::SUB,
        int64Type,
        stream,
        mr);
    auto magnitude = cudf::binary_operation(
        yearDiff->view(),
        decrement->view(),
        cudf::binary_operator::SUB,
        int64Type,
        stream,
        mr);
    return cudf::binary_operation(
        magnitude->view(),
        sign->view(),
        cudf::binary_operator::MUL,
        int64Type,
        stream,
        mr);
  }

  auto decrement = cudf::cast(decrementBool->view(), int64Type, stream, mr);
  // (y2 - y1) * 12 + (m2 - m1)
  auto yearDiff = cudf::binary_operation(
      y2->view(),
      y1->view(),
      cudf::binary_operator::SUB,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto twelve = cudf::numeric_scalar<int64_t>(12, true, stream, mr);
  auto yearMonths = cudf::binary_operation(
      yearDiff->view(),
      twelve,
      cudf::binary_operator::MUL,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto monthDiff = cudf::binary_operation(
      m2->view(),
      m1->view(),
      cudf::binary_operator::SUB,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto naiveDiff = cudf::binary_operation(
      yearMonths->view(),
      monthDiff->view(),
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto magnitude = cudf::binary_operation(
      naiveDiff->view(),
      decrement->view(),
      cudf::binary_operator::SUB,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  return cudf::binary_operation(
      magnitude->view(),
      sign->view(),
      cudf::binary_operator::MUL,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
