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
#include "velox/experimental/cudf/expression/prestosql/DateDiffFunction.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace facebook::velox::cudf_velox::prestosql {

bool DateDiffFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() != 3) {
    return false;
  }

  // A null unit is admitted by the constructor's VELOX_CHECK_NOT_NULL as a
  // hard failure, rather than falling back to CPU's default-null result -
  // reject it here instead.
  auto unitString = constantVarcharValue(expr->inputs()[0]);
  if (!unitString.has_value()) {
    return false;
  }

  // eval() has no path for two constant date/timestamp operands (see
  // binaryOp's "Both date_diff operands are scalar" failure) - require at
  // least one to be a non-constant column.
  if (expr->inputs()[1]->isConstantKind() &&
      expr->inputs()[2]->isConstantKind()) {
    return false;
  }

  const bool isDate = expr->inputs()[1]->type()->isDate();
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
  std::string unit = unitString->str();
  std::transform(unit.begin(), unit.end(), unit.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  const auto& supportedUnits = isDate ? kDateUnits : kTimestampUnits;
  return supportedUnits.find(unit) != supportedUnits.end();
}

DateDiffFunction::DateDiffFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 3, "date_diff expects exactly 3 inputs");

  auto unitString = constantVarcharValue(expr->inputs()[0]);
  VELOX_CHECK(unitString.has_value(), "date_diff unit must be a constant");
  unit_ = unitString->str();
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

  // kDateUnits happens to be exactly the set of units whose result depends
  // on the calendar date/time-of-day of each operand (see the member
  // comment in the header for why hour/minute/second/millisecond don't),
  // so it doubles as the timezone-sensitivity check.
  isTimezoneSensitiveUnit_ = kDateUnits.find(unit_) != kDateUnits.end();

  // Either date argument may be a constant (e.g. DATE '2025-03-01' or
  // CURRENT_DATE). Literals are excluded from inputColumns by the framework,
  // so we capture them here as cuDF scalars and pass them directly to
  // cudf::binary_operation's scalar overloads to avoid materializing
  // full columns on every eval() call.
  if (expr->inputs()[1]->isConstantKind()) {
    leftScalar_ = makeScalarFromConstantExpr(expr->inputs()[1], pool);
    leftIsConst_ = true;
  }
  if (expr->inputs()[2]->isConstantKind()) {
    rightScalar_ = makeScalarFromConstantExpr(expr->inputs()[2], pool);
    rightIsConst_ = true;
  }

  // These scalars are used unconditionally by diffByComponent()/eval()
  // regardless of unit_ or input data; cache them once here instead of
  // reallocating on every eval() call (see DateTruncFunction for the same
  // pattern).
  auto stream = cudf::get_default_stream(cudf::allow_default_stream);
  auto mr = get_temp_mr();
  threeScalar_ =
      std::make_unique<cudf::numeric_scalar<int64_t>>(3, true, stream, mr);
  twelveScalar_ =
      std::make_unique<cudf::numeric_scalar<int64_t>>(12, true, stream, mr);
  plusOneScalar_ =
      std::make_unique<cudf::numeric_scalar<int64_t>>(1, true, stream, mr);
  minusOneScalar_ =
      std::make_unique<cudf::numeric_scalar<int64_t>>(-1, true, stream, mr);
  stream.synchronize();
}

ColumnOrView DateDiffFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cudf::size_type numRows,
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

  // day/week/month/quarter/year on TIMESTAMP need both operands converted to
  // session-local wall-clock time before diffing, matching Velox CPU's
  // diffTimestamp(unit, ts1, ts2, timeZone) (see the isTimezoneSensitiveUnit_
  // comment in the header for why). Materializing scalars into columns here
  // only happens in this (session-timezone-on) path; the common case below
  // is untouched. The shifted columns are owned by leftLocalOwned/
  // rightLocalOwned below so the views assigned into left/right stay valid
  // for the rest of this call.
  std::unique_ptr<cudf::column> leftLocalOwned, rightLocalOwned;
  if (!isDate_ && isTimezoneSensitiveUnit_ &&
      context_.appliesSessionTimezone()) {
    std::unique_ptr<cudf::column> leftMaterialized, rightMaterialized;
    auto leftCol = ensureColumn(left, numRows, leftMaterialized, stream, mr);
    auto rightCol = ensureColumn(right, numRows, rightMaterialized, stream, mr);
    leftLocalOwned =
        toLocalTimestamp(leftCol, context_.sessionTimezone, stream, mr);
    rightLocalOwned =
        toLocalTimestamp(rightCol, context_.sessionTimezone, stream, mr);
    left = Operand{leftLocalOwned->view(), nullptr};
    right = Operand{rightLocalOwned->view(), nullptr};
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
    return cudf::binary_operation(
        monthsView,
        *threeScalar_,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
  } else if (unit_ == "year") {
    return diffByComponent(left, right, /*isYear=*/true, stream, mr);
  } else if (!isDate_) {
    static constexpr int64_t kMsPerSecond = 1000LL;
    static constexpr int64_t kMsPerMinute = 60LL * kMsPerSecond;
    static constexpr int64_t kMsPerHour = 60LL * kMsPerMinute;
    if (unit_ == "second") {
      return diffTimestamp(left, right, kMsPerSecond, stream, mr);
    } else if (unit_ == "millisecond") {
      return diffTimestamp(left, right, 1, stream, mr);
    } else if (unit_ == "minute") {
      return diffTimestamp(left, right, kMsPerMinute, stream, mr);
    } else if (unit_ == "hour") {
      return diffTimestamp(left, right, kMsPerHour, stream, mr);
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
    // DATE columns are TIMESTAMP_DAYS, whose matching DURATION_DAYS is
    // int32 - subtracting the day counts directly can overflow for dates
    // near INT32_MIN/MAX days-since-epoch, which DATE (itself INT32
    // days-since-epoch) can represent. Widen each operand to
    // TIMESTAMP_SECONDS (int64) first - exact, since a whole day is always
    // a whole number of seconds - so the subtraction itself can't overflow.
    auto n = getSize(left, right);
    std::unique_ptr<cudf::column> leftOwned, rightOwned;
    auto leftCol = ensureColumn(left, n, leftOwned, stream, mr);
    auto rightCol = ensureColumn(right, n, rightOwned, stream, mr);
    auto secType = cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
    auto leftSec = cudf::cast(leftCol, secType, stream, mr);
    auto rightSec = cudf::cast(rightCol, secType, stream, mr);
    auto duration = cudf::binary_operation(
        rightSec->view(),
        leftSec->view(),
        cudf::binary_operator::SUB,
        cudf::data_type(cudf::type_id::DURATION_SECONDS),
        stream,
        mr);
    auto diffSeconds = cudf::cast(
        duration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
    static constexpr int64_t kSecondsPerDay = 86400LL;
    auto div = cudf::numeric_scalar<int64_t>(
        divisor * kSecondsPerDay, true, stream, mr);
    // Use C-style truncating DIV (not FLOOR_DIV) to match Velox CPU's
    // sign-symmetric behavior: the unsigned magnitude is divided and the
    // sign is re-applied, which is equivalent to truncation toward zero
    // rather than flooring toward -infinity.
    return cudf::binary_operation(
        diffSeconds->view(),
        div,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
  }
  static constexpr int64_t kMsPerDay = 86400LL * 1000LL;
  return diffTimestamp(left, right, divisor * kMsPerDay, stream, mr);
}

ColumnOrView DateDiffFunction::diffTimestamp(
    const Operand& left,
    const Operand& right,
    int64_t msPerUnit,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto n = getSize(left, right);
  std::unique_ptr<cudf::column> leftOwned, rightOwned;
  auto leftCol = ensureColumn(left, n, leftOwned, stream, mr);
  auto rightCol = ensureColumn(right, n, rightOwned, stream, mr);

  // Floor each operand to millisecond precision - not just cast, since
  // cudf::cast()'s rounding direction across timestamp resolutions is
  // unspecified - before subtracting, matching Velox CPU's diffTimestamp()
  // (DateTimeUtil.h), which converts both endpoints via Timestamp::
  // toMillis() before taking the difference. Subtracting at full input
  // resolution (e.g. nanoseconds) and dividing by msPerUnit afterward gives
  // a different (wrong) answer whenever a millisecond boundary falls
  // strictly between the operands' sub-millisecond components: e.g.
  // Timestamp(0, 999'999) to Timestamp(0, 1'000'000) is a true 1ms diff,
  // but subtracting first at nanosecond resolution and dividing by 1e6
  // truncates 1ns/1e6 to 0. Flooring first also keeps the subtraction
  // itself within a range that can't overflow INT64 - millisecond
  // timestamps stay far inside INT64 for any representable Velox
  // Timestamp, unlike subtracting at nanosecond resolution, which can
  // overflow for endpoints many billions of seconds apart.
  auto leftFloored = cudf::datetime::floor_datetimes(
      leftCol, cudf::datetime::rounding_frequency::MILLISECOND, stream, mr);
  auto rightFloored = cudf::datetime::floor_datetimes(
      rightCol, cudf::datetime::rounding_frequency::MILLISECOND, stream, mr);
  auto msType = cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  auto leftMs = cudf::cast(leftFloored->view(), msType, stream, mr);
  auto rightMs = cudf::cast(rightFloored->view(), msType, stream, mr);

  auto duration = cudf::binary_operation(
      rightMs->view(),
      leftMs->view(),
      cudf::binary_operator::SUB,
      cudf::data_type(cudf::type_id::DURATION_MILLISECONDS),
      stream,
      mr);
  auto diffMs = cudf::cast(
      duration->view(), cudf::data_type(cudf::type_id::INT64), stream, mr);
  if (msPerUnit == 1) {
    return diffMs;
  }
  auto div = cudf::numeric_scalar<int64_t>(msPerUnit, true, stream, mr);
  // See diffBySubtraction() for why DIV (not FLOOR_DIV) is required.
  return cudf::binary_operation(
      diffMs->view(),
      div,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
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

std::unique_ptr<cudf::column> DateDiffFunction::extractYearAsInt64(
    cudf::column_view daysCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      daysCol.type().id() == cudf::type_id::TIMESTAMP_DAYS,
      "extractYearAsInt64 expects a TIMESTAMP_DAYS column, got type id {}",
      static_cast<int>(daysCol.type().id()));
  static_assert(
      sizeof(cudf::timestamp_D) == sizeof(int32_t),
      "timestamp_D must be int32-sized for zero-copy reinterpret");
  cudf::column_view daysAsInt32(
      cudf::data_type{cudf::type_id::INT32},
      daysCol.size(),
      daysCol.head(),
      daysCol.null_mask(),
      daysCol.null_count(),
      daysCol.offset());

  auto i64 = cudf::data_type(cudf::type_id::INT64);
  auto boolType = cudf::data_type(cudf::type_id::BOOL8);
  auto scalar = [&](int64_t v) {
    return cudf::numeric_scalar<int64_t>(v, true, stream, mr);
  };
  auto sub = [&](cudf::column_view a, cudf::column_view b) {
    return cudf::binary_operation(
        a, b, cudf::binary_operator::SUB, i64, stream, mr);
  };
  auto add = [&](cudf::column_view a, cudf::column_view b) {
    return cudf::binary_operation(
        a, b, cudf::binary_operator::ADD, i64, stream, mr);
  };
  auto addS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::ADD, i64, stream, mr);
  };
  auto subS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::SUB, i64, stream, mr);
  };
  auto mulS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::MUL, i64, stream, mr);
  };
  auto divS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::DIV, i64, stream, mr);
  };
  auto ltS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::LESS, boolType, stream, mr);
  };
  auto geS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a,
        scalar(b),
        cudf::binary_operator::GREATER_EQUAL,
        boolType,
        stream,
        mr);
  };
  auto leS = [&](cudf::column_view a, int64_t b) {
    return cudf::binary_operation(
        a, scalar(b), cudf::binary_operator::LESS_EQUAL, boolType, stream, mr);
  };

  // Howard Hinnant's civil_from_days, adapted to columnar INT64 arithmetic
  // (http://howardhinnant.github.io/date_algorithms.html). Every
  // intermediate below operates on INT64, unlike cuDF's
  // extract_datetime_component() (always INT16), so this stays correct for
  // any year a TIMESTAMP_DAYS/INT32 day count can represent.
  auto z = cudf::cast(daysAsInt32, i64, stream, mr);
  auto zShifted = addS(z->view(), 719468);

  // era = (z >= 0 ? z : z - 146096) / 146097
  auto zNonNegative = geS(zShifted->view(), 0);
  auto zMinus146096 = subS(zShifted->view(), 146096);
  auto eraNumerator = cudf::copy_if_else(
      zShifted->view(), zMinus146096->view(), zNonNegative->view(), stream, mr);
  auto era = divS(eraNumerator->view(), 146097);

  // doe = zShifted - era * 146097, in [0, 146096].
  auto doe = sub(zShifted->view(), mulS(era->view(), 146097)->view());

  // yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365, in [0, 399].
  auto yoeNum = sub(
      add(doe->view(), divS(doe->view(), 36524)->view())->view(),
      add(divS(doe->view(), 1460)->view(), divS(doe->view(), 146096)->view())
          ->view());
  auto yoe = divS(yoeNum->view(), 365);

  // y = yoe + era * 400.
  auto y = add(yoe->view(), mulS(era->view(), 400)->view());

  // doy = doe - (365*yoe + yoe/4 - yoe/100), in [0, 365].
  auto doy =
      sub(doe->view(),
          sub(add(mulS(yoe->view(), 365)->view(), divS(yoe->view(), 4)->view())
                  ->view(),
              divS(yoe->view(), 100)->view())
              ->view());

  // mp = (5*doy + 2)/153, in [0, 11].
  auto mp = divS(addS(mulS(doy->view(), 5)->view(), 2)->view(), 153);

  // m = mp < 10 ? mp + 3 : mp - 9, in [1, 12].
  auto m = cudf::copy_if_else(
      addS(mp->view(), 3)->view(),
      subS(mp->view(), 9)->view(),
      ltS(mp->view(), 10)->view(),
      stream,
      mr);

  // The algorithm's calendar year starts in March, so the true civil year
  // is y + 1 for January/February (m <= 2) and y otherwise.
  return cudf::copy_if_else(
      addS(y->view(), 1)->view(),
      y->view(),
      leS(m->view(), 2)->view(),
      stream,
      mr);
}

std::unique_ptr<cudf::column> DateDiffFunction::timeOfDayMicros(
    cudf::column_view ts,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const bool isNanos = ts.type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS;
  // Floor to day precision via floor_datetimes() rather than a round-trip
  // cast through TIMESTAMP_DAYS: it's a documented, explicit floor (matching
  // DateTruncFunction's floorToDay for the same reason) and avoids depending
  // on cudf::cast()'s unspecified rounding direction for timestamp
  // resolution changes, and it also collapses the two casts into one call.
  auto dayFloorSameRes = cudf::datetime::floor_datetimes(
      ts, cudf::datetime::rounding_frequency::DAY, stream, mr);
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
  auto sign = cudf::copy_if_else(
      *plusOneScalar_, *minusOneScalar_, leftLessEqual->view(), stream, mr);

  // YEAR is extracted via extractYearAsInt64() (not
  // extractComponentAsInt64()) since cuDF's extract_datetime_component()
  // always returns INT16, silently wrapping any year outside
  // [-32768, 32767] - representable by DATE/TIMESTAMP's INT32
  // days-since-epoch. Floor to DAY first for TIMESTAMP inputs (matching
  // the calendar day extract_datetime_component() itself uses for
  // MONTH/DAY below) since cudf::cast()'s rounding direction across
  // timestamp resolutions is otherwise unspecified.
  auto toDays = [&](cudf::column_view col) -> std::unique_ptr<cudf::column> {
    if (col.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
      return std::make_unique<cudf::column>(col, stream, mr);
    }
    auto floored = cudf::datetime::floor_datetimes(
        col, cudf::datetime::rounding_frequency::DAY, stream, mr);
    return cudf::cast(
        floored->view(),
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  };
  auto loDays = toDays(loCol->view());
  auto hiDays = toDays(hiCol->view());
  auto y1 = extractYearAsInt64(loDays->view(), stream, mr);
  auto y2 = extractYearAsInt64(hiDays->view(), stream, mr);
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
  auto yearMonths = cudf::binary_operation(
      yearDiff->view(),
      *twelveScalar_,
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
