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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include <cudf/binaryop.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>

#include <optional>
#include <string>

namespace facebook::velox::cudf_velox::prestosql {

/// Presto date_diff(unit, date1/ts1, date2/ts2) -> bigint.
/// The unit argument is always a constant VARCHAR resolved at construction
/// time.
/// Supported units for DATE: day, week, month, quarter, year.
/// Supported units for TIMESTAMP: millisecond, second, minute, hour, day,
///   week, month, quarter, year.
/// cuDF stores dates as TIMESTAMP_DAYS (int32 days since epoch) and
/// timestamps as TIMESTAMP_MICROSECONDS (int64 us since epoch), so for
/// simple duration units we can subtract and scale. Calendar-aware units
/// (month, quarter, year) extract components and compute differences.
class DateDiffFunction : public CudfFunction {
 public:
  DateDiffFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool);

  /// Rejects forms cuDF can't evaluate: a null unit (the constructor
  /// requires a non-null constant unit and would throw), and the
  /// all-constant-operands form (eval() has no path for two scalar date/
  /// timestamp operands - see DateDiffFunction::binaryOp). Matches the
  /// canEvaluate pattern used by DateAddFunction/DateTruncFunction.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override;

 private:
  // Lightweight handle for a date_diff operand that is either a column or a
  // pre-captured scalar. Allows dispatching to the correct
  // cudf::binary_operation overload without materializing columns from
  // scalars.
  struct Operand {
    std::optional<cudf::column_view> col;
    const cudf::scalar* sc = nullptr;
  };

  // Dispatches to the correct cudf::binary_operation overload based on
  // whether each operand is a column or scalar.
  static std::unique_ptr<cudf::column> binaryOp(
      const Operand& lhs,
      const Operand& rhs,
      cudf::binary_operator op,
      cudf::data_type out,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  static cudf::size_type getSize(const Operand& a, const Operand& b);

  // Materializes a scalar operand into a column when a column_view is
  // required (e.g. for extract_datetime_component).
  static cudf::column_view ensureColumn(
      const Operand& op,
      cudf::size_type size,
      std::unique_ptr<cudf::column>& owned,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  // For day/week: subtract DATE columns to get DURATION_DAYS, then
  // optionally divide. For TIMESTAMP columns, delegates to diffTimestamp.
  ColumnOrView diffBySubtraction(
      const Operand& left,
      const Operand& right,
      int64_t divisor,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Floors both operands to millisecond precision (matching Velox CPU's
  // Timestamp::toMillis(), which every diffTimestamp() call goes through
  // before subtracting - see DateTimeUtil.h), subtracts them as
  // DURATION_MILLISECONDS, casts to INT64, then scales to the requested
  // unit via truncating DIV. Flooring each operand individually before
  // subtracting (rather than subtracting at full nanosecond/microsecond
  // precision and dividing afterward) is required for correctness, not
  // just overflow-avoidance: the two orders of operations can disagree
  // whenever a unit boundary falls strictly between the operands' non-ms
  // components. Millisecond precision also keeps the subtraction itself
  // far inside INT64 range for any representable Timestamp, unlike
  // subtracting at nanosecond resolution.
  ColumnOrView diffTimestamp(
      const Operand& left,
      const Operand& right,
      int64_t msPerUnit,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Extracts a datetime component and casts the result to INT64.
  static std::unique_ptr<cudf::column> extractComponentAsInt64(
      cudf::column_view col,
      cudf::datetime::datetime_component component,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  // Extracts the calendar year as INT64 without going through cuDF's
  // extract_datetime_component(), which always returns INT16 (max 32767)
  // regardless of component - silently wrapping for any date whose year
  // falls outside that range, which Velox's DATE/TIMESTAMP (INT32
  // days-since-epoch) can otherwise represent. Computes the year directly
  // from the whole-day count via Howard Hinnant's civil_from_days
  // algorithm (http://howardhinnant.github.io/date_algorithms.html),
  // entirely in INT64 arithmetic, so it's correct across the full range a
  // TIMESTAMP_DAYS column can represent.
  static std::unique_ptr<cudf::column> extractYearAsInt64(
      cudf::column_view daysCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  // Returns the number of microseconds elapsed since local midnight for a
  // TIMESTAMP column, by flooring to TIMESTAMP_DAYS and subtracting back.
  // This is used only for the time-of-day tie-break in the day-of-month
  // correction below, so it does not need to match wall-clock semantics
  // across timezones - only self-consistent ordering within the column.
  static std::unique_ptr<cudf::column> timeOfDayMicros(
      cudf::column_view ts,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  // Calendar-aware diff for year (isYear=true), or month (isYear=false,
  // used directly for "month" and via post-divide for "quarter").
  //
  // Matches Velox CPU's diffTimestamp() with respectLastDay=true (see
  // velox/functions/lib/DateTimeUtil.h): the two operands are first sorted
  // into (lo, hi) = (earlier, later), an unsigned "from(lo) to to(hi)" diff
  // is computed and decremented by one when lo's day-of-month/time-of-day
  // falls later in the month than hi's, unless hi is the last day of its
  // month (the respectLastDay exception); finally the sign of the original
  // (left, right) order is re-applied.
  ColumnOrView diffByComponent(
      const Operand& left,
      const Operand& right,
      bool isYear,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::string unit_;
  bool isDate_;
  // True for day/week/month/quarter/year: units whose result depends on the
  // *calendar* date/time-of-day of each operand, so - per Velox CPU's
  // diffTimestamp(unit, ts1, ts2, timeZone) in DateTimeUtil.h - each operand
  // must be converted to session-local wall-clock time before diffing.
  // hour/minute/second/millisecond are deliberately excluded: CPU shifts
  // both operands by the *same* offset for these units specifically to
  // cancel out any DST difference, so the result always equals the raw UTC
  // duration this class already computes; converting them would be a
  // no-op at best and wasted work at worst. See eval().
  bool isTimezoneSensitiveUnit_;
  std::unique_ptr<cudf::scalar> leftScalar_;
  std::unique_ptr<cudf::scalar> rightScalar_;
  bool leftIsConst_ = false;
  bool rightIsConst_ = false;

  // Scalars used unconditionally by diffByComponent()/eval() regardless of
  // unit_ or input data; cached once at construction instead of
  // reallocating on every eval() call. See DateTruncFunction for the same
  // pattern.
  std::unique_ptr<cudf::scalar> threeScalar_; // quarter = months / 3
  std::unique_ptr<cudf::scalar> twelveScalar_; // years -> months
  std::unique_ptr<cudf::scalar> plusOneScalar_; // sign when left <= right
  std::unique_ptr<cudf::scalar> minusOneScalar_; // sign when left > right
};

} // namespace facebook::velox::cudf_velox::prestosql
