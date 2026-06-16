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
#include "velox/experimental/cudf/expression/prestosql/DatePlusIntervalFunction.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/type/Time.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>

#include <string_view>

namespace facebook::velox::cudf_velox::prestosql {

namespace {

// Throws a VeloxUserError with userMessage if any non-null entry of cond is
// false. cond must be a BOOL8 column. Does nothing for empty or all-null
// columns.
void checkAllTrue(
    cudf::column_view cond,
    std::string_view userMessage,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (cond.is_empty() || cond.null_count() == cond.size()) {
    return;
  }

  const auto boolType = cudf::data_type(cudf::type_id::BOOL8);
  auto allTrue = cudf::reduce(
      cond,
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      boolType,
      stream,
      mr);
  auto* result = static_cast<cudf::scalar_type_t<bool>*>(allTrue.get());
  VELOX_USER_CHECK(
      result->is_valid(stream) && result->value(stream), "{}", userMessage);
}

} // namespace

DatePlusIntervalFunction::DatePlusIntervalFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(),
      2,
      "plus(date, interval) expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->type()->isDate(),
      "First argument to plus must be a date");
  VELOX_CHECK(
      expr->inputs()[1]->type()->isIntervalDayTime(),
      "Second argument to plus must be an interval day to second");

  auto stream = cudf::get_default_stream(cudf::allow_default_stream);
  auto mr = get_temp_mr();

  // If the interval is a constant, extract it at construction time and
  // convert to a duration_days scalar. A constant-null interval leaves both
  // durationDaysLiteral_ and the column-path scalars unset; eval() short-
  // circuits to an all-null result.
  if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
          expr->inputs()[1])) {
    auto constValue = constExpr->value();
    if (!constValue->isNullAt(0)) {
      auto millis = constValue->as<ConstantVector<int64_t>>()->value();
      VELOX_USER_CHECK_EQ(
          millis % kMillisInDay,
          0,
          "Cannot add hours, minutes, seconds or milliseconds to a date");
      auto days = static_cast<int32_t>(millis / kMillisInDay);
      durationDaysLiteral_ =
          std::make_unique<cudf::duration_scalar<cudf::duration_D>>(
              days, true, stream, mr);
    }
  } else {
    // Column interval path: pre-compute the kMillisInDay and zero scalars
    // used by eval() so they aren't reallocated on every batch.
    millisPerDayScalar_ = std::make_unique<cudf::numeric_scalar<int64_t>>(
        kMillisInDay, true, stream, mr);
    zeroScalar_ = std::make_unique<cudf::numeric_scalar<int64_t>>(
        int64_t{0}, true, stream, mr);
  }
  stream.synchronize();
}

ColumnOrView DatePlusIntervalFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto dateCol = asView(inputColumns[0]);

  if (durationDaysLiteral_) {
    // Constant non-null interval.
    return cudf::binary_operation(
        dateCol,
        *durationDaysLiteral_,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

  if (millisPerDayScalar_) {
    // Column interval path: interval arrives as a column of millis. Validate
    // that all intervals are whole days, then convert to duration_D and add.
    auto intervalCol = asView(inputColumns[1]);

    auto remainder = cudf::binary_operation(
        intervalCol,
        *millisPerDayScalar_,
        cudf::binary_operator::MOD,
        cudf::data_type(cudf::type_id::INT64),
        stream,
        mr);
    auto isWholeDays = cudf::binary_operation(
        remainder->view(),
        *zeroScalar_,
        cudf::binary_operator::EQUAL,
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        mr);
    checkAllTrue(
        isWholeDays->view(),
        "Cannot add hours, minutes, seconds or milliseconds to a date",
        stream,
        mr);

    auto daysInt = cudf::binary_operation(
        intervalCol,
        *millisPerDayScalar_,
        cudf::binary_operator::DIV,
        cudf::data_type(cudf::type_id::INT32),
        stream,
        mr);

    auto daysDuration = cudf::cast(
        daysInt->view(),
        cudf::data_type(cudf::type_id::DURATION_DAYS),
        stream,
        mr);
    return cudf::binary_operation(
        dateCol,
        daysDuration->view(),
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

  // Constant-null interval: result is null for every row.
  cudf::timestamp_scalar<cudf::timestamp_D> nullDate(0, false, stream, mr);
  return cudf::make_column_from_scalar(nullDate, dateCol.size(), stream, mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
