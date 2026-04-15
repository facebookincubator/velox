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
#include "velox/experimental/cudf/expression/DateArithmeticFunctions.h"

#include "velox/experimental/cudf/expression/AstUtils.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/type/Time.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>
#include <cudf/reduction.hpp>

namespace facebook::velox::cudf_velox {

DateAddFunction::DateAddFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->type()->isDate(),
      "First argument to date_add must be a date");
  VELOX_CHECK_NULL(
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
          expr->inputs()[0]));
  value_ = makeScalarFromConstantExpr(
      expr->inputs()[1], cudf::type_id::DURATION_DAYS);
}

ColumnOrView DateAddFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto inputCol = asView(inputColumns[0]);
  return cudf::binary_operation(
      inputCol,
      *value_,
      cudf::binary_operator::ADD,
      cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
      stream,
      mr);
}

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
}

ColumnOrView DatePlusIntervalFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto dateCol = asView(inputColumns[0]);
  auto intervalCol = asView(inputColumns[1]);

  // Validate that all intervals are whole days.
  auto divisor = cudf::numeric_scalar<int64_t>(kMillisInDay, true, stream);
  auto remainder = cudf::binary_operation(
      intervalCol,
      divisor,
      cudf::binary_operator::MOD,
      cudf::data_type(cudf::type_id::INT64),
      stream,
      mr);
  auto zero = cudf::numeric_scalar<int64_t>(0, true, stream);
  auto isWholeDays = cudf::binary_operation(
      remainder->view(),
      zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto allWholeDays = cudf::reduce(
      isWholeDays->view(),
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type(cudf::type_id::BOOL8),
      stream,
      mr);
  auto* result =
      static_cast<cudf::scalar_type_t<bool>*>(allWholeDays.get());
  VELOX_USER_CHECK(
      result->is_valid(stream) && result->value(stream),
      "Cannot add hours, minutes, seconds or milliseconds to a date");

  // Divide millis by kMillisInDay to get days.
  auto daysInt = cudf::binary_operation(
      intervalCol,
      divisor,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::INT32),
      stream,
      mr);

  // Cast days to duration_days and add to date.
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

} // namespace facebook::velox::cudf_velox
