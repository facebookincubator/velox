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
#include "velox/experimental/cudf/expression/prestosql/ToUnixtimeFunction.h"

#include "velox/expression/ConstantExpr.h"

#include <cudf/binaryop.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace facebook::velox::cudf_velox::prestosql {

bool ToUnixtimeFunction::canEvaluate(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 1) {
    return false;
  }
  return std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
             expr->inputs()[0]) == nullptr;
}

ToUnixtimeFunction::ToUnixtimeFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 1, "to_unixtime expects exactly 1 input");
}

ColumnOrView ToUnixtimeFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    [[maybe_unused]] cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty(), "to_unixtime expects a column input");
  auto inputCol = asView(inputColumns[0]);

  // Pick the divisor from the input's own resolution instead of casting to
  // a fixed resolution (e.g. TIMESTAMP_MICROSECONDS) first: a cast ahead of
  // the reinterpret below would silently drop any precision finer than
  // that fixed resolution actually has - e.g. Timestamp(0, 999) (999ns) is
  // 0 once cast to TIMESTAMP_MICROSECONDS, so to_unixtime would return 0
  // instead of 0.000000999. The cuDF bridge can represent TIMESTAMP_
  // NANOSECONDS (its default), so this precision is observable in
  // practice, not just a theoretical concern.
  double divisorValue;
  switch (inputCol.type().id()) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      divisorValue = 1.0;
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      divisorValue = 1000.0;
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      divisorValue = 1000000.0;
      break;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      divisorValue = 1000000000.0;
      break;
    default:
      VELOX_USER_FAIL(
          "to_unixtime: unsupported timestamp resolution on GPU (type id {})",
          static_cast<int>(inputCol.type().id()));
  }

  // All four resolutions above store an int64 count since epoch. Reinterpret
  // the underlying data as INT64 without copying.
  static_assert(
      sizeof(cudf::timestamp_s) == sizeof(int64_t) &&
          sizeof(cudf::timestamp_ms) == sizeof(int64_t) &&
          sizeof(cudf::timestamp_us) == sizeof(int64_t) &&
          sizeof(cudf::timestamp_ns) == sizeof(int64_t),
      "every timestamp resolution handled above must be int64-sized for "
      "zero-copy reinterpret");
  cudf::column_view countView(
      cudf::data_type{cudf::type_id::INT64},
      inputCol.size(),
      inputCol.head(),
      inputCol.null_mask(),
      inputCol.null_count(),
      inputCol.offset());

  // Dividing INT64 by a FLOAT64 scalar with FLOAT64 output type produces
  // the correct floating-point result without truncation.
  auto divisor = cudf::numeric_scalar<double>(divisorValue, true, stream, mr);
  return cudf::binary_operation(
      countView,
      divisor,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::FLOAT64),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
