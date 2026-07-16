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

#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox::prestosql {

ToUnixtimeFunction::ToUnixtimeFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 1, "to_unixtime expects exactly 1 input");
}

ColumnOrView ToUnixtimeFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto inputCol = asView(inputColumns[0]);

  // Cast to TIMESTAMP_MICROSECONDS if the input has a different resolution.
  std::unique_ptr<cudf::column> castOwned;
  if (inputCol.type().id() != cudf::type_id::TIMESTAMP_MICROSECONDS) {
    castOwned = cudf::cast(
        inputCol,
        cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS),
        stream,
        mr);
    inputCol = castOwned->view();
  }

  // TIMESTAMP_MICROSECONDS stores int64 microseconds since epoch.
  // Reinterpret the underlying data as INT64 without copying.
  static_assert(
      sizeof(cudf::timestamp_us) == sizeof(int64_t),
      "timestamp_us must be int64-sized for zero-copy reinterpret");
  cudf::column_view usView(
      cudf::data_type{cudf::type_id::INT64},
      inputCol.size(),
      inputCol.head(),
      inputCol.null_mask(),
      inputCol.null_count(),
      inputCol.offset());

  // Dividing INT64 by a FLOAT64 scalar with FLOAT64 output type produces
  // the correct floating-point result without truncation.
  auto divisor = cudf::numeric_scalar<double>(1000000.0, true, stream, mr);
  return cudf::binary_operation(
      usView,
      divisor,
      cudf::binary_operator::DIV,
      cudf::data_type(cudf::type_id::FLOAT64),
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
