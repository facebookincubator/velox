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
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/sparksql/DateAddFunction.h"

#include "velox/expression/ConstantExpr.h"

#include <cudf/binaryop.hpp>

namespace facebook::velox::cudf_velox::sparksql {

DateAddFunction::DateAddFunction(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->type()->isDate(),
      "First argument to date_add must be a date");
  VELOX_CHECK_NULL(
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]));
  // The date_add second argument could be int8_t, int16_t, int32_t.
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

} // namespace facebook::velox::cudf_velox::sparksql
