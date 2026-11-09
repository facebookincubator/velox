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
#include "velox/experimental/cudf/expression/prestosql/SplitFunction.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/split/split.hpp>

#include <limits>

namespace facebook::velox::cudf_velox::prestosql {
namespace {

int64_t constantLimitValue(const core::TypedExprPtr& expr) {
  const auto* constant = expr->asUnchecked<core::ConstantTypedExpr>();
  const auto value = constant->hasValueVector()
      ? constant->valueVector()->variantAt(0)
      : constant->value();
  return value.kind() == TypeKind::INTEGER ? value.value<TypeKind::INTEGER>()
                                           : value.value<TypeKind::BIGINT>();
}

} // namespace

bool SplitFunction::canEvaluate(const core::TypedExprPtr& expr) {
  // cuDF splits on whitespace runs for an empty delimiter, where Presto
  // splits between characters.
  const auto delimiter = constantVarcharValue(expr->inputs()[1]);
  if (!delimiter.has_value() || delimiter->empty()) {
    return false;
  }
  if (expr->inputs().size() == 2) {
    return true;
  }
  const auto& limit = expr->inputs()[2];
  if (!limit->isConstantKind() ||
      limit->asUnchecked<core::ConstantTypedExpr>()->isNull()) {
    return false;
  }
  // Presto returns the input for a limit of 1 and raises below, where cuDF
  // splits without a limit. cudf::size_type is int32.
  const auto value = constantLimitValue(limit);
  return value >= 2 && value <= std::numeric_limits<int32_t>::max();
}

SplitFunction::SplitFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* /*pool*/) {
  VELOX_CHECK(
      canEvaluate(expr),
      "split expression cannot be evaluated by prestosql::SplitFunction");
  delimiter_ = std::string(*constantVarcharValue(expr->inputs()[1]));
  if (expr->inputs().size() == 3) {
    // Presto limits the array size; cuDF limits the number of splits.
    maxSplitCount_ =
        static_cast<cudf::size_type>(constantLimitValue(expr->inputs()[2]) - 1);
  }
}

ColumnOrView SplitFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  const cudf::string_scalar delimiter(delimiter_, true, stream, mr);
  return cudf::strings::split_record(
      asView(inputColumns[0]), delimiter, maxSplitCount_, stream, mr);
}

} // namespace facebook::velox::cudf_velox::prestosql
