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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

cudf::table_view convertToTableView(std::vector<ColumnOrView>& inputColumns) {
  std::vector<cudf::column_view> columns;
  columns.reserve(inputColumns.size());
  for (auto& col : inputColumns) {
    columns.push_back(asView(col));
  }
  return cudf::table_view(columns);
}

// Returns true if cuDF's Spark hash rejects the type: a TIMESTAMP whose cuDF
// unit is not microseconds, or an ARRAY of ROW, at any nesting depth.
bool containsUnsupportedType(const TypePtr& type) {
  if (type->isTimestamp() &&
      CudfConfig::getInstance().timestampUnit !=
          cudf::type_id::TIMESTAMP_MICROSECONDS) {
    return true;
  }
  if (type->isArray() && type->childAt(0)->isRow()) {
    return true;
  }
  for (size_t i = 0; i < type->size(); ++i) {
    if (containsUnsupportedType(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

} // namespace

bool HashFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() < 2) {
    return false;
  }

  for (size_t i = 1; i < expr->inputs().size(); ++i) {
    const auto& input = expr->inputs()[i];
    if (containsUnsupportedType(input->type())) {
      return false;
    }
    // eval() receives no column for a constant input. Dropping a NULL constant
    // is correct, because Spark skips nulls.
    if (input->isConstantKind() &&
        !input->asUnchecked<core::ConstantTypedExpr>()->isNull()) {
      return false;
    }
  }
  return true;
}

HashFunction::HashFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_GE(expr->inputs().size(), 2, "hash expects at least 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->isConstantKind(), "hash seed must be a constant");
  const auto* seedExpr =
      expr->inputs()[0]->asUnchecked<core::ConstantTypedExpr>();
  const auto vec = seedExpr->hasValueVector()
      ? seedExpr->valueVector()
      : seedExpr->toConstantVector(pool);
  int32_t seedValue = vec->as<SimpleVector<int32_t>>()->valueAt(0);
  VELOX_CHECK_GE(seedValue, 0);
  seedValue_ = seedValue;
}

ColumnOrView HashFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty());
  auto inputTableView = convertToTableView(inputColumns);
  return cudf::hashing::spark_murmurhash3_x86_32(
      inputTableView, seedValue_, stream, mr);
}

} // namespace facebook::velox::cudf_velox::sparksql
