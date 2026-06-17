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

} // namespace

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
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty());
  auto inputTableView = convertToTableView(inputColumns);
  return cudf::hashing::murmurhash3_x86_32(
      inputTableView, seedValue_, stream, mr);
}

} // namespace facebook::velox::cudf_velox::sparksql
