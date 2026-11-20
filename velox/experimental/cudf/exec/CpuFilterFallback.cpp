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
#include "velox/experimental/cudf/exec/CpuFilterFallback.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/expression/EvalCtx.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {
namespace {

// The rows the device was filtering, as a Velox vector named by `rowType`: a
// FieldReference in the filter resolves by name.
RowVectorPtr toHostRows(
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  auto hostRows = with_arrow::toVeloxColumn(
      cudf::table_view{columns},
      pool,
      std::static_pointer_cast<const Type>(rowType),
      stream,
      get_temp_mr());
  stream.sync();
  return hostRows;
}

// Copies the CPU's verdict to the device as the boolean column the caller
// expects from the kernel.
std::unique_ptr<cudf::column> toDeviceMask(
    const VectorPtr& verdict,
    vector_size_t numRows,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  VELOX_CHECK(
      verdict->type()->isBoolean(),
      "A filter must evaluate to a boolean, got {}",
      verdict->type()->toString());

  auto wrapped = std::make_shared<RowVector>(
      pool,
      ROW({"mask"}, {BOOLEAN()}),
      /*nulls=*/nullptr,
      numRows,
      std::vector<VectorPtr>{verdict});
  auto table = with_arrow::toCudfTable(wrapped, pool, stream, get_output_mr());
  auto columns = table->release();
  VELOX_CHECK_EQ(columns.size(), 1);
  return std::move(columns.front());
}

} // namespace

std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExecCtx* execCtx,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  auto hostRows = toHostRows(rowType, columns, pool, stream);

  if (cached == nullptr) {
    cached = velox::exec::makeExprSetFromFlag({filter}, execCtx);
  }

  velox::exec::LocalSelectivityVector rowsHolder(*execCtx, hostRows->size());
  auto* const rows = rowsHolder.get();
  rows->setAll();
  velox::exec::EvalCtx evalCtx(execCtx, cached.get(), hostRows.get());

  std::vector<VectorPtr> results;
  cached->eval(*rows, evalCtx, results);
  VELOX_CHECK_EQ(results.size(), 1);

  return toDeviceMask(results.front(), hostRows->size(), pool, stream);
}

std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExpressionEvaluator* evaluator,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  auto hostRows = toHostRows(rowType, columns, pool, stream);

  if (cached == nullptr) {
    cached = evaluator->compile(filter);
  }

  SelectivityVector rows(hostRows->size());
  VectorPtr verdict;
  evaluator->evaluate(cached.get(), rows, *hostRows, verdict);

  return toDeviceMask(verdict, hostRows->size(), pool, stream);
}

} // namespace facebook::velox::cudf_velox
