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

#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Re-runs a filter through Velox after a GPU SFI kernel declined a row, so
/// that Velox raises the error with its own class and message. Re-runs the
/// whole expression rather than the declining subexpression: a GPU conditional
/// evaluates both branches for every row, so the declined row may be one Velox
/// never evaluates, in which case nothing is raised.
///
/// Returns the verdict as a boolean cudf column; throws whatever Velox throws.
/// `cached` holds the compiled expression, built on first use.
std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExecCtx* execCtx,
    memory::MemoryPool* pool,
    cuda::stream_ref stream);

/// The connector form, for a scan's remaining filter, which reaches Velox's
/// evaluator through core::ExpressionEvaluator.
std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExpressionEvaluator* evaluator,
    memory::MemoryPool* pool,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
