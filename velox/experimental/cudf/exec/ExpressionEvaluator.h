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
#include "velox/expression/Expr.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>

#include <memory>
#include <vector>
#include <tuple>

namespace facebook::velox::cudf_velox {

cudf::ast::expression const& create_ast_tree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<std::tuple<int, std::string, int>>& precompute_instructions);

void addPrecomputedColumns(
    std::vector<std::unique_ptr<cudf::column>>& input_table_columns,
    const std::vector<std::tuple<int, std::string, int>>& precompute_instructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
