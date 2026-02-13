/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace ast {
class tree;
}
} // namespace cudf

namespace facebook::velox::cudf_velox {

// Convert subfield filters to cudf AST
cudf::ast::expression const& createAstFromSubfieldFilter(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema);

// Build a single AST expression representing logical AND of all filters in
// 'subfieldFilters'. The resulting expression reference is owned by the passed
// 'tree'.
cudf::ast::expression const& createAstFromSubfieldFilters(
    const common::SubfieldFilters& subfieldFilters,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema);

} // namespace facebook::velox::cudf_velox
