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

#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace ast {
class tree;
}
} // namespace cudf

namespace facebook::velox::cudf_velox {

/// Convert a single subfield filter to a cuDF AST expression.
/// @param subfield Column the filter applies to; resolved to a column index in
///   'inputRowSchema'.
/// @param filter Velox filter to translate (range, values-list, null check,
///   etc.).
/// @param tree AST arena that owns the returned expression and every
///   intermediate node; must outlive any evaluation of the result.
/// @param scalars Storage that owns the literal scalars the AST references;
///   must outlive evaluation. Newly created literals are appended.
/// @param inputRowSchema Row type of the columns the AST is evaluated against;
///   supplies the filtered column's index and type.
/// @param timestampUnit The cuDF timestamp type_id used by the Parquet reader
///   (e.g. TIMESTAMP_MILLISECONDS). Timestamp range scalars are created with
///   this resolution so they match the column data types at evaluation time.
/// @return Reference to the root expression, owned by 'tree'.
cudf::ast::expression const& createAstFromSubfieldFilter(
    const common::Subfield& subfield,
    const common::Filter& filter,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS);

/// Build a single AST expression representing the logical AND of all filters in
/// 'subfieldFilters'.
/// @param subfieldFilters Per-column filters to combine; each is converted with
///   createAstFromSubfieldFilter and the results are joined with
///   NULL_LOGICAL_AND.
/// @param tree AST arena that owns the returned expression and every
///   intermediate node; must outlive any evaluation of the result.
/// @param scalars Storage that owns the literal scalars the AST references;
///   must outlive evaluation. Newly created literals are appended.
/// @param inputRowSchema Row type of the columns the AST is evaluated against.
/// @param timestampUnit The cuDF timestamp type_id used by the Parquet reader;
///   forwarded to each per-filter conversion.
/// @return Reference to the combined root expression, owned by 'tree'.
cudf::ast::expression const& createAstFromSubfieldFilters(
    const common::SubfieldFilters& subfieldFilters,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS);

} // namespace facebook::velox::cudf_velox
