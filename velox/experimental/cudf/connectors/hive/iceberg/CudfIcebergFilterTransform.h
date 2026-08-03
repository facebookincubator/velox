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

#include <cudf/ast/expressions.hpp>

#include <span>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// @brief Filter over the columns the parquet reader projects, derived from a
/// filter over the assembled table.
struct TransformedFilter {
  /// Owns the expression nodes created while transforming. Nodes may also point
  /// into the input filter, which must outlive the transformed result.
  cudf::ast::tree tree;

  /// Root of the transformed filter, or nullptr when the transformed
  /// filter is always true.
  const cudf::ast::expression* expr{nullptr};

  /// Whether the input filter references an injected column
  bool referencesInjectedColumn{false};
};

/// @brief Transforms the input filter into a sub-filter over the columns
/// actually projected by the parquet reader.
///
/// Transforms the filter by dropping predicates on injected columns and
/// rebasing remaining column indices past the dropped columns.
///
/// @param filter The input filter over the assembled table.
/// @param sortedInjectedColumnIndices Ascending, unique indices into the
/// assembled table that 'filter' was built against, not indices into the data
/// file.
/// @return The transformed filter over the columns actually projected by the
/// parquet reader.
TransformedFilter transformFilterForInjectedColumns(
    const cudf::ast::expression& filter,
    const std::span<cudf::size_type const> sortedInjectedColumnIndices);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
