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
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

/// Filter over the columns the parquet reader projects, derived from a filter
/// over the assembled table.
///
/// Move-only. Owns the expression nodes created while transforming. Those nodes
/// may also point into the input filter, which must therefore outlive the
/// transformed result.
struct TransformedFilter {
  /// Expression nodes created while transforming.
  cudf::ast::tree nodes;

  /// Root of the transformed filter. Null when the transformed filter is
  /// always true.
  const cudf::ast::expression* expr;

  /// Whether the input filter references injected column(s).
  bool referencesInjectedColumn;

  /// Whether the transformed filter retains a decimal literal whose storage
  /// width must match the split.
  bool requiresSplitSpecificDecimalTypes;

  /// Ascending, unique indices of the injected columns whose predicates were
  /// dropped, in the index space of the input filter.
  std::vector<cudf::size_type> droppedInjectedColumns;

  /// Whether the input filter is equivalent to `expr` conjoined with the
  /// predicates on `droppedInjectedColumns`. When true, proving every dropped
  /// predicate true for a split makes `expr` an exact filter for it, and
  /// proving one false makes the filter reject every row of the split.
  ///
  /// False when a predicate was dropped from a position where it is not a
  /// conjunct of the filter root, for example under a disjunction, or when
  /// dropping it also discarded predicates over columns read from the data
  /// file.
  bool exactIfDroppedAreTrue;
};

/// Transforms the input filter into a sub-filter over the columns actually
/// projected by the parquet reader.
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
    std::span<const cudf::size_type> sortedInjectedColumnIndices);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
