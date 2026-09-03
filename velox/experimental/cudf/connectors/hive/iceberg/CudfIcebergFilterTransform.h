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

/// Outcome of evaluating a split's filter on an injected column against the
/// constant value that column holds for the whole split.
enum class ConstantFilterFold {
  /// The filter rejects the constant, so it rejects every row of the split.
  /// A NULL constant the filter rejects folds here too, in which case the
  /// predicate the fold stands for is NULL rather than false.
  kAlwaysFalse,
  /// The filter accepts the constant, so it accepts every row of the split.
  kAlwaysTrue,
  /// The filter could not be evaluated on the host.
  kUnknown,
};

/// Filter over the columns the parquet reader projects, derived from a filter
/// over the assembled table.
///
/// Move-only. Owns the expression nodes created while transforming. Both
/// `pushedExpr` and `deferredExpr` may point into the input filter, which must
/// therefore outlive the transformed result.
struct TransformedFilter {
  /// Expression nodes created while transforming.
  cudf::ast::tree nodes;

  /// Root of the pushed down filter into the parquet reader, over the columns
  /// it projects. Null when nothing can be pushed.
  const cudf::ast::expression* pushedExpr;

  /// Root of the deferred filter. Null when `pushedExpr` already enforces the
  /// input filter exactly and no deferred pass is needed.
  const cudf::ast::expression* deferredExpr;

  /// Whether the input filter rejects this split.
  bool skipSplit;

  /// Whether `pushedExpr` retains a decimal literal whose storage width must
  /// match the split.
  bool requiresSplitSpecificDecimalTypes;
};

/// Transforms the input filter into a filter over the columns actually
/// projected by the parquet reader and a deferred filter for whatever the
/// reader cannot enforce.
///
/// Predicates over an injected column are replaced by the fold of that column,
/// which either folds the predicate away, folds the whole filter to false, or,
/// when the fold is unknown, moves it into `deferredExpr`. Column indices in
/// the pushed filter are rebased past the injected columns.
///
/// A `kAlwaysFalse` fold means "no row passes". The predicate behind it may be
/// false or NULL. AND and OR operators accept both, but a negation over a
/// folded operand must be deferred rather than inverted.
///
/// @param filter The input filter over the assembled table.
/// @param sortedInjectedColumnIndices Ascending, unique indices into the
/// assembled table that 'filter' was built against, not indices into the data
/// file.
/// @param injectedColumnFolds Fold of the split's filter on each injected
/// column, parallel to 'sortedInjectedColumnIndices'.
/// @return The transformed filter over the columns actually projected by the
/// parquet reader.
TransformedFilter transformFilterForInjectedColumns(
    const cudf::ast::expression& filter,
    std::span<const cudf::size_type> sortedInjectedColumnIndices,
    std::span<const ConstantFilterFold> injectedColumnFolds);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
