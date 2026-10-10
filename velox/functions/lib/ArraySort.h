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

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/SimpleComparisonMatcher.h"

namespace facebook::velox::functions {

/// Creates array_sort function.
///
/// @param ascending If true, sort in ascending order; otherwise, sort in
/// descending order.
/// @param nullsFirst If true, nulls are placed first; otherwise, nulls are
/// placed last.
/// @param throwOnNestedNull If true, throw an exception if a nested null is
/// encountered.
std::shared_ptr<exec::VectorFunction> makeArraySort(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    bool ascending,
    bool nullsFirst,
    bool throwOnNestedNull);

/// Options for array_sort without a lambda.
struct ArraySortOptions {
  /// If true, sort in ascending order; otherwise, sort in descending order.
  bool ascending{true};
  /// If true, top-level nulls are placed first; otherwise, they are placed
  /// last.
  bool nullsFirst{false};
  /// If true, nulls nested inside complex values are ordered first;
  /// otherwise, they are ordered last.
  bool nestedNullsFirst{false};
  /// If true, throw an exception if a nested null is encountered.
  bool throwOnNestedNull{true};
  /// If true, preserve the original order of elements that compare equal but
  /// are distinguishable, such as -0.0 and 0.0. Ignored for element types
  /// whose equal values are indistinguishable.
  bool stable{false};
};

/// Creates array_sort function.
std::shared_ptr<exec::VectorFunction> makeArraySort(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    const ArraySortOptions& options);

/// Creates array_sort with a lambda function.
///
/// @param ascending If true, sort in ascending order; otherwise, sort in
/// descending order.
/// @param throwOnNestedNull If true, throw an exception if a nested null is
/// encountered.
std::shared_ptr<exec::VectorFunction> makeArraySortLambdaFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    bool ascending,
    bool throwOnNestedNull);

/// Options for array_sort with a sort-key lambda.
struct ArraySortLambdaOptions {
  /// If true, sort in ascending order; otherwise, sort in descending order.
  bool ascending{true};
  /// If true, throw an exception if a nested null is encountered.
  bool throwOnNestedNull{true};
  /// If true, throw a user error when the lambda produces a null sort key for
  /// an array with at least two elements.
  bool rejectNullSortKeys{false};
  /// If true, don't evaluate the lambda for arrays with fewer than two
  /// elements.
  bool skipLambdaForTrivialArrays{false};
};

/// Creates array_sort with a lambda function.
std::shared_ptr<exec::VectorFunction> makeArraySortLambdaFunction(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config,
    const ArraySortLambdaOptions& options);

/// Returns signatures for array_sort function.
///
/// @param withComparator If true, includes a signature for sorting with a
/// comparator function.
std::vector<std::shared_ptr<exec::FunctionSignature>> arraySortSignatures(
    bool withComparator);

/// Analyzes array_sort(array, lambda) call to determine whether it can be
/// re-written into a simpler call that specifies sort-by expression.
///
/// For example, rewrites
///     array_sort(a, (x, y) -> if(length(x) < length(y), -1, if(length(x) >
///     length(y), 1, 0))
/// into
///     array_sort(a, x -> length(x))
///
/// Returns the rewritten expression. If rewrite is not possible, throw a user
/// error.
core::TypedExprPtr rewriteArraySortCall(
    const std::string& prefix,
    const core::TypedExprPtr& expr,
    const std::shared_ptr<SimpleComparisonChecker> checker);

/// Options for rewriting array_sort comparator lambdas.
struct ArraySortRewriteOptions {
  /// If true, comparators may return any negative value, zero, and any
  /// positive value. Otherwise, only -1, 0, and 1 are accepted.
  bool supportsArbitraryComparatorResults{false};
  /// If true, rewrite into '$internal$array_sort_comparator[_desc]', which
  /// rejects null sort keys at runtime. Otherwise, rewrite into
  /// 'array_sort[_desc]'.
  bool rejectNullSortKeys{false};
};

/// Same as above, but with configurable rewrite options.
core::TypedExprPtr rewriteArraySortCall(
    const std::string& prefix,
    const core::TypedExprPtr& expr,
    const std::shared_ptr<SimpleComparisonChecker> checker,
    const ArraySortRewriteOptions& options);

} // namespace facebook::velox::functions
