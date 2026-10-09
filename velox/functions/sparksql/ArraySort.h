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

#include "velox/functions/lib/ArraySort.h"

namespace facebook::velox::functions::sparksql {

std::shared_ptr<exec::VectorFunction> makeArraySortAsc(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

// Sorts an array in descending order. The first argument is the array to sort,
// and the second argument is a lambda function that maps the array elements
// to the values to be sorted by.
//
// This function is only used in rewriteArraySortCall.
std::shared_ptr<exec::VectorFunction> makeArraySortDesc(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

/// Sorts using a comparator rewritten to an ascending sort key. Rejects NULL
/// sort keys unless 'spark.array_sort.reject_null_comparator_keys' is false.
std::shared_ptr<exec::VectorFunction> makeArraySortComparatorAsc(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

/// Sorts using a comparator rewritten to a descending sort key. Rejects NULL
/// sort keys unless 'spark.array_sort.reject_null_comparator_keys' is false.
std::shared_ptr<exec::VectorFunction> makeArraySortComparatorDesc(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::shared_ptr<exec::VectorFunction> makeSortArray(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::vector<std::shared_ptr<exec::FunctionSignature>> arraySortDescSignatures();

/// Signatures for the internal comparator sort functions. Unlike
/// array_sort_desc, the element type does not need to be orderable because
/// elements are ordered by the rewritten sort key.
std::vector<std::shared_ptr<exec::FunctionSignature>>
arraySortComparatorSignatures();

std::vector<std::shared_ptr<exec::FunctionSignature>> sortArraySignatures();

} // namespace facebook::velox::functions::sparksql
