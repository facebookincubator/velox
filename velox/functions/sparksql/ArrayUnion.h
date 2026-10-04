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

namespace facebook::velox::functions::sparksql {

/// Returns the signatures of array_union(array(T), array(T)) -> array(T).
std::vector<std::shared_ptr<exec::FunctionSignature>> arrayUnionSignatures();

/// Creates array_union, which returns the distinct elements of both arrays in
/// the order they first appear, with at most one null. -0.0 and 0.0 are equal,
/// and so are all NaNs. REAL and DOUBLE values in the result, including nested
/// ones, are returned in canonical form: -0.0 as 0.0 and every NaN as the
/// canonical NaN.
std::shared_ptr<exec::VectorFunction> makeArrayUnion(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

} // namespace facebook::velox::functions::sparksql
