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

/// Spark's elt(n, input1, input2, ...) returns the n-th input (1-based). The
/// inputs are either all VARCHAR or all VARBINARY. Returns NULL if n is NULL or
/// the selected input is NULL. An out-of-range n returns NULL when Spark ANSI
/// mode is disabled and throws when it is enabled.
std::vector<std::shared_ptr<exec::FunctionSignature>> eltSignatures();

/// Creates the elt function. Reads Spark ANSI mode from 'config'.
std::shared_ptr<exec::VectorFunction> makeElt(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

} // namespace facebook::velox::functions::sparksql
