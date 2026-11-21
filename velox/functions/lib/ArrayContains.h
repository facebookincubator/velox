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

namespace facebook::velox::functions {

/// Returns the signatures of the function made by makeArrayContains():
/// array(T), T -> boolean.
std::vector<std::shared_ptr<exec::FunctionSignature>> arrayContainsSignatures();

/// Makes a function that returns true if the array contains the value. If it
/// does not, returns NULL when the array has a NULL element and false
/// otherwise. If 'throwOnNestedNull' is true, throws when NULLs nested in
/// complex-type elements make a comparison indeterminate, e.g. comparing
/// ARRAY[1, NULL] with ARRAY[1, NULL]. Otherwise, compares nested NULLs as
/// values, i.e. NULL equals NULL.
std::unique_ptr<exec::VectorFunction> makeArrayContains(bool throwOnNestedNull);

} // namespace facebook::velox::functions
