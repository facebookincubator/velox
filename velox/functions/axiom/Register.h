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

#include <string_view>

namespace facebook::velox::functions::axiom {

/// Registers the functions that plans from the Axiom optimizer call, whatever
/// the SQL dialect, each as 'prefix' followed by its name. A process that runs
/// such plans registers them and tells the optimizer the prefix.
///
/// - bigint_add(bigint, bigint) -> bigint: the sum. Fails on overflow.
/// - array_append(array(T), T) -> array(T): the array with the element added
///   at the end. A NULL element adds a NULL entry. A NULL array gives NULL.
/// - array_prepend(T, array(T)) -> array(T): the array with the element added
///   at the start, with the same rules for NULLs.
void registerFunctions(std::string_view prefix);

} // namespace facebook::velox::functions::axiom
