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

#include <string>

namespace facebook::velox::functions::sparksql {

/// Registers Spark's element_at(array(T), index) -> T and
/// element_at(map(K, V), key) -> V.
///
/// Array indices start at 1, negative indices access elements from the last to
/// the first, and index 0 throws. An out-of-bound array index returns NULL, or
/// throws when Spark ANSI mode is enabled. A missing map key always returns
/// NULL.
void registerElementAtFunction(const std::string& name);

} // namespace facebook::velox::functions::sparksql
