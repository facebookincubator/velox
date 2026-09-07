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

#include <cstddef>
#include <optional>
#include <string>

namespace facebook::velox::functions::sparksql {
namespace xpath {

/// Evaluate XPath boolean expression on XML string.
/// Throws a user error on invalid XML or invalid XPath (Spark-faithful).
/// Returns nullopt only when the evaluation yields no value.
std::optional<bool>
evalBoolean(const char* xml, size_t xmlLen, const char* path, size_t pathLen);

/// Evaluate XPath string expression on XML string.
/// Throws a user error on invalid XML or invalid XPath (Spark-faithful).
/// Returns nullopt only when the evaluation yields no value.
std::optional<std::string>
evalString(const char* xml, size_t xmlLen, const char* path, size_t pathLen);

} // namespace xpath
} // namespace facebook::velox::functions::sparksql
