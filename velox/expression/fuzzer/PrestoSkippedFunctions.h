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
#include <unordered_set>

namespace facebook::velox::fuzzer {

/// Functions the Presto expression fuzzer skips in every run, regardless of
/// the reference engine. These functions at some point crash or fail and need
/// to be fixed before we can enable.
///
/// Entries are either a function name, which excludes every signature of that
/// function, or a full function signature, which excludes only that signature.
const std::unordered_set<std::string>& prestoSkippedFunctions();

/// Functions skipped on top of prestoSkippedFunctions() when Presto is used as
/// the source of truth. These are cases where Velox and Presto legitimately
/// disagree, or where the function is not registered in Presto at all.
const std::unordered_set<std::string>& prestoSkippedFunctionsSOT();

} // namespace facebook::velox::fuzzer
