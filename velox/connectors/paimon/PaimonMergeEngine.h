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

#include <fmt/format.h>
#include <string>
#include <string_view>

namespace facebook::velox::connector::paimon {

/// Paimon merge engine of a primary-key table, which decides how merge-on-read
/// combines the rows of one key. Set by the 'merge-engine' table option.
enum class PaimonMergeEngine {
  /// Keeps the latest row of each key.
  kDeduplicate,
  /// Keeps the latest non-null value of each column.
  kPartialUpdate,
  /// Aggregates each column with its configured function.
  kAggregation,
  /// Keeps the first row of each key.
  kFirstRow,
};

/// Returns the 'merge-engine' option value of the merge engine, e.g.
/// "first-row".
std::string paimonMergeEngineString(PaimonMergeEngine mergeEngine);

/// Parses a 'merge-engine' option value, ignoring capitalization as Paimon
/// Java does.
PaimonMergeEngine paimonMergeEngineFromString(std::string_view value);

} // namespace facebook::velox::connector::paimon

template <>
struct fmt::formatter<facebook::velox::connector::paimon::PaimonMergeEngine>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::paimon::PaimonMergeEngine mergeEngine,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::paimon::paimonMergeEngineString(
            mergeEngine),
        ctx);
  }
};
