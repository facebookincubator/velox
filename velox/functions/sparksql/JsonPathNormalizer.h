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

#include "velox/type/StringView.h"

namespace facebook::velox::functions::sparksql {

/// Normalizes the JSON path to be Spark-compatible.
///
/// Rules applied:
/// 1. Removes single quotes in bracket notation (e.g., "$['a']" -> "$[a]").
/// 2. Removes spaces after dots (e.g., "$. a" -> "$.a").
/// 3. Removes trailing spaces after root symbol (e.g., "$ " -> "$").
/// 4. Invalid cases return "-1":
///    - Empty path or path not starting with '$'.
///    - Space between ($ or ]) and dot (e.g., "$ .a").
///    - Space between [ and field name (e.g., "$[' a']").
///    - Consecutive dots (e.g., "$..a").
///    - Dot at the end (e.g., "$.a. ").
class JsonPathNormalizer {
 public:
  // The state transitions： kAfterDollar --> kAfterDot <--> kToken
  //                            |               /|\            |
  //                            |                |             |
  //                            |----> kInSquareBrackets <-----|
  enum class State { kAfterDollar, kAfterDot, kToken, kInSquareBrackets };

  std::string normalize(StringView jsonPath);

 private:
  std::string removeSingleQuotes(StringView jsonPath);

  // Try to convert state to kAfterDot if currentChar is '.'.
  bool tryConvertToAfterDot(char currentChar, char previousChar);
  // Try to convert state to kInSquareBrackets if currentChar is '['.
  bool tryConvertToInSquareBrackets(char currentChar, char nextChar);

  State state_{State::kAfterDollar};
};

} // namespace facebook::velox::functions::sparksql
