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

#include "velox/functions/sparksql/JsonPathNormalizer.h"

namespace facebook::velox::functions::sparksql {

std::string JsonPathNormalizer::normalize(StringView jsonPath) {
  // First, remove single quotes for bracket notation
  std::string path = removeSingleQuotes(jsonPath);
  if (path.empty() || path[0] != '$') {
    return "-1";
  }
  std::string normalized;
  normalized.reserve(path.size() - 1);

  // Initialize state.
  state_ = State::kAfterDollar;
  for (size_t i = 1; i < path.size(); ++i) {
    const char c = path[i];
    if (c == ' ') {
      if (state_ == State::kToken || state_ == State::kInSquareBrackets) {
        // Spaces within tokens and quare brackets are preserved.
        normalized.push_back(c);
      }
      continue;
    }
    switch (state_) {
      case State::kAfterDollar: {
        if (!tryConvertToAfterDot(c, path[i - 1])) {
          return "-1";
        }
        // i + 1 is safe because removeSingleQuotes ensures at least one
        // char (']') after.
        if (!tryConvertToInSquareBrackets(c, path[i + 1])) {
          return "-1";
        }
        break;
      }
      case State::kAfterDot: {
        if (c == '.') {
          // Consecutive dots are invalid.
          return "-1";
        }
        state_ = State::kToken;
        break;
      }
      case State::kToken: {
        if (c == '.') {
          state_ = State::kAfterDot;
        } else if (!tryConvertToInSquareBrackets(c, path[i + 1])) {
          return "-1";
        }
        break;
      }
      case State::kInSquareBrackets: {
        if (!tryConvertToAfterDot(c, path[i - 1])) {
          return "-1";
        }
        break;
      }
    }
    normalized.push_back(c);
  }

  if (state_ == State::kAfterDot) {
    // Trailing dot is invalid.
    return "-1";
  }

  return normalized;
}

// Spark's json path requires field name surrounded by single quotes if it is
// specified in "[]". But simdjson lib requires not. This method just removes
// such single quotes to adapt to simdjson lib, e.g., converts "['a']['b']" to
// "[a][b]".
std::string JsonPathNormalizer::removeSingleQuotes(StringView jsonPath) {
  std::string result(jsonPath.data(), jsonPath.size());
  size_t pairEnd = 0;
  while (true) {
    auto pairBegin = result.find("['", pairEnd);
    if (pairBegin == std::string::npos) {
      break;
    }
    pairEnd = result.find("]", pairBegin);
    // If expected pattern, like ['a'], is not found.
    if (pairEnd == std::string::npos || result[pairEnd - 1] != '\'') {
      return "-1";
    }
    result.erase(pairEnd - 1, 1);
    result.erase(pairBegin + 1, 1);
    pairEnd -= 2;
  }
  return result;
}

bool JsonPathNormalizer::tryConvertToAfterDot(
    char currentChar,
    char previousChar) {
  if (currentChar == '.') {
    if (previousChar == ' ' && state_ != State::kToken) {
      // Spaces before '.' are invalid.
      return false;
    }
    state_ = State::kAfterDot;
  }
  return true;
}

bool JsonPathNormalizer::tryConvertToInSquareBrackets(
    char currentChar,
    char nextChar) {
  if (currentChar == '[') {
    if (nextChar == ' ') {
      // Spaces between '[' and field are invalid.
      return false;
    }
    state_ = State::kInSquareBrackets;
  }
  return true;
}

} // namespace facebook::velox::functions::sparksql
