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

#include <folly/Likely.h>
#include <cstring>

#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/JsonUtil.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"
#include "velox/type/Conversions.h"

namespace facebook::velox::functions::sparksql {

/// Parses a JSON string and returns the value at the specified path.
/// Simdjson On-Demand API is used to parse JSON string.
/// get_json_object(jsonString, path) -> value
template <typename T>
struct GetJsonObjectFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*json*/,
      const arg_type<Varchar>* jsonPath) {
    if (jsonPath != nullptr && checkJsonPath(*jsonPath)) {
      jsonPath_ = normalizeJsonPath(*jsonPath);
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& json,
      const arg_type<Varchar>& jsonPath) {
    // Spark requires the first char in jsonPath is '$'.
    if (!checkJsonPath(jsonPath)) {
      return false;
    }
    // Check if json has invalid escape sequence.
    if (hasInvalidEscapedChar(json.data(), json.size())) {
      return false;
    }
    const auto formattedJsonPath =
        jsonPath_.has_value() ? jsonPath_.value() : normalizeJsonPath(jsonPath);
    // jsonPath is "$".
    if (formattedJsonPath.empty()) {
      result.append(json);
      return true;
    }
    simdjson::ondemand::document jsonDoc;
    simdjson::padded_string paddedJson(json.data(), json.size());
    if (simdjsonParseIncomplete(paddedJson).get(jsonDoc)) {
      return false;
    }
    try {
      // Can return error result or throw exception possibly.
      auto rawResult = jsonDoc.at_path(formattedJsonPath);
      if (rawResult.error()) {
        return false;
      }

      if (!extractStringResult(rawResult, result)) {
        return false;
      }
    } catch (simdjson::simdjson_error&) {
      return false;
    }

    const char* currentPos;
    if (jsonDoc.current_location().get(currentPos)) {
      return false;
    }

    return isValidEndingCharacter(currentPos);
  }

 private:
  FOLLY_ALWAYS_INLINE bool checkJsonPath(StringView jsonPath) {
    // Spark requires the first char in jsonPath is '$'.
    return std::string_view{jsonPath}.starts_with('$');
  }

  // Spark's json path requires field name surrounded by single quotes if it is
  // specified in "[]". But simdjson lib requires not. This method just removes
  // such single quotes to adapt to simdjson lib, e.g., converts "['a']['b']" to
  // "[a][b]".
  std::string removeSingleQuotes(StringView jsonPath) {
    std::string result(jsonPath.data(), jsonPath.size());
    size_t pairEnd = 0;
    while (true) {
      auto pairBegin = result.find("['", pairEnd);
      if (pairBegin == std::string::npos) {
        break;
      }
      pairEnd = result.find(']', pairBegin);
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

  // Normalizes the JSON path to be Spark-compatible.
  //
  // Rules applied:
  // 1. Removes single quotes in bracket notation (e.g., "$['a']" -> "$[a]").
  // 2. Removes spaces after dots (e.g., "$. a" -> "$.a").
  // 3. Removes trailing spaces after root symbol (e.g., "$ " -> "$").
  // 4. Invalid cases return "-1":
  //    - Empty path or path not starting with '$'.
  //    - Space between $ and dot (e.g., "$ .a").
  //    - Consecutive dots (e.g., "$..a").
  //    - Dot at the end (e.g., "$.a. ").
  std::string normalizeJsonPath(StringView jsonPath) {
    // First, remove single quotes for bracket notation
    std::string path = removeSingleQuotes(jsonPath);
    if (path.empty() || path[0] != '$') {
      return "-1";
    }

    enum class State {
      kAfterDollar,
      kAfterDot,
      kToken
    } state = State::kAfterDollar;

    std::string normalized;
    normalized.reserve(path.size() - 1);

    for (size_t i = 1; i < path.size(); ++i) {
      const char c = path[i];
      if (c == ' ') {
        if (state == State::kToken) {
          // Spaces within tokens are preserved.
          normalized.push_back(c);
        }
        continue;
      }
      switch (state) {
        case State::kAfterDollar: {
          if (c == '.') {
            state = State::kAfterDot;
            if (path[i - 1] == ' ') {
              // Spaces between '$' and '.' are invalid.
              return "-1";
            }
          }
          normalized.push_back(c);
          break;
        }
        case State::kAfterDot: {
          if (c == '.') {
            // Consecutive dots are invalid.
            return "-1";
          }
          normalized.push_back(c);
          state = State::kToken;
          break;
        }
        case State::kToken: {
          if (c == '.') {
            normalized.push_back(c);
            state = State::kAfterDot;
          } else {
            normalized.push_back(c);
          }
          break;
        }
      }
    }

    if (state == State::kAfterDot) {
      // Trailing dot is invalid.
      return "-1";
    }

    return normalized;
  }

  // Extracts a string representation from a simdjson result. Handles various
  // JSON types including numbers, booleans, strings, objects, and arrays.
  // Returns true if the conversion is successful. Otherwise, returns false.
  bool extractStringResult(
      simdjson::simdjson_result<simdjson::ondemand::value> rawResult,
      out_type<Varchar>& result) {
    std::stringstream ss;
    switch (rawResult.type()) {
      // For number and bool types, we need to explicitly get the value
      // for specific types instead of using `ss << rawResult`. Thus, we
      // can make simdjson's internal parsing position moved and then we
      // can check the validity of ending character.
      case simdjson::ondemand::json_type::number: {
        switch (rawResult.get_number_type()) {
          case simdjson::ondemand::number_type::floating_point_number: {
            double numberResult;
            if (!rawResult.get_double().get(numberResult)) {
              result.append(
                  util::Converter<TypeKind::VARCHAR>::tryCast(numberResult)
                      .value());
              return true;
            }
            return false;
          }
          default: {
            std::string_view intResult = trimToken(rawResult.raw_json_token());
            // Spark uses Jackson to parse JSON, which does not preserve the
            // negative sign for -0. See the implementation here:
            // https://github.com/FasterXML/jackson-core/blob/jackson-core-2.19.2/src/main/java/com/fasterxml/jackson/core/util/TextBuffer.java#L699-L702
            if (intResult == "-0") {
              intResult = "0";
            }
            result.append(intResult);
            // Advance the simdjson parsing position.
            return !rawResult.get_double().error();
          }
        }
      }
      case simdjson::ondemand::json_type::boolean: {
        bool boolResult;
        if (!rawResult.get_bool().get(boolResult)) {
          result.append(boolResult ? "true" : "false");
          return true;
        }
        return false;
      }
      case simdjson::ondemand::json_type::string: {
        std::string_view stringResult;
        if (!rawResult.get_string().get(stringResult)) {
          result.append(stringResult);
          return true;
        }
        return false;
      }
      // For nested case, e.g., for "{"my": {"hello": 10}}",
      // "$.my" will return an object type.
      case simdjson::ondemand::json_type::object:
      case simdjson::ondemand::json_type::array: {
        ss << rawResult;
        result.append(ss.str());
        return true;
      }
      default:
        return false;
    }
  }

  // Checks whether the obtained result is followed by valid char. Because
  // On-Demand API we are using ignores json format validation for characters
  // following the current parsing position. As json doc is padded with NULL
  // characters, it's safe to do recursively check.
  bool isValidEndingCharacter(const char* currentPos) {
    char endingChar = *currentPos;
    if (endingChar == ',' || endingChar == '}' || endingChar == ']') {
      return true;
    }
    // These chars can be prior to a valid ending char. See reference:
    // https://github.com/simdjson/simdjson/blob/v3.9.0/dependencies/jsoncppdist/jsoncpp.cpp
    if (endingChar == ' ' || endingChar == '\r' || endingChar == '\n' ||
        endingChar == '\t') {
      return isValidEndingCharacter(++currentPos);
    }
    return false;
  }

  // Checks for invalid escape sequences in JSON string.
  // Note: We only search for '\' which is ASCII (0x5C) and cannot appear
  // as a continuation byte in valid UTF-8 (continuation bytes are 0x80-0xBF).
  // So we can safely scan byte-by-byte regardless of encoding.
  // See the valid escape sequences in Jackson's JSON parser:
  // https://github.com/FasterXML/jackson-core/blob/jackson-core-2.19.2/src/main/java/com/fasterxml/jackson/core/json/ReaderBasedJsonParser.java#L2648
  FOLLY_ALWAYS_INLINE bool hasInvalidEscapedChar(
      const char* json,
      size_t size) {
    const char* end = json + size;
    const char* pos = json;

    while ((pos = static_cast<const char*>(std::memchr(
                pos, '\\', static_cast<size_t>(end - pos)))) != nullptr) {
      const auto remaining = static_cast<size_t>(end - pos);
      if (FOLLY_UNLIKELY(remaining < 2)) {
        return false; // Incomplete escape at end, let parser handle it.
      }

      switch (pos[1]) {
        case '"':
          [[fallthrough]];
        case '\\':
          [[fallthrough]];
        case '/':
          [[fallthrough]];
        case 'b':
          [[fallthrough]];
        case 'f':
          [[fallthrough]];
        case 'n':
          [[fallthrough]];
        case 'r':
          [[fallthrough]];
        case 't':
          pos += 2;
          break;
        case 'u':
          // Validate \uXXXX.
          if (FOLLY_UNLIKELY(remaining < 6) || !isHexDigit(pos[2]) ||
              !isHexDigit(pos[3]) || !isHexDigit(pos[4]) ||
              !isHexDigit(pos[5])) {
            return true;
          }
          pos += 6;
          break;
        default:
          return true; // Invalid escape character.
      }
    }
    return false;
  }

  static FOLLY_ALWAYS_INLINE bool isHexDigit(char c) {
    auto uc = static_cast<unsigned char>(c);
    return (uc - '0' < 10U) || ((uc | 0x20) - 'a' < 6U);
  }

  // Used for constant json path.
  std::optional<std::string> jsonPath_;
};

} // namespace facebook::velox::functions::sparksql
