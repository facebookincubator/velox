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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>

#include "velox/core/QueryConfig.h"
#include "velox/expression/StringWriter.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/JsonUtil.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"
#include "velox/type/Conversions.h"

namespace facebook::velox::functions::sparksql {

namespace detail {
// Normalizes the JSON path to be Spark-compatible.
//
// Rules applied:
// 1. Removes single quotes in bracket notation (e.g., "$['a']" -> "$[a]").
// 2. Removes spaces after dots (e.g., "$. a" -> "$.a").
// 3. Removes trailing spaces after root symbol (e.g., "$ " -> "$").
// 4. Invalid cases return "-1":
//    - Empty path or path not starting with '$'.
//    - Space between ($ or ]) and dot (e.g., "$ .a").
//    - Space between [ and field name (e.g., "$[' a']").
//    - Consecutive dots (e.g., "$..a").
//    - Dot at the end (e.g., "$.a. ").
class JsonPathNormalizer {
 public:
  // The state transitions: kAfterDollar --> kAfterDot <--> kToken
  //                            |               /|\            |
  //                            |                |             |
  //                            |----> kInSquareBrackets <-----|
  enum class State { kAfterDollar, kAfterDot, kToken, kInSquareBrackets };

  std::string normalize(StringView jsonPath) {
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

 private:
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

  // Try to convert state to kAfterDot if currentChar is '.'.
  bool tryConvertToAfterDot(char currentChar, char previousChar) {
    if (currentChar == '.') {
      if (previousChar == ' ' && state_ != State::kToken) {
        // Spaces before '.' are invalid.
        return false;
      }
      state_ = State::kAfterDot;
    }
    return true;
  }

  // Try to convert state to kInSquareBrackets if currentChar is '['.
  bool tryConvertToInSquareBrackets(char currentChar, char nextChar) {
    if (currentChar == '[') {
      if (nextChar == ' ') {
        // Spaces between '[' and field are invalid.
        return false;
      }
      state_ = State::kInSquareBrackets;
    }
    return true;
  }

  State state_{State::kAfterDollar};
};

// Appends the raw JSON object/array slice `raw` to `out`, rewriting characters
// in the Unicode supplementary planes (code points >= U+10000, encoded as
// 4-byte UTF-8) into '\\uXXXX\\uXXXX' UTF-16 surrogate-pair escapes. All other
// bytes (ASCII, 2-/3-byte BMP sequences, existing escapes) are appended
// verbatim, in bulk runs. When `raw` has no supplementary-plane character it is
// appended directly with no extra allocation.
//
// Spark's get_json_object re-serializes an object/array result through a
// Jackson JsonGenerator, which escapes supplementary-plane characters as
// surrogate-pair '\\u' escapes while leaving BMP characters (e.g. CJK) literal.
// The raw simdjson slice keeps them literal, so this aligns Velox with Spark.
// See https://github.com/facebookincubator/velox/issues/17923.
void appendWithSupplementaryEscapes(
    std::string_view raw,
    exec::StringWriter& out);

// Writes the object or array slice `raw` to `out` as Spark's Jackson generator
// writes it: no whitespace between tokens, numbers as in the input, and string
// escapes decoded, except that quotes, backslashes, control characters and
// surrogates are written as Jackson escapes them. Returns false if `raw` has an
// incomplete escape.
bool compactJson(std::string_view raw, std::string& out);

// Returns whether `value` is well formed at every nesting level. Spark's
// Jackson parser reads the whole root value, so an error anywhere in it gives
// NULL, while text after the root value is never read.
bool isValidJsonValue(simdjson::ondemand::value value);

// Spark accepts single-quoted JSON strings by default. Detect them cheaply so
// standard JSON remains allocation-free.
bool normalizeSingleQuotedJson(std::string_view input, std::string& output);

// Writes `input` to `output` with the control characters inside its strings
// escaped, since Spark's parser accepts them unescaped, and leaves `output`
// empty if there are none. Returns false if a bracket inside the root value is
// closed by a bracket of the other kind, which Spark rejects.
bool escapeControlCharacters(std::string_view input, std::string& output);

// Returns the scalar at the start of `json` without the text after it, or an
// empty view if `json` does not start with a scalar.
std::string_view rootScalar(std::string_view json);

// Returns whether the character is a JSON digit, '0' to '9'.
inline bool isJsonDigit(char character) {
  return character >= '0' && character <= '9';
}

// Validates strict JSON number syntax before using simdjson's looser parser.
bool isValidJsonNumber(std::string_view input);

} // namespace detail

/// Parses a JSON string and returns the value at the specified path.
/// Simdjson On-Demand API is used to parse JSON string.
/// get_json_object(jsonString, path) -> value
template <typename T>
struct GetJsonObjectFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<Varchar>* /*json*/,
      const arg_type<Varchar>* jsonPath) {
    if (jsonPath != nullptr && checkJsonPath(*jsonPath)) {
      jsonPath_ = pathNormalizer_.normalize(*jsonPath);
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
    unconsumedSize_ = 0;
    std::string_view input{json};
    if (!detail::escapeControlCharacters(input, escapedJson_)) {
      return false;
    }
    if (!escapedJson_.empty()) {
      input = escapedJson_;
    }
    if (detail::normalizeSingleQuotedJson(input, normalizedJson_)) {
      input = normalizedJson_;
    }
    // Check if json has invalid escape sequence.
    if (hasInvalidEscapedChar(input.data(), input.size())) {
      return false;
    }
    const auto formattedJsonPath = jsonPath_.has_value()
        ? jsonPath_.value()
        : pathNormalizer_.normalize(jsonPath);
    simdjson::ondemand::document jsonDoc;
    simdjson::padded_string paddedJson(input.data(), input.size());
    if (simdjsonParseIncomplete(paddedJson).get(jsonDoc)) {
      return false;
    }
    try {
      simdjson::ondemand::value root;
      if (jsonDoc.get_value().get(root)) {
        return formattedJsonPath.empty() && extractRootScalar(input, result);
      }
      if (!detail::isValidJsonValue(root)) {
        return false;
      }
      jsonDoc.rewind();
      // jsonPath is "$".
      if (formattedJsonPath.empty()) {
        if (jsonDoc.get_value().get(root)) {
          return false;
        }
        return extractStringResult(std::move(root), result);
      }
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

    return isValidEndingCharacter(currentPos + unconsumedSize_);
  }

 private:
  FOLLY_ALWAYS_INLINE bool checkJsonPath(StringView jsonPath) {
    // Spark requires the first char in jsonPath is '$'.
    return std::string_view{jsonPath}.starts_with('$');
  }

  // Writes the root of 'json', a scalar, to 'result' as Spark does for '$': the
  // text after the scalar is ignored.
  bool extractRootScalar(std::string_view json, out_type<Varchar>& result) {
    const auto scalar = detail::rootScalar(json);
    if (scalar == "null") {
      result.append(scalar);
      return true;
    }
    simdjson::padded_string array("[" + std::string(scalar) + "]");
    simdjson::ondemand::document document;
    simdjson::ondemand::value value;
    return !scalar.empty() && !simdjsonParse(array).get(document) &&
        !document.at(0).get(value) &&
        extractStringResult(std::move(value), result);
  }

  // Extracts a string representation from a simdjson result. Handles various
  // JSON types including numbers, booleans, strings, objects, and arrays.
  // Returns true if the conversion is successful. Otherwise, returns false.
  bool extractStringResult(
      simdjson::simdjson_result<simdjson::ondemand::value> rawResult,
      out_type<Varchar>& result) {
    switch (rawResult.type()) {
      // For number and bool types, we need to explicitly get the value
      // for specific types instead of taking their raw token. Thus, we
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
            const auto token = trimToken(rawResult.raw_json_token());
            if (!detail::isValidJsonNumber(token)) {
              return false;
            }
            const auto castResult =
                util::Converter<TypeKind::DOUBLE>::tryCast(token);
            if (castResult.hasError() || !std::isinf(castResult.value())) {
              return false;
            }
            unconsumedSize_ = token.size();
            result.append(
                std::signbit(castResult.value()) ? R"("-Infinity")"
                                                 : R"("Infinity")");
            return true;
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
        // simdjson returns the object/array as its raw input slice, verbatim
        // (no unescaping) -- exactly what we want.
        std::string_view raw;
        if (rawResult.raw_json().get(raw) ||
            !detail::compactJson(raw, minified_)) {
          return false;
        }
        detail::appendWithSupplementaryEscapes(minified_, result);
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
      if (remaining < 2) [[unlikely]] {
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

  detail::JsonPathNormalizer pathNormalizer_;

  // Holds a minified object or array result.
  std::string minified_;
  std::string escapedJson_;
  std::string normalizedJson_;
  size_t unconsumedSize_{0};
};

} // namespace facebook::velox::functions::sparksql
