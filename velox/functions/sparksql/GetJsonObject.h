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

namespace detail {
template <typename T>
bool appendSparkFormattedValue(
    T& rawResult,
    std::string& out,
    bool quoteString) {
  simdjson::ondemand::json_type type;
  if (rawResult.type().get(type)) {
    return false;
  }

  switch (type) {
    case simdjson::ondemand::json_type::null:
      out.append("null");
      return true;
    case simdjson::ondemand::json_type::number:
      switch (rawResult.get_number_type()) {
        case simdjson::ondemand::number_type::floating_point_number: {
          double numberResult;
          if (rawResult.get_double().get(numberResult)) {
            return false;
          }
          out.append(
              util::Converter<TypeKind::VARCHAR>::tryCast(numberResult)
                  .value());
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
          out.append(intResult);
          // Advance the simdjson parsing position.
          return !rawResult.get_double().error();
        }
      }
    case simdjson::ondemand::json_type::boolean: {
      bool boolResult;
      if (rawResult.get_bool().get(boolResult)) {
        return false;
      }
      out.append(boolResult ? "true" : "false");
      return true;
    }
    case simdjson::ondemand::json_type::string:
      if (!quoteString) {
        std::string_view stringResult;
        if (rawResult.get_string().get(stringResult)) {
          return false;
        }
        out.append(stringResult);
        return true;
      }
      [[fallthrough]];
    case simdjson::ondemand::json_type::object:
    case simdjson::ondemand::json_type::array: {
      std::string_view raw;
      if (simdjson::to_json_string(rawResult).get(raw)) {
        return false;
      }
      out.append(raw);
      return true;
    }
    default:
      return false;
  }
}

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

// Parses a normalized path string into tokens and evaluates the path
// against a simdjson document.
class GetJsonObjectEvaluator {
 private:
  struct NamedToken {
    std::string name;
  };

  struct IndexToken {
    int64_t index;
  };

  struct WildcardToken {};

  using PathToken = std::variant<NamedToken, IndexToken, WildcardToken>;

  enum class WriteStyle { kRaw, kQuoted, kFlatten };

 public:
  explicit GetJsonObjectEvaluator(const std::string& normalizedPath) {
    parse(normalizedPath);
  }

  template <typename T>
  bool evaluate(T& doc, std::string& out, WriteStyle style = WriteStyle::kRaw) {
    try {
      return evaluatePath(doc, out, style, tokens_, 0);
    } catch (const simdjson::simdjson_error&) {
      out.clear();
      return false;
    }
  }

 private:
  FOLLY_ALWAYS_INLINE void throwIfError(simdjson::error_code error) {
    if (error != simdjson::SUCCESS) {
      throw simdjson::simdjson_error(error);
    }
  }

  void parse(const std::string& normalizedPath) {
    size_t i = 0;
    while (i < normalizedPath.size()) {
      if (normalizedPath[i] == '.') {
        i++;
        size_t start = i;
        while (i < normalizedPath.size() && normalizedPath[i] != '.' &&
               normalizedPath[i] != '[') {
          i++;
        }
        if (i > start) {
          tokens_.emplace_back(
              NamedToken{normalizedPath.substr(start, i - start)});
        }
      } else if (normalizedPath[i] == '[') {
        i++;
        if (i < normalizedPath.size() && normalizedPath[i] == '*') {
          tokens_.emplace_back(WildcardToken{});
          i += 2; // skip '*]'
        } else {
          size_t start = i;
          while (i < normalizedPath.size() && normalizedPath[i] != ']') {
            i++;
          }
          std::string content = normalizedPath.substr(start, i - start);
          bool isNum = !content.empty();
          for (char c : content) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
              isNum = false;
              break;
            }
          }
          if (isNum) {
            tokens_.emplace_back(IndexToken{std::stoll(content)});
          } else {
            tokens_.emplace_back(NamedToken{content});
          }
          if (i < normalizedPath.size()) {
            i++; // skip ']'
          }
        }
      } else {
        i++;
      }
    }
  }

  FOLLY_ALWAYS_INLINE void appendCommaIfNeeded(std::string& out) {
    if (!out.empty() && out.back() != '[') {
      out += ',';
    }
  }

  // Emits the current JSON subtree according to Spark's write style rules.
  // - kRaw: writes string leaves without quotes
  // - kFlatten: flattens an array into its parent output
  // - all other cases: copy the current subtree verbatim.
  template <typename T>
  bool writeMatchedValue(
      T& val,
      std::string& out,
      WriteStyle style,
      const std::vector<PathToken>& tokens,
      size_t tokenIdx) {
    simdjson::ondemand::json_type type;
    throwIfError(val.type().get(type));

    if (type == simdjson::ondemand::json_type::array &&
        style == WriteStyle::kFlatten) {
      simdjson::ondemand::array arr;
      throwIfError(val.get_array().get(arr));
      bool dirty = false;
      for (auto elem : arr) {
        throwIfError(elem.error());
        auto elemVal = elem.value_unsafe();
        dirty |= evaluatePath(elemVal, out, style, tokens, tokenIdx);
      }
      return dirty;
    }

    appendCommaIfNeeded(out);
    return appendSparkFormattedValue(val, out, style != WriteStyle::kRaw);
  }

  template <typename T>
  bool evaluateNamedToken(
      T& val,
      std::string& out,
      WriteStyle style,
      const std::vector<PathToken>& tokens,
      size_t tokenIdx,
      simdjson::ondemand::json_type type) {
    if (type != simdjson::ondemand::json_type::object || val.is_null()) {
      return false;
    }

    const auto& name = std::get<NamedToken>(tokens[tokenIdx]).name;
    simdjson::ondemand::object object;
    throwIfError(val.get_object().get(object));
    bool dirty = false;
    for (auto field : object) {
      throwIfError(field.error());
      std::string_view key;
      throwIfError(field.unescaped_key().get(key));
      if (dirty || key != name) {
        continue;
      }
      simdjson::ondemand::value fieldValue;
      throwIfError(field.value().get(fieldValue));
      if (fieldValue.is_null()) {
        continue;
      }
      dirty = evaluatePath(fieldValue, out, style, tokens, tokenIdx + 1);
    }
    return dirty;
  }

  template <typename T>
  bool evaluateIndexToken(
      T& val,
      std::string& out,
      WriteStyle style,
      const std::vector<PathToken>& tokens,
      size_t tokenIdx,
      simdjson::ondemand::json_type type) {
    if (val.is_null()) {
      return false;
    }
    int64_t targetIdx = std::get<IndexToken>(tokens[tokenIdx]).index;
    bool nextIsWildcard =
        (tokenIdx + 1 < tokens.size() &&
         std::holds_alternative<WildcardToken>(tokens[tokenIdx + 1]));
    WriteStyle nextStyle = nextIsWildcard ? WriteStyle::kQuoted : style;

    if (type == simdjson::ondemand::json_type::array) {
      simdjson::ondemand::array arr;
      throwIfError(val.get_array().get(arr));
      int64_t idx = 0;
      bool dirty = false;
      for (auto elem : arr) {
        throwIfError(elem.error());
        if (idx == targetIdx) {
          auto elemVal = elem.value_unsafe();
          if (elemVal.is_null()) {
            return false;
          }
          dirty = evaluatePath(elemVal, out, nextStyle, tokens, tokenIdx + 1);
        }
        idx++;
      }
      return dirty;
    }

    if (type == simdjson::ondemand::json_type::object) {
      auto fieldName = std::to_string(targetIdx);
      simdjson::ondemand::object object;
      throwIfError(val.get_object().get(object));
      bool dirty = false;
      for (auto field : object) {
        throwIfError(field.error());
        std::string_view key;
        throwIfError(field.unescaped_key().get(key));
        if (dirty || key != fieldName) {
          continue;
        }
        simdjson::ondemand::value fieldValue;
        throwIfError(field.value().get(fieldValue));
        if (fieldValue.is_null()) {
          continue;
        }
        dirty = evaluatePath(fieldValue, out, nextStyle, tokens, tokenIdx + 1);
      }
      return dirty;
    }

    return false;
  }

  template <typename T>
  bool evaluateWildcardToken(
      T& val,
      std::string& out,
      WriteStyle style,
      const std::vector<PathToken>& tokens,
      size_t tokenIdx,
      simdjson::ondemand::json_type type) {
    if (type != simdjson::ondemand::json_type::array || val.is_null()) {
      return false;
    }

    simdjson::ondemand::array arr;
    throwIfError(val.get_array().get(arr));

    bool isDoubleWildcard =
        (tokenIdx + 1 < tokens.size() &&
         std::holds_alternative<WildcardToken>(tokens[tokenIdx + 1]));

    if (isDoubleWildcard) {
      std::string buffer;
      bool dirty = false;
      for (auto elem : arr) {
        throwIfError(elem.error());
        auto elemVal = elem.value_unsafe();
        dirty |= evaluatePath(
            elemVal, buffer, WriteStyle::kFlatten, tokens, tokenIdx + 2);
      }
      if (dirty) {
        appendCommaIfNeeded(out);
        out += '[';
        out += buffer;
        out += ']';
      }
      return dirty;
    }

    if (style != WriteStyle::kQuoted) {
      WriteStyle nextStyle = (style == WriteStyle::kFlatten)
          ? WriteStyle::kFlatten
          : WriteStyle::kQuoted;
      std::string buffer;
      int dirty = 0;
      for (auto elem : arr) {
        throwIfError(elem.error());
        auto elemVal = elem.value_unsafe();
        dirty += evaluatePath(elemVal, buffer, nextStyle, tokens, tokenIdx + 1);
      }
      if (dirty > 1) {
        appendCommaIfNeeded(out);
        out += '[';
        out += buffer;
        out += ']';
      } else if (dirty == 1) {
        appendCommaIfNeeded(out);
        out += buffer;
      }
      return dirty > 0;
    }

    appendCommaIfNeeded(out);
    out += '[';
    int dirty = 0;
    for (auto elem : arr) {
      throwIfError(elem.error());
      auto elemVal = elem.value_unsafe();
      dirty +=
          evaluatePath(elemVal, out, WriteStyle::kQuoted, tokens, tokenIdx + 1);
    }
    out += ']';
    return dirty > 0;
  }

  // Recursively evaluates the normalized path against the current JSON node.
  // The recursion has two stages:
  // 1. If all path tokens have been consumed, emit the current subtree using
  //    Spark-compatible write rules.
  // 2. Otherwise dispatch to the current token handler:
  //    NamedToken scans object fields in order, IndexToken scans array/object
  //    entries in order, and WildcardToken expands array elements with Spark's
  //    flattening and single-element elision behavior.
  // Any simdjson parse failure while consuming the current container throws and
  // is converted to "no result" by evaluate().
  template <typename T>
  bool evaluatePath(
      T& val,
      std::string& out,
      WriteStyle style,
      const std::vector<PathToken>& tokens,
      size_t tokenIdx) {
    if (tokenIdx >= tokens.size()) {
      return writeMatchedValue(val, out, style, tokens, tokenIdx);
    }

    const auto& token = tokens[tokenIdx];
    simdjson::ondemand::json_type type;
    throwIfError(val.type().get(type));

    if (std::holds_alternative<NamedToken>(token)) {
      return evaluateNamedToken(val, out, style, tokens, tokenIdx, type);
    }

    if (std::holds_alternative<IndexToken>(token)) {
      return evaluateIndexToken(val, out, style, tokens, tokenIdx, type);
    }

    return evaluateWildcardToken(val, out, style, tokens, tokenIdx, type);
  }

  std::vector<PathToken> tokens_;
};

} // namespace detail

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
      jsonPath_ = pathNormalizer_.normalize(*jsonPath);
      if (hasWildcard(jsonPath_.value())) {
        evaluator_.emplace(detail::GetJsonObjectEvaluator(jsonPath_.value()));
      }
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
    const auto formattedJsonPath = jsonPath_.has_value()
        ? jsonPath_.value()
        : pathNormalizer_.normalize(jsonPath);
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
    if (hasWildcard(formattedJsonPath)) {
      std::optional<detail::GetJsonObjectEvaluator> localEvaluator;
      auto& evaluator = jsonPath_.has_value()
          ? evaluator_.value()
          : localEvaluator.emplace(formattedJsonPath);
      std::string out;
      bool matched = evaluator.evaluate(jsonDoc, out);
      if (!matched || out.empty()) {
        return false;
      }
      result.append(out);
      return true;
    }

    try {
      // Can return error result or throw exception possibly.
      auto rawResult = jsonDoc.at_path(formattedJsonPath);
      if (rawResult.error()) {
        return false;
      }
      if (rawResult.type() == simdjson::ondemand::json_type::null) {
        return false;
      }

      std::string out;
      if (!detail::appendSparkFormattedValue(rawResult, out, false)) {
        return false;
      }
      result.append(out);
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

  FOLLY_ALWAYS_INLINE bool hasWildcard(const std::string& path) {
    return path.find("[*]") != std::string::npos;
  }

  // Used for constant json path.
  std::optional<std::string> jsonPath_;

  // Cached evaluator for constant paths.
  std::optional<detail::GetJsonObjectEvaluator> evaluator_;

  detail::JsonPathNormalizer pathNormalizer_;
};

} // namespace facebook::velox::functions::sparksql
