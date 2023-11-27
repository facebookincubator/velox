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

#include "velox/functions/prestosql/SIMDJsonFunctions.h"

using namespace simdjson;

namespace facebook::velox::functions::sparksql {

template <typename T>
struct SIMDGetJsonObjectFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  std::optional<std::string> formattedJsonPath_;

  // ASCII input always produces ASCII result.
  static constexpr bool is_default_ascii_behavior = true;

  // Makes a conversion from spark's json path, e.g., converts
  // "$.a.b" to "/a/b".
  FOLLY_ALWAYS_INLINE std::string getFormattedJsonPath(
      const arg_type<Varchar>& jsonPath) {
    char formattedJsonPath[jsonPath.size() + 1];
    int j = 0;
    for (int i = 0; i < jsonPath.size(); i++) {
      if (jsonPath.data()[i] == '$' || jsonPath.data()[i] == ']' ||
          jsonPath.data()[i] == '\'') {
        continue;
      } else if (jsonPath.data()[i] == '[' || jsonPath.data()[i] == '.') {
        formattedJsonPath[j] = '/';
        j++;
      } else {
        formattedJsonPath[j] = jsonPath.data()[i];
        j++;
      }
    }
    formattedJsonPath[j] = '\0';
    return std::string(formattedJsonPath, j + 1);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*json*/,
      const arg_type<Varchar>* jsonPath) {
    if (jsonPath != nullptr) {
      formattedJsonPath_ = getFormattedJsonPath(*jsonPath);
    }
  }

  FOLLY_ALWAYS_INLINE simdjson::error_code extractStringResult(
      simdjson_result<ondemand::value> rawResult,
      out_type<Varchar>& result) {
    simdjson::error_code error;
    std::stringstream ss;
    switch (rawResult.type()) {
      // For number and bool types, we need to explicitly get the value
      // for specific types instead of using `ss << rawResult`. Thus, we
      // can make simdjson's internal parsing position moved and then we
      // can check the validity of ending character.
      case ondemand::json_type::number: {
        switch (rawResult.get_number_type()) {
          case ondemand::number_type::unsigned_integer: {
            uint64_t numberResult;
            error = rawResult.get_uint64().get(numberResult);
            if (!error) {
              ss << numberResult;
              result.append(ss.str());
            }
            return error;
          }
          case ondemand::number_type::signed_integer: {
            int64_t numberResult;
            error = rawResult.get_int64().get(numberResult);
            if (!error) {
              ss << numberResult;
              result.append(ss.str());
            }
            return error;
          }
          case ondemand::number_type::floating_point_number: {
            double numberResult;
            error = rawResult.get_double().get(numberResult);
            if (!error) {
              ss << numberResult;
              result.append(ss.str());
            }
            return error;
          }
          default:
            VELOX_UNREACHABLE();
        }
      }
      case ondemand::json_type::boolean: {
        bool boolResult;
        error = rawResult.get_bool().get(boolResult);
        if (!error) {
          result.append(boolResult ? "true" : "false");
        }
        return error;
      }
      case ondemand::json_type::string: {
        std::string_view stringResult;
        error = rawResult.get_string().get(stringResult);
        result.append(stringResult);
        return error;
      }
      case ondemand::json_type::object: {
        // For nested case, e.g., for "{"my": {"hello": 10}}", "$.my" will
        // return an object type.
        ss << rawResult;
        result.append(ss.str());
        return SUCCESS;
      }
      case ondemand::json_type::array: {
        ss << rawResult;
        result.append(ss.str());
        return SUCCESS;
      }
      default: {
        return UNSUPPORTED_ARCHITECTURE;
      }
    }
  }

  // This is a simple validation by checking whether the obtained result is
  // followed by valid char. Because ondemand parsing we are using ignores json
  // format validation for characters following the current parsing position.
  bool isValidEndingCharacter(const char* currentPos) {
    char endingChar = *currentPos;
    if (endingChar == ',' || endingChar == '}' || endingChar == ']') {
      return true;
    }
    if (endingChar == ' ' || endingChar == '\r' || endingChar == '\n' ||
        endingChar == '\t') {
      // These chars can be prior to a valid ending char.
      return isValidEndingCharacter(currentPos++);
    }
    return false;
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& json,
      const arg_type<Varchar>& jsonPath) {
    ParserContext ctx(json.data(), json.size());
    try {
      ctx.parseDocument();
      simdjson_result<ondemand::value> rawResult =
          formattedJsonPath_.has_value()
          ? ctx.jsonDoc.at_pointer(formattedJsonPath_.value().data())
          : ctx.jsonDoc.at_pointer(getFormattedJsonPath(jsonPath).data());
      // Field not found.
      if (rawResult.error() == NO_SUCH_FIELD) {
        return false;
      }
      auto error = extractStringResult(rawResult, result);
      if (error) {
        return false;
      }
    } catch (simdjson_error& e) {
      return false;
    }

    const char* currentPos;
    ctx.jsonDoc.current_location().get(currentPos);
    return isValidEndingCharacter(currentPos);
  }
};

} // namespace facebook::velox::functions::sparksql
