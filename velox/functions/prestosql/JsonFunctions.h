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
#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/prestosql/json/JsonExtractor.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions {

template <typename T>
struct IsJsonScalarFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(bool& result, const arg_type<Json>& json) {
    auto parsedJson = folly::parseJson(json);
    result = parsedJson.isNumber() || parsedJson.isString() ||
        parsedJson.isBool() || parsedJson.isNull();
  }
};

// jsonExtractScalar(json, json_path) -> varchar
// Current implementation support UTF-8 in json, but not in json_path.
// Like jsonExtract(), but returns the result value as a string (as opposed
// to being encoded as JSON). The value referenced by json_path must be a scalar
// (boolean, number or string)
template <typename T>
struct JsonExtractScalarFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    const folly::StringPiece& jsonStringPiece = json;
    const folly::StringPiece& jsonPathStringPiece = jsonPath;
    auto extractResult =
        jsonExtractScalar(jsonStringPiece, jsonPathStringPiece);
    if (extractResult.hasValue()) {
      UDFOutputString::assign(result, *extractResult);
      return true;

    } else {
      return false;
    }
  }
};

template <typename T>
struct JsonArrayLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int64_t& result, const arg_type<Json>& json) {
    auto parsedJson = folly::parseJson(json);
    if (!parsedJson.isArray()) {
      return false;
    }

    result = parsedJson.size();
    return true;
  }
};

template <typename T>
struct JsonArrayContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool
  call(bool& result, const arg_type<Json>& json, const TInput& value) {
    auto parsedJson = folly::parseJson(json);
    if (!parsedJson.isArray()) {
      return false;
    }

    result = false;
    for (const auto& v : parsedJson) {
      if constexpr (std::is_same_v<TInput, bool>) {
        if (v.isBool() && v == value) {
          result = true;
          break;
        }
      } else if constexpr (std::is_same_v<TInput, int64_t>) {
        if (v.isInt() && v == value) {
          result = true;
          break;
        }
      } else if constexpr (std::is_same_v<TInput, double>) {
        if (v.isDouble() && v == value) {
          result = true;
          break;
        }
      } else {
        if (v.isString() && v == value) {
          result = true;
          break;
        }
      }
    }
    return true;
  }
};

template <typename T>
struct JsonSizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    const folly::StringPiece& jsonStringPiece = json;
    const folly::StringPiece& jsonPathStringPiece = jsonPath;
    auto extractResult = jsonExtract(jsonStringPiece, jsonPathStringPiece);
    if (!extractResult.has_value()) {
      return false;
    }
    // The size of the object or array is the number of members, otherwise the
    // size is zero
    if (extractResult->isArray() || extractResult->isObject()) {
      result = extractResult->size();
    } else {
      result = 0;
    }

    return true;
  }
};

} // namespace facebook::velox::functions
