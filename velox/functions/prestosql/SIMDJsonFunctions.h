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
#include "simdjson.h"
#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/prestosql/json/JsonExtractor.h"
#include "velox/functions/prestosql/json/SimdJsonExtractor.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions {

template <typename T>
struct SIMDIsJsonScalarFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(bool& result, const arg_type<Json>& json) {
    ParserContext ctx(json.data(), json.size());
    result = false;

    ctx.parseDocument();
    if (ctx.jsonDoc.type() == simdjson::ondemand::json_type::number ||
        ctx.jsonDoc.type() == simdjson::ondemand::json_type::string ||
        ctx.jsonDoc.type() == simdjson::ondemand::json_type::boolean ||
        ctx.jsonDoc.type() == simdjson::ondemand::json_type::null) {
      result = true;
    }
  }
};

template <typename T>
struct SIMDJsonArrayContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool
  call(bool& result, const arg_type<Json>& json, const TInput& value) {
    ParserContext ctx(json.data(), json.size());
    result = false;

    try {
      ctx.parseDocument();
    } catch (simdjson::simdjson_error& e) {
      return false;
    }

    if (ctx.jsonDoc.type() != simdjson::ondemand::json_type::array) {
      return false;
    }

    try {
      for (auto&& v : ctx.jsonDoc) {
        if constexpr (std::is_same_v<TInput, bool>) {
          if (v.type() == simdjson::ondemand::json_type::boolean &&
              v.get_bool() == value) {
            result = true;
            break;
          }
        } else if constexpr (std::is_same_v<TInput, int64_t>) {
          if (v.type() == simdjson::ondemand::json_type::number &&
              ((v.get_number_type() ==
                    simdjson::ondemand::number_type::signed_integer &&
                v.get_int64() == value) ||
               (v.get_number_type() ==
                    simdjson::ondemand::number_type::unsigned_integer &&
                v.get_uint64() == value))) {
            result = true;
            break;
          }
        } else if constexpr (std::is_same_v<TInput, double>) {
          if (v.type() == simdjson::ondemand::json_type::number &&
              v.get_number_type() ==
                  simdjson::ondemand::number_type::floating_point_number &&
              v.get_double() == value) {
            result = true;
            break;
          }
        } else {
          if (v.type() == simdjson::ondemand::json_type::string) {
            std::string_view rlt = v.get_string();
            std::string str_value = value.getString();
            if (rlt.compare(str_value) == 0) {
              result = true;
              break;
            }
          }
        }
      }
    } catch (simdjson::simdjson_error& e) {
      return false;
    }
    return true;
  }
};

template <typename T>
struct SIMDJsonParseFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to the first input strings parameter buffer.
  static constexpr int32_t reuse_strings_from_arg = 0;

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& json) {
    ParserContext ctx(json.data(), json.size());
    ctx.parseElement();
    std::string rlt = simdjson::to_string(ctx.jsonEle);
    result.setNoCopy(facebook::velox::StringView(rlt));
    return true;
  }
};

template <typename T>
struct SIMDJsonExtractScalarFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    std::string jsonPathStr = jsonPath;
    bool retVal = false;

    auto extractResult = simdJsonExtractScalar(json, jsonPathStr);

    if (extractResult.has_value()) {
      UDFOutputString::assign(result, extractResult.value());
      retVal = true;
    }
    return retVal;
  }
};

template <typename T>
struct SIMDJsonArrayLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int64_t& len, const arg_type<Json>& json) {
    ParserContext ctx(json.data(), json.size());
    bool result = false;

    try {
      ctx.parseDocument();
    } catch (simdjson::simdjson_error& e) {
      throw e;
    }

    if (ctx.jsonDoc.type() != simdjson::ondemand::json_type::array) {
      return result;
    }

    len = 0;
    try {
      for (auto&& v : ctx.jsonDoc) {
        len++;
      }
      result = true;
    } catch (simdjson::simdjson_error& e) {
      throw e;
    }
    return result;
  }
};

template <typename T>
struct SIMDJsonSizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    ParserContext ctx(json.data(), json.size());
    std::string jsonPathStr = jsonPath;
    result = 0;

    try {
      ctx.parseDocument();
      auto rlt = simdJsonSize(json, jsonPathStr);
      if (rlt.has_value()) {
        result = rlt.value();
      } else {
        return false;
      }
    } catch (simdjson::simdjson_error& e) {
      return false;
    }

    return true;
  }
};

} // namespace facebook::velox::functions
