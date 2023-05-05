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
#include "velox/functions/prestosql/json/JsonPathTokenizer.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions {

struct ParserContext {
 public:
  explicit ParserContext() noexcept;
  explicit ParserContext(const char* data, size_t length) noexcept
      : padded_json(data, length) {}
  void parseElement(){
    jsonEle = domParser.parse(padded_json);
  }
  void parseDocument(){
    jsonDoc = ondemandParser.iterate(padded_json);
  }
  simdjson::dom::element jsonEle;
  simdjson::ondemand::document jsonDoc;

 private:
  simdjson::padded_string padded_json;
  simdjson::dom::parser domParser;
  simdjson::ondemand::parser ondemandParser;
};

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
    // TODO
    return false;
  }
};

template <typename T>
struct SIMDJsonParseFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Json>& result,
      const arg_type<Varchar>& json) {
    // TODO
  }
};

template <typename T>
struct SIMDJsonExtractFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Json>& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    // TODO
    return false;
  }
};

template <typename T>
struct SIMDJsonExtractScalarFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    // TODO
    return false;
  }
};

template <typename T>
struct SIMDJsonArrayLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int64_t& len, const arg_type<Json>& json) {
    // TODO
    return false;
  }
};

template <typename T>
struct SIMDJsonSizeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    // TODO
    return false;
  }
};

} // namespace facebook::velox::functions
