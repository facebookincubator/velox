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

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct JsonArrayLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(int32_t& len, const arg_type<Json>& json) {
    simdjson::ondemand::document jsonDoc;

    simdjson::padded_string paddedJson(json.data(), json.size());
    if (simdjsonParse(paddedJson).get(jsonDoc)) {
      return false;
    }
    if (jsonDoc.type().error()) {
      return false;
    }

    if (jsonDoc.type() != simdjson::ondemand::json_type::array) {
      return false;
    }

    size_t numElements;
    if (jsonDoc.count_elements().get(numElements)) {
      return false;
    }

    len = numElements;
    return true;
  }
};
}  // namespace facebook::velox::functions::sparksql

