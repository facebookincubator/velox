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

#include <cstring>

#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/XPathUtil.h"

namespace facebook::velox::functions::sparksql {

/// NOTE: Spark rejects non-foldable (non-literal) path arguments at analysis
/// time. Velox accepts dynamic paths - this is a minor divergence but safe
/// since Gluten only passes constant path literals.

/// xpath_boolean(xml, path) -> boolean
template <typename T>
struct XPathBooleanFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<bool>& result,
      const arg_type<Varchar>& xml,
      const arg_type<Varchar>& path) {
    if (xml.empty() || path.empty()) {
      return false;
    }
    auto value =
        xpath::evalBoolean(xml.data(), xml.size(), path.data(), path.size());
    if (!value) {
      return false;
    }
    result = *value;
    return true;
  }
};

/// xpath_string(xml, path) -> varchar
template <typename T>
struct XPathStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& xml,
      const arg_type<Varchar>& path) {
    if (xml.empty() || path.empty()) {
      return false;
    }
    auto value =
        xpath::evalString(xml.data(), xml.size(), path.data(), path.size());
    if (!value) {
      return false;
    }
    result.resize(value->size());
    if (!value->empty()) {
      std::memcpy(result.data(), value->data(), value->size());
    }
    return true;
  }
};

} // namespace facebook::velox::functions::sparksql
