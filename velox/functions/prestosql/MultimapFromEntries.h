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

#include <folly/container/F14Map.h>
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"

namespace facebook::velox::functions {

template <typename TExec>
struct MultimapFromEntriesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Map<Generic<T1>, Array<Generic<T2>>>>& out,
      const arg_type<Array<Row<Generic<T1>, Generic<T2>>>>& inputArray) {
    // Use std::unordered_map to ensure deterministic order of keys in the
    // result. Without ensuring deterministic order of keys, the results of
    // expressions like map_keys(multimap_from_entries(...)) will be
    // non-deterministic and trigger Fuzzer failures. F14Map is faster, but in
    // debug builds it returns keys in non-deterministic order (on purpose).
    std::unordered_map<
        exec::GenericView,
        std::vector<std::optional<exec::GenericView>>>
        keyValuesMap;

    for (const auto& entry : inputArray.skipNulls()) {
      const auto& key = entry.template at<0>();
      const auto& value = entry.template at<1>();

      VELOX_USER_CHECK(key.has_value(), "map key cannot be null")

      auto result = keyValuesMap.insert({key.value(), {}});
      result.first->second.push_back(value);
    }

    for (const auto& [key, values] : keyValuesMap) {
      auto [keyWriter, valueWriter] = out.add_item();
      keyWriter.copy_from(key);
      for (const auto& value : values) {
        if (value.has_value()) {
          valueWriter.add_item().copy_from(value.value());
        } else {
          valueWriter.add_null();
        }
      }
    }
  }
};

} // namespace facebook::velox::functions
