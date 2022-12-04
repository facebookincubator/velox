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
#include <string>
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/MapConcat.h"
#include "velox/functions/prestosql/MapFunctions.h"

namespace facebook::velox::functions {

template <typename K, typename V>
inline void registerMapFromEntriesFunctions() {
  registerFunction<
      ParameterBinder<MapFromEntriesFunction, K, V>,
      Map<K, V>,
      Array<Row<K, V>>>({"map_from_entries"});
}

template <typename V>
void registerMapFromEntriesWithFixedKeyType() {
  registerMapFromEntriesFunctions<int8_t, V>();
  registerMapFromEntriesFunctions<int16_t, V>();
  registerMapFromEntriesFunctions<int32_t, V>();
  registerMapFromEntriesFunctions<int64_t, V>();
  registerMapFromEntriesFunctions<float, V>();
  registerMapFromEntriesFunctions<double, V>();
  registerMapFromEntriesFunctions<bool, V>();
  registerMapFromEntriesFunctions<Varchar, V>();
  registerMapFromEntriesFunctions<Timestamp, V>();
  registerMapFromEntriesFunctions<Date, V>();
}

void registerMapFromEntries() {
  registerMapFromEntriesWithFixedKeyType<int8_t>();
  registerMapFromEntriesWithFixedKeyType<int16_t>();
  registerMapFromEntriesWithFixedKeyType<int32_t>();
  registerMapFromEntriesWithFixedKeyType<int64_t>();
  registerMapFromEntriesWithFixedKeyType<float>();
  registerMapFromEntriesWithFixedKeyType<double>();
  registerMapFromEntriesWithFixedKeyType<bool>();
  registerMapFromEntriesWithFixedKeyType<Varchar>();
  registerMapFromEntriesWithFixedKeyType<Timestamp>();
  registerMapFromEntriesWithFixedKeyType<Date>();
}

void registerMapFunctions() {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_filter, "map_filter");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_transform_keys, "transform_keys");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_transform_values, "transform_values");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map, "map");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_entries, "map_entries");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_keys, "map_keys");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_values, "map_values");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_zip_with, "map_zip_with");

  registerMapConcatFunction("map_concat");
  registerMapFromEntries();
}

void registerMapAllowingDuplicates(const std::string& name) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_allow_duplicates, name);
}
} // namespace facebook::velox::functions
