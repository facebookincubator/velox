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
#include "velox/functions/prestosql/Map.h"
#include "velox/functions/prestosql/MapAppend.h"
#include "velox/functions/prestosql/MapExcept.h"
#include "velox/functions/prestosql/MapFunctions.h"
#include "velox/functions/prestosql/MapIntersect.h"
#include "velox/functions/prestosql/MapKeysByTopNValues.h"
#include "velox/functions/prestosql/MapNormalize.h"
#include "velox/functions/prestosql/MapSubset.h"
#include "velox/functions/prestosql/MapTopN.h"
#include "velox/functions/prestosql/MapTopNKeys.h"
#include "velox/functions/prestosql/MapTopNValues.h"
#include "velox/functions/prestosql/MultimapFromEntries.h"
#include "velox/functions/prestosql/RemapKeys.h"

namespace facebook::velox::functions {

namespace {
template <typename T>
void registerRemapKeysPrimitive(const std::string& prefix) {
  registerFunction<
      ParameterBinder<RemapKeysPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>,
      Array<T>>({prefix + "remap_keys"});
}

void registerRemapKeys(const std::string& prefix) {
  registerRemapKeysPrimitive<bool>(prefix);
  registerRemapKeysPrimitive<int8_t>(prefix);
  registerRemapKeysPrimitive<int16_t>(prefix);
  registerRemapKeysPrimitive<int32_t>(prefix);
  registerRemapKeysPrimitive<int64_t>(prefix);
  registerRemapKeysPrimitive<float>(prefix);
  registerRemapKeysPrimitive<double>(prefix);
  registerRemapKeysPrimitive<Timestamp>(prefix);
  registerRemapKeysPrimitive<Date>(prefix);

  registerFunction<
      RemapKeysVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>,
      Array<Varchar>>({prefix + "remap_keys"});

  registerFunction<
      RemapKeysFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>,
      Array<Generic<T1>>>({prefix + "remap_keys"});
}

template <typename T>
void registerMapIntersectPrimitive(const std::string& prefix) {
  registerFunction<
      ParameterBinder<MapIntersectPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({prefix + "map_intersect"});
}

void registerMapIntersect(const std::string& prefix) {
  registerMapIntersectPrimitive<bool>(prefix);
  registerMapIntersectPrimitive<int8_t>(prefix);
  registerMapIntersectPrimitive<int16_t>(prefix);
  registerMapIntersectPrimitive<int32_t>(prefix);
  registerMapIntersectPrimitive<int64_t>(prefix);
  registerMapIntersectPrimitive<float>(prefix);
  registerMapIntersectPrimitive<double>(prefix);
  registerMapIntersectPrimitive<Timestamp>(prefix);
  registerMapIntersectPrimitive<Date>(prefix);

  registerFunction<
      MapIntersectVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({prefix + "map_intersect"});

  registerFunction<
      MapIntersectFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({prefix + "map_intersect"});
}

template <typename T>
void registerMapExceptPrimitive(const std::string& prefix) {
  registerFunction<
      ParameterBinder<MapExceptPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({prefix + "map_except"});
}

void registerMapExcept(const std::string& prefix) {
  registerMapExceptPrimitive<bool>(prefix);
  registerMapExceptPrimitive<int8_t>(prefix);
  registerMapExceptPrimitive<int16_t>(prefix);
  registerMapExceptPrimitive<int32_t>(prefix);
  registerMapExceptPrimitive<int64_t>(prefix);
  registerMapExceptPrimitive<float>(prefix);
  registerMapExceptPrimitive<double>(prefix);
  registerMapExceptPrimitive<Timestamp>(prefix);
  registerMapExceptPrimitive<Date>(prefix);

  registerFunction<
      MapExceptVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({prefix + "map_except"});

  registerFunction<
      MapExceptFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({prefix + "map_except"});
}

template <typename Key, typename Value>
void registerMapAppendPrimitive(const std::string& prefix) {
  registerFunction<
      ParameterBinder<MapAppendPrimitiveFunction, Key, Value>,
      Map<Key, Value>,
      Map<Key, Value>,
      Array<Key>,
      Array<Value>>({prefix + "map_append"});
}

template <typename Value>
void registerMapAppendVarchar(const std::string& prefix) {
  registerFunction<
      ParameterBinder<MapAppendVarcharFunction, Value>,
      Map<Varchar, Value>,
      Map<Varchar, Value>,
      Array<Varchar>,
      Array<Value>>({prefix + "map_append"});
}

void registerMapAppend(const std::string& prefix) {
  // Register for all combinations of primitive key and value types
  registerMapAppendPrimitive<bool, bool>(prefix);
  registerMapAppendPrimitive<bool, int8_t>(prefix);
  registerMapAppendPrimitive<bool, int16_t>(prefix);
  registerMapAppendPrimitive<bool, int32_t>(prefix);
  registerMapAppendPrimitive<bool, int64_t>(prefix);
  registerMapAppendPrimitive<bool, float>(prefix);
  registerMapAppendPrimitive<bool, double>(prefix);
  registerMapAppendPrimitive<bool, Timestamp>(prefix);
  registerMapAppendPrimitive<bool, Date>(prefix);

  registerMapAppendPrimitive<int8_t, bool>(prefix);
  registerMapAppendPrimitive<int8_t, int8_t>(prefix);
  registerMapAppendPrimitive<int8_t, int16_t>(prefix);
  registerMapAppendPrimitive<int8_t, int32_t>(prefix);
  registerMapAppendPrimitive<int8_t, int64_t>(prefix);
  registerMapAppendPrimitive<int8_t, float>(prefix);
  registerMapAppendPrimitive<int8_t, double>(prefix);
  registerMapAppendPrimitive<int8_t, Timestamp>(prefix);
  registerMapAppendPrimitive<int8_t, Date>(prefix);

  registerMapAppendPrimitive<int16_t, bool>(prefix);
  registerMapAppendPrimitive<int16_t, int8_t>(prefix);
  registerMapAppendPrimitive<int16_t, int16_t>(prefix);
  registerMapAppendPrimitive<int16_t, int32_t>(prefix);
  registerMapAppendPrimitive<int16_t, int64_t>(prefix);
  registerMapAppendPrimitive<int16_t, float>(prefix);
  registerMapAppendPrimitive<int16_t, double>(prefix);
  registerMapAppendPrimitive<int16_t, Timestamp>(prefix);
  registerMapAppendPrimitive<int16_t, Date>(prefix);

  registerMapAppendPrimitive<int32_t, bool>(prefix);
  registerMapAppendPrimitive<int32_t, int8_t>(prefix);
  registerMapAppendPrimitive<int32_t, int16_t>(prefix);
  registerMapAppendPrimitive<int32_t, int32_t>(prefix);
  registerMapAppendPrimitive<int32_t, int64_t>(prefix);
  registerMapAppendPrimitive<int32_t, float>(prefix);
  registerMapAppendPrimitive<int32_t, double>(prefix);
  registerMapAppendPrimitive<int32_t, Timestamp>(prefix);
  registerMapAppendPrimitive<int32_t, Date>(prefix);

  registerMapAppendPrimitive<int64_t, bool>(prefix);
  registerMapAppendPrimitive<int64_t, int8_t>(prefix);
  registerMapAppendPrimitive<int64_t, int16_t>(prefix);
  registerMapAppendPrimitive<int64_t, int32_t>(prefix);
  registerMapAppendPrimitive<int64_t, int64_t>(prefix);
  registerMapAppendPrimitive<int64_t, float>(prefix);
  registerMapAppendPrimitive<int64_t, double>(prefix);
  registerMapAppendPrimitive<int64_t, Timestamp>(prefix);
  registerMapAppendPrimitive<int64_t, Date>(prefix);

  registerMapAppendPrimitive<float, bool>(prefix);
  registerMapAppendPrimitive<float, int8_t>(prefix);
  registerMapAppendPrimitive<float, int16_t>(prefix);
  registerMapAppendPrimitive<float, int32_t>(prefix);
  registerMapAppendPrimitive<float, int64_t>(prefix);
  registerMapAppendPrimitive<float, float>(prefix);
  registerMapAppendPrimitive<float, double>(prefix);
  registerMapAppendPrimitive<float, Timestamp>(prefix);
  registerMapAppendPrimitive<float, Date>(prefix);

  registerMapAppendPrimitive<double, bool>(prefix);
  registerMapAppendPrimitive<double, int8_t>(prefix);
  registerMapAppendPrimitive<double, int16_t>(prefix);
  registerMapAppendPrimitive<double, int32_t>(prefix);
  registerMapAppendPrimitive<double, int64_t>(prefix);
  registerMapAppendPrimitive<double, float>(prefix);
  registerMapAppendPrimitive<double, double>(prefix);
  registerMapAppendPrimitive<double, Timestamp>(prefix);
  registerMapAppendPrimitive<double, Date>(prefix);

  registerMapAppendPrimitive<Timestamp, bool>(prefix);
  registerMapAppendPrimitive<Timestamp, int8_t>(prefix);
  registerMapAppendPrimitive<Timestamp, int16_t>(prefix);
  registerMapAppendPrimitive<Timestamp, int32_t>(prefix);
  registerMapAppendPrimitive<Timestamp, int64_t>(prefix);
  registerMapAppendPrimitive<Timestamp, float>(prefix);
  registerMapAppendPrimitive<Timestamp, double>(prefix);
  registerMapAppendPrimitive<Timestamp, Timestamp>(prefix);
  registerMapAppendPrimitive<Timestamp, Date>(prefix);

  registerMapAppendPrimitive<Date, bool>(prefix);
  registerMapAppendPrimitive<Date, int8_t>(prefix);
  registerMapAppendPrimitive<Date, int16_t>(prefix);
  registerMapAppendPrimitive<Date, int32_t>(prefix);
  registerMapAppendPrimitive<Date, int64_t>(prefix);
  registerMapAppendPrimitive<Date, float>(prefix);
  registerMapAppendPrimitive<Date, double>(prefix);
  registerMapAppendPrimitive<Date, Timestamp>(prefix);
  registerMapAppendPrimitive<Date, Date>(prefix);

  // Register varchar key with various value types
  registerMapAppendVarchar<bool>(prefix);
  registerMapAppendVarchar<int8_t>(prefix);
  registerMapAppendVarchar<int16_t>(prefix);
  registerMapAppendVarchar<int32_t>(prefix);
  registerMapAppendVarchar<int64_t>(prefix);
  registerMapAppendVarchar<float>(prefix);
  registerMapAppendVarchar<double>(prefix);
  registerMapAppendVarchar<Timestamp>(prefix);
  registerMapAppendVarchar<Date>(prefix);
  registerMapAppendVarchar<Varchar>(prefix);

  // Register generic implementation for complex types
  registerFunction<
      MapAppendFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>,
      Array<Generic<T2>>>({prefix + "map_append"});
}

void registerMapRemoveNullValues(const std::string& prefix) {
  registerFunction<
      MapRemoveNullValues,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>>({prefix + "map_remove_null_values"});
}

void registerMapKeyExists(const std::string& prefix) {
  registerFunction<
      MapKeyExists,
      bool,
      Map<Generic<T1>, Generic<T2>>,
      Generic<T1>>({prefix + "map_key_exists"});
}

} // namespace

void registerMapFunctions(const std::string& prefix) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_filter, prefix + "map_filter");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_transform_keys, prefix + "transform_keys");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_transform_values, prefix + "transform_values");
  registerMapFunction(prefix + "map", false /*allowDuplicateKeys*/);
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_entries, prefix + "map_entries");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_map_from_entries, prefix + "map_from_entries");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_keys, prefix + "map_keys");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_values, prefix + "map_values");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_zip_with, prefix + "map_zip_with");

  VELOX_REGISTER_VECTOR_FUNCTION(udf_all_keys_match, prefix + "all_keys_match");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_any_keys_match, prefix + "any_keys_match");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_no_keys_match, prefix + "no_keys_match");

  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_any_values_match, prefix + "any_values_match");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_no_values_match, prefix + "no_values_match");

  registerMapConcatFunction(prefix + "map_concat");

  registerFunction<
      MultimapFromEntriesFunction,
      Map<Generic<T1>, Array<Generic<T2>>>,
      Array<Row<Generic<T1>, Generic<T2>>>>({prefix + "multimap_from_entries"});

  registerFunction<
      MapTopNFunction,
      Map<Orderable<T1>, Orderable<T2>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_top_n"});

  registerFunction<
      MapTopNKeysFunction,
      Array<Orderable<T1>>,
      Map<Orderable<T1>, Generic<T2>>,
      int64_t>({prefix + "map_top_n_keys"});

  registerFunction<
      MapKeysByTopNValuesFunction,
      Array<Orderable<T1>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_keys_by_top_n_values"});

  registerFunction<
      MapTopNValuesFunction,
      Array<Orderable<T2>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_top_n_values"});

  registerMapSubset(prefix + "map_subset");

  registerRemapKeys(prefix);

  registerMapIntersect(prefix);

  registerMapExcept(prefix);

  registerMapAppend(prefix);

  registerMapRemoveNullValues(prefix);

  registerMapKeyExists(prefix);

  registerFunction<
      MapNormalizeFunction,
      Map<Varchar, double>,
      Map<Varchar, double>>({prefix + "map_normalize"});
}

void registerMapAllowingDuplicates(
    const std::string& name,
    const std::string& prefix) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_map_allow_duplicates, prefix + name);
}
} // namespace facebook::velox::functions
