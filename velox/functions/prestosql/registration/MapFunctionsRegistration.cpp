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
#include "velox/functions/lib/MapFromEntries.h"
#include "velox/functions/prestosql/Map.h"
#include "velox/functions/prestosql/MapAppend.h"
#include "velox/functions/prestosql/MapExcept.h"
#include "velox/functions/prestosql/MapFunctions.h"
#include "velox/functions/prestosql/MapIntersect.h"
#include "velox/functions/prestosql/MapKeysByTopNValues.h"
#include "velox/functions/prestosql/MapKeysOverlap.h"
#include "velox/functions/prestosql/MapNormalize.h"
#include "velox/functions/prestosql/MapSubset.h"
#include "velox/functions/prestosql/MapSubsetKeyInRange.h"
#include "velox/functions/prestosql/MapTopN.h"
#include "velox/functions/prestosql/MapTopNKeys.h"
#include "velox/functions/prestosql/MapTopNValues.h"
#include "velox/functions/prestosql/MapTrimValues.h"
#include "velox/functions/prestosql/MapUpdate.h"
#include "velox/functions/prestosql/MapValuesInRange.h"
#include "velox/functions/prestosql/MultimapFromEntries.h"
#include "velox/functions/prestosql/RemapKeys.h"

namespace facebook::velox::functions {

namespace {
template <typename T>
void registerRemapKeysPrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<RemapKeysPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>,
      Array<T>>({prefix + "remap_keys"}, true, defaultOwner);
}

void registerRemapKeys(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerRemapKeysPrimitive<bool>(prefix, defaultOwner);
  registerRemapKeysPrimitive<int8_t>(prefix, defaultOwner);
  registerRemapKeysPrimitive<int16_t>(prefix, defaultOwner);
  registerRemapKeysPrimitive<int32_t>(prefix, defaultOwner);
  registerRemapKeysPrimitive<int64_t>(prefix, defaultOwner);
  registerRemapKeysPrimitive<float>(prefix, defaultOwner);
  registerRemapKeysPrimitive<double>(prefix, defaultOwner);
  registerRemapKeysPrimitive<Timestamp>(prefix, defaultOwner);
  registerRemapKeysPrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      RemapKeysVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>,
      Array<Varchar>>({prefix + "remap_keys"}, {}, true, defaultOwner);

  registerFunction<
      RemapKeysFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>,
      Array<Generic<T1>>>({prefix + "remap_keys"}, {}, true, defaultOwner);
}

template <typename T>
void registerMapIntersectPrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapIntersectPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({prefix + "map_intersect"}, true, defaultOwner);
}

void registerMapIntersect(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerMapIntersectPrimitive<bool>(prefix, defaultOwner);
  registerMapIntersectPrimitive<int8_t>(prefix, defaultOwner);
  registerMapIntersectPrimitive<int16_t>(prefix, defaultOwner);
  registerMapIntersectPrimitive<int32_t>(prefix, defaultOwner);
  registerMapIntersectPrimitive<int64_t>(prefix, defaultOwner);
  registerMapIntersectPrimitive<float>(prefix, defaultOwner);
  registerMapIntersectPrimitive<double>(prefix, defaultOwner);
  registerMapIntersectPrimitive<Timestamp>(prefix, defaultOwner);
  registerMapIntersectPrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      MapIntersectVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({prefix + "map_intersect"}, {}, true, defaultOwner);
}

template <typename T>
void registerMapExceptPrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapExceptPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({prefix + "map_except"}, true, defaultOwner);
}

void registerMapExcept(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerMapExceptPrimitive<bool>(prefix, defaultOwner);
  registerMapExceptPrimitive<int8_t>(prefix, defaultOwner);
  registerMapExceptPrimitive<int16_t>(prefix, defaultOwner);
  registerMapExceptPrimitive<int32_t>(prefix, defaultOwner);
  registerMapExceptPrimitive<int64_t>(prefix, defaultOwner);
  registerMapExceptPrimitive<float>(prefix, defaultOwner);
  registerMapExceptPrimitive<double>(prefix, defaultOwner);
  registerMapExceptPrimitive<Timestamp>(prefix, defaultOwner);
  registerMapExceptPrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      MapExceptVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({prefix + "map_except"}, {}, true, defaultOwner);
}

template <typename T>
void registerMapKeysOverlapPrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapKeysOverlapPrimitiveFunction, T>,
      bool,
      Map<T, Generic<T1>>,
      Array<T>>({prefix + "map_keys_overlap"}, true, defaultOwner);
}

void registerMapKeysOverlap(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerMapKeysOverlapPrimitive<bool>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<int8_t>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<int16_t>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<int32_t>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<int64_t>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<float>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<double>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<Timestamp>(prefix, defaultOwner);
  registerMapKeysOverlapPrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      MapKeysOverlapVarcharFunction,
      bool,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({prefix + "map_keys_overlap"}, {}, true, defaultOwner);

  registerFunction<
      MapKeysOverlapFunction,
      bool,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>(
      {prefix + "map_keys_overlap"}, {}, true, defaultOwner);
}

template <typename Key>
void registerMapAppendPrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapAppendPrimitiveFunction, Key>,
      Map<Key, Generic<T1>>,
      Map<Key, Generic<T1>>,
      Array<Key>,
      Array<Generic<T1>>>({prefix + "map_append"}, true, defaultOwner);
}

void registerMapAppend(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerMapAppendPrimitive<bool>(prefix, defaultOwner);
  registerMapAppendPrimitive<int8_t>(prefix, defaultOwner);
  registerMapAppendPrimitive<int16_t>(prefix, defaultOwner);
  registerMapAppendPrimitive<int32_t>(prefix, defaultOwner);
  registerMapAppendPrimitive<int64_t>(prefix, defaultOwner);
  registerMapAppendPrimitive<float>(prefix, defaultOwner);
  registerMapAppendPrimitive<double>(prefix, defaultOwner);
  registerMapAppendPrimitive<Timestamp>(prefix, defaultOwner);
  registerMapAppendPrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      MapAppendVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>,
      Array<Generic<T1>>>({prefix + "map_append"}, {}, true, defaultOwner);

  registerFunction<
      MapAppendFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>,
      Array<Generic<T2>>>({prefix + "map_append"}, {}, true, defaultOwner);
}

template <typename Key>
void registerMapUpdatePrimitive(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapUpdatePrimitiveFunction, Key>,
      Map<Key, Generic<T1>>,
      Map<Key, Generic<T1>>,
      Array<Key>,
      Array<Generic<T1>>>({prefix + "map_update"}, true, defaultOwner);
}

void registerMapUpdate(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerMapUpdatePrimitive<bool>(prefix, defaultOwner);
  registerMapUpdatePrimitive<int8_t>(prefix, defaultOwner);
  registerMapUpdatePrimitive<int16_t>(prefix, defaultOwner);
  registerMapUpdatePrimitive<int32_t>(prefix, defaultOwner);
  registerMapUpdatePrimitive<int64_t>(prefix, defaultOwner);
  registerMapUpdatePrimitive<float>(prefix, defaultOwner);
  registerMapUpdatePrimitive<double>(prefix, defaultOwner);
  registerMapUpdatePrimitive<Timestamp>(prefix, defaultOwner);
  registerMapUpdatePrimitive<Date>(prefix, defaultOwner);

  registerFunction<
      MapUpdateVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>,
      Array<Generic<T1>>>({prefix + "map_update"}, {}, true, defaultOwner);

  registerFunction<
      MapUpdateFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>,
      Array<Generic<T2>>>({prefix + "map_update"}, {}, true, defaultOwner);
}

void registerMapRemoveNullValues(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      MapRemoveNullValues,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>>(
      {prefix + "map_remove_null_values"}, {}, true, defaultOwner);
}

void registerMapKeyExists(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      MapKeyExists,
      bool,
      Map<Generic<T1>, Generic<T2>>,
      Generic<T1>>({prefix + "map_key_exists"}, {}, true, defaultOwner);
}

} // namespace

void registerMapFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_filter, prefix + "map_filter", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_transform_keys, prefix + "transform_keys", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_transform_values, prefix + "transform_values", defaultOwner);
  registerMapFunction(
      prefix + "map", false /*allowDuplicateKeys*/, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_entries, prefix + "map_entries", defaultOwner);
  registerMapFromEntriesFunction(
      prefix + "map_from_entries", /*throwForNull=*/true, defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_keys, prefix + "map_keys", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_values, prefix + "map_values", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_zip_with, prefix + "map_zip_with", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_all_keys_match, prefix + "all_keys_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_any_keys_match, prefix + "any_keys_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_no_keys_match, prefix + "no_keys_match", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_any_values_match, prefix + "any_values_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_no_values_match, prefix + "no_values_match", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_values_all_match, prefix + "map_values_all_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_values_any_match, prefix + "map_values_any_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_values_none_match,
      prefix + "map_values_none_match",
      defaultOwner);

  registerMapConcatFunction(prefix + "map_concat", defaultOwner);

  registerFunction<
      MultimapFromEntriesFunction,
      Map<Generic<T1>, Array<Generic<T2>>>,
      Array<Row<Generic<T1>, Generic<T2>>>>(
      {prefix + "multimap_from_entries"}, {}, true, defaultOwner);

  registerFunction<
      MapTopNFunction,
      Map<Orderable<T1>, Orderable<T2>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_top_n"}, {}, true, defaultOwner);

  registerFunction<
      MapTopNKeysFunction,
      Array<Orderable<T1>>,
      Map<Orderable<T1>, Generic<T2>>,
      int64_t>({prefix + "map_top_n_keys"}, {}, true, defaultOwner);

  registerFunction<
      MapKeysByTopNValuesFunction,
      Array<Orderable<T1>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_keys_by_top_n_values"}, {}, true, defaultOwner);

  registerFunction<
      MapTopNValuesFunction,
      Array<Orderable<T2>>,
      Map<Orderable<T1>, Orderable<T2>>,
      int64_t>({prefix + "map_top_n_values"}, {}, true, defaultOwner);

  registerMapSubset(prefix + "map_subset", defaultOwner);

  registerRemapKeys(prefix, defaultOwner);

  registerMapIntersect(prefix, defaultOwner);

  registerMapExcept(prefix, defaultOwner);

  registerMapKeysOverlap(prefix, defaultOwner);

  registerMapAppend(prefix, defaultOwner);

  registerMapUpdate(prefix, defaultOwner);

  registerMapRemoveNullValues(prefix, defaultOwner);

  registerMapKeyExists(prefix, defaultOwner);

  registerFunction<
      MapNormalizeFunction,
      Map<Varchar, double>,
      Map<Varchar, double>>({prefix + "map_normalize"}, {}, true, defaultOwner);

  // Register map_values_in_range for various key/value type combinations
  // Helper lambda to reduce registration boilerplate
  auto registerMapValuesInRange = [&prefix,
                                   defaultOwner]<typename K, typename V>() {
    registerFunction<
        ParameterBinder<MapValuesInRangeFunction, K, V>,
        Map<K, V>,
        Map<K, V>,
        V,
        V>({prefix + "map_values_in_range"}, true, defaultOwner);
  };

  registerMapValuesInRange.template operator()<int64_t, int64_t>();
  registerMapValuesInRange.template operator()<int64_t, double>();
  registerMapValuesInRange.template operator()<int64_t, float>();
  registerMapValuesInRange.template operator()<int32_t, int32_t>();
  registerMapValuesInRange.template operator()<int32_t, double>();
  registerMapValuesInRange.template operator()<int32_t, float>();
  registerMapValuesInRange.template operator()<Varchar, int64_t>();
  registerMapValuesInRange.template operator()<Varchar, double>();
  registerMapValuesInRange.template operator()<Varchar, float>();

  // Generic fallback for complex key types
  registerFunction<
      MapValuesInRangeGenericFunction,
      Map<Generic<T1>, double>,
      Map<Generic<T1>, double>,
      double,
      double>({prefix + "map_values_in_range"}, {}, true, defaultOwner);

  registerFunction<
      MapTrimValuesFunction,
      Map<Generic<T1>, Array<Generic<T2>>>,
      Map<Generic<T1>, Array<Generic<T2>>>,
      int64_t>({prefix + "map_trim_values"}, {}, true, defaultOwner);

  // Register map_subset_key_in_range for primitive key types.
  // Boolean keys are intentionally not supported because element_at on the
  // two possible keys provides the same functionality.
  auto registerMapSubsetKeyInRangePrimitive = [&prefix,
                                               defaultOwner]<typename K>() {
    registerFunction<
        ParameterBinder<MapSubsetKeyInRangeFunction, K>,
        Map<K, Generic<T1>>,
        Map<K, Generic<T1>>,
        K,
        K>({prefix + "map_subset_key_in_range"}, true, defaultOwner);
  };

  registerMapSubsetKeyInRangePrimitive.template operator()<int8_t>();
  registerMapSubsetKeyInRangePrimitive.template operator()<int16_t>();
  registerMapSubsetKeyInRangePrimitive.template operator()<int32_t>();
  registerMapSubsetKeyInRangePrimitive.template operator()<int64_t>();
  registerMapSubsetKeyInRangePrimitive.template operator()<float>();
  registerMapSubsetKeyInRangePrimitive.template operator()<double>();
  registerMapSubsetKeyInRangePrimitive.template operator()<Timestamp>();
  registerMapSubsetKeyInRangePrimitive.template operator()<Date>();

  registerFunction<
      MapSubsetKeyInRangeVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Varchar,
      Varchar>({prefix + "map_subset_key_in_range"}, {}, true, defaultOwner);

  registerFunction<
      MapSubsetKeyInRangeGenericFunction,
      Map<Orderable<T1>, Generic<T2>>,
      Map<Orderable<T1>, Generic<T2>>,
      Orderable<T1>,
      Orderable<T1>>(
      {prefix + "map_subset_key_in_range"}, {}, true, defaultOwner);
}

void registerMapAllowingDuplicates(
    const std::string& name,
    const std::string& prefix,
    std::string_view defaultOwner) {
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_map_allow_duplicates, prefix + name, defaultOwner);
}
} // namespace facebook::velox::functions
