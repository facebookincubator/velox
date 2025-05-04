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

#include "velox/functions/Registerer.h"
#include "velox/functions/lib/ArrayRemoveNullFunction.h"
#include "velox/functions/lib/ArrayShuffle.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/lib/Repeat.h"
#include "velox/functions/lib/Slice.h"
#include "velox/functions/prestosql/ArrayConstructor.h"
#include "velox/functions/prestosql/ArrayFunctions.h"
#include "velox/functions/prestosql/ArraySort.h"
#include "velox/functions/prestosql/WidthBucketArray.h"
#include "velox/functions/prestosql/types/JsonRegistration.h"

namespace facebook::velox::functions {
extern void registerArrayConcatFunctions(const std::string& prefix);
extern void registerArrayNGramsFunctions(const std::string& prefix);

template <typename T>
inline void registerArrayMinMaxFunctions(const std::string& prefix) {
  registerFunction<ArrayMinFunction, T, Array<T>>({prefix + "array_min"});
  registerFunction<ArrayMaxFunction, T, Array<T>>({prefix + "array_max"});
}

template <typename T>
inline void registerArrayJoinFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayJoinFunction, T>,
      Varchar,
      Array<T>,
      Varchar>({prefix + "array_join"});

  registerFunction<
      ParameterBinder<ArrayJoinFunction, T>,
      Varchar,
      Array<T>,
      Varchar,
      Varchar>({prefix + "array_join"});
}

template <typename T>
inline void registerArrayCombinationsFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<CombinationsFunction, T>,
      Array<Array<T>>,
      Array<T>,
      int32_t>({prefix + "combinations"});
}

template <typename T>
inline void registerArrayCumSumFunction(const std::string& prefix) {
  registerFunction<ParameterBinder<ArrayCumSumFunction, T>, Array<T>, Array<T>>(
      {prefix + "array_cum_sum"});
}

template <typename T>
inline void registerArrayHasDuplicatesFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayHasDuplicatesFunction, T>,
      bool,
      Array<T>>({prefix + "array_has_duplicates"});
}

template <typename T>
inline void registerArrayFrequencyFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayFrequencyFunction, T>,
      Map<T, int>,
      Array<T>>({prefix + "array_frequency"});
}

template <typename T>
inline void registerArrayNormalizeFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayNormalizeFunction, T>,
      Array<T>,
      Array<T>,
      T>({prefix + "array_normalize"});
}

template <typename T>
inline void registerArrayTrimFunctions(const std::string& prefix) {
  registerFunction<ArrayTrimFunction, Array<T>, Array<T>, int64_t>(
      {prefix + "trim_array"});
}

template <typename T>
inline void registerArrayTopNFunction(const std::string& prefix) {
  registerFunction<ArrayTopNFunction, Array<T>, Array<T>, int32_t>(
      {prefix + "array_top_n"});
}

template <typename T>
inline void registerArrayRemoveNullFunctions(const std::string& prefix) {
  registerFunction<ArrayRemoveNullFunction, Array<T>, Array<T>>(
      {prefix + "remove_nulls"});
}

template <typename T>
inline void registerArrayUnionFunctions(const std::string& prefix) {
  registerFunction<ArrayUnionFunction, Array<T>, Array<T>, Array<T>>(
      {prefix + "array_union"});
}

template <typename T>
inline void registerArrayRemoveFunctions(const std::string& prefix) {
  registerFunction<ArrayRemoveFunction, Array<T>, Array<T>, T>(
      {prefix + "array_remove"});
}

void registerInternalArrayFunctions() {
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_$internal$canonicalize, "$internal$canonicalize");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_$internal$contains, "$internal$contains");
}

void registerArrayFunctions(const std::string& prefix) {
  registerJsonType();
  registerArrayConstructor(prefix + "array_constructor");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_all_match, prefix + "all_match");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_any_match, prefix + "any_match");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_none_match, prefix + "none_match");

  VELOX_REGISTER_VECTOR_FUNCTION(udf_find_first, prefix + "find_first");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_find_first_index, prefix + "find_first_index");

  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_distinct, prefix + "array_distinct");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_array_duplicates, prefix + "array_duplicates");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_array_intersect, prefix + "array_intersect");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_contains, prefix + "contains");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_except, prefix + "array_except");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_arrays_overlap, prefix + "arrays_overlap");
  registerBigintSliceFunction(prefix);
  VELOX_REGISTER_VECTOR_FUNCTION(udf_zip, prefix + "zip");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_zip_with, prefix + "zip_with");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_position, prefix + "array_position");
  exec::registerStatefulVectorFunction(
      prefix + "shuffle",
      arrayShuffleSignatures(),
      makeArrayShuffle,
      getMetadataForArrayShuffle());

  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_sort, prefix + "array_sort");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_array_sort_desc, prefix + "array_sort_desc");

  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_max_by, prefix + "array_max_by");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_min_by, prefix + "array_min_by");

  exec::registerExpressionRewrite([prefix](const auto& expr) {
    return rewriteArraySortCall(prefix, expr);
  });

  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_sum, prefix + "array_sum");
  exec::registerStatefulVectorFunction(
      prefix + "repeat", repeatSignatures(), makeRepeat, repeatMetadata());
  VELOX_REGISTER_VECTOR_FUNCTION(udf_sequence, prefix + "sequence");

  exec::registerStatefulVectorFunction(
      prefix + "width_bucket",
      widthBucketArraySignature(),
      makeWidthBucketArray);

  REGISTER_NUMERIC_FUNCTIONS(registerArrayMinMaxFunctions, prefix);
  registerArrayMinMaxFunctions<bool>(prefix);
  registerArrayMinMaxFunctions<Varchar>(prefix);
  registerArrayMinMaxFunctions<Timestamp>(prefix);
  registerArrayMinMaxFunctions<Date>(prefix);
  registerArrayMinMaxFunctions<Orderable<T1>>(prefix);

  REGISTER_NUMERIC_FUNCTIONS(registerArrayJoinFunctions, prefix);
  registerArrayJoinFunctions<bool>(prefix);
  registerArrayJoinFunctions<Varchar>(prefix);
  registerArrayJoinFunctions<Timestamp>(prefix);
  registerArrayJoinFunctions<Date>(prefix);
  registerArrayJoinFunctions<Json>(prefix);
  registerArrayJoinFunctions<UnknownValue>(prefix);

  registerFunction<ArrayAverageFunction, double, Array<double>>(
      {prefix + "array_average"});

  registerArrayConcatFunctions(prefix);
  registerArrayNGramsFunctions(prefix);

  registerFunction<
      ArrayFlattenFunction,
      Array<Generic<T1>>,
      Array<Array<Generic<T1>>>>({prefix + "flatten"});

  REGISTER_SCALAR_FUNCTIONS_WITHOUT_VARCHAR(
      registerArrayRemoveFunctions, prefix);
  registerFunction<
      ArrayRemoveFunctionString,
      Array<Varchar>,
      Array<Varchar>,
      Varchar>({prefix + "array_remove"});
  registerArrayRemoveFunctions<Generic<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS_WITHOUT_VARCHAR(registerArrayTrimFunctions, prefix);
  registerArrayTrimFunctions<Generic<T1>>(prefix);
  registerFunction<
      ArrayTrimFunctionString,
      Array<Varchar>,
      Array<Varchar>,
      int64_t>({prefix + "trim_array"});

  REGISTER_NUMERIC_FUNCTIONS(registerArrayTopNFunction, prefix);
  registerArrayTopNFunction<Varchar>(prefix);
  registerArrayTopNFunction<Timestamp>(prefix);
  registerArrayTopNFunction<Date>(prefix);
  registerArrayTopNFunction<Varbinary>(prefix);
  registerArrayTopNFunction<Orderable<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS_WITHOUT_VARCHAR(
      registerArrayRemoveNullFunctions, prefix);
  registerFunction<
      ArrayRemoveNullFunctionString,
      Array<Varchar>,
      Array<Varchar>>({prefix + "remove_nulls"});
  registerArrayRemoveNullFunctions<Generic<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayUnionFunctions, prefix);
  registerArrayUnionFunctions<Generic<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayCombinationsFunctions, prefix);
  registerArrayCombinationsFunctions<Generic<T1>>(prefix);

  registerArrayCumSumFunction<int8_t>(prefix);
  registerArrayCumSumFunction<int16_t>(prefix);
  registerArrayCumSumFunction<int32_t>(prefix);
  registerArrayCumSumFunction<int64_t>(prefix);
  registerArrayCumSumFunction<int128_t>(prefix);
  registerArrayCumSumFunction<float>(prefix);
  registerArrayCumSumFunction<double>(prefix);

  registerArrayHasDuplicatesFunctions<int8_t>(prefix);
  registerArrayHasDuplicatesFunctions<int16_t>(prefix);
  registerArrayHasDuplicatesFunctions<int32_t>(prefix);
  registerArrayHasDuplicatesFunctions<int64_t>(prefix);
  registerArrayHasDuplicatesFunctions<int128_t>(prefix);
  registerArrayHasDuplicatesFunctions<Varchar>(prefix);
  registerArrayHasDuplicatesFunctions<Json>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayFrequencyFunctions, prefix);

  registerArrayNormalizeFunctions<int8_t>(prefix);
  registerArrayNormalizeFunctions<int16_t>(prefix);
  registerArrayNormalizeFunctions<int32_t>(prefix);
  registerArrayNormalizeFunctions<int64_t>(prefix);
  registerArrayNormalizeFunctions<float>(prefix);
  registerArrayNormalizeFunctions<double>(prefix);
}
} // namespace facebook::velox::functions
