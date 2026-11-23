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

#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/ArrayRemoveNullFunction.h"
#include "velox/functions/lib/ArrayShuffle.h"
#include "velox/functions/lib/Repeat.h"
#include "velox/functions/lib/Slice.h"
#include "velox/functions/prestosql/ArrayConstructor.h"
#include "velox/functions/prestosql/ArrayFunctions.h"
#include "velox/functions/prestosql/ArraySort.h"
#include "velox/functions/prestosql/ArraySubset.h"
#include "velox/functions/prestosql/DotProduct.h"
#include "velox/functions/prestosql/L2Norm.h"
#include "velox/functions/prestosql/WidthBucketArray.h"
#include "velox/functions/prestosql/types/JsonRegistration.h"
#include "velox/type/SimpleFunctionApi.h"

namespace facebook::velox::functions {
extern void registerArrayConcatFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerArrayNGramsFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);
extern void registerArraySplitIntoChunksFunctions(
    const std::string& prefix,
    std::string_view defaultOwner);

template <typename T>
inline void registerArrayMinMaxFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayMinFunction, T, Array<T>>(
      {prefix + "array_min"}, {}, true, defaultOwner);
  registerFunction<ArrayMaxFunction, T, Array<T>>(
      {prefix + "array_max"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArrayJoinFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArrayJoinFunction, T>,
      Varchar,
      Array<T>,
      Varchar>({prefix + "array_join"}, true, defaultOwner);

  registerFunction<
      ParameterBinder<ArrayJoinFunction, T>,
      Varchar,
      Array<T>,
      Varchar,
      Varchar>({prefix + "array_join"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayCombinationsFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<CombinationsFunction, T>,
      Array<Array<T>>,
      Array<T>,
      int32_t>({prefix + "combinations"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayCumSumFunction(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ParameterBinder<ArrayCumSumFunction, T>, Array<T>, Array<T>>(
      {prefix + "array_cum_sum"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayHasDuplicatesFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArrayHasDuplicatesFunction, T>,
      bool,
      Array<T>>({prefix + "array_has_duplicates"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayFrequencyFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArrayFrequencyFunction, T>,
      Map<T, int>,
      Array<T>>({prefix + "array_frequency"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayNormalizeFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArrayNormalizeFunction, T>,
      Array<T>,
      Array<T>,
      T>({prefix + "array_normalize"}, true, defaultOwner);
}

template <typename T>
inline void registerArrayTrimFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayTrimFunction, Array<T>, Array<T>, int64_t>(
      {prefix + "trim_array"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArrayTopNFunction(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayTopNFunction, Array<T>, Array<T>, int32_t>(
      {prefix + "array_top_n"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArrayRemoveNullFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayRemoveNullFunction, Array<T>, Array<T>>(
      {prefix + "remove_nulls"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArrayUnionFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayUnionFunction, Array<T>, Array<T>, Array<T>>(
      {prefix + "array_union"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArrayRemoveFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayRemoveFunction, Array<T>, Array<T>, T>(
      {prefix + "array_remove"}, {}, true, defaultOwner);
}

template <typename T>
inline void registerArraySubsetFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArraySubsetFunction, T>,
      Array<T>,
      Array<T>,
      Array<int32_t>>({prefix + "array_subset"}, true, defaultOwner);
}

void registerInternalArrayFunctions(std::string_view defaultOwner) {
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_$internal$canonicalize, "$internal$canonicalize", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_$internal$contains, "$internal$contains", defaultOwner);
}

void registerArrayFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerJsonType();
  registerArrayConstructor(prefix + "array_constructor", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_all_match, prefix + "all_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_any_match, prefix + "any_match", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_none_match, prefix + "none_match", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_find_first, prefix + "find_first", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_find_first_index, prefix + "find_first_index", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_distinct, prefix + "array_distinct", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_duplicates, prefix + "array_duplicates", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_intersect, prefix + "array_intersect", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_contains, prefix + "contains", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_except, prefix + "array_except", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_arrays_overlap, prefix + "arrays_overlap", defaultOwner);
  registerBigintSliceFunction(prefix, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_zip, prefix + "zip", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_zip_with, prefix + "zip_with", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_position, prefix + "array_position", defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "shuffle",
      arrayShuffleSignatures(),
      makeArrayShuffle,
      getMetadataForArrayShuffle(),
      /*overwrite=*/true,
      defaultOwner);

  exec::registerStatefulVectorFunction(
      prefix + "array_sort",
      arraySortSignatures(true),
      makeArraySortAsc,
      {},
      /*overwrite=*/true,
      defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "array_sort_desc",
      arraySortSignatures(false),
      makeArraySortDesc,
      {},
      /*overwrite=*/true,
      defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_max_by, prefix + "array_max_by", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_min_by, prefix + "array_min_by", defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_flatten, prefix + "flatten", defaultOwner);

  auto checker = std::make_shared<SimpleComparisonChecker>();
  expression::ExprRewriteRegistry::instance().registerRewrite(
      [prefix, checker](const auto& expr) {
        return rewriteArraySortCall(prefix, expr, checker);
      });

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_sum, prefix + "array_sum", defaultOwner);
  exec::registerStatefulVectorFunction(
      prefix + "repeat",
      repeatSignatures(),
      makeRepeat,
      repeatMetadata(),
      /*overwrite=*/true,
      defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_sequence, prefix + "sequence", defaultOwner);

  exec::registerStatefulVectorFunction(
      prefix + "width_bucket",
      widthBucketArraySignature(),
      makeWidthBucketArray,
      {},
      /*overwrite=*/true,
      defaultOwner);

  registerArrayMinMaxFunctions<int8_t>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<int16_t>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<int32_t>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<int64_t>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<int128_t>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<float>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<double>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<bool>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<Varchar>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<Date>(prefix, defaultOwner);
  registerArrayMinMaxFunctions<Orderable<T1>>(prefix, defaultOwner);

  registerArrayJoinFunctions<int8_t>(prefix, defaultOwner);
  registerArrayJoinFunctions<int16_t>(prefix, defaultOwner);
  registerArrayJoinFunctions<int32_t>(prefix, defaultOwner);
  registerArrayJoinFunctions<int64_t>(prefix, defaultOwner);
  registerArrayJoinFunctions<int128_t>(prefix, defaultOwner);
  registerArrayJoinFunctions<float>(prefix, defaultOwner);
  registerArrayJoinFunctions<double>(prefix, defaultOwner);
  registerArrayJoinFunctions<bool>(prefix, defaultOwner);
  registerArrayJoinFunctions<Varchar>(prefix, defaultOwner);
  registerArrayJoinFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayJoinFunctions<Date>(prefix, defaultOwner);
  registerArrayJoinFunctions<Json>(prefix, defaultOwner);
  registerArrayJoinFunctions<UnknownValue>(prefix, defaultOwner);

  registerFunction<ArrayAverageFunction, double, Array<double>>(
      {prefix + "array_average"}, {}, true, defaultOwner);

  registerArrayConcatFunctions(prefix, defaultOwner);
  registerArrayNGramsFunctions(prefix, defaultOwner);
  registerArraySplitIntoChunksFunctions(prefix, defaultOwner);

  registerArrayRemoveFunctions<int8_t>(prefix, defaultOwner);
  registerArrayRemoveFunctions<int16_t>(prefix, defaultOwner);
  registerArrayRemoveFunctions<int32_t>(prefix, defaultOwner);
  registerArrayRemoveFunctions<int64_t>(prefix, defaultOwner);
  registerArrayRemoveFunctions<int128_t>(prefix, defaultOwner);
  registerArrayRemoveFunctions<float>(prefix, defaultOwner);
  registerArrayRemoveFunctions<double>(prefix, defaultOwner);
  registerArrayRemoveFunctions<bool>(prefix, defaultOwner);
  registerArrayRemoveFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayRemoveFunctions<Date>(prefix, defaultOwner);
  registerArrayRemoveFunctions<Varbinary>(prefix, defaultOwner);
  registerArrayRemoveFunctions<Generic<T1>>(prefix, defaultOwner);
  registerFunction<
      ArrayRemoveFunctionString,
      Array<Varchar>,
      Array<Varchar>,
      Varchar>({prefix + "array_remove"}, {}, true, defaultOwner);

  registerArrayTrimFunctions<int8_t>(prefix, defaultOwner);
  registerArrayTrimFunctions<int16_t>(prefix, defaultOwner);
  registerArrayTrimFunctions<int32_t>(prefix, defaultOwner);
  registerArrayTrimFunctions<int64_t>(prefix, defaultOwner);
  registerArrayTrimFunctions<int128_t>(prefix, defaultOwner);
  registerArrayTrimFunctions<float>(prefix, defaultOwner);
  registerArrayTrimFunctions<double>(prefix, defaultOwner);
  registerArrayTrimFunctions<bool>(prefix, defaultOwner);
  registerArrayTrimFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayTrimFunctions<Date>(prefix, defaultOwner);
  registerArrayTrimFunctions<Varbinary>(prefix, defaultOwner);
  registerArrayTrimFunctions<Generic<T1>>(prefix, defaultOwner);
  registerFunction<
      ArrayTrimFunctionString,
      Array<Varchar>,
      Array<Varchar>,
      int64_t>({prefix + "trim_array"}, {}, true, defaultOwner);

  registerArrayTopNFunction<int8_t>(prefix, defaultOwner);
  registerArrayTopNFunction<int16_t>(prefix, defaultOwner);
  registerArrayTopNFunction<int32_t>(prefix, defaultOwner);
  registerArrayTopNFunction<int64_t>(prefix, defaultOwner);
  registerArrayTopNFunction<int128_t>(prefix, defaultOwner);
  registerArrayTopNFunction<float>(prefix, defaultOwner);
  registerArrayTopNFunction<double>(prefix, defaultOwner);
  registerArrayTopNFunction<Varchar>(prefix, defaultOwner);
  registerArrayTopNFunction<Timestamp>(prefix, defaultOwner);
  registerArrayTopNFunction<Date>(prefix, defaultOwner);
  registerArrayTopNFunction<Varbinary>(prefix, defaultOwner);
  registerArrayTopNFunction<Orderable<T1>>(prefix, defaultOwner);

  registerArrayRemoveNullFunctions<int8_t>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<int16_t>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<int32_t>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<int64_t>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<int128_t>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<float>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<double>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<bool>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<Date>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<Varbinary>(prefix, defaultOwner);
  registerArrayRemoveNullFunctions<Generic<T1>>(prefix, defaultOwner);
  registerFunction<
      ArrayRemoveNullFunctionString,
      Array<Varchar>,
      Array<Varchar>>({prefix + "remove_nulls"}, {}, true, defaultOwner);

  registerArrayUnionFunctions<int8_t>(prefix, defaultOwner);
  registerArrayUnionFunctions<int16_t>(prefix, defaultOwner);
  registerArrayUnionFunctions<int32_t>(prefix, defaultOwner);
  registerArrayUnionFunctions<int64_t>(prefix, defaultOwner);
  registerArrayUnionFunctions<int128_t>(prefix, defaultOwner);
  registerArrayUnionFunctions<float>(prefix, defaultOwner);
  registerArrayUnionFunctions<double>(prefix, defaultOwner);
  registerArrayUnionFunctions<bool>(prefix, defaultOwner);
  registerArrayUnionFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayUnionFunctions<Date>(prefix, defaultOwner);
  registerArrayUnionFunctions<Varbinary>(prefix, defaultOwner);
  registerArrayUnionFunctions<Generic<T1>>(prefix, defaultOwner);

  registerArrayCombinationsFunctions<int8_t>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<int16_t>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<int32_t>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<int64_t>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<int128_t>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<float>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<double>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<bool>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<Varchar>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<Date>(prefix, defaultOwner);
  registerArrayCombinationsFunctions<Generic<T1>>(prefix, defaultOwner);

  registerArrayCumSumFunction<int8_t>(prefix, defaultOwner);
  registerArrayCumSumFunction<int16_t>(prefix, defaultOwner);
  registerArrayCumSumFunction<int32_t>(prefix, defaultOwner);
  registerArrayCumSumFunction<int64_t>(prefix, defaultOwner);
  registerArrayCumSumFunction<int128_t>(prefix, defaultOwner);
  registerArrayCumSumFunction<float>(prefix, defaultOwner);
  registerArrayCumSumFunction<double>(prefix, defaultOwner);
  registerArrayCumSumFunction<LongDecimal<P1, S1>>(prefix, defaultOwner);
  registerArrayCumSumFunction<ShortDecimal<P1, S1>>(prefix, defaultOwner);

  registerArrayHasDuplicatesFunctions<int8_t>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<int16_t>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<int32_t>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<int64_t>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<int128_t>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<Varchar>(prefix, defaultOwner);
  registerArrayHasDuplicatesFunctions<Json>(prefix, defaultOwner);

  registerArrayFrequencyFunctions<bool>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<int8_t>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<int16_t>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<int32_t>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<int64_t>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<int128_t>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<float>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<double>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<Date>(prefix, defaultOwner);
  registerArrayFrequencyFunctions<Varchar>(prefix, defaultOwner);

  registerArrayNormalizeFunctions<int8_t>(prefix, defaultOwner);
  registerArrayNormalizeFunctions<int16_t>(prefix, defaultOwner);
  registerArrayNormalizeFunctions<int32_t>(prefix, defaultOwner);
  registerArrayNormalizeFunctions<int64_t>(prefix, defaultOwner);
  registerArrayNormalizeFunctions<float>(prefix, defaultOwner);
  registerArrayNormalizeFunctions<double>(prefix, defaultOwner);

  registerArraySubsetFunctions<int8_t>(prefix, defaultOwner);
  registerArraySubsetFunctions<int16_t>(prefix, defaultOwner);
  registerArraySubsetFunctions<int32_t>(prefix, defaultOwner);
  registerArraySubsetFunctions<int64_t>(prefix, defaultOwner);
  registerArraySubsetFunctions<int128_t>(prefix, defaultOwner);
  registerArraySubsetFunctions<float>(prefix, defaultOwner);
  registerArraySubsetFunctions<double>(prefix, defaultOwner);
  registerArraySubsetFunctions<bool>(prefix, defaultOwner);
  registerArraySubsetFunctions<Timestamp>(prefix, defaultOwner);
  registerArraySubsetFunctions<Date>(prefix, defaultOwner);
  registerArraySubsetFunctions<Varbinary>(prefix, defaultOwner);
  registerArraySubsetFunctions<Generic<T1>>(prefix, defaultOwner);
  registerFunction<
      ArraySubsetVarcharFunction,
      Array<Varchar>,
      Array<Varchar>,
      Array<int32_t>>({prefix + "array_subset"}, {}, true, defaultOwner);
  registerFunction<
      ArraySubsetGenericFunction,
      Array<Generic<T1>>,
      Array<Generic<T1>>,
      Array<int32_t>>({prefix + "array_subset"}, {}, true, defaultOwner);

  // Register l2_norm function for arrays
  registerFunction<ArrayL2NormFunction, double, Array<int8_t>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);
  registerFunction<ArrayL2NormFunction, double, Array<int16_t>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);
  registerFunction<ArrayL2NormFunction, double, Array<int32_t>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);
  registerFunction<ArrayL2NormFunction, double, Array<int64_t>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);
  registerFunction<ArrayL2NormFunction, double, Array<float>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);
  registerFunction<ArrayL2NormFunction, double, Array<double>>(
      {prefix + "l2_norm"}, {}, true, defaultOwner);

  // Register l2_norm function for maps with numeric values
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, int8_t>,
      double,
      Map<Varchar, int8_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, int16_t>,
      double,
      Map<Varchar, int16_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, int32_t>,
      double,
      Map<Varchar, int32_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, int64_t>,
      double,
      Map<Varchar, int64_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, float>,
      double,
      Map<Varchar, float>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, Varchar, double>,
      double,
      Map<Varchar, double>>({prefix + "l2_norm"}, true, defaultOwner);

  // Register l2_norm function for maps with integer keys
  registerFunction<
      ParameterBinder<MapL2NormFunction, int32_t, int32_t>,
      double,
      Map<int32_t, int32_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, int32_t, int64_t>,
      double,
      Map<int32_t, int64_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, int32_t, float>,
      double,
      Map<int32_t, float>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, int32_t, double>,
      double,
      Map<int32_t, double>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, int64_t, int64_t>,
      double,
      Map<int64_t, int64_t>>({prefix + "l2_norm"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapL2NormFunction, int64_t, double>,
      double,
      Map<int64_t, double>>({prefix + "l2_norm"}, true, defaultOwner);

  // Register dot_product for integer arrays only.
  // Float and double array versions already exist in
  // MathematicalFunctionsRegistration.cpp (DotProductArray,
  // DotProductFloatArray) with different semantics: they return NaN for empty
  // arrays to maintain compatibility with cosine_similarity and other distance
  // functions there. Integer versions here return 0 for empty arrays.
  registerFunction<
      ParameterBinder<DotProductFunction, int8_t>,
      int64_t,
      Array<int8_t>,
      Array<int8_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<DotProductFunction, int16_t>,
      int64_t,
      Array<int16_t>,
      Array<int16_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<DotProductFunction, int32_t>,
      int64_t,
      Array<int32_t>,
      Array<int32_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<DotProductFunction, int64_t>,
      int64_t,
      Array<int64_t>,
      Array<int64_t>>({prefix + "dot_product"}, true, defaultOwner);

  // Register dot_product for maps with integer keys
  registerFunction<
      ParameterBinder<MapDotProductFunction, int32_t, int64_t>,
      int64_t,
      Map<int32_t, int64_t>,
      Map<int32_t, int64_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapDotProductFunction, int64_t, int64_t>,
      int64_t,
      Map<int64_t, int64_t>,
      Map<int64_t, int64_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapDotProductFunction, int32_t, double>,
      double,
      Map<int32_t, double>,
      Map<int32_t, double>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapDotProductFunction, int64_t, double>,
      double,
      Map<int64_t, double>,
      Map<int64_t, double>>({prefix + "dot_product"}, true, defaultOwner);

  // Register dot_product for maps with varchar keys
  registerFunction<
      ParameterBinder<MapDotProductFunction, Varchar, int64_t>,
      int64_t,
      Map<Varchar, int64_t>,
      Map<Varchar, int64_t>>({prefix + "dot_product"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<MapDotProductFunction, Varchar, double>,
      double,
      Map<Varchar, double>,
      Map<Varchar, double>>({prefix + "dot_product"}, true, defaultOwner);
}
} // namespace facebook::velox::functions
