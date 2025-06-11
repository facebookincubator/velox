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
#include "velox/functions/lib/ArrayRemoveNullFunction.h"
#include "velox/functions/lib/ArrayShuffle.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/lib/Repeat.h"
#include "velox/functions/lib/Slice.h"
#include "velox/functions/prestosql/ArrayFunctions.h"
#include "velox/functions/sparksql/ArrayAppend.h"
#include "velox/functions/sparksql/ArrayConcat.h"
#include "velox/functions/sparksql/ArrayFlattenFunction.h"
#include "velox/functions/sparksql/ArrayInsert.h"
#include "velox/functions/sparksql/ArrayMinMaxFunction.h"
#include "velox/functions/sparksql/ArrayPrepend.h"
#include "velox/functions/sparksql/ArraySort.h"

namespace facebook::velox::functions {

// VELOX_REGISTER_VECTOR_FUNCTION must be invoked in the same namespace as the
// vector function definition.
// Higher order functions.
void registerSparkArrayFunctions(const std::string& prefix) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_transform, prefix + "transform");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_reduce, prefix + "aggregate");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_constructor, prefix + "array");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_contains, prefix + "array_contains");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_distinct, prefix + "array_distinct");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_except, prefix + "array_except");
  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_array_intersect, prefix + "array_intersect");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_position, prefix + "array_position");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_zip, prefix + "arrays_zip");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_any_match, prefix + "exists");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_filter, prefix + "filter");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_all_match, prefix + "forall");
  VELOX_REGISTER_VECTOR_FUNCTION(udf_zip_with, prefix + "zip_with");
}

namespace sparksql {
template <typename T>
inline void registerArrayConcatFunction(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayConcatFunction, T>,
      Array<T>,
      Variadic<Array<T>>>({prefix + "concat"});
}

inline void registerArrayJoinFunctions(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayJoinFunction, Varchar>,
      Varchar,
      Array<Varchar>,
      Varchar>({prefix + "array_join"});

  registerFunction<
      ParameterBinder<ArrayJoinFunction, Varchar>,
      Varchar,
      Array<Varchar>,
      Varchar,
      Varchar>({prefix + "array_join"});
}

template <typename T>
void registerArrayMinMaxFunctions(const std::string& prefix) {
  registerFunction<ArrayMinFunction, T, Array<T>>({prefix + "array_min"});
  registerFunction<ArrayMaxFunction, T, Array<T>>({prefix + "array_max"});
}

template <typename T>
void registerArrayRemoveFunctions(const std::string& prefix) {
  registerFunction<ArrayRemoveFunction, Array<T>, Array<T>, T>(
      {prefix + "array_remove"});
}

template <typename T>
void registerArrayUnionFunction(const std::string& prefix) {
  registerFunction<ArrayUnionFunction, Array<T>, Array<T>, Array<T>>(
      {prefix + "array_union"});
}

template <typename T>
void registerArrayPrependFunctions(const std::string& prefix) {
  registerFunction<ArrayPrependFunction, Array<T>, Array<T>, T>(
      {prefix + "array_prepend"});
}

template <typename T>
void registerArrayUnionFunction(const std::string& prefix) {
  registerFunction<ArrayUnionFunction, Array<T>, Array<T>, Array<T>>(
      {prefix + "array_union"});
}

template <typename T>
void registerArrayCompactFunction(const std::string& prefix) {
  registerFunction<ArrayRemoveNullFunction, Array<T>, Array<T>>(
      {prefix + "array_compact"});
}

template <typename T>
void registerArrayConcatFunction(const std::string& prefix) {
  registerFunction<
      ParameterBinder<ArrayConcatFunction, T>,
      Array<T>,
      Variadic<Array<T>>>({prefix + "concat"});
}

void registerArrayFunctions(const std::string& prefix) {
  REGISTER_SCALAR_FUNCTIONS(registerArrayConcatFunctions, prefix);
  registerArrayConcatFunction<Generic<T1>>(prefix);
  registerArrayConcatFunction<UnknownValue>(prefix);

  registerArrayJoinFunctions(prefix);
  REGISTER_SCALAR_FUNCTIONS(registerArrayMinMaxFunctions, prefix);
  registerArrayMinMaxFunctions<Orderable<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayPrependFunctions, prefix);
  registerArrayPrependFunctions<Generic<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayUnionFunction, prefix);
  registerArrayUnionFunction<Generic<T1>>(prefix);
  REGISTER_SCALAR_FUNCTIONS_WITHOUT_VARCHAR(
      registerArrayRemoveFunctions, prefix);
  registerArrayRemoveFunctions<Generic<T1>>(prefix);
  registerFunction<
      ArrayRemoveFunctionString,
      Array<Varchar>,
      Array<Varchar>,
      Varchar>({prefix + "array_remove"});
  REGISTER_SCALAR_FUNCTIONS_WITHOUT_VARCHAR(
      ArrayRemoveNullFunctionString, prefix);
  registerFunction<
      ArrayRemoveNullFunctionString,
      Array<Varchar>,
      Array<Varchar>>({prefix + "array_compact"});
  registerSparkArrayFunctions(prefix);
  // Register array sort functions.
  exec::registerStatefulVectorFunction(
      prefix + "array_sort", arraySortSignatures(), makeArraySort);
  exec::registerStatefulVectorFunction(
      prefix + "sort_array", sortArraySignatures(), makeSortArray);
  exec::registerStatefulVectorFunction(
      prefix + "array_repeat",
      repeatSignatures(),
      makeRepeatAllowNegativeCount,
      repeatMetadata());
  registerFunction<
      ArrayFlattenFunction,
      Array<Generic<T1>>,
      Array<Array<Generic<T1>>>>({prefix + "flatten"});
  registerFunction<
      ArrayInsert,
      Array<Generic<T1>>,
      Array<Generic<T1>>,
      int32_t,
      Generic<T1>,
      bool>({prefix + "array_insert"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_array_get, prefix + "get");
  exec::registerStatefulVectorFunction(
      prefix + "shuffle",
      arrayShuffleWithCustomSeedSignatures(),
      makeArrayShuffleWithCustomSeed,
      getMetadataForArrayShuffle());
  registerIntegerSliceFunction(prefix);
  registerFunction<
      ArrayAppendFunction,
      Array<Generic<T1>>,
      Array<Generic<T1>>,
      Generic<T1>>({prefix + "array_append"});

  REGISTER_SCALAR_FUNCTIONS(registerArrayUnionFunction, prefix);
  registerArrayUnionFunction<Generic<T1>>(prefix);

  REGISTER_SCALAR_FUNCTIONS(registerArrayCompactFunction, prefix);
  registerFunction<
      ArrayRemoveNullFunctionString,
      Array<Varchar>,
      Array<Varchar>>({prefix + "array_compact"});
}

} // namespace sparksql
} // namespace facebook::velox::functions
