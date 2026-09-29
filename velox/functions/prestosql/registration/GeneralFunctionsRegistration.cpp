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
#include "velox/expression/ExprConstants.h"
#include "velox/expression/RegisterSpecialForm.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/IsNull.h"
#include "velox/functions/prestosql/Cardinality.h"
#include "velox/functions/prestosql/Fail.h"
#include "velox/functions/prestosql/GreatestLeast.h"
#include "velox/functions/prestosql/InPredicate.h"
#include "velox/functions/prestosql/Reduce.h"
#include "velox/functions/prestosql/types/IPAddressType.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::functions {

namespace {

void registerFailFunction(
    const std::vector<std::string>& names,
    std::string_view defaultOwner) {
  registerFunction<FailFunction, UnknownValue, Varchar>(
      names, {}, true, defaultOwner);
  registerFunction<FailFunction, UnknownValue, int32_t, Varchar>(
      names, {}, true, defaultOwner);
  registerFunction<FailFromJsonFunction, UnknownValue, Json>(
      names, {}, true, defaultOwner);
  registerFunction<FailFromJsonFunction, UnknownValue, int32_t, Json>(
      names, {}, true, defaultOwner);
}

template <typename T>
void registerGreatestLeastFunction(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ParameterBinder<GreatestFunction, T>, T, T, Variadic<T>>(
      {prefix + "greatest"}, true, defaultOwner);

  registerFunction<ParameterBinder<LeastFunction, T>, T, T, Variadic<T>>(
      {prefix + "least"}, true, defaultOwner);
}

void registerAllGreatestLeastFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerGreatestLeastFunction<bool>(prefix, defaultOwner);
  registerGreatestLeastFunction<int8_t>(prefix, defaultOwner);
  registerGreatestLeastFunction<int16_t>(prefix, defaultOwner);
  registerGreatestLeastFunction<int32_t>(prefix, defaultOwner);
  registerGreatestLeastFunction<int64_t>(prefix, defaultOwner);
  registerGreatestLeastFunction<float>(prefix, defaultOwner);
  registerGreatestLeastFunction<double>(prefix, defaultOwner);
  registerGreatestLeastFunction<Varchar>(prefix, defaultOwner);
  registerGreatestLeastFunction<LongDecimal<P1, S1>>(prefix, defaultOwner);
  registerGreatestLeastFunction<ShortDecimal<P1, S1>>(prefix, defaultOwner);
  registerGreatestLeastFunction<Date>(prefix, defaultOwner);
  registerGreatestLeastFunction<Timestamp>(prefix, defaultOwner);
  registerGreatestLeastFunction<TimestampWithTimezone>(prefix, defaultOwner);
  registerGreatestLeastFunction<IPAddress>(prefix, defaultOwner);
  registerGreatestLeastFunction<Time>(prefix, defaultOwner);
  registerGreatestLeastFunction<TimeWithTimezone>(prefix, defaultOwner);
}
} // namespace

extern void registerSubscriptFunction(
    const std::string& name,
    bool enableCaching,
    std::string_view defaultOwner);
extern void registerElementAtFunction(
    const std::string& name,
    bool enableCaching,
    std::string_view defaultOwner);

// Special form functions don't have any prefix.
void registerAllSpecialFormGeneralFunctions(std::string_view defaultOwner) {
  exec::registerFunctionCallToSpecialForms();
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(udf_in, "in", defaultOwner);
  registerFunction<
      GenericInPredicateFunction,
      bool,
      Generic<T1>,
      Variadic<Generic<T1>>>({"in"}, {}, true, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_concat_row, expression::kRowConstructor, defaultOwner);
  registerIsNullFunction("is_null", defaultOwner);
}

void registerGeneralFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerSubscriptFunction(prefix + "subscript", true, defaultOwner);
  registerElementAtFunction(prefix + "element_at", true, defaultOwner);

  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_transform, prefix + "transform", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_transform_with_index, prefix + "transform_with_index", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_reduce, prefix + "reduce", defaultOwner);
  registerReduceRewrites(prefix);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_array_filter, prefix + "filter", defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_typeof, prefix + "typeof", defaultOwner);

  registerAllGreatestLeastFunctions(prefix, defaultOwner);

  registerFunction<CardinalityFunction, int64_t, Array<Generic<T1>>>(
      {prefix + "cardinality"}, {}, true, defaultOwner);
  registerFunction<CardinalityFunction, int64_t, Map<Generic<T1>, Generic<T2>>>(
      {prefix + "cardinality"}, {}, true, defaultOwner);

  registerFailFunction({prefix + "fail"}, defaultOwner);

  registerAllSpecialFormGeneralFunctions(defaultOwner);
}

} // namespace facebook::velox::functions
