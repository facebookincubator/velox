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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Comparisons.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions {

void registerComparisonFunctions(const std::string& prefix) {
  registerNonSimdizableScalar<EqFunction, bool>({prefix + "eq"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_eq, prefix + "eq");
  registerFunction<EqFunction, bool, Generic<T1>, Generic<T1>>({prefix + "eq"});

  registerNonSimdizableScalar<NeqFunction, bool>({prefix + "neq"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_neq, prefix + "neq");

  registerNonSimdizableScalar<LtFunction, bool>({prefix + "lt"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_lt, prefix + "lt");

  registerNonSimdizableScalar<GtFunction, bool>({prefix + "gt"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_gt, prefix + "gt");

  registerNonSimdizableScalar<LteFunction, bool>({prefix + "lte"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_lte, prefix + "lte");

  registerNonSimdizableScalar<GteFunction, bool>({prefix + "gte"});
  VELOX_REGISTER_VECTOR_FUNCTION(udf_simd_comparison_gte, prefix + "gte");

  registerBinaryScalar<DistinctFromFunction, bool>({prefix + "distinct_from"});

  registerFunction<BetweenFunction, bool, int8_t, int8_t, int8_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int16_t, int16_t, int16_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int32_t, int32_t, int32_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, int64_t, int64_t, int64_t>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, double, double, double>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, float, float, float>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, Varchar, Varchar, Varchar>(
      {prefix + "between"});
  registerFunction<BetweenFunction, bool, Date, Date, Date>(
      {prefix + "between"});
  registerFunction<
      BetweenFunction,
      bool,
      UnscaledShortDecimal,
      UnscaledShortDecimal,
      UnscaledShortDecimal>({prefix + "between"});
  registerFunction<
      BetweenFunction,
      bool,
      UnscaledLongDecimal,
      UnscaledLongDecimal,
      UnscaledLongDecimal>({prefix + "between"});
  registerFunction<
      GtFunction,
      bool,
      UnscaledShortDecimal,
      UnscaledShortDecimal>({prefix + "gt"});
  registerFunction<GtFunction, bool, UnscaledLongDecimal, UnscaledLongDecimal>(
      {prefix + "gt"});
  registerFunction<
      LtFunction,
      bool,
      UnscaledShortDecimal,
      UnscaledShortDecimal>({prefix + "lt"});
  registerFunction<LtFunction, bool, UnscaledLongDecimal, UnscaledLongDecimal>(
      {prefix + "lt"});

  registerFunction<
      GteFunction,
      bool,
      UnscaledShortDecimal,
      UnscaledShortDecimal>({prefix + "gte"});
  registerFunction<GteFunction, bool, UnscaledLongDecimal, UnscaledLongDecimal>(
      {prefix + "gte"});
  registerFunction<
      LteFunction,
      bool,
      UnscaledShortDecimal,
      UnscaledShortDecimal>({prefix + "lte"});
  registerFunction<LteFunction, bool, UnscaledLongDecimal, UnscaledLongDecimal>(
      {prefix + "lte"});
}

} // namespace facebook::velox::functions
