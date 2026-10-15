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
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/DecimalFunctions.h"
#include "velox/functions/prestosql/DistanceFunctions.h"
#include "velox/functions/prestosql/Rand.h"

namespace facebook::velox::functions {

namespace {

void registerTruncate(
    const std::vector<std::string>& names,
    std::string_view defaultOwner) {
  registerFunction<TruncateFunction, double, double>(
      names, {}, true, defaultOwner);
  registerFunction<TruncateFunction, float, float>(
      names, {}, true, defaultOwner);
  registerFunction<TruncateFunction, double, double, int32_t>(
      names, {}, true, defaultOwner);
  registerFunction<TruncateFunction, float, float, int32_t>(
      names, {}, true, defaultOwner);
}

void registerMathFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerUnaryNumeric<CeilFunction>(
      {prefix + "ceil", prefix + "ceiling"}, defaultOwner);
  registerUnaryNumeric<FloorFunction>({prefix + "floor"}, defaultOwner);

  registerUnaryNumeric<AbsFunction>({prefix + "abs"}, defaultOwner);
  registerFunction<
      DecimalAbsFunction,
      LongDecimal<P1, S1>,
      LongDecimal<P1, S1>>({prefix + "abs"}, {}, true, defaultOwner);
  registerFunction<
      DecimalAbsFunction,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>>({prefix + "abs"}, {}, true, defaultOwner);

  registerUnaryFloatingPoint<NegateFunction>({prefix + "negate"}, defaultOwner);
  registerFunction<NegateFunction, LongDecimal<P1, S1>, LongDecimal<P1, S1>>(
      {prefix + "negate"}, {}, true, defaultOwner);
  registerFunction<NegateFunction, ShortDecimal<P1, S1>, ShortDecimal<P1, S1>>(
      {prefix + "negate"}, {}, true, defaultOwner);

  registerFunction<RadiansFunction, double, double>(
      {prefix + "radians"}, {}, true, defaultOwner);
  registerFunction<DegreesFunction, double, double>(
      {prefix + "degrees"}, {}, true, defaultOwner);
  registerUnaryNumeric<RoundFunction>({prefix + "round"}, defaultOwner);
  registerFunction<RoundFunction, int8_t, int8_t, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<RoundFunction, int16_t, int16_t, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<RoundFunction, int32_t, int32_t, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<RoundFunction, int64_t, int64_t, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<RoundFunction, double, double, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<RoundFunction, float, float, int32_t>(
      {prefix + "round"}, {}, true, defaultOwner);
  registerFunction<PowerFunction, double, double, double>(
      {prefix + "power", prefix + "pow"}, {}, true, defaultOwner);
  registerFunction<PowerFunction, double, int64_t, int64_t>(
      {prefix + "power", prefix + "pow"}, {}, true, defaultOwner);
  registerFunction<ExpFunction, double, double>(
      {prefix + "exp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, int8_t, int8_t, int8_t, int8_t>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, int16_t, int16_t, int16_t, int16_t>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, int32_t, int32_t, int32_t, int32_t>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, int64_t, int64_t, int64_t, int64_t>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, double, double, double, double>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<ClampFunction, float, float, float, float>(
      {prefix + "clamp"}, {}, true, defaultOwner);
  registerFunction<LnFunction, double, double>(
      {prefix + "ln"}, {}, true, defaultOwner);
  registerFunction<Log2Function, double, double>(
      {prefix + "log2"}, {}, true, defaultOwner);
  registerFunction<Log10Function, double, double>(
      {prefix + "log10"}, {}, true, defaultOwner);
  registerFunction<SqrtFunction, double, double>(
      {prefix + "sqrt"}, {}, true, defaultOwner);
  registerFunction<CbrtFunction, double, double>(
      {prefix + "cbrt"}, {}, true, defaultOwner);
  registerFunction<
      WidthBucketFunction,
      int64_t,
      double,
      double,
      double,
      int64_t>({prefix + "width_bucket"}, {}, true, defaultOwner);

  registerUnaryNumeric<SignFunction>({prefix + "sign"}, defaultOwner);
  registerFunction<InfinityFunction, double>(
      {prefix + "infinity"}, {}, true, defaultOwner);
  registerFunction<IsFiniteFunction, bool, double>(
      {prefix + "is_finite"}, {}, true, defaultOwner);
  registerFunction<IsInfiniteFunction, bool, double>(
      {prefix + "is_infinite"}, {}, true, defaultOwner);
  registerFunction<IsNanFunction, bool, double>(
      {prefix + "is_nan"}, {}, true, defaultOwner);
  registerFunction<NanFunction, double>(
      {prefix + "nan"}, {}, true, defaultOwner);
  registerFunction<RandFunction, double>(
      {prefix + "rand", prefix + "random"}, {}, true, defaultOwner);
  registerUnaryIntegral<RandFunction>(
      {prefix + "rand", prefix + "random"}, defaultOwner);
  registerFunction<SecureRandFunction, double>(
      {prefix + "secure_rand", prefix + "secure_random"},
      {},
      true,
      defaultOwner);
  registerBinaryNumeric<SecureRandFunction>(
      {prefix + "secure_rand", prefix + "secure_random"}, defaultOwner);
  registerFunction<FromBaseFunction, int64_t, Varchar, int64_t>(
      {prefix + "from_base"}, {}, true, defaultOwner);
  registerFunction<ToBaseFunction, Varchar, int64_t, int64_t>(
      {prefix + "to_base"}, {}, true, defaultOwner);
  registerFunction<PiFunction, double>({prefix + "pi"}, {}, true, defaultOwner);
  registerFunction<EulerConstantFunction, double>(
      {prefix + "e"}, {}, true, defaultOwner);

  registerTruncate({prefix + "truncate"}, defaultOwner);

  registerFunction<
      CosineSimilarityFunctionMap,
      double,
      Map<Varchar, double>,
      Map<Varchar, double>>(
      {prefix + "cosine_similarity"}, {}, true, defaultOwner);
  registerFunction<
      CosineSimilarityFunctionArray,
      double,
      Array<double>,
      Array<double>>({prefix + "cosine_similarity"}, {}, true, defaultOwner);
  registerFunction<DotProductArray, double, Array<double>, Array<double>>(
      {prefix + "dot_product"}, {}, true, defaultOwner);
#ifdef VELOX_ENABLE_FAISS
  registerFunction<
      CosineSimilarityFunctionFloatArray,
      float,
      Array<float>,
      Array<float>>({prefix + "cosine_similarity"}, {}, true, defaultOwner);
  registerFunction<
      L2SquaredFunctionFloatArray,
      float,
      Array<float>,
      Array<float>>({prefix + "l2_squared"}, {}, true, defaultOwner);
  registerFunction<
      L2SquaredFunctionDoubleArray,
      double,
      Array<double>,
      Array<double>>({prefix + "l2_squared"}, {}, true, defaultOwner);
  registerFunction<DotProductFloatArray, float, Array<float>, Array<float>>(
      {prefix + "dot_product"}, {}, true, defaultOwner);
#endif
}

} // namespace

void registerMathematicalFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {}) {
  registerMathFunctions(prefix, defaultOwner);
  VELOX_REGISTER_VECTOR_FUNCTION_WITH_OWNER(
      udf_not, prefix + "not", defaultOwner);

  registerDecimalFloor(prefix, defaultOwner);
  registerDecimalCeil(prefix, defaultOwner);
  registerDecimalRound(prefix, defaultOwner);
  registerDecimalTruncate(prefix, defaultOwner);
}

} // namespace facebook::velox::functions
