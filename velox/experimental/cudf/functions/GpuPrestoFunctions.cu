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

// Registers Velox Presto simple functions for native GPU execution.
// Each registerGpuFunction instantiates the GpuSimpleFunction bridge
// for that specific function+type combo.

#include "velox/experimental/cudf/functions/CudfLibraryFunctions.h"
#include "velox/experimental/cudf/functions/GpuSimpleFunction.cuh"
#include "velox/common/base/BitUtil.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::gpu {

void registerPrestoArithmetic() {
  using namespace facebook::velox::functions;

  // Plus
  registerGpuFunction<PlusFunction<GpuExec>, double, double, double>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, float, float, float>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int64_t, int64_t, int64_t>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int32_t, int32_t, int32_t>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int16_t, int16_t, int16_t>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int8_t, int8_t, int8_t>("plus");

  // Minus
  registerGpuFunction<MinusFunction<GpuExec>, double, double, double>("minus");
  registerGpuFunction<MinusFunction<GpuExec>, float, float, float>("minus");
  registerGpuFunction<MinusFunction<GpuExec>, int64_t, int64_t, int64_t>(
      "minus");
  registerGpuFunction<MinusFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "minus");
  registerGpuFunction<MinusFunction<GpuExec>, int16_t, int16_t, int16_t>(
      "minus");
  registerGpuFunction<MinusFunction<GpuExec>, int8_t, int8_t, int8_t>("minus");

  // Multiply
  registerGpuFunction<MultiplyFunction<GpuExec>, double, double, double>(
      "multiply");
  registerGpuFunction<MultiplyFunction<GpuExec>, float, float, float>(
      "multiply");
  registerGpuFunction<MultiplyFunction<GpuExec>, int64_t, int64_t, int64_t>(
      "multiply");
  registerGpuFunction<MultiplyFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "multiply");

  // Divide
  registerGpuFunction<DivideFunction<GpuExec>, double, double, double>(
      "divide");
  registerGpuFunction<DivideFunction<GpuExec>, float, float, float>("divide");

  // Modulus
  registerGpuFunction<ModulusFunction<GpuExec>, double, double, double>(
      "modulus");
  registerGpuFunction<ModulusFunction<GpuExec>, float, float, float>("modulus");
  registerGpuFunction<ModulusFunction<GpuExec>, int64_t, int64_t, int64_t>(
      "modulus");
  registerGpuFunction<ModulusFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "modulus");

  // Negate
  registerGpuFunction<NegateFunction<GpuExec>, double, double>("negate");
  registerGpuFunction<NegateFunction<GpuExec>, float, float>("negate");
  registerGpuFunction<NegateFunction<GpuExec>, int64_t, int64_t>("negate");
  registerGpuFunction<NegateFunction<GpuExec>, int32_t, int32_t>("negate");

  // Abs
  registerGpuFunction<AbsFunction<GpuExec>, double, double>("abs");
  registerGpuFunction<AbsFunction<GpuExec>, float, float>("abs");
  registerGpuFunction<AbsFunction<GpuExec>, int64_t, int64_t>("abs");
  registerGpuFunction<AbsFunction<GpuExec>, int32_t, int32_t>("abs");

  // Ceil/Floor
  registerGpuFunction<CeilFunction<GpuExec>, double, double>("ceil");
  registerGpuFunction<CeilFunction<GpuExec>, float, float>("ceil");
  registerGpuFunction<FloorFunction<GpuExec>, double, double>("floor");
  registerGpuFunction<FloorFunction<GpuExec>, float, float>("floor");

  // Truncate (1-arg: trunc to zero decimal places)
  registerGpuFunction<TruncateFunction<GpuExec>, double, double>("truncate");
  registerGpuFunction<TruncateFunction<GpuExec>, float, float>("truncate");

  // Math functions
  registerGpuFunction<ExpFunction<GpuExec>, double, double>("exp");
  registerGpuFunction<LnFunction<GpuExec>, double, double>("ln");
  registerGpuFunction<Log2Function<GpuExec>, double, double>("log2");
  registerGpuFunction<Log10Function<GpuExec>, double, double>("log10");
  registerGpuFunction<SqrtFunction<GpuExec>, double, double>("sqrt");
  registerGpuFunction<CbrtFunction<GpuExec>, double, double>("cbrt");

  // Trig
  registerGpuFunction<SinFunction<GpuExec>, double, double>("sin");
  registerGpuFunction<CosFunction<GpuExec>, double, double>("cos");
  registerGpuFunction<TanFunction<GpuExec>, double, double>("tan");
  registerGpuFunction<AsinFunction<GpuExec>, double, double>("asin");
  registerGpuFunction<AcosFunction<GpuExec>, double, double>("acos");
  registerGpuFunction<AtanFunction<GpuExec>, double, double>("atan");
  registerGpuFunction<Atan2Function<GpuExec>, double, double, double>("atan2");
  registerGpuFunction<CoshFunction<GpuExec>, double, double>("cosh");
  registerGpuFunction<TanhFunction<GpuExec>, double, double>("tanh");

  // Power
  registerGpuFunction<PowerFunction<GpuExec>, double, double, double>("power");

  // Sign
  registerGpuFunction<SignFunction<GpuExec>, double, double>("sign");
  registerGpuFunction<SignFunction<GpuExec>, float, float>("sign");
  registerGpuFunction<SignFunction<GpuExec>, int64_t, int64_t>("sign");
  registerGpuFunction<SignFunction<GpuExec>, int32_t, int32_t>("sign");

  // Radians / Degrees
  registerGpuFunction<RadiansFunction<GpuExec>, double, double>("radians");
  registerGpuFunction<DegreesFunction<GpuExec>, double, double>("degrees");

  // Predicates
  registerGpuFunction<IsFiniteFunction<GpuExec>, bool, double>("is_finite");
  registerGpuFunction<IsInfiniteFunction<GpuExec>, bool, double>("is_infinite");
  registerGpuFunction<IsNanFunction<GpuExec>, bool, double>("is_nan");

  // Clamp
  registerGpuFunction<ClampFunction<GpuExec>, double, double, double, double>(
      "clamp");
  registerGpuFunction<ClampFunction<GpuExec>, float, float, float, float>(
      "clamp");
  registerGpuFunction<
      ClampFunction<GpuExec>, int64_t, int64_t, int64_t, int64_t>("clamp");
  registerGpuFunction<
      ClampFunction<GpuExec>, int32_t, int32_t, int32_t, int32_t>("clamp");
}

void registerPrestoComparisons() {
  using namespace facebook::velox::functions;

  // eq/neq: handled by CudfFallbackRegistry (cudf::binary_operator::EQUAL /
  // NOT_EQUAL) since EqFunction/NeqFunction lack FOLLY_ALWAYS_INLINE.

  // Lt, Gt, Lte, Gte
  registerGpuFunction<LtFunction<GpuExec>, bool, double, double>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, float, float>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, int64_t, int64_t>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, int32_t, int32_t>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, int16_t, int16_t>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, int8_t, int8_t>("lt");

  registerGpuFunction<GtFunction<GpuExec>, bool, double, double>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, float, float>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, int64_t, int64_t>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, int32_t, int32_t>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, int16_t, int16_t>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, int8_t, int8_t>("gt");

  registerGpuFunction<LteFunction<GpuExec>, bool, double, double>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, float, float>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, int64_t, int64_t>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, int32_t, int32_t>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, int16_t, int16_t>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, int8_t, int8_t>("lte");

  registerGpuFunction<GteFunction<GpuExec>, bool, double, double>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, float, float>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, int64_t, int64_t>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, int32_t, int32_t>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, int16_t, int16_t>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, int8_t, int8_t>("gte");

  // Between
  registerGpuFunction<BetweenFunction<GpuExec>, bool, double, double, double>(
      "between");
  registerGpuFunction<BetweenFunction<GpuExec>, bool, float, float, float>(
      "between");
  registerGpuFunction<
      BetweenFunction<GpuExec>, bool, int64_t, int64_t, int64_t>("between");
  registerGpuFunction<
      BetweenFunction<GpuExec>, bool, int32_t, int32_t, int32_t>("between");
}

void registerPrestoBitwise() {
  using namespace facebook::velox::functions;

  registerGpuFunction<
      BitwiseAndFunction<GpuExec>, int64_t, int64_t, int64_t>("bitwise_and");
  registerGpuFunction<
      BitwiseAndFunction<GpuExec>, int64_t, int32_t, int32_t>("bitwise_and");
  registerGpuFunction<
      BitwiseAndFunction<GpuExec>, int64_t, int16_t, int16_t>("bitwise_and");
  registerGpuFunction<
      BitwiseAndFunction<GpuExec>, int64_t, int8_t, int8_t>("bitwise_and");

  registerGpuFunction<
      BitwiseOrFunction<GpuExec>, int64_t, int64_t, int64_t>("bitwise_or");
  registerGpuFunction<
      BitwiseOrFunction<GpuExec>, int64_t, int32_t, int32_t>("bitwise_or");
  registerGpuFunction<
      BitwiseOrFunction<GpuExec>, int64_t, int16_t, int16_t>("bitwise_or");
  registerGpuFunction<
      BitwiseOrFunction<GpuExec>, int64_t, int8_t, int8_t>("bitwise_or");

  registerGpuFunction<
      BitwiseXorFunction<GpuExec>, int64_t, int64_t, int64_t>("bitwise_xor");
  registerGpuFunction<
      BitwiseXorFunction<GpuExec>, int64_t, int32_t, int32_t>("bitwise_xor");

  registerGpuFunction<
      BitwiseNotFunction<GpuExec>, int64_t, int64_t>("bitwise_not");
  registerGpuFunction<
      BitwiseNotFunction<GpuExec>, int64_t, int32_t>("bitwise_not");

  registerGpuFunction<
      BitwiseLeftShiftFunction<GpuExec>, int64_t, int64_t, int32_t>(
      "bitwise_left_shift");
  registerGpuFunction<
      BitwiseLeftShiftFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "bitwise_left_shift");

  registerGpuFunction<
      BitwiseRightShiftFunction<GpuExec>, int64_t, int64_t, int32_t>(
      "bitwise_right_shift");
  registerGpuFunction<
      BitwiseRightShiftFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "bitwise_right_shift");

  registerGpuFunction<
      BitwiseRightShiftArithmeticFunction<GpuExec>,
      int64_t, int64_t, int32_t>("bitwise_arithmetic_shift_right");
  registerGpuFunction<
      BitwiseRightShiftArithmeticFunction<GpuExec>,
      int32_t, int32_t, int32_t>("bitwise_arithmetic_shift_right");
}

void registerAllPrestoGpuFunctions() {
  registerPrestoArithmetic();
  registerPrestoComparisons();
  registerPrestoBitwise();
  registerCudfStringFunctions();
  registerCudfDateTimeFunctions();
  registerCudfHashFunctions();
}

} // namespace facebook::velox::gpu
