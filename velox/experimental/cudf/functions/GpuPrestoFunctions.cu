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

// PrestoSQL simple functions compiled for GPU, registered the way
// velox/functions/prestosql/registration/*.cpp registers them on the CPU.
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuRegistrationHelpers.cuh"

// Bitwise.h calls bits::countBits without including BitUtil.h.
#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"

#include "velox/common/base/BitUtil.h"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

void registerPrestoGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  // Type sets follow MathematicalOperatorsRegistration.cpp and
  // MathematicalFunctionsRegistration.cpp helper for helper: the plain structs
  // for floating point, the Checked* ones for the integral overloads.
  registerGpuBinaryFloatingPoint<PlusFunction>({prefix + "plus"});
  registerGpuBinaryFloatingPoint<MinusFunction>({prefix + "minus"});
  registerGpuBinaryFloatingPoint<MultiplyFunction>({prefix + "multiply"});
  registerGpuBinaryFloatingPoint<DivideFunction>({prefix + "divide"});
  registerGpuBinaryFloatingPoint<ModulusFunction>({prefix + "mod"});

  // negate is floating point only upstream; its integral overloads come from
  // CheckedNegateFunction below.
  registerGpuUnaryFloatingPoint<NegateFunction>({prefix + "negate"});

  registerGpuUnaryNumeric<AbsFunction>({prefix + "abs"});
  registerGpuUnaryNumeric<CeilFunction>({prefix + "ceil", prefix + "ceiling"});
  registerGpuUnaryNumeric<FloorFunction>({prefix + "floor"});
  registerGpuUnaryNumeric<SignFunction>({prefix + "sign"});
  registerGpuTernaryNumeric<ClampFunction>({prefix + "clamp"});

  registerGpuFunction<PowerFunction, double, double, double>(
      {prefix + "power", prefix + "pow"});
  registerGpuFunction<PowerFunction, double, int64_t, int64_t>(
      {prefix + "power", prefix + "pow"});

  // One call() with a defaulted trailing parameter serves both arities
  // upstream, so each type appears once per arity.
  registerGpuUnaryNumeric<RoundFunction>({prefix + "round"});
  registerGpuNumericWithDecimals<RoundFunction>({prefix + "round"});
  // truncate is floating point only, both arities.
  registerGpuUnaryFloatingPoint<TruncateFunction>({prefix + "truncate"});
  registerGpuFunction<TruncateFunction, double, double, int32_t>(
      {prefix + "truncate"});
  registerGpuFunction<TruncateFunction, float, float, int32_t>(
      {prefix + "truncate"});

  // --- Checked integral arithmetic ----------------------------------------
  registerGpuBinaryIntegral<CheckedPlusFunction>({prefix + "plus"});
  registerGpuBinaryIntegral<CheckedMinusFunction>({prefix + "minus"});
  registerGpuBinaryIntegral<CheckedMultiplyFunction>({prefix + "multiply"});
  registerGpuBinaryIntegral<CheckedDivideFunction>({prefix + "divide"});
  registerGpuBinaryIntegral<CheckedModulusFunction>({prefix + "mod"});
  registerGpuUnaryIntegral<CheckedNegateFunction>({prefix + "negate"});

  // --- Math and trigonometry ---------------------------------------------
  registerGpuFunction<ExpFunction, double, double>({prefix + "exp"});
  registerGpuFunction<LnFunction, double, double>({prefix + "ln"});
  registerGpuFunction<Log2Function, double, double>({prefix + "log2"});
  registerGpuFunction<Log10Function, double, double>({prefix + "log10"});
  registerGpuFunction<SqrtFunction, double, double>({prefix + "sqrt"});
  registerGpuFunction<CbrtFunction, double, double>({prefix + "cbrt"});
  registerGpuFunction<SinFunction, double, double>({prefix + "sin"});
  registerGpuFunction<CosFunction, double, double>({prefix + "cos"});
  registerGpuFunction<TanFunction, double, double>({prefix + "tan"});
  registerGpuFunction<AsinFunction, double, double>({prefix + "asin"});
  registerGpuFunction<AcosFunction, double, double>({prefix + "acos"});
  registerGpuFunction<AtanFunction, double, double>({prefix + "atan"});
  registerGpuFunction<CoshFunction, double, double>({prefix + "cosh"});
  registerGpuFunction<TanhFunction, double, double>({prefix + "tanh"});
  registerGpuFunction<DegreesFunction, double, double>({prefix + "degrees"});
  registerGpuFunction<RadiansFunction, double, double>({prefix + "radians"});
  registerGpuFunction<Atan2Function, double, double, double>(
      {prefix + "atan2"});

  // --- Floating-point predicates -----------------------------------------
  registerGpuFunction<IsNanFunction, bool, double>({prefix + "is_nan"});
  registerGpuFunction<IsFiniteFunction, bool, double>({prefix + "is_finite"});
  registerGpuFunction<IsInfiniteFunction, bool, double>(
      {prefix + "is_infinite"});

  // --- Comparisons --------------------------------------------------------
  registerGpuBinaryNumericWithTReturn<LtFunction, bool>({prefix + "lt"});
  registerGpuBinaryNumericWithTReturn<LteFunction, bool>({prefix + "lte"});
  registerGpuBinaryNumericWithTReturn<GtFunction, bool>({prefix + "gt"});
  registerGpuBinaryNumericWithTReturn<GteFunction, bool>({prefix + "gte"});
  registerGpuTernaryNumericWithTReturn<BetweenFunction, bool>(
      {prefix + "between"});

  // --- Bitwise ------------------------------------------------------------
  registerGpuFunction<BitwiseAndFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_and"});
  registerGpuFunction<BitwiseOrFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_or"});
  registerGpuFunction<BitwiseXorFunction, int64_t, int64_t, int64_t>(
      {prefix + "bitwise_xor"});
  registerGpuFunction<BitwiseNotFunction, int64_t, int64_t>(
      {prefix + "bitwise_not"});
  registerGpuFunction<BitwiseLeftShiftFunction, int64_t, int64_t, int32_t>(
      {prefix + "bitwise_left_shift"});
  registerGpuFunction<BitwiseRightShiftFunction, int64_t, int64_t, int32_t>(
      {prefix + "bitwise_right_shift"});
  registerGpuFunction<
      BitwiseRightShiftArithmeticFunction,
      int64_t,
      int64_t,
      int32_t>({prefix + "bitwise_right_shift_arithmetic"});
  // The checked bitwise functions follow BitwiseFunctionsRegistration.cpp:
  // bit_count widens every integral pair to bigint, the rest are bigint only.
  registerGpuFunction<BitCountFunction, int64_t, int8_t, int8_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int16_t, int16_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int32_t, int32_t>(
      {prefix + "bit_count"});
  registerGpuFunction<BitCountFunction, int64_t, int64_t, int64_t>(
      {prefix + "bit_count"});
  registerGpuFunction<
      BitwiseArithmeticShiftRightFunction,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_arithmetic_shift_right"});
  registerGpuFunction<
      BitwiseLogicalShiftRightFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_logical_shift_right"});
  registerGpuFunction<
      BitwiseShiftLeftFunction,
      int64_t,
      int64_t,
      int64_t,
      int64_t>({prefix + "bitwise_shift_left"});

  // --- Logical -------------------------------------------------------------
  // See GpuLogicalFunctions.cuh. TODO: register is_null for every input type,
  // not only BOOLEAN.
  registerGpuFunction<GpuAndFunction, bool, Variadic<bool>>({prefix + "and"});
  registerGpuFunction<GpuOrFunction, bool, Variadic<bool>>({prefix + "or"});
  registerGpuFunction<GpuNotFunction, bool, bool>({prefix + "not"});
  registerGpuFunction<GpuIsNullFunction, bool, bool>({prefix + "is_null"});

  // eq and neq are absent because their call() bodies are host-only.
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
