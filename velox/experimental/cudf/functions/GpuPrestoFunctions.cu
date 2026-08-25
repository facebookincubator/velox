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

// PrestoSQL simple functions compiled for GPU.
//
// The analogue of velox/functions/prestosql/registration/*.cpp, and
// deliberately shaped like it: each line names a Velox function struct and its
// types, and the template instantiates here. A SparkSQL file registering its
// own struct under the same name would produce a different kernel, which is how
// the two dialects diverge on the CPU side too.
//
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuRegistrationHelpers.cuh"

// Bitwise.h calls bits::countBits but includes only Macros.h, so it relies on
// an includer having pulled BitUtil.h first. Included explicitly rather than
// depending on that.
#include "velox/common/base/BitUtil.h"
#include "velox/experimental/cudf/functions/GpuDateTimeFunctions.cuh"
#include "velox/experimental/cudf/functions/GpuLogicalFunctions.cuh"
#include "velox/functions/lib/CheckedArithmetic.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

void registerPrestoGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  // Type sets follow MathematicalOperatorsRegistration.cpp and
  // MathematicalFunctionsRegistration.cpp exactly, helper for helper, so a
  // divergence in coverage shows up as a different helper here rather than as a
  // missing overload nobody notices.
  //
  // Floating point uses the plain structs; the integral overloads use the
  // Checked* ones, which is how Presto binds these names on the CPU -- see
  // registerCheckedArithmeticFunctions.
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
  //
  // TODO(gpu-sfi): these do not raise on overflow yet. checkedPlus and friends
  // detect overflow correctly on device -- the CCCL intrinsics behind them are
  // bit-identical to the host builtins -- but they report it by expanding
  // VELOX_ARITHMETIC_ERROR, which the gpu_shadows Exceptions.h reduces to
  // VELOX_GPU_SHADOW_NOOP_CHECK. So an overflowing row silently yields the
  // wrapped value here where Presto on CPU throws. Registered anyway for
  // coverage; once the P2 per-row error buffer lands, VELOX_ARITHMETIC_ERROR
  // writes an error code and these become faithful with no change at this call
  // site. The same gap holds back bit_count and the three shift functions
  // listed at the bottom of this file.
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
  registerGpuTernaryNumericWithTReturn<BetweenFunction, bool>({prefix + "between"});

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

  // --- Datetime -----------------------------------------------------------
  // DATE only. The TIMESTAMP overloads of these functions read a session
  // timezone, which is host state; see GpuDateTimeFunctions.cuh for why these
  // wrap the shared accessors instead of the Velox structs.
  registerGpuFunction<GpuYearFunction, int64_t, Date>({prefix + "year"});
  registerGpuFunction<GpuMonthFunction, int64_t, Date>({prefix + "month"});
  registerGpuFunction<GpuDayFunction, int64_t, Date>(
      {prefix + "day", prefix + "day_of_month"});
  registerGpuFunction<GpuQuarterFunction, int64_t, Date>({prefix + "quarter"});
  registerGpuFunction<GpuDayOfYearFunction, int64_t, Date>(
      {prefix + "day_of_year", prefix + "doy"});
  registerGpuFunction<GpuDayOfWeekFunction, int64_t, Date>(
      {prefix + "day_of_week", prefix + "dow"});

  // --- Logical -------------------------------------------------------------
  // Special forms on the CPU, ordinary functions here; see
  // GpuLogicalFunctions.cuh. is_null is registered for BOOLEAN only for now --
  // it is type-generic, and giving it every input type is what the argument
  // breadth work is for.
  registerGpuFunction<GpuAndFunction, bool, Variadic<bool>>({prefix + "and"});
  registerGpuFunction<GpuOrFunction, bool, Variadic<bool>>({prefix + "or"});
  registerGpuFunction<GpuNotFunction, bool, bool>({prefix + "not"});
  registerGpuFunction<GpuIsNullFunction, bool, bool>({prefix + "is_null"});

  // Deliberately not registered. These compile and are device-callable, but
  // their bodies use VELOX_USER_CHECK, which the Exceptions.h shadow reduces to
  // a no-op, so on GPU they would accept input the CPU path rejects and return
  // a wrong answer rather than an error:
  //
  //   bit_count, bitwise_arithmetic_shift_right, bitwise_shift_left,
  //   bitwise_logical_shift_right
  //
  // They are covered by the compile test so the day per-row error reporting
  // exists, registering them is a four-line change. Also absent: eq and neq,
  // host-only until #18385 lands, and truncate, which reads
  // DoubleUtil::kPowersOfTen with a runtime index.
  //
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
