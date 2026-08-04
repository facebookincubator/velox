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
// The analogue of velox/functions/prestosql/registration/*.cpp, and deliberately
// shaped like it: each line names a Velox function struct and its types, and the
// template instantiates here. A SparkSQL file registering its own struct under
// the same name would produce a different kernel, which is how the two dialects
// diverge on the CPU side too.
//
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"

// Bitwise.h calls bits::countBits but includes only Macros.h, so it relies on
// an includer having pulled BitUtil.h first. Included explicitly rather than
// depending on that.
#include "velox/common/base/BitUtil.h"

#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

void registerPrestoGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  registerGpuFunction<PlusFunction, double, double, double>({prefix + "plus"});
  registerGpuFunction<PlusFunction, int64_t, int64_t, int64_t>(
      {prefix + "plus"});
  registerGpuFunction<MinusFunction, double, double, double>(
      {prefix + "minus"});
  registerGpuFunction<MinusFunction, int64_t, int64_t, int64_t>(
      {prefix + "minus"});
  registerGpuFunction<MultiplyFunction, double, double, double>(
      {prefix + "multiply"});
  registerGpuFunction<MultiplyFunction, int64_t, int64_t, int64_t>(
      {prefix + "multiply"});
  registerGpuFunction<DivideFunction, double, double, double>(
      {prefix + "divide"});
  registerGpuFunction<ModulusFunction, double, double, double>(
      {prefix + "mod"});
  registerGpuFunction<NegateFunction, double, double>({prefix + "negate"});
  registerGpuFunction<NegateFunction, int64_t, int64_t>({prefix + "negate"});
  registerGpuFunction<AbsFunction, double, double>({prefix + "abs"});
  registerGpuFunction<CeilFunction, double, double>({prefix + "ceil"});
  registerGpuFunction<FloorFunction, double, double>({prefix + "floor"});
  registerGpuFunction<SignFunction, double, double>({prefix + "sign"});
  registerGpuFunction<PowerFunction, double, double, double>(
      {prefix + "power", prefix + "pow"});

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
  registerGpuFunction<LtFunction, bool, double, double>({prefix + "lt"});
  registerGpuFunction<LtFunction, bool, int64_t, int64_t>({prefix + "lt"});
  registerGpuFunction<LteFunction, bool, double, double>({prefix + "lte"});
  registerGpuFunction<LteFunction, bool, int64_t, int64_t>({prefix + "lte"});
  registerGpuFunction<GtFunction, bool, double, double>({prefix + "gt"});
  registerGpuFunction<GtFunction, bool, int64_t, int64_t>({prefix + "gt"});
  registerGpuFunction<GteFunction, bool, double, double>({prefix + "gte"});
  registerGpuFunction<GteFunction, bool, int64_t, int64_t>({prefix + "gte"});
  registerGpuFunction<BetweenFunction, bool, double, double, double>(
      {prefix + "between"});
  registerGpuFunction<ClampFunction, double, double, double, double>(
      {prefix + "clamp"});

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
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
