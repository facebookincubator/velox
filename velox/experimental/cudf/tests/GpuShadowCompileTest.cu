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

// Compile-only test. Each verify* instantiates a Velox simple-function call()
// with TExec = GpuExec from a __device__ function, and probeKernel calls every
// verify*: nvcc generates device code only for what a kernel reaches, so a new
// verify* must be added there or it checks nothing. Results are discarded.

#include "velox/experimental/cudf/functions/GpuExec.h"

#include "velox/common/base/BitUtil.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"
#include "velox/type/FloatingPointUtil.h"

// Not covered: sparksql/Arithmetic.h needs the host-only ToHexUtil,
// sparksql/Comparisons.h mixes in VectorFunction factories, and
// prestosql/StringFunctions.h includes Udf.h and reaches host SIMD through
// StringImpl.h.

using namespace facebook::velox::gpu;

namespace {

namespace vfn = facebook::velox::functions;

#define VERIFY_UNARY(NAME, FN, OUT, IN) \
  __device__ void verify_##NAME() {     \
    vfn::FN<GpuExec> fn;                \
    OUT r{};                            \
    fn.call(r, IN);                     \
    (void)r;                            \
  }

#define VERIFY_BINARY(NAME, FN, OUT, A, B) \
  __device__ void verify_##NAME() {        \
    vfn::FN<GpuExec> fn;                   \
    OUT r{};                               \
    fn.call(r, A, B);                      \
    (void)r;                               \
  }

#define VERIFY_TERNARY(NAME, FN, OUT, A, B, C) \
  __device__ void verify_##NAME() {            \
    vfn::FN<GpuExec> fn;                       \
    OUT r{};                                   \
    fn.call(r, A, B, C);                       \
    (void)r;                                   \
  }

// --- Arithmetic: unary ---
VERIFY_UNARY(abs, AbsFunction, double, 1.0)
VERIFY_UNARY(ceil, CeilFunction, double, 1.5)
VERIFY_UNARY(floor, FloorFunction, double, 1.5)
VERIFY_UNARY(negate, NegateFunction, double, 1.0)
VERIFY_UNARY(sign, SignFunction, double, -2.0)
VERIFY_UNARY(truncate, TruncateFunction, double, 1.5)

// Both take a digit count, so both index DoubleUtil::powerOfTen()'s device
// table with a runtime value.
VERIFY_BINARY(round_to_digits, RoundFunction, double, 1.5, 2)
VERIFY_BINARY(truncate_to_digits, TruncateFunction, double, 1.5, 2)

// nvcc evaluates this in its device pass too, where powerOfTen() reads
// detail::kDeviceDoublePowersOfTen, so every entry of the device copy is
// checked against the host table at compile time.
constexpr bool devicePowersOfTenMatchHost() {
  using facebook::velox::DoubleUtil;
  for (int32_t i = 0; i < DoubleUtil::kNumPowersOfTen; ++i) {
    if (DoubleUtil::powerOfTen(i) !=
        facebook::velox::detail::kDoublePowersOfTen.values[i]) {
      return false;
    }
  }
  return true;
}
static_assert(devicePowersOfTenMatchHost());

// --- Arithmetic: binary ---
VERIFY_BINARY(plus, PlusFunction, double, 1.0, 2.0)
VERIFY_BINARY(minus, MinusFunction, double, 5.0, 3.0)
VERIFY_BINARY(multiply, MultiplyFunction, double, 2.0, 3.0)
VERIFY_BINARY(divide, DivideFunction, double, 6.0, 2.0)
VERIFY_BINARY(modulus, ModulusFunction, double, 7.0, 3.0)
VERIFY_BINARY(power, PowerFunction, double, 2.0, 8.0)

// --- Math and trigonometry ---
VERIFY_UNARY(exp, ExpFunction, double, 1.0)
VERIFY_UNARY(ln, LnFunction, double, 2.0)
VERIFY_UNARY(log2, Log2Function, double, 8.0)
VERIFY_UNARY(log10, Log10Function, double, 100.0)
VERIFY_UNARY(sqrt, SqrtFunction, double, 4.0)
VERIFY_UNARY(cbrt, CbrtFunction, double, 27.0)
VERIFY_UNARY(sin, SinFunction, double, 1.0)
VERIFY_UNARY(cos, CosFunction, double, 1.0)
VERIFY_UNARY(tan, TanFunction, double, 1.0)
VERIFY_UNARY(asin, AsinFunction, double, 0.5)
VERIFY_UNARY(acos, AcosFunction, double, 0.5)
VERIFY_UNARY(atan, AtanFunction, double, 1.0)
VERIFY_UNARY(cosh, CoshFunction, double, 1.0)
VERIFY_UNARY(tanh, TanhFunction, double, 1.0)
VERIFY_UNARY(degrees, DegreesFunction, double, 1.0)
VERIFY_UNARY(radians, RadiansFunction, double, 90.0)
VERIFY_BINARY(atan2, Atan2Function, double, 1.0, 1.0)

// --- Float predicates ---
VERIFY_UNARY(is_nan, IsNanFunction, bool, 1.0)
VERIFY_UNARY(is_finite, IsFiniteFunction, bool, 1.0)
VERIFY_UNARY(is_infinite, IsInfiniteFunction, bool, 1.0)

// --- Comparisons ---
VERIFY_BINARY(eq, EqFunction, bool, 1.0, 2.0)
VERIFY_BINARY(neq, NeqFunction, bool, 1.0, 2.0)
VERIFY_BINARY(lt, LtFunction, bool, 1.0, 2.0)
VERIFY_BINARY(lte, LteFunction, bool, 1.0, 2.0)
VERIFY_BINARY(gt, GtFunction, bool, 1.0, 2.0)
VERIFY_BINARY(gte, GteFunction, bool, 1.0, 2.0)
VERIFY_TERNARY(between, BetweenFunction, bool, 1.0, 0.0, 2.0)
VERIFY_TERNARY(clamp, ClampFunction, double, 1.0, 0.0, 2.0)

// --- Bitwise ---
VERIFY_BINARY(bitwise_and, BitwiseAndFunction, int64_t, int64_t{7}, int64_t{3})
VERIFY_BINARY(bitwise_or, BitwiseOrFunction, int64_t, int64_t{7}, int64_t{3})
VERIFY_BINARY(bitwise_xor, BitwiseXorFunction, int64_t, int64_t{7}, int64_t{3})
VERIFY_UNARY(bitwise_not, BitwiseNotFunction, int64_t, int64_t{7})
VERIFY_BINARY(
    bitwise_left_shift,
    BitwiseLeftShiftFunction,
    int64_t,
    int64_t{7},
    int32_t{2})
VERIFY_BINARY(
    bitwise_right_shift,
    BitwiseRightShiftFunction,
    int64_t,
    int64_t{7},
    int32_t{2})
VERIFY_BINARY(
    bitwise_right_shift_arithmetic,
    BitwiseRightShiftArithmeticFunction,
    int64_t,
    int64_t{7},
    int32_t{2})

// Device-callable, but their bodies validate input with VELOX_USER_CHECK,
// which the Exceptions.h shadow discards, so they are not registered.
// TODO(gpu-sfi-checks).
VERIFY_BINARY(bit_count, BitCountFunction, int64_t, int64_t{7}, int32_t{8})
VERIFY_BINARY(
    bitwise_arithmetic_shift_right,
    BitwiseArithmeticShiftRightFunction,
    int64_t,
    int64_t{7},
    int64_t{2})
VERIFY_TERNARY(
    bitwise_shift_left,
    BitwiseShiftLeftFunction,
    int64_t,
    int64_t{7},
    int64_t{2},
    int64_t{64})
VERIFY_TERNARY(
    bitwise_logical_shift_right,
    BitwiseLogicalShiftRightFunction,
    int64_t,
    int64_t{7},
    int64_t{2},
    int64_t{64})

// Calls the BitUtil shadow from device code. Not a static_assert, since
// __builtin_popcountll is not constexpr on every compiler.
__device__ int32_t
verifyCountBits(const uint64_t* bits, int32_t begin, int32_t end) {
  return facebook::velox::bits::countBits(bits, begin, end);
}

} // namespace

// Forces device codegen for every verifier above. Sinks results through a
// pointer so nothing is optimized away before it is type-checked.
__global__ void probeKernel(double* sink, const uint64_t* bits) {
  verify_abs();
  verify_ceil();
  verify_floor();
  verify_negate();
  verify_sign();
  verify_truncate();
  verify_round_to_digits();
  verify_truncate_to_digits();

  verify_plus();
  verify_minus();
  verify_multiply();
  verify_divide();
  verify_modulus();
  verify_power();

  verify_exp();
  verify_ln();
  verify_log2();
  verify_log10();
  verify_sqrt();
  verify_cbrt();
  verify_sin();
  verify_cos();
  verify_tan();
  verify_asin();
  verify_acos();
  verify_atan();
  verify_cosh();
  verify_tanh();
  verify_degrees();
  verify_radians();
  verify_atan2();

  verify_is_nan();
  verify_is_finite();
  verify_is_infinite();

  verify_eq();
  verify_neq();
  verify_lt();
  verify_lte();
  verify_gt();
  verify_gte();
  verify_between();
  verify_clamp();

  verify_bitwise_and();
  verify_bitwise_or();
  verify_bitwise_xor();
  verify_bitwise_not();
  verify_bitwise_left_shift();
  verify_bitwise_right_shift();
  verify_bitwise_right_shift_arithmetic();

  verify_bit_count();
  verify_bitwise_arithmetic_shift_right();
  verify_bitwise_shift_left();
  verify_bitwise_logical_shift_right();

  *sink = static_cast<double>(verifyCountBits(bits, 0, 64));
}
