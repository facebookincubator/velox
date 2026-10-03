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

// Compile-only test for simple functions that are compiled for the device but
// not registered; registration already compiles every registered function as a
// kernel under the same flags. Each verify* instantiates call() with
// TExec = GpuExec, and probeKernel calls every verify*, since nvcc generates
// device code only for what a kernel reaches.

#include "velox/experimental/cudf/functions/GpuExec.h"

#include "velox/common/base/BitUtil.h"
// Bitwise.h uses VELOX_USER_CHECK without including Exceptions.h.
#include "velox/common/base/Exceptions.h"
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

VERIFY_BINARY(eq, EqFunction, bool, 1.0, 2.0)
VERIFY_BINARY(neq, NeqFunction, bool, 1.0, 2.0)

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
  verify_eq();
  verify_neq();
  verify_bit_count();
  verify_bitwise_arithmetic_shift_right();
  verify_bitwise_shift_left();
  verify_bitwise_logical_shift_right();

  *sink = static_cast<double>(verifyCountBits(bits, 0, 64));
}
