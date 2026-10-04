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

// Compile-only test. Proves that nvcc, with gpu_shadows/ ahead of the Velox
// source root and --diag-error=20011, generates device code for each mechanism
// a Velox simple function relies on when it runs on the GPU: one probe per
// shadowed header or annotation. Registration compiles every registered
// function's call() as a kernel under the same flags, so there are no
// per-function probes here. nvcc generates device code only for what a kernel
// reaches, so probeKernel must call every probe or it checks nothing.

#include "velox/experimental/cudf/functions/GpuExec.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/detail/DateTimeCalendarFunctions.h"
#include "velox/type/FloatingPointUtil.h"
// Included here so the SimpleFunctionApi.h shadow is reached whatever the
// headers above include.
#include "velox/type/SimpleFunctionApi.h"

// Not covered: sparksql/Arithmetic.h needs the host-only ToHexUtil,
// sparksql/Comparisons.h mixes in VectorFunction factories, and
// prestosql/StringFunctions.h includes Udf.h and reaches host SIMD through
// StringImpl.h.

namespace {

using namespace facebook::velox;
using gpu::GpuExec;

// A VELOX_GPU_COMPATIBLE call() from a Velox function header, instantiated
// with TExec = GpuExec. truncate(x, n) also indexes DoubleUtil::powerOfTen()'s
// device table with a runtime digit count.
__device__ double verifyVeloxCall(double value, int32_t digits) {
  double result{};
  functions::TruncateFunction<GpuExec>{}.call(result, value, digits);
  return result;
}

// nvcc evaluates this in its device pass too, where powerOfTen() reads
// detail::kDeviceDoublePowersOfTen, so every entry of the device copy is
// checked against the host table at compile time.
constexpr bool devicePowersOfTenMatchHost() {
  for (int32_t i = 0; i < DoubleUtil::kNumPowersOfTen; ++i) {
    if (DoubleUtil::powerOfTen(i) !=
        facebook::velox::detail::kDoublePowersOfTen.values[i]) {
      return false;
    }
  }
  return true;
}
static_assert(devicePowersOfTenMatchHost());

// The Exceptions.h shadow records a failed check in the launch's shared memory
// and lets the body run on, so a check site compiles with nothing passed to it.
__device__ void verifyChecks(int64_t value) {
  VELOX_USER_CHECK_GE(value, 0, "Value must not be negative: {}", value);
  VELOX_CHECK_LT(value, 100);
  VELOX_USER_FAIL("Unsupported value: {}", value);
}

// Resolves the Date and Timestamp tags through GpuExec: a DATE arrives as its
// day count and a TIMESTAMP as Velox's own Timestamp, whose constructors and
// comparisons run on the device.
template <typename TExec>
struct LaterThanFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  VELOX_GPU_COMPATIBLE void call(
      out_type<bool>& result,
      const arg_type<Timestamp>& timestamp,
      const arg_type<Date>& date) {
    result = timestamp > Timestamp::fromDate(date);
  }
};

__device__ bool verifyTypeTags(int64_t seconds, int32_t date) {
  bool result{};
  LaterThanFunction<GpuExec>{}.call(result, Timestamp(seconds, 0), date);
  return result;
}

// Instantiates one of Velox's calendar functions for the device, over the
// real Timestamp and the shadowed time zone: a core change that no longer
// parses behind the shadow, or that reaches a host-only call, fails here
// before any registration does.
__device__ int64_t verifyCalendar(const Timestamp& timestamp) {
  int64_t result{};
  functions::YearFunction<GpuExec>{}.call(result, timestamp);
  return result;
}

} // namespace

// Reaches every probe above and sinks the results, so nothing is optimized
// away before it is type-checked. countBits is the BitUtil.h shadow.
__global__ void probeKernel(double* sink, const uint64_t* bits) {
  const auto value = static_cast<int64_t>(bits[0]);
  verifyChecks(value);
  *sink = verifyVeloxCall(sink[0], static_cast<int32_t>(value)) +
      (verifyTypeTags(value, static_cast<int32_t>(bits[1])) ? 1.0 : 0.0) +
      static_cast<double>(verifyCalendar(Timestamp(value, 0))) +
      static_cast<double>(bits::countBits(bits, 0, 64));
}
