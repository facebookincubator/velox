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

// Runs each device-annotated decimal routine on the GPU and checks that it
// returns what the host returns. Built against gpu_shadows/, since the real
// Exceptions.h reaches folly code nvcc rejects; the checks are no-ops here, so
// an overflow is observable only as the seam's flag, never as a throw.

#include "velox/experimental/cudf/tests/MapOnDevice.h"

#include "velox/functions/prestosql/detail/DecimalMathFunctions.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ostream>
#include <string_view>
#include <vector>

namespace facebook::velox {
namespace {

template <typename T>
struct Operands {
  T a;
  T b;
};

// What a routine produced and whether it overflowed, which the host reports by
// throwing and the device cannot.
struct Outcome {
  bool overflow;
  int128_t value;
  bool operator==(const Outcome&) const = default;
};

// Approximate, since neither fmt nor folly prints int128 under nvcc.
void PrintTo(const Outcome& outcome, std::ostream* os) {
  *os << (outcome.overflow ? "overflow " : "")
      << static_cast<double>(outcome.value);
}

template <typename In, typename Out, typename Routine>
__global__ void
applyOnDevice(Routine routine, const In* in, Out* out, int count) {
  for (int i = 0; i < count; ++i) {
    out[i] = routine(in[i]);
  }
}

// Runs `routine` over `inputs` on the GPU and expects each result to equal what
// its host reference, `Routine::host`, returns for the same input.
template <typename In, typename Routine>
void expectDeviceMatchesHost(
    std::string_view name,
    Routine routine,
    const std::vector<In>& inputs) {
  using Out = decltype(Routine::host(inputs[0]));
  SCOPED_TRACE(name);
  const auto outcomes = cudf_velox::mapOnDevice<In, Out>(
      inputs, [&](const In* in, Out* out, int count) {
        // One byte of dynamic shared memory per thread for the error sink.
        applyOnDevice<<<1, 1, sizeof(uint8_t)>>>(routine, in, out, count);
      });
  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(outcomes[i], Routine::host(inputs[i]));
  }
}

// checkedPlus, checkedMinus and checkedMultiply on one pair of operands, each
// beside the flag of the seam it is built on, against the compiler builtins
// that seam replaces.
struct Arithmetic {
  template <typename T>
  static std::array<Outcome, 3> host(Operands<T> o) {
    T sum;
    T difference;
    T product;
    const bool sumOverflow = __builtin_add_overflow(o.a, o.b, &sum);
    const bool differenceOverflow =
        __builtin_sub_overflow(o.a, o.b, &difference);
    const bool productOverflow = __builtin_mul_overflow(o.a, o.b, &product);
    return {
        Outcome{sumOverflow, sum},
        Outcome{differenceOverflow, difference},
        Outcome{productOverflow, product}};
  }
  template <typename T>
  __device__ std::array<Outcome, 3> operator()(Operands<T> o) const {
    T wrapped;
    const bool sumOverflow = addWithOverflow(o.a, o.b, &wrapped);
    const bool differenceOverflow = subWithOverflow(o.a, o.b, &wrapped);
    const bool productOverflow = mulWithOverflow(o.a, o.b, &wrapped);
    return {
        Outcome{sumOverflow, checkedPlus(o.a, o.b)},
        Outcome{differenceOverflow, checkedMinus(o.a, o.b)},
        Outcome{productOverflow, checkedMultiply(o.a, o.b)}};
  }
};

// Operands that overflow each operation in both directions, and one pair that
// overflows none.
template <typename T>
std::vector<Operands<T>> boundaries() {
  constexpr T kMin = std::numeric_limits<T>::min();
  constexpr T kMax = std::numeric_limits<T>::max();
  return {{7, 3}, {kMax, 1}, {kMin, -1}, {kMax, -1}, {kMin, 2}, {kMax, kMax}};
}

// The device's copy of the powers-of-ten table against the host's.
struct PowerOfTen {
  static Outcome host(uint8_t exponent) {
    return {false, DecimalArithmetic::kPowersOfTen[exponent]};
  }
  __device__ Outcome operator()(uint8_t exponent) const {
    return {false, DecimalArithmetic::powerOfTen(exponent)};
  }
};

// Stands in for exec::VectorExec, whose header nvcc cannot parse. The function
// types only name its resolver, and nothing here uses them.
struct IdentityExec {
  template <typename T>
  struct resolver {
    using in_type = T;
    using out_type = T;
  };
};

// A Presto decimal call() body, run from the same source on either side, on
// operands already at the result scale: value-initialization leaves the
// rescale factors zero, which keeps initialize() and the runtime types it needs
// out of a device test.
template <template <typename> class Function>
struct Presto {
  static Outcome host(Operands<int64_t> o) {
    return Presto{}(o);
  }
  VELOX_GPU_COMPATIBLE Outcome operator()(Operands<int64_t> o) const {
    int128_t result;
    Function<IdentityExec>{}.call(result, o.a, o.b);
    return {false, result};
  }
};

// Each routine must come back from the device with the host's outcome.
TEST(DecimalArithmeticDeviceTest, deviceMatchesHost) {
  expectDeviceMatchesHost(
      "checked arithmetic<int64_t>", Arithmetic{}, boundaries<int64_t>());
  expectDeviceMatchesHost(
      "checked arithmetic<int128_t>", Arithmetic{}, boundaries<int128_t>());

  std::vector<uint8_t> exponents(DecimalArithmetic::kMaxLongPrecision + 1);
  std::iota(exponents.begin(), exponents.end(), uint8_t{0});
  expectDeviceMatchesHost("powerOfTen", PowerOfTen{}, exponents);

  expectDeviceMatchesHost(
      "DecimalPlusFunction::call",
      Presto<functions::detail::DecimalPlusFunction>{},
      std::vector<Operands<int64_t>>{
          {700, 30}, {-1, 1}, {999'999'999'999'999'999, 1}});
  expectDeviceMatchesHost(
      "DecimalDivideFunction::call",
      Presto<functions::detail::DecimalDivideFunction>{},
      std::vector<Operands<int64_t>>{
          {7, 2}, {-7, 2}, {10, 3}, {2, 3}, {7, -2}});
}

} // namespace
} // namespace facebook::velox
