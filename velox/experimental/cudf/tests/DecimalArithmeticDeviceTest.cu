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

// Runs the decimal arithmetic core on the device and checks it against the
// host: the CCCL overflow seam and the powerOfTen() device table. Built against
// gpu_shadows/, since the real Exceptions.h reaches folly code nvcc rejects.

#include "velox/experimental/cudf/tests/MapOnDevice.h"

#include "velox/type/DecimalArithmetic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

// CheckedArithmetic.h falls back to __builtin_*_overflow silently when CCCL
// lacks cuda::*_overflow. Without this, an older CCCL would leave the device
// branch uncompiled and every test below still passing.
#ifndef VELOX_HAS_DEVICE_OVERFLOW_INTRINSICS
#error "cuda::*_overflow is unavailable, so the device branch is not compiled"
#endif

namespace facebook::velox {
namespace {

using cudf_velox::mapOnDevice;

// nvcc evaluates these in its device pass as well as its host pass, so they
// pin both branches of powerOfTen() to the same literals at compile time.
static_assert(DecimalArithmetic::powerOfTen(0) == 1);
static_assert(
    DecimalArithmetic::powerOfTen(DecimalArithmetic::kMaxShortPrecision) ==
    1'000'000'000'000'000'000);
static_assert(
    DecimalArithmetic::powerOfTen(DecimalArithmetic::kMaxLongPrecision) ==
    1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
        (int128_t)100);

template <typename T>
struct Operands {
  T a;
  T b;
};

// What the seam reports for a + b, a - b and a * b.
template <typename T>
struct OverflowResult {
  T sum;
  T difference;
  T product;
  bool sumOverflowed;
  bool differenceOverflowed;
  bool productOverflowed;
};

template <typename T>
__global__ void
overflowOnDevice(const Operands<T>* in, OverflowResult<T>* out, int count) {
  for (int i = 0; i < count; ++i) {
    const auto [a, b] = in[i];
    out[i].sumOverflowed = detail::addOverflow(a, b, &out[i].sum);
    out[i].differenceOverflowed = detail::subOverflow(a, b, &out[i].difference);
    out[i].productOverflowed = detail::mulOverflow(a, b, &out[i].product);
  }
}

// Within this file the seam itself takes the CCCL branch on the host too, so
// the reference is the compiler builtin it replaces.
template <typename T>
OverflowResult<T> overflowOnHost(const Operands<T>& operands) {
  const auto [a, b] = operands;
  OverflowResult<T> result;
  result.sumOverflowed = __builtin_add_overflow(a, b, &result.sum);
  result.differenceOverflowed =
      __builtin_sub_overflow(a, b, &result.difference);
  result.productOverflowed = __builtin_mul_overflow(a, b, &result.product);
  return result;
}

// Both sides are wrapped values on overflow, so this checks the stored result
// as well as the flag.
template <typename T>
void testOverflowMatchesBuiltins(const std::vector<Operands<T>>& operands) {
  const auto device = mapOnDevice<Operands<T>, OverflowResult<T>>(
      operands, [](const Operands<T>* in, OverflowResult<T>* out, int count) {
        overflowOnDevice<<<1, 1>>>(in, out, count);
      });

  for (size_t i = 0; i < operands.size(); ++i) {
    SCOPED_TRACE(i);
    const auto host = overflowOnHost(operands[i]);
    EXPECT_EQ(device[i].sumOverflowed, host.sumOverflowed);
    EXPECT_EQ(device[i].differenceOverflowed, host.differenceOverflowed);
    EXPECT_EQ(device[i].productOverflowed, host.productOverflowed);
    // __int128 has no stream operator, so compare rather than EXPECT_EQ.
    EXPECT_TRUE(device[i].sum == host.sum);
    EXPECT_TRUE(device[i].difference == host.difference);
    EXPECT_TRUE(device[i].product == host.product);
  }
}

TEST(DecimalArithmeticDeviceTest, overflowSeamMatchesBuiltins) {
  constexpr auto kMin64 = std::numeric_limits<int64_t>::min();
  constexpr auto kMax64 = std::numeric_limits<int64_t>::max();
  testOverflowMatchesBuiltins<int64_t>({
      {7, 3},
      {kMax64, 1},
      {kMin64, 1},
      {kMax64, 2},
      {kMin64, -1},
  });

  constexpr auto kMin128 = std::numeric_limits<int128_t>::min();
  constexpr auto kMax128 = std::numeric_limits<int128_t>::max();
  testOverflowMatchesBuiltins<int128_t>({
      {7, 3},
      {kMax128, 1},
      {kMin128, 1},
      {kMax128, 2},
      {kMin128, -1},
      {DecimalArithmetic::kLongDecimalMax, DecimalArithmetic::kLongDecimalMax},
  });
}

__global__ void
decimalPowersOfTenOnDevice(const uint8_t* exponents, int128_t* out, int count) {
  for (int i = 0; i < count; ++i) {
    out[i] = DecimalArithmetic::powerOfTen(exponents[i]);
  }
}

// The exponents are read from device memory, so the device branch runs rather
// than folding, and is compared against the table the host indexes.
TEST(DecimalArithmeticDeviceTest, decimalPowerOfTenMatchesHostTable) {
  std::vector<uint8_t> exponents;
  for (uint8_t exponent = 0; exponent <= DecimalArithmetic::kMaxLongPrecision;
       ++exponent) {
    exponents.push_back(exponent);
  }
  const auto device = mapOnDevice<uint8_t, int128_t>(
      exponents, [](const uint8_t* in, int128_t* out, int count) {
        decimalPowersOfTenOnDevice<<<1, 1>>>(in, out, count);
      });

  for (const auto exponent : exponents) {
    SCOPED_TRACE(static_cast<int>(exponent));
    EXPECT_TRUE(device[exponent] == DecimalArithmetic::kPowersOfTen[exponent]);
  }
}

struct Division {
  int128_t dividend;
  int128_t divisor;
  uint8_t dividendRescale;
  int128_t quotient;
};

__global__ void
divideOnDevice(const Division* divisions, int128_t* quotients, int count) {
  for (int i = 0; i < count; ++i) {
    DecimalArithmetic::divideWithRoundUp<int128_t, int128_t, int128_t>(
        quotients[i],
        divisions[i].dividend,
        divisions[i].divisor,
        /*noRoundUp=*/false,
        divisions[i].dividendRescale,
        /*bRescale=*/0);
  }
}

// divideWithRoundUp rescales through checkedMultiply, so this exercises the
// seam the way the decimal call() bodies use it, against known answers.
TEST(DecimalArithmeticDeviceTest, divideWithRoundUpRoundsHalfAwayFromZero) {
  const std::vector<Division> divisions = {
      {7, 2, 0, 4}, // 3.5 rounds up.
      {-7, 2, 0, -4}, // And away from zero when negative.
      {10, 3, 0, 3}, // 3.33 rounds down.
      {2, 3, 0, 1}, // 0.67 rounds up to the first whole unit.
      {1, 3, 2, 33}, // Rescaled by 10^2 first: 100 / 3.
  };
  const auto quotients = mapOnDevice<Division, int128_t>(
      divisions, [](const Division* in, int128_t* out, int count) {
        divideOnDevice<<<1, 1>>>(in, out, count);
      });

  for (size_t i = 0; i < divisions.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_TRUE(quotients[i] == divisions[i].quotient);
  }
}

} // namespace
} // namespace facebook::velox
