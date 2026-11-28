/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. Copyright (c) Facebook, Inc. and its affiliates.
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

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>

// The conversion helpers require the fixed-point types and powers above.
#include <cudf/fixed_point/detail/floating_conversion.hpp>

#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>

namespace facebook::velox::cudf_velox::sparksql::detail {

// Adapted from scaled_round in NVIDIA/cudf-spark-jni's decimal_utils.cu:
// https://github.com/NVIDIA/cudf-spark-jni/blob/
// 761c184ff0a0dc5b678b1f5f163c579068beb133/src/main/cpp/src/decimal_utils.cu
// This opt-in conversion uses binary-significand rounding, not Java decimal
// digit generation. Unlike the original, it checks storage bounds before
// conversion and keeps the result unsigned and wide until checking precision.
template <typename Input>
__device__ inline bool numericFloatingPointToDecimal(
    Input input,
    unsigned __int128 maxValue,
    int32_t scale,
    double factor,
    __int128& output) {
  static_assert(
      cuda::std::is_same_v<Input, float> ||
      cuda::std::is_same_v<Input, double>);
  using UInt128 = unsigned __int128;
  using Converter = numeric::detail::floating_converter<double>;

  const double value = static_cast<double>(input);
  if (!cuda::std::isfinite(value) || scale < 0 || scale > 38) {
    return false;
  }
  const double absolute = cuda::std::fabs(value);
  const double scaled = absolute * factor;
  // A valid DECIMAL(38, s) magnitude is below 10^38, well below 2^127.
  // This loose bound leaves room for floating multiplication error and prevents
  // overflow in the unsigned 128-bit shifting algorithm. The exact precision
  // check happens after rounding, before any narrowing or signed conversion.
  if (!cuda::std::isfinite(scaled) || scaled >= 0x1p127) {
    return false;
  }

  const auto bits = Converter::bit_cast_to_integer(value);
  if (Converter::is_zero(bits)) {
    output = 0;
    return true;
  }
  const auto [significand, floatingPow2] =
      Converter::get_significand_and_pow2(bits);
  const bool wholeNumber = cuda::std::floor(value) == value;
  // Add half a binary64 unit only for fractional DOUBLE inputs. REAL
  // retains its original type even though the shifting arithmetic uses DOUBLE.
  const uint64_t base2Value = (significand << 1) +
      static_cast<uint64_t>(cuda::std::is_same_v<Input, double> &&
                            !wholeNumber);
  const int32_t pow2 = floatingPow2 - 1;
  // Keep multiplication order stable at the extra-digit storage boundary.
  const bool canRound = 10.0 * absolute * factor < 0x1p127;
  const int32_t shiftingPow10 = -scale - static_cast<int32_t>(canRound);
  UInt128 magnitude =
      numeric::detail::convert_floating_to_integral_shifting<__int128, double>(
          base2Value, shiftingPow10, pow2);

  const bool roundUp = cuda::std::is_same_v<Input, double> &&
      absolute > static_cast<double>(cuda::std::numeric_limits<int64_t>::max());
  const int32_t floorPow10 =
      (3 * pow2 + 10 * scale + 9 * static_cast<int32_t>(roundUp)) / 10;
  constexpr UInt128 kUnsignedMax = ~UInt128{0};
  if (canRound) {
    if (floorPow10 < 0) {
      if (magnitude > kUnsignedMax - 5) {
        return false;
      }
      magnitude = (magnitude + 5) / 10;
    } else {
      // The storage guard bounds this exponent below 38. Keep the check here
      // to protect the power-of-ten helpers if the accepted range changes.
      if (floorPow10 >= 38) {
        return false;
      }
      if (cuda::std::is_same_v<Input, float> || wholeNumber) {
        const auto increment =
            numeric::detail::multiply_power10<__int128>(UInt128{5}, floorPow10);
        if (magnitude > kUnsignedMax - increment) {
          return false;
        }
        magnitude += increment;
      }
      const auto truncated =
          numeric::detail::divide_power10<__int128>(magnitude, floorPow10 + 1);
      magnitude =
          numeric::detail::multiply_power10<__int128>(truncated, floorPow10);
    }
  } else if (floorPow10 > 0) {
    if (floorPow10 > 38) {
      return false;
    }
    const auto truncated =
        numeric::detail::divide_power10<__int128>(magnitude, floorPow10);
    magnitude =
        numeric::detail::multiply_power10<__int128>(truncated, floorPow10);
  }

  if (magnitude > maxValue || magnitude > (kUnsignedMax >> 1)) {
    return false;
  }
  output = static_cast<__int128>(magnitude);
  if (Converter::get_is_negative(bits)) {
    output = -output;
  }
  return true;
}

} // namespace facebook::velox::cudf_velox::sparksql::detail
