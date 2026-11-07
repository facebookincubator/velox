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
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace facebook::velox::functions::sparksql {

/// Computes Spark's primitive PMOD for a nonzero divisor.
template <typename T>
T computePmod(T dividend, T divisor) {
  if constexpr (std::is_floating_point_v<T>) {
    const T remainder = std::fmod(dividend, divisor);
    // Preserve signed zero and NaN. Keep REAL arithmetic at float precision.
    if (remainder < 0) {
      const T adjusted = remainder + divisor;
      return std::fmod(adjusted, divisor);
    }
    return remainder;
  } else {
    // Java defines MIN % -1 as zero; C++ signed division would overflow.
    if (divisor == -1) {
      return 0;
    }
    const T remainder = dividend % divisor;
    if (remainder >= 0) {
      return remainder;
    }
    if constexpr (sizeof(T) < sizeof(int32_t)) {
      // Java promotes byte and short arithmetic to int before narrowing.
      return (remainder + divisor) % divisor;
    } else {
      // Java int and long addition wraps, including in ANSI PMOD.
      using Unsigned = std::make_unsigned_t<T>;
      const Unsigned adjusted =
          static_cast<Unsigned>(remainder) + static_cast<Unsigned>(divisor);
      return std::bit_cast<T>(adjusted) % divisor;
    }
  }
}

} // namespace facebook::velox::functions::sparksql
