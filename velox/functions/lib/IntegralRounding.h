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
#include <cstdint>
#include <limits>
#include <type_traits>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Status.h"
#include "velox/type/DecimalArithmetic.h"

namespace facebook::velox::functions::detail {

// Rounds signed inputs up to BIGINT using the supplied rounded division.
// Shares scale bounds, ANSI overflow checks and wraparound across policies.
template <typename T, typename Divide>
FOLLY_ALWAYS_INLINE Status roundIntegral(
    T& result,
    T value,
    int32_t scale,
    bool ansiEnabled,
    Divide divide,
    fmt::string_view overflowMessage) {
  static_assert(
      std::is_integral_v<T> && std::is_signed_v<T> &&
      sizeof(T) <= sizeof(int64_t));
  if (scale >= 0 || value == 0) {
    result = value;
    return Status::OK();
  }
  const auto digitsToDrop = -static_cast<int64_t>(scale);
  if (digitsToDrop > std::numeric_limits<T>::digits10 + 1) {
    result = 0;
    return Status::OK();
  }

  // The divisor is at most 10^19, so division and rounding fit in int128_t.
  const auto divisor = DecimalArithmetic::kPowersOfTen[digitsToDrop];
  const int128_t rounded =
      divide(static_cast<int128_t>(value), divisor) * divisor;
  if (ansiEnabled &&
      (rounded < std::numeric_limits<T>::min() ||
       rounded > std::numeric_limits<T>::max())) {
    return threadSkipErrorDetails()
        ? Status::UserError()
        : Status::UserError(
              overflowMessage, static_cast<int64_t>(value), scale);
  }

  // Unsigned conversion followed by bit_cast gives Java's two's-complement
  // narrowing without signed overflow.
  result = std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(rounded));
  return Status::OK();
}

} // namespace facebook::velox::functions::detail
