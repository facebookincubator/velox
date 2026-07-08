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

#include "velox/type/HugeInt.h"

namespace facebook::velox {

int128_t HugeInt::parse(const std::string& str) {
  int128_t result = 0;
  bool negative = false;
  size_t idx = 0;

  VELOX_CHECK(!str.empty(), "Empty string cannot be converted to int128_t.");

  for (; idx < str.length() && str.at(idx) == ' '; ++idx) {
  }

  if (idx < str.length() && str.at(idx) == '+') {
    ++idx;
  } else if (idx < str.length() && str.at(idx) == '-') {
    ++idx;
    negative = true;
  }

  int128_t max = std::numeric_limits<int128_t>::max();
  int128_t min = std::numeric_limits<int128_t>::min();
  for (; idx < str.size(); ++idx) {
    VELOX_CHECK(
        std::isdigit(str[idx]),
        "Invalid character {} in the string.",
        str[idx]);

    // Throw error if the result is out of the range of int128_t, and return the
    // result before computing the last digit if the digit string would be the
    // min or max value of int128_t to avoid the potential overflow issue making
    // it more robust.
    int128_t cur = str[idx] - '0';
    if ((result > max / 10)) {
      VELOX_FAIL("Value is out of range of int128_t: {}", str);
    }

    int128_t num = cur - (max % 10);
    if (result == (max / 10)) {
      if (negative) {
        if (num > 1) {
          VELOX_FAIL("Value is out of range of int128_t: {}", str);
        } else if (num == 1) {
          return min;
        }
      } else {
        if (num > 0) {
          VELOX_FAIL("Value is out of range of int128_t: {}", str);
        } else if (num == 0) {
          return max;
        }
      }
    }

    result = result * 10 + cur;
  }

  return negative ? -result : result;
}
} // namespace facebook::velox

namespace std {

#ifdef _MSC_VER
string to_string(const facebook::velox::int128_t& x) {
#else
string to_string(facebook::velox::int128_t x) {
#endif
  if (x == 0) {
    return "0";
  }
  string ans;
  bool negative = x < 0;
#ifdef _MSC_VER
  // Use correct 128-bit division by 10 without relying on Int128 operator/.
  // The Int128 operator/ uses double conversion which loses precision.
  uint64_t hi, lo;
  if (negative) {
    // Negate: ~x + 1
    auto nx = -x;
    hi = static_cast<uint64_t>(nx.high());
    lo = nx.low();
  } else {
    hi = static_cast<uint64_t>(x.high());
    lo = x.low();
  }
  while (hi != 0 || lo != 0) {
    // Divide 128-bit (hi:lo) by 10, getting quotient and remainder.
    // First divide high part.
    uint64_t q_hi = hi / 10;
    uint64_t r_hi = hi % 10;
    // Combine remainder with low part: (r_hi * 2^64 + lo) / 10
    // Use MSVC _udiv128 or manual two-step division.
    // r_hi is at most 9, so r_hi * 2^64 + lo fits in 128 bits.
    // We can split: (r_hi * 2^64 + lo) = (r_hi * (2^64 / 10) * 10 + r_hi * (2^64 % 10) + lo)
    // Simpler: use the fact that r_hi < 10, so we can do two 64-bit divides.
    // high_part = r_hi * (2^63) => need to be careful with overflow.
    // Use: combined = r_hi * 2^64 + lo. Since r_hi < 10, this is < 10 * 2^64.
    // We split into: (r_hi * (UINT64_MAX / 10 + 1)) * 10 + (r_hi * (UINT64_MAX % 10 + 1) - r_hi * 10 * (UINT64_MAX / 10 + 1)) + lo
    // Actually, simpler: for each step, r_hi <= 9.
    // We can compute: temp = r_hi * (1ULL << 32) + (lo >> 32), then
    //   q_mid = temp / 10, r_mid = temp % 10
    //   temp2 = r_mid * (1ULL << 32) + (lo & 0xFFFFFFFF)
    //   q_lo_low = temp2 / 10, digit = temp2 % 10
    //   q_lo = q_mid * (1ULL << 32) + q_lo_low
    // This avoids 128-bit arithmetic entirely.
    uint64_t lo_high = lo >> 32;
    uint64_t lo_low = lo & 0xFFFFFFFFULL;
    uint64_t temp1 = (r_hi << 32) | lo_high;
    uint64_t q_mid = temp1 / 10;
    uint64_t r_mid = temp1 % 10;
    uint64_t temp2 = (r_mid << 32) | lo_low;
    uint64_t q_lo_low = temp2 / 10;
    uint64_t digit = temp2 % 10;
    uint64_t q_lo = (q_mid << 32) | q_lo_low;
    ans += '0' + static_cast<char>(digit);
    hi = q_hi;
    lo = q_lo;
  }
#else
  while (x != 0) {
    ans += '0' + abs(static_cast<int>(x % 10));
    x /= 10;
  }
#endif
  if (negative) {
    ans += '-';
  }
  reverse(ans.begin(), ans.end());
  return ans;
}

} // namespace std
