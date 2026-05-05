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

#include <cstdint>
#include <cstring>

namespace facebook::velox {

/// Lookup table of two-digit decimal strings packed contiguously, so
/// that `kTwoDigitChars + 2 * n` points at the two ASCII bytes of `n`
/// for any n in [0, 99]. Storing as char[] lets callers emit a fixed-
/// width two-digit field with a single 16-bit memcpy per pair, faster
/// than std::to_chars + std::memset for zero-padding or per-byte
/// stores via division/modulo. Used by the date / timestamp formatters
/// in DateType::toIso8601 and Timestamp::tsToStringView.
inline constexpr char kTwoDigitChars[] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

/// Writes the two ASCII bytes of `value` (which must be in [0, 99])
/// to the two bytes starting at `out` via a single 16-bit memcpy.
inline void writeTwoDigit(char* out, uint32_t value) {
  std::memcpy(out, kTwoDigitChars + 2u * value, 2);
}

} // namespace facebook::velox
