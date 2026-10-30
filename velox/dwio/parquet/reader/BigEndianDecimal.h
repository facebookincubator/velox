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

#include <algorithm>
#include <cstring>
#include <string_view>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox::parquet {

/// True for the integer types that hold decimal values read from Parquet:
/// int64_t for short decimals and int128_t for long decimals.
template <typename T>
inline constexpr bool kIsDecimalValueType =
    std::is_same_v<T, int64_t> || std::is_same_v<T, int128_t>;

/// Converts the unscaled value of a Parquet decimal, stored as big-endian
/// two's complement bytes, to 'T'. Values shorter than 'T' are sign-extended.
/// The Parquet spec allows a FIXED_LEN_BYTE_ARRAY wider than the precision
/// needs, e.g. DECIMAL(10, 2) in 16 bytes. Such values are read if the extra
/// leading bytes are sign extension, and throw otherwise. 'bytes' must not be
/// empty; the reader requires a positive FIXED_LEN_BYTE_ARRAY length.
template <typename T>
FOLLY_ALWAYS_INLINE T decimalFromBigEndian(std::string_view bytes) {
  static_assert(
      kIsDecimalValueType<T>, "Decimal values are int64_t or int128_t");
  VELOX_DCHECK_GT(bytes.size(), 0);

  const auto* rawBytes = reinterpret_cast<const uint8_t*>(bytes.data());
  auto numBytes = bytes.size();
  if (numBytes > sizeof(T)) [[unlikely]] {
    const auto numExtraBytes = numBytes - sizeof(T);
    const uint8_t signByte = (rawBytes[numExtraBytes] & 0x80) != 0 ? 0xFF : 0;
    VELOX_CHECK(
        std::all_of(
            rawBytes,
            rawBytes + numExtraBytes,
            [signByte](uint8_t byte) { return byte == signByte; }),
        "Decimal value of {} bytes does not fit in {} bytes",
        numBytes,
        sizeof(T));
    rawBytes += numExtraBytes;
    numBytes = sizeof(T);
  }

  T result = (rawBytes[0] & 0x80) != 0 ? -1 : 0;
  memcpy(
      reinterpret_cast<uint8_t*>(&result) + sizeof(T) - numBytes,
      rawBytes,
      numBytes);
  if constexpr (std::is_same_v<T, int128_t>) {
    return bits::builtin_bswap128(result);
  } else {
    return __builtin_bswap64(result);
  }
}

} // namespace facebook::velox::parquet
