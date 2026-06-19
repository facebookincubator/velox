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
#include <string>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"

#pragma once

#ifdef _MSC_VER
#include "velox/type/windows/Int128.h"
#else
namespace facebook::velox {
using int128_t = __int128_t;
using uint128_t = __uint128_t;
}
#endif

namespace facebook::velox {

class HugeInt {
 public:
  static constexpr FOLLY_ALWAYS_INLINE int128_t
  build(uint64_t hi, uint64_t lo) {
#ifdef _MSC_VER
    return int128_t(static_cast<int64_t>(hi), lo);
#else
    // GCC does not allow left shift negative value.
    return (static_cast<__uint128_t>(hi) << 64) | lo;
#endif
  }

  static constexpr FOLLY_ALWAYS_INLINE uint64_t lower(int128_t value) {
#ifdef _MSC_VER
    return value.low();
#else
    return static_cast<uint64_t>(value);
#endif
  }

  static constexpr FOLLY_ALWAYS_INLINE uint64_t upper(int128_t value) {
#ifdef _MSC_VER
    return static_cast<uint64_t>(value.high());
#else
    return static_cast<uint64_t>(value >> 64);
#endif
  }

  static FOLLY_ALWAYS_INLINE int128_t deserialize(const char* serializedData) {
    int128_t value;
    memcpy(&value, serializedData, sizeof(int128_t));
    return value;
  }

  static FOLLY_ALWAYS_INLINE void serialize(
      const int128_t& value,
      char* serializedData) {
    memcpy(serializedData, &value, sizeof(int128_t));
  }

  /// Read 16 bytes from memory into int128_t with canonical byte mapping.
  /// On all platforms, src[0:8] maps to the lower 64 bits and src[8:16] maps
  /// to the upper 64 bits (matching GCC's __int128_t little-endian layout).
  /// MSVC Int128 now uses [low_, high_] layout to match.
  static FOLLY_ALWAYS_INLINE int128_t deserializeNative(const void* src) {
    int128_t value;
    memcpy(&value, src, sizeof(int128_t));
    return value;
  }

  /// Write int128_t to 16 bytes of memory with canonical byte mapping.
  /// On all platforms, lower 64 bits go to dst[0:8] and upper 64 bits go to
  /// dst[8:16] (matching GCC's __int128_t little-endian layout).
  /// MSVC Int128 now uses [low_, high_] layout to match.
  static FOLLY_ALWAYS_INLINE void serializeNative(
      const int128_t& value,
      void* dst) {
    memcpy(dst, &value, sizeof(int128_t));
  }

  static int128_t parse(const std::string& str);
};

} // namespace facebook::velox

namespace std {
#ifdef _MSC_VER
string to_string(const facebook::velox::int128_t& x);
#else
string to_string(__int128_t x);
#endif
} // namespace std
