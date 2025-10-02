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

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include <folly/lang/Bits.h>
#include "velox/common/base/BitUtil.h"

namespace facebook::velox {

namespace detail::hasher {

namespace {

// https://github.com/stbrumme/xxhash/blob/c2866db364b6ea3a11933e62235ddc166ba18565/xxhash64.h#L167
constexpr uint64_t kPrime1 = 11400714785074694791ULL;
constexpr uint64_t kPrime2 = 14029467366897019727ULL;
constexpr uint64_t kPrime3 = 1609587929392839161ULL;

// https://github.com/stbrumme/xxhash/blob/c2866db364b6ea3a11933e62235ddc166ba18565/xxhash64.h#L182
inline uint64_t rotateLeft(uint64_t x, unsigned char bits) {
  return (x << bits) | (x >> (64 - bits));
}

// https://github.com/stbrumme/xxhash/blob/c2866db364b6ea3a11933e62235ddc166ba18565/xxhash64.h#L188
inline uint64_t processSingle(uint64_t input) {
  return rotateLeft(kPrime1 + input * kPrime3, 31) * kPrime2;
}

// https://github.com/stbrumme/xxhash/blob/c2866db364b6ea3a11933e62235ddc166ba18565/xxhash64.h#L143
inline uint64_t mix(uint64_t k) noexcept {
  k ^= k >> 33;
  k *= kPrime2;
  k ^= k >> 29;
  k *= kPrime3;
  k ^= k >> 32;
  return k;
}

inline uint64_t hash64XXHash(uint64_t value) noexcept {
  return mix(processSingle(value));
}

inline uint64_t hash64Avalanche(uint64_t key) {
  key ^= (key >> 30);
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= (key >> 27);
  key *= 0x94d049bb133111ebULL;
  key ^= (key >> 31);
  return key;
}

} // namespace

inline uint64_t hash64(uint64_t value) noexcept {
  return hash64Avalanche(value);
}

inline uint64_t combine128(uint64_t first, uint64_t second) noexcept {
  return first ^ (second * 0x9ddfea08eb382d69ULL);
}

template <typename T>
struct integral_hasher {
  inline constexpr uint64_t operator()(T value) const noexcept {
    constexpr auto size = sizeof(T);
    static_assert(size <= 16, "Input type is too wide");
    if constexpr (size <= 8) {
      return hash64(static_cast<uint64_t>(value));
    } else {
      using uint128_t = unsigned __int128;
      const auto u = static_cast<uint128_t>(value);
      const auto hi = static_cast<uint64_t>(u >> 64);
      const auto lo = static_cast<uint64_t>(u);
      return combine128(hash64(hi), hash64(lo));
    }
  }
};

template <typename T>
struct float_hasher {
  constexpr uint64_t operator()(T value) const noexcept {
    static_assert(sizeof(T) <= 8, "Input type is too wide");

    if (value == T{}) {
      // Ensure 0 and -0 get the same hash.
      value = 0;
    }

    if (std::isnan(value)) {
      // Ensure quiet and signaling NaNs get the same hash.
      value = std::numeric_limits<T>::quiet_NaN();
    }

    uint64_t u64 = 0;
    memcpy(&u64, &value, sizeof(T));
    return hash64(u64);
  }
};
} // namespace detail::hasher

template <typename T, typename Enable = void>
struct hasher;

template <>
struct hasher<bool> {
  constexpr uint64_t operator()(bool value) const noexcept {
    return value ? std::numeric_limits<uint64_t>::max() : 0;
  }
};

template <typename T>
struct hasher<T, std::enable_if_t<std::is_integral_v<T>>>
    : public detail::hasher::integral_hasher<T> {};

template <typename T>
struct hasher<T, std::enable_if_t<std::is_floating_point_v<T>>>
    : public detail::hasher::float_hasher<T> {};

template <>
struct hasher<__int128> : public detail::hasher::integral_hasher<__int128> {};

template <>
struct hasher<unsigned __int128>
    : public detail::hasher::integral_hasher<unsigned __int128> {};

template <typename T>
struct hasher<T*> {
  constexpr uint64_t operator()(T* value) const {
    return hasher<std::uintptr_t>{}(folly::bit_cast<std::uintptr_t>(value));
  }
};

template <typename T>
struct hasher<std::unique_ptr<T>> {
  constexpr uint64_t operator()(const std::unique_ptr<T>& value) const {
    return hasher<T*>{}(value.get());
  }
};

template <typename T>
struct hasher<std::shared_ptr<T>> {
  constexpr uint64_t operator()(const std::shared_ptr<T>& value) const {
    return hasher<T*>{}(value.get());
  }
};

template <>
struct hasher<std::string> {
  uint64_t operator()(const std::string& value) const {
    return bits::hashBytes(1, value.data(), value.size());
  }
};

struct Hash {
  template <class T>
  constexpr uint64_t operator()(const T& value) const
      noexcept(noexcept(hasher<T>()(value))) {
    return hasher<T>()(value);
  }

  template <class T, class... Ts>
  constexpr uint64_t operator()(const T& t, const Ts&... ts) const {
    return detail::hasher::combine128((*this)(t), (*this)(ts...));
  }

  constexpr uint64_t operator()() const noexcept {
    return 0;
  }
};

} // namespace facebook::velox
