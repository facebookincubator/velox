/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <type_traits>

namespace facebook::nimble {

template <typename T, typename = void>
class NimbleCompare {
 public:
  static bool equals(const T& a, const T& b);
};

template <typename T, typename V>
inline bool NimbleCompare<T, V>::equals(const T& a, const T& b) {
  return a == b;
}

template <typename T>
class NimbleCompare<
    T,
    std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>> {
 public:
  // double or float
  using FloatingType =
      typename std::conditional<std::is_same_v<T, double>, double, float>::type;
  // 64bit integer or 32 bit integer
  using IntegralType = typename std::
      conditional<std::is_same_v<T, double>, int64_t, int32_t>::type;

  static_assert(sizeof(FloatingType) == sizeof(IntegralType));

  static bool equals(const T& a, const T& b);

  // This will be convenient when for debug, logging.
  static IntegralType asInteger(const T& a);
};

template <typename T>
inline bool NimbleCompare<
    T,
    std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>::
    equals(const T& a, const T& b) {
  // For floating point types, we do bit-wise comparison, for other types,
  // just use the original ==.
  // TODO: handle NaN.
  return *(reinterpret_cast<const IntegralType*>(&(a))) ==
      *(reinterpret_cast<const IntegralType*>(&(b)));
}

template <typename T>
inline typename NimbleCompare<
    T,
    std::enable_if_t<
        std::is_same_v<T, double> || std::is_same_v<T, float>>>::IntegralType
NimbleCompare<
    T,
    std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, float>>>::
    asInteger(const T& a) {
  return *(reinterpret_cast<const IntegralType*>(&(a)));
}

template <typename T>
struct NimbleComparator {
  constexpr bool operator()(const T& a, const T& b) const {
    return NimbleCompare<T>::equals(a, b);
  }
};

} // namespace facebook::nimble
