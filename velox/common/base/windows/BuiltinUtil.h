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

#ifdef _MSC_VER

#include <limits>
#include <type_traits>

namespace facebook::velox::windows {

template <typename T>
inline bool builtin_add_overflow(T a, T b, T* result) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  using Limits = std::numeric_limits<T>;

  if constexpr (std::is_signed_v<T>) {
    if ((b > 0 && a > Limits::max() - b) ||
        (b < 0 && a < Limits::min() - b)) {
      return true;
    }
  } else if (a > Limits::max() - b) {
    return true;
  }

  *result = static_cast<T>(a + b);
  return false;
}

template <typename T>
inline bool builtin_sub_overflow(T a, T b, T* result) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  using Limits = std::numeric_limits<T>;

  if constexpr (std::is_signed_v<T>) {
    if ((b < 0 && a > Limits::max() + b) ||
        (b > 0 && a < Limits::min() + b)) {
      return true;
    }
  } else if (a < b) {
    return true;
  }

  *result = static_cast<T>(a - b);
  return false;
}

template <typename T>
inline bool builtin_mul_overflow(T a, T b, T* result) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  using Limits = std::numeric_limits<T>;

  if constexpr (std::is_signed_v<T>) {
    if (a == 0 || b == 0) {
      *result = 0;
      return false;
    }
    if ((a == -1 && b == Limits::min()) ||
        (b == -1 && a == Limits::min())) {
      return true;
    }
    if (a > 0) {
      if ((b > 0 && a > Limits::max() / b) ||
          (b < 0 && b < Limits::min() / a)) {
        return true;
      }
    } else if (b > 0) {
      if (a < Limits::min() / b) {
        return true;
      }
    } else if (b < Limits::max() / a) {
      return true;
    }
  } else if (b != 0 && a > Limits::max() / b) {
    return true;
  }

  *result = static_cast<T>(a * b);
  return false;
}

} // namespace facebook::velox::windows

#endif // _MSC_VER