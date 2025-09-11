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

#include <cstring>

#include "velox/common/base/BitUtil.h"

namespace facebook::velox::functions {

class HashCodeUtils {
 public:
  template <typename T>
  static int64_t hashInteger(T value) {
    return bits::rotateLeft64(
               static_cast<uint64_t>(value) * 0xC2B2AE3D27D4EB4FL, 31) *
        0x9E3779B185EBCA87L;
  }

  static inline int64_t hashReal(float value) {
    canonicalizeZero<float>(value);
    return hashInteger<int32_t>(floatToIntBits(value));
  }

  static inline int64_t hashDouble(double value) {
    canonicalizeZero<double>(value);
    return hashInteger<int64_t>(doubleToLongBits(value));
  }

 private:
  // Canonicalize +0 and -0 to a single value.
  template <typename T>
  static void canonicalizeZero(T& value) {
    if (value == 0.0f && std::signbit(value)) {
      value = 0.0f;
    }
  }

  // Canonicalize all NaNs to the same representation.
  static int32_t floatToIntBits(float x) {
    if (std::isnan(x)) {
      return 0x7fc00000;
    } else {
      int32_t bits;
      std::memcpy(&bits, &x, sizeof(bits));
      return bits;
    }
  }

  // Canonicalize all NaNs to the same representation.
  static int64_t doubleToLongBits(double x) {
    if (std::isnan(x)) {
      return 0x7ff8000000000000ULL;
    } else {
      int64_t bits;
      std::memcpy(&bits, &x, sizeof(bits));
      return bits;
    }
  }
};
} // namespace facebook::velox::functions
