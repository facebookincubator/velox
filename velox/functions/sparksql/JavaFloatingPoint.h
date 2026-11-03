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
#include <limits>
#include <type_traits>

#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

namespace facebook::velox::functions::sparksql {

/// Equality, hashing and ordering of float and double the way Java's boxed
/// Float and Double define them, which is how Spark compares top-level map
/// keys and Java code compares values with equals and Double.compare. Every
/// NaN is one value, as in Velox and Presto, but -0.0 and 0.0 are distinct,
/// where Velox and Presto treat them as equal.
class JavaFloatingPoint {
 public:
  /// Unsigned integer type of the same width as the floating point type T.
  template <typename T>
  using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;

  /// Signed integer type of the same width as the floating point type T.
  template <typename T>
  using SignedBits = std::make_signed_t<Bits<T>>;

  /// Returns the bits Java floatToIntBits or doubleToLongBits returns for
  /// 'value': its bit pattern, with every NaN replaced by the canonical quiet
  /// NaN.
  template <typename T>
  static Bits<T> bits(T value) {
    static_assert(std::is_floating_point_v<T>);
    return std::bit_cast<Bits<T>>(
        std::isnan(value) ? std::numeric_limits<T>::quiet_NaN() : value);
  }

  /// Equality of Java Float.equals and Double.equals.
  template <typename T>
  struct Equals {
    bool operator()(T lhs, T rhs) const {
      return bits(lhs) == bits(rhs);
    }
  };

  /// Hash consistent with Equals.
  template <typename T>
  struct Hash {
    size_t operator()(T value) const noexcept {
      return folly::hasher<Bits<T>>{}(bits(value));
    }
  };

  /// Hash of a Java HashMap key of type T: Hash<T> for float and double, and
  /// folly's default hash for every other type.
  template <typename T>
  using KeyHash = std::conditional_t<
      std::is_floating_point_v<T>,
      Hash<T>,
      folly::f14::DefaultHasher<T>>;

  /// Equality of a Java HashMap key of type T, consistent with KeyHash.
  template <typename T>
  using KeyEqualTo = std::conditional_t<
      std::is_floating_point_v<T>,
      Equals<T>,
      folly::f14::DefaultKeyEqual<T>>;

  /// Three-way comparison of Java Float.compare and Double.compare. Returns a
  /// negative value, zero or a positive value as 'lhs' ranks below, level with
  /// or above 'rhs'. -0.0 ranks below 0.0, and NaN ranks above positive
  /// infinity and level with every other NaN.
  template <typename T>
  static int32_t compare(T lhs, T rhs) {
    if (lhs < rhs) {
      return -1;
    }
    if (lhs > rhs) {
      return 1;
    }
    const auto lhsBits = static_cast<SignedBits<T>>(bits(lhs));
    const auto rhsBits = static_cast<SignedBits<T>>(bits(rhs));
    return lhsBits < rhsBits ? -1 : (lhsBits > rhsBits ? 1 : 0);
  }
};

} // namespace facebook::velox::functions::sparksql
