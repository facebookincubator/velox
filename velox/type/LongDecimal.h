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
#include <folly/dynamic.h>
#include <sstream>
#include <string>
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"

#pragma once

#define BUILD_INT128(X, Y) ((int128_t)X << 64 | Y)
namespace facebook::velox {

using int128_t = __int128_t;

struct LongDecimal {
  // Default required for creating vector with NULL values.
  constexpr explicit LongDecimal() : unscaledValue({0, 0}) {}
  constexpr explicit LongDecimal(uint64_t upper, uint64_t lower)
      : unscaledValue{upper, lower} {}
  static constexpr size_t kArrayLen = 2;

  bool operator==(const LongDecimal& other) const {
    return unscaledValue[0] == other.unscaledValue[0] &&
        unscaledValue[1] == other.unscaledValue[1];
  }

  bool operator!=(const LongDecimal& other) const {
    VELOX_NYI();
  }

  bool operator<(const LongDecimal& other) const {
    int128_t lhs = BUILD_INT128(unscaledValue[0], unscaledValue[1]);
    int128_t rhs = BUILD_INT128(other.unscaledValue[0], other.unscaledValue[1]);
    return lhs < rhs;
  }

  bool operator<=(const LongDecimal& other) const {
    VELOX_NYI();
  }

  bool operator>(const LongDecimal& other) const {
    VELOX_NYI();
  }

  bool operator>=(const LongDecimal& other) const {
    VELOX_NYI();
  }

  // Needed for serialization of FlatVector<LongDecimal>
  operator StringView() const {
    VELOX_NYI();
  }

  std::string toString() const;

  operator std::string() const {
    return toString();
  }

  operator folly::dynamic() const {
    VELOX_NYI();
  }

  std::array<uint64_t, kArrayLen> unscaledValue;
}; // struct LongDecimal

} // namespace facebook::velox

namespace std {
template <>
struct hash<::facebook::velox::LongDecimal> {
  size_t operator()(const ::facebook::velox::LongDecimal& value) const {
    VELOX_NYI();
  }
};

std::string to_string(const ::facebook::velox::LongDecimal& ts);

} // namespace std

namespace folly {
template <>
struct hasher<::facebook::velox::LongDecimal> {
  size_t operator()(const ::facebook::velox::LongDecimal& value) const {
    VELOX_NYI();
  }
};
} // namespace folly
