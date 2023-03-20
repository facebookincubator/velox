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
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"
#include "velox/type/UnscaledShortDecimal.h"

#pragma once

namespace facebook::velox {

using int128_t = __int128_t;

constexpr int128_t buildInt128(uint64_t hi, uint64_t lo) {
  // GCC does not allow left shift negative value.
  return (static_cast<__uint128_t>(hi) << 64) | lo;
}

#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
__attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
inline int128_t
mul(int128_t x, const int128_t y) {
  return x * y;
}

#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
__attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
inline int128_t
add(int128_t x, const int128_t y) {
  return x + y;
}

struct UnscaledLongDecimal {
 public:
  inline static bool valueInRange(int128_t value) {
    return (value >= kMin) && (value <= kMax);
  }

  // Default required for creating vector with NULL values.
  UnscaledLongDecimal() = default;

  explicit UnscaledLongDecimal(int128_t value) : unscaledValue_(value) {
    VELOX_DCHECK(
        valueInRange(unscaledValue_),
        "Value '{}' is not in the range of LongDecimal Type",
        unscaledValue_);
  }

  constexpr explicit UnscaledLongDecimal(int64_t value)
      : unscaledValue_(value) {}

  constexpr explicit UnscaledLongDecimal(int value) : unscaledValue_(value) {}

  constexpr explicit UnscaledLongDecimal(unsigned int value)
      : unscaledValue_(value) {}

  explicit UnscaledLongDecimal(UnscaledShortDecimal value)
      : unscaledValue_(value.unscaledValue()) {}

  static UnscaledLongDecimal min() {
    return UnscaledLongDecimal(kMin);
  }

  static UnscaledLongDecimal max() {
    return UnscaledLongDecimal(kMax);
  }

  int128_t unscaledValue() const {
    return unscaledValue_;
  }

  void setUnscaledValue(const int128_t& unscaledValue) {
    unscaledValue_ = unscaledValue;
  }

  bool operator==(const UnscaledLongDecimal& other) const {
    return unscaledValue_ == other.unscaledValue_;
  }

  bool operator!=(const UnscaledLongDecimal& other) const {
    return unscaledValue_ != other.unscaledValue_;
  }

  bool operator!=(int other) const {
    return unscaledValue_ != other;
  }

  bool operator<(const UnscaledLongDecimal& other) const {
    return unscaledValue_ < other.unscaledValue_;
  }

  bool operator<=(const UnscaledLongDecimal& other) const {
    return unscaledValue_ <= other.unscaledValue_;
  }

  bool operator>=(const UnscaledLongDecimal& other) const {
    return unscaledValue_ >= other.unscaledValue_;
  }

  bool operator<(int other) const {
    return unscaledValue_ < other;
  }

  bool operator>(const UnscaledLongDecimal& other) const {
    return unscaledValue_ > other.unscaledValue_;
  }

  UnscaledLongDecimal operator+(const UnscaledLongDecimal& other) const {
    return UnscaledLongDecimal(add(unscaledValue_, other.unscaledValue_));
  }

  UnscaledLongDecimal operator-(const UnscaledLongDecimal& other) const {
    return UnscaledLongDecimal(unscaledValue_ - other.unscaledValue_);
  }

  UnscaledLongDecimal operator=(int value) const {
    return UnscaledLongDecimal(static_cast<int64_t>(value));
  }

  UnscaledLongDecimal& operator+=(const UnscaledLongDecimal& value);

  UnscaledLongDecimal& operator+=(const UnscaledShortDecimal& value);

  UnscaledLongDecimal& operator*=(int value) {
    unscaledValue_ *= value;
    return *this;
  }

  UnscaledLongDecimal operator/(const UnscaledLongDecimal& other) const {
    return UnscaledLongDecimal(unscaledValue_ / other.unscaledValue_);
  }

  UnscaledLongDecimal operator%(const UnscaledLongDecimal& other) const {
    return UnscaledLongDecimal(unscaledValue_ % other.unscaledValue_);
  }

  UnscaledLongDecimal& operator++() {
    unscaledValue_++;
    return *this;
  }

  static FOLLY_ALWAYS_INLINE void serialize(
      const UnscaledLongDecimal& longDecimal,
      char* serializedData) {
    memcpy(serializedData, &longDecimal.unscaledValue_, sizeof(int128_t));
  }

  static FOLLY_ALWAYS_INLINE UnscaledLongDecimal
  deserialize(const char* serializedData) {
    UnscaledLongDecimal ans;
    memcpy(&ans.unscaledValue_, serializedData, sizeof(int128_t));
    return ans;
  }

 private:
  static constexpr int128_t kMin =
      -(1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
        (int128_t)100) +
      1;
  static constexpr int128_t kMax =
      (1'000'000'000'000'000'000 * (int128_t)1'000'000'000'000'000'000 *
       (int128_t)100) -
      1;
  int128_t unscaledValue_;
}; // struct UnscaledLongDecimal

static inline UnscaledLongDecimal operator/(
    const UnscaledLongDecimal& a,
    int b) {
  VELOX_CHECK_NE(b, 0, "Divide by zero is not supported");
  return UnscaledLongDecimal(a.unscaledValue() / b);
}

static inline UnscaledLongDecimal operator*(
    const UnscaledLongDecimal& a,
    int b) {
  return UnscaledLongDecimal(mul(a.unscaledValue(), b));
}

template <>
inline UnscaledLongDecimal checkedPlus(
    const UnscaledLongDecimal& a,
    const UnscaledLongDecimal& b) {
  int128_t result;
  bool overflow =
      __builtin_add_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledLongDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} + {}", a.unscaledValue(), b.unscaledValue());
  }
  return UnscaledLongDecimal(result);
}

template <>
inline UnscaledLongDecimal checkedMinus(
    const UnscaledLongDecimal& a,
    const UnscaledLongDecimal& b) {
  int128_t result;
  bool overflow =
      __builtin_sub_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledLongDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} - {}", a.unscaledValue(), b.unscaledValue());
  }

  return UnscaledLongDecimal(result);
}

template <>
inline UnscaledLongDecimal checkedMultiply(
    const UnscaledLongDecimal& a,
    const UnscaledLongDecimal& b) {
  int128_t result;
  bool overflow =
      __builtin_mul_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledLongDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} * {}", a.unscaledValue(), b.unscaledValue());
  }
  return UnscaledLongDecimal(result);
}

} // namespace facebook::velox

namespace folly {
template <>
struct hasher<::facebook::velox::UnscaledLongDecimal> {
  size_t operator()(const ::facebook::velox::UnscaledLongDecimal& value) const {
    auto upperHash = folly::hasher<uint64_t>{}(
        static_cast<uint64_t>(value.unscaledValue() >> 64));
    auto lowerHash =
        folly::hasher<uint64_t>{}(static_cast<uint64_t>(value.unscaledValue()));
    return facebook::velox::bits::hashMix(upperHash, lowerHash);
  }
};
} // namespace folly

namespace std {
string to_string(facebook::velox::int128_t x);

// Required for STL containers like unordered_map.
template <>
struct hash<facebook::velox::UnscaledLongDecimal> {
  size_t operator()(const facebook::velox::UnscaledLongDecimal& val) const {
    return hash<__int128_t>()(val.unscaledValue());
  }
};

template <>
class numeric_limits<facebook::velox::UnscaledLongDecimal> {
 public:
  static facebook::velox::UnscaledLongDecimal min() {
    return facebook::velox::UnscaledLongDecimal::min();
  }
  static facebook::velox::UnscaledLongDecimal max() {
    return facebook::velox::UnscaledLongDecimal::max();
  }

  static facebook::velox::UnscaledLongDecimal lowest() {
    return facebook::velox::UnscaledLongDecimal::min();
  }
};
} // namespace std

/// fmt::formatter<> specialization required for error message formatting
/// in VELOX checks.
template <>
struct fmt::formatter<facebook::velox::UnscaledLongDecimal> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(
      const facebook::velox::UnscaledLongDecimal& d,
      FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "{}", std::to_string(d.unscaledValue()));
  }
};
