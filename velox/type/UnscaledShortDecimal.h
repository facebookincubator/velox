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
#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"

#pragma once

namespace facebook::velox {

struct UnscaledShortDecimal {
 public:
  inline static bool valueInRange(int64_t value) {
    return (value >= kMin) && (value <= kMax);
  }

  // Default required for creating vector with NULL values.
  UnscaledShortDecimal() = default;

  explicit UnscaledShortDecimal(int64_t value) : unscaledValue_(value) {
    VELOX_DCHECK(
        valueInRange(unscaledValue_),
        "Value '{}' is not in the range of ShortDecimal Type",
        unscaledValue_);
  }

  static UnscaledShortDecimal min() {
    return UnscaledShortDecimal(kMin);
  }

  static UnscaledShortDecimal max() {
    return UnscaledShortDecimal(kMax);
  }

  int64_t unscaledValue() const {
    return unscaledValue_;
  }
  bool operator==(const UnscaledShortDecimal& other) const {
    return unscaledValue_ == other.unscaledValue_;
  }

  bool operator!=(const UnscaledShortDecimal& other) const {
    return unscaledValue_ != other.unscaledValue_;
  }

  bool operator!=(int other) const {
    return unscaledValue_ != other;
  }

  bool operator<(const UnscaledShortDecimal& other) const {
    return unscaledValue_ < other.unscaledValue_;
  }

  bool operator<=(const UnscaledShortDecimal& other) const {
    return unscaledValue_ <= other.unscaledValue_;
  }

  bool operator<(int other) const {
    return unscaledValue_ < other;
  }

  bool operator>(const UnscaledShortDecimal& other) const {
    return unscaledValue_ > other.unscaledValue_;
  }

  UnscaledShortDecimal operator-(const UnscaledShortDecimal& other) const {
    return UnscaledShortDecimal(unscaledValue_ - other.unscaledValue_);
  }

  UnscaledShortDecimal operator+(const UnscaledShortDecimal& other) const {
    return UnscaledShortDecimal(unscaledValue_ + other.unscaledValue_);
  }

  UnscaledShortDecimal operator*(int value) const {
    return UnscaledShortDecimal(unscaledValue_ * value);
  }

  UnscaledShortDecimal& operator+=(const UnscaledShortDecimal& value) {
    unscaledValue_ += value.unscaledValue_;
    return *this;
  }

  UnscaledShortDecimal& operator*=(int value) {
    unscaledValue_ *= value;
    return *this;
  }

  bool operator>=(const UnscaledShortDecimal& other) const {
    return unscaledValue_ >= other.unscaledValue_;
  }

  UnscaledShortDecimal operator=(int value) const {
    return UnscaledShortDecimal(static_cast<int64_t>(value));
  }

  UnscaledShortDecimal operator/(const UnscaledShortDecimal& other) const {
    return UnscaledShortDecimal(unscaledValue_ / other.unscaledValue_);
  }

  UnscaledShortDecimal operator%(const UnscaledShortDecimal& other) const {
    return UnscaledShortDecimal(unscaledValue_ % other.unscaledValue_);
  }

  UnscaledShortDecimal& operator++() {
    unscaledValue_++;
    return *this;
  }

 private:
  static constexpr int64_t kMin = -1'000'000'000'000'000'000 + 1;
  static constexpr int64_t kMax = 1'000'000'000'000'000'000 - 1;
  int64_t unscaledValue_;
};

static inline UnscaledShortDecimal operator/(
    const UnscaledShortDecimal& a,
    int b) {
  VELOX_CHECK_NE(b, 0, "Divide by zero is not supported");
  return UnscaledShortDecimal(a.unscaledValue() / b);
}

template <>
inline UnscaledShortDecimal checkedPlus(
    const UnscaledShortDecimal& a,
    const UnscaledShortDecimal& b) {
  int64_t result;
  bool overflow =
      __builtin_add_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledShortDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} + {}", a.unscaledValue(), b.unscaledValue());
  }
  return UnscaledShortDecimal(result);
}

template <>
inline UnscaledShortDecimal checkedMinus(
    const UnscaledShortDecimal& a,
    const UnscaledShortDecimal& b) {
  int64_t result;
  bool overflow =
      __builtin_sub_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledShortDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} - {}", a.unscaledValue(), b.unscaledValue());
  }
  return UnscaledShortDecimal(result);
}

template <>
inline UnscaledShortDecimal checkedMultiply(
    const UnscaledShortDecimal& a,
    const UnscaledShortDecimal& b) {
  int64_t result;
  bool overflow =
      __builtin_mul_overflow(a.unscaledValue(), b.unscaledValue(), &result);
  if (UNLIKELY(overflow || !UnscaledShortDecimal::valueInRange(result))) {
    VELOX_ARITHMETIC_ERROR(
        "Decimal overflow: {} * {}", a.unscaledValue(), b.unscaledValue());
  }
  return UnscaledShortDecimal(result);
}
} // namespace facebook::velox

namespace folly {
template <>
struct hasher<::facebook::velox::UnscaledShortDecimal> {
  size_t operator()(
      const ::facebook::velox::UnscaledShortDecimal& value) const {
    return std::hash<int64_t>{}(value.unscaledValue());
  }
};
} // namespace folly

namespace std {

// Required for STL containers like unordered_map.
template <>
struct hash<facebook::velox::UnscaledShortDecimal> {
  size_t operator()(const facebook::velox::UnscaledShortDecimal& val) const {
    return hash<int64_t>()(val.unscaledValue());
  }
};

template <>
class numeric_limits<facebook::velox::UnscaledShortDecimal> {
 public:
  static facebook::velox::UnscaledShortDecimal min() {
    return facebook::velox::UnscaledShortDecimal::min();
  }
  static facebook::velox::UnscaledShortDecimal max() {
    return facebook::velox::UnscaledShortDecimal::max();
  }

  static facebook::velox::UnscaledShortDecimal lowest() {
    return facebook::velox::UnscaledShortDecimal::min();
  }
};
} // namespace std

/// fmt::formatter<> specialization required for error message formatting
/// in VELOX checks.
template <>
struct fmt::formatter<facebook::velox::UnscaledShortDecimal> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(
      const facebook::velox::UnscaledShortDecimal& d,
      FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "{}", std::to_string(d.unscaledValue()));
  }
};
