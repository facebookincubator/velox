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

#include <folly/dynamic.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"

namespace facebook::velox {

#define PRECISION(typmod) ((uint8_t)(typmod >> 8))
#define SCALE(typmod) ((uint8_t)(typmod))
#define TYPMOD(X, Y) ((X << 8) | y)
#define NUM_DECIMAL_PARAMETERS 2

using int128_t = __int128_t;
static constexpr uint8_t kMaxPrecisionInt128 = 38;
static constexpr uint8_t kDefaultScale = 0;
static constexpr uint8_t kDefaultPrecision = kMaxPrecisionInt128;

/*
 * This class defines the Velox DECIMAL type support to store
 * fixed-point rational numbers.
 */
class BigDecimal {
 public:
  inline const uint8_t getPrecision() const {
    return precision_;
  }

  inline const uint8_t getScale() const {
    return scale_;
  }

  inline int128_t getUnscaledValue() const {
    return unscaledValue_;
  }

  inline void setUnscaledValue(const int128_t& value) {
    unscaledValue_ = value;
  }

  // Needed for serialization of FlatVector<BigDecimal>
  operator StringView() const {VELOX_NYI()}

  std::string toString() const;

  operator std::string() const {
    return toString();
  }

  bool operator==(const BigDecimal& other) const {
    return (
        this->unscaledValue_ == other.getUnscaledValue() &&
        this->precision_ == other.getPrecision() &&
        this->scale_ == other.getScale());
  }
  bool operator!=(const BigDecimal& other) const {
    VELOX_NYI();
  }

  bool operator<(const BigDecimal& other) const {
    VELOX_NYI();
  }

  bool operator<=(const BigDecimal& other) const {
    VELOX_NYI();
  }

  bool operator>(const BigDecimal& other) const {
    VELOX_NYI();
  }

  BigDecimal(
      int128_t value,
      uint8_t precision = kDefaultPrecision,
      uint8_t scale = kDefaultScale)
      : unscaledValue_(value), precision_(precision), scale_(scale) {}
  BigDecimal() = default;

 private:
  int128_t unscaledValue_; // The actual unscaled value with
                           // max precision 38.
  uint8_t precision_ = kDefaultPrecision; // The number of digits in unscaled
                                          // decimal value
  uint8_t scale_ = kDefaultScale; // The number of digits on the right
                                  // of radix point.
};

class ShortDecimal {
 public:
  ShortDecimal(
      int128_t value,
      uint8_t precision = kDefaultPrecision,
      uint8_t scale = kDefaultScale)
      : unscaledValue_(value), precision_(precision), scale_(scale) {}

  ShortDecimal() = default;

 private:
  int64_t unscaledValue_; // The actual unscaled value with
                          // max precision 38.
  uint8_t precision_ = kDefaultPrecision; // The number of digits in unscaled
                                          // decimal value
  uint8_t scale_ = kDefaultScale; // The number of digits on the right
                                  // of radix point.
};

class DecimalCasts {
 public:
  static BigDecimal parseStringToDecimal(const std::string& value) {
    // throws overflow exception if length is > 38
    VELOX_CHECK_GT(
        value.length(), 0, "BigDecimal string must have at least 1 char")
    int128_t unscaledValue;
    uint8_t precision;
    uint8_t scale;
    try {
      parseToInt128(value, unscaledValue, precision, scale);
    } catch (VeloxRuntimeError const& e) {
      VELOX_USER_CHECK(false, "BigDecimal overflow");
    }
    return BigDecimal(unscaledValue, precision, scale);
  }

  /**
   */
  static void parseToInt128(
      std::string value,
      int128_t& result,
      uint8_t& precision,
      uint8_t& scale) {
    uint8_t pos = 0;
    bool isNegative = false;
    // Remove leading zeroes.
    if (!isdigit(value[pos])) {
      // Presto allows string literals that start with +123.45
      VELOX_USER_CHECK(
          value[pos] == '-' || value[pos] == '+',
          "Illegal decimal value {}",
          value);
      isNegative = value[pos] == '-';
      value = value.erase(0, 1);
    }
    value = value.erase(0, value.find_first_not_of('0'));
    precision = 0;
    scale = 0;
    bool hasScale = false;
    int128_t digit;
    int128_t exponent = 10;
    while (pos < value.length()) {
      if (value[pos] == '.') {
        hasScale = true;
        pos++;
        continue;
      }
      VELOX_USER_CHECK(std::isdigit(value[pos]), "Invalid decimal string");
      digit = value[pos] - '0';
      if (isNegative) {
        result = result * exponent - digit;
      } else {
        result = result * exponent + digit;
      }
      if (hasScale) {
        scale++;
      }
      precision++;
      pos++;
    }
  }
};

void parseTo(folly::StringPiece in, BigDecimal& out);

template <typename T>
void toAppend(const ::facebook::velox::BigDecimal& value, T* result) {
  result->append(value.toString());
}
} // namespace facebook::velox

namespace std {
template <>
struct hash<::facebook::velox::BigDecimal> {
  size_t operator()(const ::facebook::velox::BigDecimal& value) const {
    VELOX_NYI();
  }
};

std::string to_string(const ::facebook::velox::BigDecimal& ts);
} // namespace std

namespace folly {
template <>
struct hasher<::facebook::velox::BigDecimal> {
  size_t operator()(const ::facebook::velox::BigDecimal& value) const {
    VELOX_NYI();
  }
};
} // namespace folly
