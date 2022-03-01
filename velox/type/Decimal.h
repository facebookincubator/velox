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
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"

namespace facebook::velox {

static constexpr uint8_t kMaxPrecisionInt16 = 4;
static constexpr uint8_t kMaxPercisionInt32 = 9;
static constexpr uint8_t kMaxPrecisionInt64 = 18;
static constexpr uint8_t kMaxPrecisionInt128 = 38;
static constexpr uint8_t kMaxPrecisionDecimal = kMaxPrecisionInt128;
static constexpr uint8_t kDefaultScale = 3;
static constexpr uint8_t kDefaultPrecision = 18;

enum DecimalInternalType { INT16, INT32, INT64, INT128 };

/*
 * This class defines the Velox DECIMAL type support to store
 * fixed-point rational numbers.
 */
class Decimal {
 public:
  inline const uint8_t getPrecision() const {
    return precision_;
  }

  inline const uint8_t getScale() const {
    return scale_;
  }

  inline const DecimalInternalType getInternalType() const {
    return internalType_;
  }

  /**
   * Returns the max number of bytes required to store a decimal of
   * given the precision.
   * @param precision Precision of the decimal value.
   * @return Max Number of bytes.
   */
  DecimalInternalType getInternalType(const uint16_t precision) {
    VELOX_USER_CHECK(precision >= 1 && precision <= kMaxPrecisionInt128);
    if (precision <= kMaxPrecisionInt16) {
      return INT16;
    } else if (precision <= kMaxPercisionInt32) {
      return INT32;
    } else if (precision <= kMaxPrecisionInt64) {
      return INT64;
    }
    return INT128;
  }

  // Needed for serialization of FlatVector<Decimal>
  operator StringView() const {VELOX_NYI()}

  std::string toString() const;

  operator std::string() const {
    return toString();
  }

  bool operator==(const Decimal& other) const {
    return true;
  }
  bool operator!=(const Decimal& other) const {
    return true;
  }

  bool operator<(const Decimal& other) const {
    return true;
  }

  bool operator<=(const Decimal& other) const {
    return true;
  }

  bool operator>(const Decimal& other) const {
    return true;
  }

  Decimal(uint8_t precision = kDefaultPrecision, uint8_t scale = kDefaultScale)
      : precision_(precision), scale_(scale) {
    // Validate string value fits in the precision and scale.
    // validate precision and scale
  }

 private:
  uint8_t precision_;
  uint8_t scale_;
  DecimalInternalType internalType_;
};

class SmallDecimal : public Decimal {
  // represents Decimal value with precision 1 to 18.
  SmallDecimal(
      std::string value,
      uint8_t precision = kDefaultPrecision,
      uint8_t scale = kDefaultScale)
      : Decimal(precision, scale) {}
};

class LargeDecimal : public Decimal {
  // represents the integer form for decimals in the range 18 to 38.
  LargeDecimal(
      std::string value,
      uint8_t precision = kDefaultPrecision,
      uint8_t scale = kDefaultScale)
      : Decimal(precision, scale) {}
};

class DecimalParser {
  static Decimal strToDecimal(
      const std::string value,
      const uint8_t precision,
      const uint8_t scale) {
    return 0;
  }
};

void parseTo(folly::StringPiece in, Decimal& out);

template <typename T>
void toAppend(const ::facebook::velox::Decimal& value, T* result) {}
} // namespace facebook::velox

namespace std {
template <>
struct hash<::facebook::velox::Decimal> {
  size_t operator()(const ::facebook::velox::Decimal& value) const {
    return 0;
  }
};

std::string to_string(const ::facebook::velox::Decimal& ts);
} // namespace std

namespace folly {
template <>
struct hasher<::facebook::velox::Decimal> {
  size_t operator()(const ::facebook::velox::Decimal& value) const {
    return 0;
  }
};
} // namespace folly
