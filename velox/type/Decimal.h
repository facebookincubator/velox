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

#ifndef VELOX_DECIMAL_H
#define VELOX_DECIMAL_H

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
  inline const uint8_t getPrecision() {
    return precision_;
  }

  inline const uint8_t getScale() {
    return scale_;
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

  std::string toString() const {
    return "";
  }

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

} // namespace facebook::velox
#endif // VELOX_DECIMAL_H
