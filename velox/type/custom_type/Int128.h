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
/**
 * Simple data structure for representing 128-bit numbers as 2 64-bit integers.
 */

#ifndef TYPE_INT128_H__
#define TYPE_INT128_H__

#include <stdint.h>
#include <utility>
#include <cmath>
#include <folly/dynamic.h>
#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <climits>

namespace facebook::velox::type {

/**
 * Simple data structure for representing 128-bit numbers as 2 64-bit integers.
 *
 * Only logical operations are included; arithmetic operations are not,
 * but they could be easily added.
 *
 * This class should not have a user-defined copy constructor, assignment
 * operator, or destructor; this way, on x86_64, arguments of type uint128
 * (passed by value) are passed around in registers instead of on the stack.
 */
class int128;


class uint128 {
 public:
  friend std::hash<uint128>;
  constexpr uint128() : hi_(0), lo_(0) {}
  constexpr uint128(std::pair<uint64_t, uint64_t> p) : hi_(p.first), lo_(p.second) {}
  constexpr uint128(uint64_t hi, uint64_t lo) : hi_(hi), lo_(lo) {}
  constexpr uint128(uint64_t lo) : hi_(0), lo_(lo) {}

  constexpr uint128 operator=(const uint128& other) {
    this->hi_ = other.hi_;
    this->lo_ = other.lo_;
    return *this;
  }
  uint64_t hi() const {
    return hi_;
  }
  void setHi(uint64_t hi) {
    hi_ = hi;
  }
  uint64_t lo() const {
    return lo_;
  }
  void setLo(uint64_t lo) {
    lo_ = lo;
  }

  uint128& operator|=(uint128 other) {
    hi_ |= other.hi_;
    lo_ |= other.lo_;
    return *this;
  }
  uint128 operator|(uint128 other) const {
    uint128 a(*this);
    return (a |= other);
  }

  uint128& operator&=(uint128 other) {
    hi_ &= other.hi_;
    lo_ &= other.lo_;
    return *this;
  }
  uint128 operator&(uint128 other) const {
    uint128 a(*this);
    return (a &= other);
  }

  constexpr uint128& operator<<=(int n) {
    if (n >= 64) {
      hi_ = lo_ << (n - 64);
      lo_ = 0;
    } else if (n > 0) {
      hi_ = (hi_ << n) | (lo_ >> (64 - n));
      lo_ <<= n;
    }
    return *this;
  }
  constexpr uint128 operator<<(int n) const {
    uint128 a(*this);
    return (a <<= n);
  }

  uint128& operator>>=(int n) {
    if (n >= 64) {
      lo_ = hi_ >> (n - 64);
      hi_ = 0;
    } else if (n > 0) {
      lo_ = (lo_ >> n) | (hi_ << (64 - n));
      hi_ >>= n;
    }
    return *this;
  }
  uint128 operator>>(int n) const {
    uint128 a(*this);
    return (a >>= n);
  }  
  uint128 operator>>(uint64_t n) const {
    uint128 a(*this);
    return (a >>= n);
  }

  uint128 operator~() const {
    return uint128(~hi_, ~lo_);
  }

  bool operator==(uint128 other) const {
    return hi_ == other.hi_ && lo_ == other.lo_;
  }    
  bool operator==(int other) const {
    return *this == uint128(other);
  }  

  bool operator!=(uint128 other) const {
    return !(*this == other);
  }

  constexpr uint128 operator+(const uint128& other) const {
    uint64_t lo = this->lo_ + other.lo_;
    // Check for overflow.
    uint64_t c = lo < this->lo_ || lo < other.lo_ ? 1 : 0;

    uint64_t hi = this->hi_ + other.hi_ + c;
    return uint128(hi, lo);
  }

  constexpr uint128 operator+(const int64_t& other) const {
    return *this + uint128(other);
  }

  constexpr uint128 operator++() const {
    return *this + uint128(1);
  }

  // TODO: implement multply=
  constexpr uint128 operator*=(const uint128& other) const {
    return uint128(0);
  }
  // TODO: implement multply=
  constexpr uint128 operator*=(const int& other) const {
    return uint128(0);
  }
  // TODO: implement multiply=
  constexpr uint128 operator*=(const int64_t& other) const {
    return uint128(0);
  }
  // TODO: davidmar implement /= operator.
  uint128 operator/=(int other) const {
    return *this;
  }


  operator int64_t() const {
    return this->lo_;
  }
  operator int128() const;
  

 private:
  uint64_t hi_;
  uint64_t lo_;
};

// TODO: Davidmar define validity of >> and << operators as those may need to
// propagate the sign.
class int128 {
 private:
  int64_t hi_;
  uint64_t lo_;

 public:
  friend std::hash<int128>;
  constexpr int128(const int128& other) {
    this->hi_ = other.hi_;
    this->lo_ = other.lo_;
  }
  constexpr int128(const boost::multiprecision::int128_t& boostValue) {
    lo_ = static_cast<uint64_t>(boostValue);
    hi_ = static_cast<int64_t>(boostValue >> 64);
  }
  constexpr int128() : hi_(0), lo_(0) {}
  constexpr int128(std::pair<int64_t, uint64_t> p)
      : hi_(p.first), lo_(p.second) {}
  constexpr int128(int64_t hi, uint64_t lo) : hi_(hi), lo_(lo) {}
  constexpr int128(int64_t lo)
      : hi_(lo < 0 ? 0xfffffffffffffffflu : 0), lo_(lo) {}
  
  int128 operator=(const int128& other) {
    this->hi_ = other.hi_;
    this->lo_ = other.lo_;
    return *this;

  }
  constexpr int64_t hi() const {
    return hi_;
  }
  void setHi(int64_t hi) {
    hi_ = hi;
  }
  constexpr uint64_t lo() const {
    return lo_;
  }
  void setLo(uint64_t lo) {
    lo_ = lo;
  }
  operator folly::dynamic() const {
    folly::dynamic dynamicObject = folly::dynamic::object;
    dynamicObject["lo"] = this->lo_;
    dynamicObject["hi"] = this->hi_;
    return dynamicObject;
  }

  //To convert from boost to custom int128
  const int128 ConvertBoostInt128ToCustomInt128(const boost::multiprecision::int128_t& boostValue) {
    int128 value;
    value.lo_ = static_cast<uint64_t>(boostValue);
    value.hi_ = static_cast<int64_t>(boostValue >> 64);
    return value;
  }

  const int128 multiply_uint64(uint64_t a, uint64_t b) const {
    // Split the input numbers into two 32-bit halves to prevent overflow during
    // multiplication.
    uint64_t a_low = a & 0xFFFFFFFF;
    uint64_t a_high = a >> 32;
    uint64_t b_low = b & 0xFFFFFFFF;
    uint64_t b_high = b >> 32;

    // Multiply the parts to get intermediate 64-bit results.
    uint64_t p0 = a_low * b_low;
    uint64_t p1 = a_low * b_high;
    uint64_t p2 = a_high * b_low;
    uint64_t p3 = a_high * b_high;

    // Calculate the carry from the lower to the higher 64-bit part.
    uint64_t middle_sum = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
    uint64_t carry = (middle_sum >> 32) + (p1 >> 32) + (p2 >> 32);

    // Assemble the final 128-bit result.
    int128 result;
    result.lo_ = (middle_sum << 32) | (p0 & 0xFFFFFFFF);
    result.hi_ = p3 + carry;

    return result;
  }

  int128& operator|=(int128 other) {
    hi_ |= other.hi_;
    lo_ |= other.lo_;
    return *this;
  }
  int128 operator|(int128 other) const {
    int128 a(*this);
    return (a |= other);
  }

  int128& operator&=(int128 other) {
    hi_ &= other.hi_;
    lo_ &= other.lo_;
    return *this;
  }
  int128 operator&(int128 other) const {
    int128 a(*this);
    return (a &= other);
  }

  int128& operator<<=(int n) {
    if (n >= 64) {
      hi_ = lo_ << (n - 64);
      lo_ = 0;
    } else if (n > 0) {
      hi_ = (hi_ << n) | (lo_ >> (64 - n));
      lo_ <<= n;
    }
    return *this;
  }
  int128 operator<<(int n) const {
    int128 a(*this);
    return (a <<= n);
  }

  int128& operator>>=(int n) {
    if (n >= 64) {
      lo_ = hi_ >> (n - 64);
      hi_ = 0;
    } else if (n > 0) {
      lo_ = (lo_ >> n) | (hi_ << (64 - n));
      hi_ >>= n;
    }
    return *this;
  }
  int128 operator>>(int n) const {
    int128 a(*this);
    return (a >>= n);
  }

  int128 operator~() const {
    return int128(~hi_, ~lo_);
  }

  bool operator==(int128 other) const {
    return hi_ == other.hi_ && lo_ == other.lo_;
  }
  bool operator==(const int &other) const {
    return *this == int128(other);
  }  
  bool operator!=(int128 other) const {
    return !(*this == other);
  }  
 
  bool operator!=(int other) const {
    return !(*this == other);
  }
 
  int128 operator/=(int other) const {
    int128 temp = *this;
    temp = *this / int128(other);
    return temp;
  }

  
  constexpr int128 operator+(const int128 &other) const{
    uint64_t lo = this->lo_ + other.lo_;
    //Check for overflow.
    int64_t c = lo < this->lo_ || lo < other.lo_ ? 1 : 0;

    int64_t hi = this->hi_ + other.hi_ + c;
    return int128(hi, lo);
  }
  constexpr int128 operator+(const int& other) const {
    return *this + int128(other);
  }

  // TODO implement +=
  constexpr int128 operator+=(const int& other) const {
    int128 temp;
    temp = *this + int128(other);
    return temp;
  } 

  constexpr int128 operator++() const {
    // predecrement
    return *this + 1;
  }

  //TODO postdecrement functions have constant problem
  //constexpr int128 operator++(int) {
  //  // postdecrement
  //  int128 temp= int128(*this); // Create a copy of the current object
  //  *this += 1;
  //  return temp; // Return the copy (previous value)
  //}

  constexpr int128 operator--() const {
    //predecrement
    return *this - 1;
  }

 //constexpr int128 operator--(int) {
 //    //postdecrement
 //   int128 temp = int128(*this); // Create a copy of the current object
 //   *this -= 1;
 //   return temp; // Return the copy (previous value)
 // }

  constexpr int128 operator-(const int128& other) const {
    uint64_t lo = this->lo_ - other.lo_;
    // Check for overflow.
    int64_t c = lo > this->lo_ ? -1 : 0;

    int64_t hi = this->hi_ - other.hi_ + c;
    return int128(hi, lo);
  }
 
  constexpr int128 operator-(const int& other) const {
    return *this - int128(other);
  }
  constexpr int128 operator-() const {
    return int128(0) - *this;
  }

  //TODO implement -=
  constexpr int128 operator-=(const int& other) {
    *this = *this - other;
    return *this;
  } 

  constexpr bool operator<(const int128& other) const {
      int128 result = *this - other;
    return result.hi_ < 0;
  }  
  constexpr bool operator<(const int64_t& other) const {
      int128 result = *this - int128(other);
    return result < int128(0);
  }
  constexpr bool operator<(const int& other) const {
      int128 result = *this - int128(other);
    return result < int128(0);
  }
 
  constexpr bool operator<=(const int128& other) const {
    int128 result = *this - other;
    if (result == int128(0)) {
      return true;
    }
    else {
        return result.hi_ < 0;
    }
  }

  constexpr bool operator<=(const int64_t& other) const {
    int128 result = *this - int128(other);
    return result <= int128(0);
  }

  constexpr bool operator<=(const int& other) const {
    int128 result = *this - int128(other);
    return result <= int128(0);
  }

  constexpr bool operator>(const int128& other) const {
    //int128 result = *this - other;
    //return result.hi_ > 0;

    if (hi_ > other.hi_) {
        return true;
    } else if (hi_ < other.hi_) {
        return false;
    } else {
        return lo_ > other.lo_;
    }

  }
  constexpr bool operator>(const int64_t& other) const {
    int128 result = *this - int128(other);
    return result > int128(0);
  }
  constexpr bool operator>(const int& other) const {
    int128 result = *this - int128(other);
    return result > int128(0);
  }


  constexpr bool operator>=(const int128& other) const {
    int128 result = *this - other;
    if (result == int128(0)) {
        return true;
    } else {
        return result > int128(0);
    }
  }
  constexpr bool operator>=(const int64_t& other) const {
    int128 result = *this - int128(other);
    return result >= int128(0);
  }
  constexpr bool operator>=(const int& other) const {
    int128 result = *this - int128(other);
    return result >= int128(0);
  }

  //Int128 to boost int128
  operator boost::multiprecision::int128_t() const {
    boost::multiprecision::int128_t result = hi_;
    result <<= 64; // Shift the high part to its correct position
    result |= static_cast<uint64_t>(lo_); // Combine with the low part
    return result;
  }

  //TODO: implement division, current implementation is not viable
  constexpr int128 operator/(const int& other) const {
    return *this / int128(other);
  }    
  constexpr int128 operator/(const int64_t& other) const {
    return *this / int128(other);
  }  
  constexpr int128 operator/(const int128& other) const {
    int128 dividend(*this);
    int128 divisor(other);

    // Handle division by zero as an error.
    if (divisor.lo_ == 0 && divisor.hi_ == 0) {
        throw std::runtime_error("Division by zero.");
    }

    // Determine the sign of the result.
    bool negative_result = (dividend.hi_ < 0) ^ (divisor.hi_ < 0);

    // Use absolute values for the division.
    int128 abs_dividend = dividend.hi_ < 0 ? -dividend : dividend;
    int128 abs_divisor = divisor.hi_ < 0 ? -divisor : divisor;

    int128 quotient(0, 0);
    int128 remainder(0, 0);

    // Long division algorithm.
    for (int i = 127; i >= 0; --i) {
        // Shift remainder left by 1 bit.
        remainder = remainder << 1;
        // Bring down the next bit of the dividend.
        if (i >= 64) {
          // We are working with the high part of the dividend.
          remainder.lo_ |= (abs_dividend.hi_ >> (i - 64)) & 1;
        } else {
          // We are working with the low part of the dividend.
          remainder.lo_ |= (abs_dividend.lo_ >> i) & 1;
        }

        // If the remainder is greater than or equal to the divisor, subtract
        // and increment quotient.
        if (!(remainder < abs_divisor)) {
          remainder = remainder - abs_divisor;
          if (i >= 64) {
            quotient.hi_ |= (1LL << (i - 64));
          } else {
            quotient.lo_ |= (1LL << i);
          }
        }
    }

    // Apply the sign to the quotient.
    if (negative_result) {
        quotient = -quotient;
    }

    return quotient;

  }

  
  constexpr int128 operator%(const int64_t& other) const {
    return *this % int128(other);
  }

  constexpr int128 operator%(const int& other) const {
    return *this % int128(other);
  }
 
  constexpr int128 operator%(const int128& other) const {
    //if (other.lo_ == 0 && other.hi_ == 0) {
    //    // Division by zero exception
    //    throw std::invalid_argument("Division by Zero Exception");
    //}
    //int128 dividend(*this); // Copy of the dividend
    //int128 divisor(other);

    ////dividend = dividend - divisor;
    //// Perform modulo operation
    //while (dividend >= divisor) {
    //    dividend = dividend - divisor;
    //}

    //return dividend;

    int128 dividend(*this);
    int128 divisor(other);

    // Handle division by zero as an error.
    if (divisor.lo_ == 0 && divisor.hi_ == 0) {
        throw std::runtime_error("Division by zero.");
    }

    // Determine the sign of the result.
    bool negative_result = (dividend.hi_ < 0) ^ (divisor.hi_ < 0);

    // Use absolute values for the division.
    int128 abs_dividend = dividend.hi_ < 0 ? -dividend : dividend;
    int128 abs_divisor = divisor.hi_ < 0 ? -divisor : divisor;

    int128 remainder(0, 0);

    // Long division algorithm.
    for (int i = 127; i >= 0; --i) {
        // Shift remainder left by 1 bit.
        remainder = remainder << 1;
        // Bring down the next bit of the dividend.
        if (i >= 64) {
          // We are working with the high part of the dividend.
          remainder.lo_ |= (abs_dividend.hi_ >> (i - 64)) & 1;
        } else {
          // We are working with the low part of the dividend.
          remainder.lo_ |= (abs_dividend.lo_ >> i) & 1;
        }

        // If the remainder is greater than or equal to the divisor, subtract
        // and increment quotient.
        if (!(remainder < abs_divisor)) {
          remainder = remainder - abs_divisor;
        }
    }

    return remainder;
  }  

 
  constexpr int128 operator*(const int64_t other) const {
    return *this * int128(other);
  }
 
  constexpr int128 operator*(const int& other) const {
    return *this * int128(other);
  }
  //TODO: implement multply, current implementation is not viable
  constexpr int128 operator*(const int128& other) const {

    // Determine the sign of the result.
    bool negative = (this->hi_ < 0) ^ (other.hi_ < 0);

    // Convert to unsigned for multiplication, taking the absolute value if
    // necessary.
    uint64_t a_low = this->lo_;
    uint64_t a_high = (this->hi_ < 0) ? -this->hi_ : this->hi_;
    uint64_t b_low = other.lo_;
    uint64_t b_high = (other.hi_ < 0) ? -other.hi_ : other.hi_;

    // Perform unsigned multiplication.
    int128 result_unsigned = multiply_uint64(a_low, b_low);
    result_unsigned.hi_ += a_low * b_high + a_high * b_low;

    // Convert back to signed, adjusting the sign as necessary.
    int128 result;
    result.lo_ = result_unsigned.lo_;
    result.hi_ = negative ? -result_unsigned.hi_ : result_unsigned.hi_;

    // If the result is negative and there's a non-zero low part, we need to
    // adjust the high part.
    if (negative && result.lo_ != 0) {
        result.hi_ -= 1;
    }

    return result;

  }  
  
  //TODO: implement multply=
  constexpr int128 operator*=(const int128& other) {
    *this = *this * other;
    return *this;
  }  
  //TODO: implement multply=
  constexpr int128 operator*=(const int& other) {
    *this = *this * int128(other);
    return *this;
  }
  // TODO: implement multiply=
  constexpr int128 operator*=(const int64_t& other) {
    *this = *this * int128(other);
    return *this;
  }


  operator int64_t() const {
    return this->lo_;
  }

  operator uint128() const {
    return uint128(this->hi_, this->lo_);
  }

inline std::string toString(const int128 value) {
    // Handle zero as a special case.
    if (value.lo_ == 0 && value.hi_ == 0) {
        return "0";
    }

    // Handle negative numbers.
    bool is_negative = value.hi_ < 0;
    int128 abs_value = is_negative
        ? -value
        : value; // Assuming you have unary minus implemented.

    std::string result;
    while (abs_value.hi_ != 0 || abs_value.lo_ != 0) {
        int128 quotient = abs_value / 10;
        int128 remainder = abs_value % 10;
        auto div_result = std::make_pair(quotient, remainder);
        abs_value = div_result.first; // Quotient becomes the new value to divide.
        int64_t digit = div_result.second; // Remainder is the digit.
        result.push_back( '0' + digit); // Convert digit to char and append to the result string.
    }

    // Since the digits are in reverse order, reverse the string to get the
    // correct order.
    std::reverse(result.begin(), result.end());

    // Add the negative sign if the original number was negative.
    if (is_negative) {
        result.insert(result.begin(), '-');
    }

    return result;
  }

  static int128 mul_check(uint64_t a, uint64_t b, bool& overflow) {
    // Split the input numbers into two 32-bit halves to prevent overflow during
    // multiplication.
    uint64_t a_low = a & 0xFFFFFFFF;
    uint64_t a_high = a >> 32;
    uint64_t b_low = b & 0xFFFFFFFF;
    uint64_t b_high = b >> 32;

    // Multiply the parts to get intermediate 64-bit results.
    uint64_t p0 = a_low * b_low;
    uint64_t p1 = a_low * b_high;
    uint64_t p2 = a_high * b_low;
    uint64_t p3 = a_high * b_high;

    // Calculate the carry from the lower to the higher 64-bit part.
    uint64_t middle_sum = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
    uint64_t carry = (middle_sum >> 32) + (p1 >> 32) + (p2 >> 32);

    // Assemble the final 128-bit result.
    int128 result;
    result.lo_ = (middle_sum << 32) | (p0 & 0xFFFFFFFF);
    result.hi_ = p3 + carry;

    // Check for overflow. The overflow occurs if p3 is non-zero or if the carry
    // from the middle sum overflows into the high part of the result.
    overflow = (p3 != 0) || (carry > std::numeric_limits<uint64_t>::max() - p3);
    return result;
  }

  // Function to check for overflow in int128 multiplication.
  static bool mul_overflow(const int128& a, const int128& b, int128* result) {
    bool overflow = false;

    // Multiply the low parts and check for overflow.
    *result = mul_check(a.lo_, b.lo_, overflow);
    if (overflow) {
        return true; // Overflow occurred.
    }

    // Multiply the high parts and cross terms, checking for overflow at each
    // step. If any of these multiplications overflow, the final result will
    // overflow.
    overflow = false;
    int128 high_high = mul_check(
        static_cast<uint64_t>(a.hi_), static_cast<uint64_t>(b.hi_), overflow);
    if (overflow || high_high.hi_ != 0) {
        return true; // Overflow occurred.
    }

    overflow = false;
    int128 high_low = mul_check(static_cast<uint64_t>(a.hi_), b.lo_, overflow);
    if (overflow) {
        return true; // Overflow occurred.
    }

    overflow = false;
    int128 low_high = mul_check(a.lo_, static_cast<uint64_t>(b.hi_), overflow);
    if (overflow) {
        return true; // Overflow occurred.
    }

    // Combine the results of the multiplications.
    // Since we're only interested in overflow, we don't need to compute the
    // exact result. We just need to check if any of the high parts would
    // contribute to the final high part.
    if (high_low.hi_ != 0 || low_high.hi_ != 0) {
        return true; // Overflow occurred.
    }

    // Add the cross terms to the result and check for overflow.
    result->hi_ += high_low.lo_ + low_high.lo_;
    if (result->hi_ < high_low.lo_ || result->hi_ < low_high.lo_) {
        return true; // Overflow occurred due to addition.
    }

    // Add the high part of the high_high multiplication if it's non-zero.
    // This would also cause an overflow.
    result->hi_ += high_high.lo_;
    if (high_high.lo_ != 0 && result->hi_ < high_high.lo_) {
        return true; // Overflow occurred due to addition.
    }

    return false;
  }

  // Function to check for overflow in int128 addition.
  static bool add_overflow(const int128& a, const int128& b, int128* result) {
    // Perform addition on the low parts and check for carry.
    uint64_t new_low = a.lo_ + b.lo_;
    bool carry = new_low < a.lo_; // Check if carry occurred.

    // Add the high parts along with the carry.
    int64_t new_high = a.hi_ + b.hi_ + (carry ? 1 : 0);

    // Check for overflow. Overflow occurs if:
    // 1. The sign of a and b are the same, and the result has a different sign.
    // 2. The high part overflowed due to carry.
    if ((a.hi_ < 0 == b.hi_ < 0) && (new_high < 0 != a.hi_ < 0)) {
        return true; // Overflow occurred.
    }

    // Set the result if no overflow occurred.
    result->lo_ = new_low;
    result->hi_ = new_high;
    return false; // No overflow.
  }

  // Function to check for overflow in int128 subtraction.
  static bool sub_overflow(const int128& a, const int128& b, int128* result) {
    // Perform subtraction on the low parts and check for borrow.
    uint64_t new_low = a.lo_ - b.lo_;
    bool borrow = a.lo_ < b.lo_; // Check if borrow occurred.

    // Subtract the high parts, taking into account the borrow.
    int64_t new_high = a.hi_ - b.hi_ - (borrow ? 1 : 0);

    // Check for overflow. Overflow occurs if:
    // 1. The sign of a and b are different, and the result has a different sign
    // than a.
    if ((a.hi_ < 0 != b.hi_ < 0) && (new_high < 0 != a.hi_ < 0)) {
        return true; // Overflow occurred.
    }

    // Set the result if no overflow occurred.
    result->lo_ = new_low;
    result->hi_ = new_high;
    return false; // No overflow.
  }


};

//bool mul_overflow(int128 a , int128 b ,int64_t result) {
//  return int128::mul_overflow(a, b, result);
//
//}
bool add_overflow(int128 a , int128 b ,int128* result) {
  return int128::add_overflow(a,b,result);

}bool sub_overflow(int128 a , int128 b ,int128* result) {
  return int128::sub_overflow(a, b, result);

}
bool mul_overflow(int128 a , int128 b ,int128 *result) {
  return int128::mul_overflow(a, b, result);

} // namespace facebook

} // namespace facebook::velox::type

namespace std {
template <>
struct hash<facebook::velox::type::uint128> {
  // TODO: davidmar implement hashing operation for 128 bits.
  size_t operator()(const facebook::velox::type::uint128& obj) const {
    return 0;
  }
};
template <>
struct hash<facebook::velox::type::int128> {
  // TODO: davidmar implement hashing operation for 128 bits.
  size_t operator()(const facebook::velox::type::int128& obj) const {
    return 0;
  }
};
// TODO: add std::make_unsiged capabilities
template <>
struct make_unsigned<facebook::velox::type::int128> {
  using type = facebook::velox::type::uint128; 
};
// TODO: Implemente std::log for int128 
double log(facebook::velox::type::int128 value) {
  return 0.0;
}


} // namespace std


namespace folly {
template <>
struct hasher<facebook::velox::type::int128> : detail::integral_hasher<facebook::velox::type::int128> {};
} // namespace folly
#endif /* TYPE_INT128_H__ */
