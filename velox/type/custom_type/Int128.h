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
#ifndef DUCK_DB
#include <folly/dynamic.h>
#endif

#include <boost/multiprecision/cpp_int.hpp>

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
  uint128 operator&(int other) const {
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
  int128(const int128& other) {
    this->hi_ = other.hi_;
    this->lo_ = other.lo_;
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
  #ifndef DUCK_DB
  operator folly::dynamic() const {
    folly::dynamic dynamicObject = folly::dynamic::object;
    dynamicObject["lo"] = this->lo_;
    dynamicObject["hi"] = this->hi_;
    return dynamicObject;
  }
  #endif


  int128& operator|=(int128 other) {
    hi_ |= other.hi_;
    lo_ |= other.lo_;
    return *this;
  }
  int128 operator|(int128 other) const {
    int128 a(*this);
    return (a |= other);
  }  
  int128 operator|(uint64_t other) const {
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
  int128 operator&(uint128 other) const {
    int128 a(*this);
    return (a &= other);
  }  
  
  int128 operator&(uint64_t other) const {
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
  
  int128 operator<<(uint64_t n) const {
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
  int128 operator>>(uint64_t n) const {
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
  // TODO: davidmar implement != operator.
  bool operator!=(int other) const {
    return !(*this == other);
  }
  //TODO: davidmar implement /= operator.
  int128 operator/=(int other) const {
    return *this;
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
  constexpr int128 operator+=(const int& other) const {
    return int128(0);
  } 
  //TODO
  constexpr int128 operator--() const {
    return int128();
  }
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
  //TODO: implement <= 
  constexpr bool operator<=(const int128& other) const {
    return true;
  }  
  //TODO: implement <= 
  constexpr bool operator<=(const int& other) const {
    return true;
  }
  //TODO: implement >=
  constexpr bool operator>=(const int128& other) const {
    return true;
  }  
  constexpr bool operator>=(const int& other) const {
    return true;
  }  

  //TODO: implement division
  constexpr int128 operator/(const int& other) const {
    return int128(0);
  }    
  constexpr int128 operator/(const int64_t& other) const {
    return int128(0);
  }  
  constexpr int128 operator/(const int128& other) const {
    return int128(0);
  }

  //TODO: implement modulo
  constexpr int128 operator%(const int64_t& other) const {
    return int128(0);
  }
  //TODO: implement modulo
  constexpr int128 operator%(const int& other) const {
    return int128(0);
  }
  //TODO: implement modulo
  constexpr int128 operator%(const int128& other) const {
    return int128(0);
  }  
  // TODO: implement multiply
  constexpr int128 operator*(const int64_t other) const {
    return int128(0);
  }
  //TODO: implement multiply
  constexpr int128 operator*(const int& other) const {
    return int128(0);
  }
  //TODO: implement multply
  constexpr int128 operator*(const int128& other) const {
    return int128(0);
  }  
  
  //TODO: implement multply=
  constexpr int128 operator*=(const int128& other) const {
    return int128(0);
  }  
  //TODO: implement multply=
  constexpr int128 operator*=(const int& other) const {
    return int128(0);
  }
  // TODO: implement multiply=
  constexpr int128 operator*=(const int64_t& other) const {
    return int128(0);
  }


  constexpr int128 operator++() const {
    return *this + int128(1);
  }

  operator int64_t() const {
    return this->lo_;
  }

  operator boost::multiprecision::int256_t() const {
    return boost::multiprecision::int256_t(0);
  }
  operator boost::multiprecision::int128_t() const {
    return boost::multiprecision::int128_t(0);
  }

  operator uint128() const {
    return uint128(this->hi_, this->lo_);
  }

};



bool mul_overflow(int128 a , int128 b ,int64_t result) {
  return true;

}
bool add_overflow(int128 a , int128 b ,int128* result) {
  return true;

}bool sub_overflow(int128 a , int128 b ,int128* result) {
  return true;

}
bool mul_overflow(int128 a , int128 b ,int128 *result) {
  return true;

} // namespace facebook


} // namespace facebook::velox::type

namespace std {

template <>
struct is_pod<facebook::velox::type::int128> : std::true_type {};

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
// TODO: add std::make_unsiged capabilities
template <>
struct make_unsigned<facebook::velox::type::uint128> {
  using type = facebook::velox::type::uint128; 
};
// TODO: add std::make_signed capabilities
template <>
struct make_signed<facebook::velox::type::uint128> {
  using type = facebook::velox::type::int128; 
};
// TODO: Implemente std::log for int128 
double log(facebook::velox::type::int128 value) {
  return 0.0;
}



} // namespace std
#ifndef DUCK_DB
namespace folly {
template <>
struct hasher<facebook::velox::type::int128> : detail::integral_hasher<facebook::velox::type::int128> {};
template <>
struct hasher<facebook::velox::type::uint128> : detail::integral_hasher<facebook::velox::type::uint128> {};
} // namespace folly
#endif


//namespace boost::multiprecision{
//
//template<>
//inline int256_t int256_t::operator*(int128_t other) {
//    return int256_t(other);
//}
//
//}


#endif /* TYPE_INT128_H__ */
