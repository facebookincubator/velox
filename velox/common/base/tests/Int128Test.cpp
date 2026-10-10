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

#include "velox/common/base/Int128.h"
#include <fmt/format.h>
#include <folly/Conv.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "velox/common/base/CountBits.h"
#include "velox/type/TypeKind.h"

namespace facebook::velox {

namespace {
constexpr int128_t buildSigned(uint64_t high, uint64_t low) {
  return static_cast<int128_t>((static_cast<uint128_t>(high) << 64) | low);
}

void testBasic(int128_t hugeInt, uint64_t upper, uint64_t lower) {
  EXPECT_EQ(hugeInt, buildSigned(upper, lower));
  EXPECT_EQ(upper, static_cast<uint64_t>(hugeInt >> 64));
  EXPECT_EQ(lower, static_cast<uint64_t>(hugeInt));
}

template <typename T>
void testMixedIntegralType() {
  static_assert(std::is_same_v<decltype(int128_t{1} + T{2}), int128_t>);
  static_assert(std::is_same_v<decltype(T{2} - int128_t{1}), int128_t>);
  static_assert(std::is_same_v<decltype(int128_t{2} * T{3}), int128_t>);
  static_assert(std::is_same_v<decltype(T{6} / int128_t{2}), int128_t>);
  static_assert(std::is_same_v<decltype(T{7} % int128_t{3}), int128_t>);

  const int128_t wide = buildSigned(1, 5);
  const T scalar = 3;
  EXPECT_EQ(wide + scalar, wide + int128_t{scalar});
  EXPECT_EQ(scalar + wide, int128_t{scalar} + wide);
  EXPECT_EQ(wide - scalar, wide - int128_t{scalar});
  EXPECT_EQ(scalar - wide, int128_t{scalar} - wide);
  EXPECT_EQ(wide * scalar, wide * int128_t{scalar});
  EXPECT_EQ(wide / scalar, wide / int128_t{scalar});
  EXPECT_EQ(wide % scalar, wide % int128_t{scalar});
  EXPECT_EQ(wide > scalar, wide > int128_t{scalar});
  EXPECT_EQ(scalar < wide, int128_t{scalar} < wide);

  auto product = wide;
  EXPECT_EQ(&(product *= scalar), &product);
  EXPECT_EQ(product, buildSigned(3, 15));

  // Unsigned maxima must reach the 128-bit operand without signed narrowing.
  const T maximum = std::numeric_limits<T>::max();
  product = int128_t{2};
  product *= maximum;
  EXPECT_EQ(product, int128_t{maximum} + int128_t{maximum});

  if constexpr (std::is_signed_v<T>) {
    product = int128_t{2};
    product *= T{-1};
    EXPECT_EQ(product, int128_t{-2});
  }
}
} // namespace

TEST(Int128Test, basic) {
  testBasic(0xDEADBEEE, 0x0, 0xDEADBEEE);

  // 0xF{16}F{16} = -1
  auto uint64Max = static_cast<int128_t>(std::numeric_limits<uint64_t>::max());
  int128_t hugeInt = -1;
  testBasic(
      hugeInt,
      static_cast<uint64_t>(uint64Max),
      static_cast<uint64_t>(uint64Max));

  hugeInt = std::numeric_limits<int128_t>::max() - 0x12345;
  uint64_t upper = 0x7FFFFFFFFFFFFFFF;
  uint64_t lower = 0xFFFFFFFFFFFEDCBA;
  testBasic(hugeInt, upper, lower);

  hugeInt = std::numeric_limits<int128_t>::min() + 0xDEADBEEFCAFECAFE;
  upper = 0x8000000000000000;
  lower = 0xDEADBEEFCAFECAFE;
  testBasic(hugeInt, upper, lower);

  // uint64Max * 0xDEADBEEF + 0xBADFEED = 0x0{8}DEADBEEEF{8}2D003FFE
  hugeInt = uint64Max * 0xDEADBEEF + 0xBADFEED;
  testBasic(hugeInt, 0xDEADBEEE, 0xFFFFFFFF2D003FFE);
}

TEST(Int128Test, floatingPointConversion) {
  const auto check = [](auto value,
                        uint64_t expectedDoubleBits,
                        uint32_t expectedFloatBits) {
    EXPECT_EQ(
        std::bit_cast<uint64_t>(static_cast<double>(value)),
        expectedDoubleBits);
    EXPECT_EQ(
        std::bit_cast<uint32_t>(static_cast<float>(value)), expectedFloatBits);
  };

  check(int128_t{0}, 0x0000000000000000, 0x00000000);
  check(int128_t{-1}, 0xbff0000000000000, 0xbf800000);
  check(buildSigned(1, 0), 0x43f0000000000000, 0x5f800000);
  check(buildSigned(1, 1ULL << 40), 0x43f0000010000000, 0x5f800000);
  check(buildSigned(1, (1ULL << 40) + 1), 0x43f0000010000000, 0x5f800001);
  check(std::numeric_limits<int128_t>::max(), 0x47e0000000000000, 0x7f000000);
  check(std::numeric_limits<int128_t>::min(), 0xc7e0000000000000, 0xff000000);
  check(std::numeric_limits<uint128_t>::max(), 0x47f0000000000000, 0x7f800000);
}

TEST(Int128Test, numericLimits) {
  static_assert(std::numeric_limits<int128_t>::digits == 127);
  static_assert(std::numeric_limits<uint128_t>::digits == 128);
  static_assert(std::numeric_limits<int128_t>::digits10 == 38);
  static_assert(std::numeric_limits<uint128_t>::digits10 == 38);
  static_assert(std::numeric_limits<int128_t>::radix == 2);
  static_assert(std::numeric_limits<uint128_t>::radix == 2);
  static_assert(std::numeric_limits<int128_t>::max_exponent == 0);
  static_assert(std::numeric_limits<uint128_t>::max_exponent == 0);
  static_assert(std::numeric_limits<int128_t>::is_bounded);
  static_assert(std::numeric_limits<uint128_t>::is_bounded);
  static_assert(!std::numeric_limits<int128_t>::is_modulo);
  static_assert(std::numeric_limits<uint128_t>::is_modulo);
}

TEST(Int128Test, aliasesAndConstantMultiplication) {
  static_assert(
      std::is_same_v<TypeTraits<TypeKind::HUGEINT>::NativeType, int128_t>);
  static_assert(std::is_same_v<int128_t, __int128_t>);
  static_assert(std::is_same_v<uint128_t, __uint128_t>);
#if !defined(__SIZEOF_INT128__)
  static_assert(std::is_same_v<int128_t, Int128>);
  static_assert(std::is_same_v<uint128_t, UInt128>);
#endif

  // DecimalArithmetic constructs its powers with compound multiplication.
  constexpr auto powerOfTen = [](uint8_t exponent) {
    int128_t result = 1;
    int128_t base = 10;
    while (exponent > 0) {
      if (exponent & 1) {
        result *= base;
      }
      exponent >>= 1;
      if (exponent > 0) {
        base *= base;
      }
    }
    return result;
  };
  static_assert(powerOfTen(0) == 1);
  static_assert(powerOfTen(18) == 1'000'000'000'000'000'000LL);
  constexpr auto widest = powerOfTen(38);
  static_assert(
      widest == buildSigned(0x4b3b4ca85a86c47aULL, 0x098a224000000000ULL));
  EXPECT_EQ(
      folly::to<std::string>(widest),
      "100000000000000000000000000000000000000");

  constexpr auto scalarProduct = [] {
    int128_t signedValue = 1;
    signedValue *= int32_t{10};
    signedValue *= int64_t{10};
    signedValue *= uint64_t{10};
    uint128_t unsignedValue = 10;
    unsignedValue *= uint128_t{100};
    return static_cast<uint128_t>(signedValue) == unsignedValue;
  }();
  static_assert(scalarProduct);
}

TEST(Int128Test, fullWidthArithmetic) {
  const auto maximum = std::numeric_limits<uint128_t>::max();
  const auto highBit = uint128_t{1} << 127;
  const auto lowMask = static_cast<uint128_t>(UINT64_MAX);
  for (const auto value :
       {uint128_t{0}, uint128_t{1}, lowMask, lowMask + 1, highBit, maximum}) {
    for (const auto divisor :
         {uint128_t{1}, uint128_t{3}, lowMask, lowMask + 1, highBit, maximum}) {
      const auto quotient = value / divisor;
      const auto remainder = value % divisor;
      EXPECT_EQ(quotient * divisor + remainder, value);
      EXPECT_LT(remainder, divisor);
      EXPECT_EQ((value + divisor) - divisor, value);
      EXPECT_EQ(value * divisor, divisor * value);
    }
  }
  EXPECT_EQ(
      maximum / uint128_t{3},
      (uint128_t{0x5555555555555555ULL} << 64) |
          uint128_t{0x5555555555555555ULL});
  EXPECT_EQ(
      lowMask * lowMask, (uint128_t{UINT64_MAX - 1} << 64) | uint128_t{1});
  EXPECT_EQ(maximum + uint128_t{1}, uint128_t{0});
  EXPECT_EQ(uint128_t{0} - uint128_t{1}, maximum);

  const auto signedDividend =
      buildSigned(0x123456789abcULL, 0xfedcba9876543210ULL);
  for (const auto sign : {-1, 1}) {
    for (const auto divisor :
         {int128_t{3}, int128_t{-3}, buildSigned(1, 1), -buildSigned(1, 1)}) {
      const auto value = signedDividend * sign;
      const auto quotient = value / divisor;
      const auto remainder = value % divisor;
      EXPECT_EQ(quotient * divisor + remainder, value);
      EXPECT_EQ(remainder < 0, sign < 0 && remainder != 0);
    }
  }

  auto value = lowMask;
  EXPECT_EQ(value++, lowMask);
  EXPECT_EQ(value, lowMask + 1);
  EXPECT_EQ(--value, lowMask);
  EXPECT_EQ(value--, lowMask);
  EXPECT_EQ(++value, lowMask);
}

TEST(Int128Test, digitCountAndShiftBoundaries) {
  uint128_t power = 1;
  EXPECT_EQ(countDigits(uint128_t{0}), 1);
  for (int digits = 1; digits <= 39; ++digits) {
    EXPECT_EQ(countDigits(power), digits);
    EXPECT_EQ(countDigits(power - 1), std::max(1, digits - 1));
    if (digits < 39) {
      power *= uint128_t{10};
    }
  }
  EXPECT_EQ(countDigits(std::numeric_limits<uint128_t>::max()), 39);

  for (const int shift : {0, 1, 63, 64, 65, 126, 127}) {
    const auto bit = uint128_t{1} << shift;
    EXPECT_EQ(bit >> shift, uint128_t{1});
    auto compound = uint128_t{1};
    compound <<= shift;
    EXPECT_EQ(compound, bit);
    compound >>= shift;
    EXPECT_EQ(compound, uint128_t{1});
    EXPECT_EQ(int128_t{-1} >> shift, int128_t{-1});
  }

#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
  // The emulation defines out-of-range shifts; native integers do not.
  for (const auto shift :
       {uint64_t{128},
        uint64_t{129},
        uint64_t{1} << 32,
        std::numeric_limits<uint64_t>::max()}) {
    EXPECT_EQ(uint128_t{1} << shift, uint128_t{0});
    EXPECT_EQ(std::numeric_limits<uint128_t>::max() >> shift, uint128_t{0});
    EXPECT_EQ(int128_t{-1} << shift, int128_t{0});
    EXPECT_EQ(int128_t{-1} >> shift, int128_t{-1});
    EXPECT_EQ(int128_t{1} >> shift, int128_t{0});
  }
  EXPECT_EQ(uint128_t{1} << -1, uint128_t{0});
  EXPECT_EQ(int128_t{-1} >> -1, int128_t{-1});
#endif
}

TEST(Int128Test, mixedIntegralCompatibility) {
  testMixedIntegralType<int8_t>();
  testMixedIntegralType<uint8_t>();
  testMixedIntegralType<int16_t>();
  testMixedIntegralType<uint16_t>();
  testMixedIntegralType<int32_t>();
  testMixedIntegralType<uint32_t>();
  testMixedIntegralType<int64_t>();
  testMixedIntegralType<uint64_t>();
  testMixedIntegralType<long>();
  testMixedIntegralType<unsigned long>();

  int128_t legacyProduct{7};
  EXPECT_EQ(&(legacyProduct *= true), &legacyProduct);
  EXPECT_EQ(legacyProduct, int128_t{7});
  enum LegacyMultiplier { kTriple = 3 };
  EXPECT_EQ(&(legacyProduct *= kTriple), &legacyProduct);
  EXPECT_EQ(legacyProduct, int128_t{21});
  legacyProduct *= false;
  EXPECT_EQ(legacyProduct, int128_t{0});

  const uint64_t wideValue = 0x1'0000'0001ULL;
  EXPECT_EQ(wideValue - int128_t{1}, int128_t{0x1'0000'0000ULL});
  EXPECT_EQ(int128_t{-1000} / uint64_t{3}, int128_t{-333});
  EXPECT_EQ(int128_t{-1000} % uint64_t{3}, int128_t{-1});

  const uint64_t wideDivisor = 0x8000'0000'0000'0001ULL;
  const int128_t dividend = buildSigned(1, 0);
  EXPECT_EQ(dividend / wideDivisor, int128_t{1});
  EXPECT_EQ(dividend % wideDivisor, int128_t{0x7fff'ffff'ffff'ffffULL});

  const uint128_t narrowSource =
      (uint128_t{1} << 100) | uint128_t{0x1234'5678'9abc'de41ULL};
  EXPECT_EQ(uint128_t{1} << uint32_t{127}, uint128_t{1} << 127);
  EXPECT_EQ(narrowSource >> uint32_t{64}, narrowSource >> 64);
  const uint64_t narrow64 = narrowSource;
  const uint32_t narrow32 = narrowSource;
  const char narrowChar = narrowSource;
  EXPECT_EQ(narrow64, 0x1234'5678'9abc'de41ULL);
  EXPECT_EQ(narrow32, 0x9abc'de41U);
  EXPECT_EQ(narrowChar, 'A');

  int64_t compoundResult = 7;
  compoundResult *= int128_t{6};
  EXPECT_EQ(compoundResult, 42);
}

#if defined(_MSC_VER) && !defined(__SIZEOF_INT128__)
namespace {

enum SmallOperand : uint8_t { kOne = 1, kThree = 3 };
enum WideOperand : uint64_t { kWideMax = UINT64_MAX };
enum NegativeOperand : int64_t { kNegative = -3 };
enum class ScopedOperand : uint64_t { kOne = 1 };

struct NativeConvertible {
  constexpr operator int64_t() const {
    return 7;
  }
};

struct Int128Convertible {
  constexpr operator int128_t() const {
    return int128_t{7};
  }
};

template <typename L, typename R>
concept HasFreeAddition = requires(const L& left, const R& right) {
  facebook::velox::operator+(left, right);
};

template <typename L, typename R>
void testIntegralComparisons(
    const L& left,
    const R& right,
    bool less,
    bool equal) {
  EXPECT_EQ(left == right, equal);
  EXPECT_EQ(right == left, equal);
  EXPECT_EQ(left != right, !equal);
  EXPECT_EQ(right != left, !equal);
  EXPECT_EQ(left < right, less);
  EXPECT_EQ(right > left, less);
  EXPECT_EQ(left <= right, less || equal);
  EXPECT_EQ(right >= left, less || equal);
  EXPECT_EQ(left > right, !less && !equal);
  EXPECT_EQ(right < left, !less && !equal);
  EXPECT_EQ(left >= right, !less);
  EXPECT_EQ(right <= left, !less);
}

template <typename I, typename T>
void testPrimitiveCompounds() {
  T value = 12;
  const I wide(1, 1);
#define CHECK_PRIMITIVE_COMPOUND(OP, RIGHT, EXPECTED)              \
  static_assert(std::is_same_v<decltype(value OP## = RIGHT), T&>); \
  static_assert(!requires(const T& a) { a OP## = RIGHT; });        \
  static_assert(!requires { std::declval<T>() OP## = RIGHT; });    \
  value = 12;                                                      \
  EXPECT_EQ(&(value OP## = RIGHT), &value);                        \
  EXPECT_EQ(value, static_cast<T>(EXPECTED));

  CHECK_PRIMITIVE_COMPOUND(+, wide, 13)
  CHECK_PRIMITIVE_COMPOUND(-, wide, 11)
  CHECK_PRIMITIVE_COMPOUND(*, wide, 12)
  CHECK_PRIMITIVE_COMPOUND(/, wide, 0)
  CHECK_PRIMITIVE_COMPOUND(%, wide, 12)
  CHECK_PRIMITIVE_COMPOUND(&, wide, 0)
  CHECK_PRIMITIVE_COMPOUND(|, wide, 13)
  CHECK_PRIMITIVE_COMPOUND(^, wide, 13)
  CHECK_PRIMITIVE_COMPOUND(/, I(1, 0), 0)
  CHECK_PRIMITIVE_COMPOUND(%, I(1, 0), 12)
  CHECK_PRIMITIVE_COMPOUND(/, I{5}, 2)
  CHECK_PRIMITIVE_COMPOUND(%, I{5}, 2)
  if constexpr (std::is_same_v<I, int128_t>) {
    CHECK_PRIMITIVE_COMPOUND(/, I{-5}, -2)
    CHECK_PRIMITIVE_COMPOUND(%, I{-5}, 2)
  }
#undef CHECK_PRIMITIVE_COMPOUND

  value = 12;
  EXPECT_THROW(value /= I{0}, std::runtime_error);
  EXPECT_EQ(value, 12);
  EXPECT_THROW(value %= I{0}, std::runtime_error);
  EXPECT_EQ(value, 12);

  if constexpr (std::is_signed_v<T>) {
    value = -13;
    value /= I{2};
    EXPECT_EQ(value, (std::is_same_v<I, int128_t> ? -6 : -7));
    value = -13;
    value %= I{2};
    EXPECT_EQ(value, (std::is_same_v<I, int128_t> ? -1 : 1));
  }
}

template <typename T>
void testCommonType() {
  static_assert(std::is_same_v<std::common_type_t<int128_t, T>, int128_t>);
  static_assert(std::is_same_v<std::common_type_t<T, int128_t>, int128_t>);
  static_assert(std::is_same_v<std::common_type_t<uint128_t, T>, uint128_t>);
  static_assert(std::is_same_v<std::common_type_t<T, uint128_t>, uint128_t>);
  static_assert(
      std::
          is_same_v<std::common_type_t<const int128_t&, volatile T>, int128_t>);
}

template <typename I, typename T>
void testClassAndScalarOperators() {
  const I wide(1, 12);
  const T three = 3;
#define CHECK_SCALAR_OPERATOR(OP, FORWARD, REVERSE)                   \
  static_assert(std::is_same_v<decltype(wide OP three), I>);          \
  static_assert(std::is_same_v<decltype(three OP wide), I>);          \
  static_assert(                                                      \
      std::is_same_v<decltype(std::declval<I&>() OP## = three), I&>); \
  EXPECT_EQ(wide OP three, FORWARD);                                  \
  EXPECT_EQ(three OP wide, REVERSE);                                  \
  {                                                                   \
    auto value = wide;                                                \
    EXPECT_EQ(&(value OP## = three), &value);                         \
    EXPECT_EQ(value, FORWARD);                                        \
  }

  CHECK_SCALAR_OPERATOR(+, I(1, 15), I(1, 15))
  CHECK_SCALAR_OPERATOR(
      -, I(1, 9), I(uint128_t(UINT64_MAX - 1, UINT64_MAX - 8)))
  CHECK_SCALAR_OPERATOR(*, I(3, 36), I(3, 36))
  CHECK_SCALAR_OPERATOR(/, I{6'148'914'691'236'517'209ULL}, I{0})
  CHECK_SCALAR_OPERATOR(%, I{1}, I{3})
  CHECK_SCALAR_OPERATOR(&, I{0}, I{0})
  CHECK_SCALAR_OPERATOR(|, I(1, 15), I(1, 15))
  CHECK_SCALAR_OPERATOR(^, I(1, 15), I(1, 15))
#undef CHECK_SCALAR_OPERATOR

  testIntegralComparisons(wide, three, false, false);
  testIntegralComparisons(I{3}, three, false, true);
  constexpr I zero{0};
  static_assert(zero == T{0});
  static_assert(T{0} == zero);
  static_assert(!(zero != T{0}));
  static_assert(!(T{0} != zero));
  static_assert(zero < T{1});
  static_assert(T{1} > zero);
  static_assert(zero <= T{0});
  static_assert(T{0} >= zero);

  volatile T volatileThree = 3;
  EXPECT_EQ(wide + volatileThree, I(1, 15));
  EXPECT_EQ(volatileThree + wide, I(1, 15));
  EXPECT_EQ(wide / volatileThree, I{6'148'914'691'236'517'209ULL});
  EXPECT_EQ(wide % volatileThree, I{1});
}

template <typename T>
void testIntegralSignedness() {
  testPrimitiveCompounds<int128_t, T>();
  testPrimitiveCompounds<uint128_t, T>();
  testClassAndScalarOperators<int128_t, T>();
  testClassAndScalarOperators<uint128_t, T>();
  testCommonType<T>();
}

template <typename I>
void testBoolCompounds() {
  bool value;
  const I wide(1, 1);
#define CHECK_BOOL_COMPOUND(OP, EXPECTED)                            \
  static_assert(std::is_same_v<decltype(value OP## = wide), bool&>); \
  value = true;                                                      \
  EXPECT_EQ(&(value OP## = wide), &value);                           \
  EXPECT_EQ(value, EXPECTED);

  CHECK_BOOL_COMPOUND(+, true)
  CHECK_BOOL_COMPOUND(-, true)
  CHECK_BOOL_COMPOUND(*, true)
  CHECK_BOOL_COMPOUND(/, false)
  CHECK_BOOL_COMPOUND(%, true)
  CHECK_BOOL_COMPOUND(&, true)
  CHECK_BOOL_COMPOUND(|, true)
  CHECK_BOOL_COMPOUND(^, true)
#undef CHECK_BOOL_COMPOUND
  value = false;
  value += I(1, 0);
  EXPECT_TRUE(value);
  value *= I(1, 0);
  EXPECT_TRUE(value);
  EXPECT_THROW(value /= I{0}, std::runtime_error);
  EXPECT_TRUE(value);
  EXPECT_THROW(value %= I{0}, std::runtime_error);
  EXPECT_TRUE(value);
}

template <typename I, typename T>
void testPromotedOperand(T one) {
  const I wide(1, 2);
#define CHECK_PROMOTED_BINARY(OP, FORWARD, REVERSE)                           \
  static_assert(std::is_same_v<decltype(wide OP one), I>);                    \
  static_assert(std::is_same_v<decltype(one OP wide), I>);                    \
  static_assert(std::is_same_v<decltype(std::declval<I&>() OP## = one), I&>); \
  EXPECT_EQ(wide OP one, FORWARD);                                            \
  EXPECT_EQ(one OP wide, REVERSE);                                            \
  {                                                                           \
    I compound = wide;                                                        \
    EXPECT_EQ(&(compound OP## = one), &compound);                             \
    EXPECT_EQ(compound, FORWARD);                                             \
  }

  CHECK_PROMOTED_BINARY(+, I(1, 3), I(1, 3))
  CHECK_PROMOTED_BINARY(-, I(1, 1), I(uint128_t(~uint64_t{1}, UINT64_MAX)))
  CHECK_PROMOTED_BINARY(*, wide, wide)
  CHECK_PROMOTED_BINARY(/, wide, I{0})
  CHECK_PROMOTED_BINARY(%, I{0}, I{1})
  CHECK_PROMOTED_BINARY(&, I{0}, I{0})
  CHECK_PROMOTED_BINARY(|, I(1, 3), I(1, 3))
  CHECK_PROMOTED_BINARY(^, I(1, 3), I(1, 3))
#undef CHECK_PROMOTED_BINARY

  EXPECT_TRUE(wide > one);
  EXPECT_TRUE(wide >= one);
  EXPECT_FALSE(wide < one);
  EXPECT_FALSE(wide <= one);
  EXPECT_FALSE(wide == one);
  EXPECT_TRUE(wide != one);
  EXPECT_TRUE(one < wide);
  EXPECT_TRUE(one <= wide);
  EXPECT_FALSE(one > wide);
  EXPECT_FALSE(one >= wide);
  EXPECT_FALSE(one == wide);
  EXPECT_TRUE(one != wide);
  EXPECT_TRUE(I{1} == one);
  EXPECT_TRUE(one == I{1});
}

template <typename I>
void testRejectedEnumOperands() {
#define CHECK_REJECTED_ENUM(OP)                                    \
  static_assert(!requires(I a, ScopedOperand b) { a OP b; });      \
  static_assert(!requires(ScopedOperand a, I b) { a OP b; });      \
  static_assert(!requires(I& a, ScopedOperand b) { a OP## = b; }); \
  static_assert(!requires(ScopedOperand& a, I b) { a OP## = b; }); \
  static_assert(!requires(SmallOperand& a, I b) { a OP## = b; });

  CHECK_REJECTED_ENUM(+)
  CHECK_REJECTED_ENUM(-)
  CHECK_REJECTED_ENUM(*)
  CHECK_REJECTED_ENUM(/)
  CHECK_REJECTED_ENUM(%)
  CHECK_REJECTED_ENUM(&)
  CHECK_REJECTED_ENUM(|)
  CHECK_REJECTED_ENUM(^)
#undef CHECK_REJECTED_ENUM
#define CHECK_REJECTED_COMPARISON(OP)                         \
  static_assert(!requires(I a, ScopedOperand b) { a OP b; }); \
  static_assert(!requires(ScopedOperand a, I b) { a OP b; });

  CHECK_REJECTED_COMPARISON(==)
  CHECK_REJECTED_COMPARISON(!=)
  CHECK_REJECTED_COMPARISON(<)
  CHECK_REJECTED_COMPARISON(<=)
  CHECK_REJECTED_COMPARISON(>)
  CHECK_REJECTED_COMPARISON(>=)
#undef CHECK_REJECTED_COMPARISON
}

template <typename I, typename F>
void testFloatingCompounds() {
  const I wide(1, 0);
  const F floatingWide = std::ldexp(F{1}, 64);
  F native;
#define CHECK_FLOATING_BINARY(OP)                                  \
  static_assert(std::is_same_v<decltype(wide OP F{2.5}), F>);      \
  static_assert(std::is_same_v<decltype(F{2.5} OP wide), F>);      \
  static_assert(std::is_same_v<decltype(native OP## = wide), F&>); \
  EXPECT_EQ(wide OP F{2.5}, floatingWide OP F{2.5});               \
  EXPECT_EQ(F{2.5} OP wide, F{2.5} OP floatingWide);               \
  native = F{2.5};                                                 \
  EXPECT_EQ(&(native OP## = wide), &native);                       \
  EXPECT_EQ(native, F{2.5} OP floatingWide);

  CHECK_FLOATING_BINARY(+)
  CHECK_FLOATING_BINARY(-)
  CHECK_FLOATING_BINARY(*)
  CHECK_FLOATING_BINARY(/)
#undef CHECK_FLOATING_BINARY

  // Check every ordering directly against IEEE arithmetic, including
  // unordered NaNs and UInt128-to-float overflow to positive infinity.
  for (const I integer :
       {I{0},
        I{7},
        wide,
        std::numeric_limits<I>::lowest(),
        std::numeric_limits<I>::max()}) {
    const F converted = static_cast<F>(integer);
    for (const F scalar :
         {F{0},
          F{7},
          F{7.5},
          std::numeric_limits<F>::infinity(),
          -std::numeric_limits<F>::infinity(),
          std::numeric_limits<F>::quiet_NaN()}) {
#define CHECK_FLOATING_COMPARISON(OP)                \
  EXPECT_EQ(integer OP scalar, converted OP scalar); \
  EXPECT_EQ(scalar OP integer, scalar OP converted);

      CHECK_FLOATING_COMPARISON(==)
      CHECK_FLOATING_COMPARISON(!=)
      CHECK_FLOATING_COMPARISON(<)
      CHECK_FLOATING_COMPARISON(<=)
      CHECK_FLOATING_COMPARISON(>)
      CHECK_FLOATING_COMPARISON(>=)
#undef CHECK_FLOATING_COMPARISON
    }
  }

  I value{7};
#define CHECK_FLOATING_COMPOUND(OP, RIGHT, EXPECTED)                  \
  static_assert(std::is_same_v<decltype(value OP## = F{RIGHT}), I&>); \
  value = I{7};                                                       \
  EXPECT_EQ(&(value OP## = F{RIGHT}), &value);                        \
  EXPECT_EQ(value, I{EXPECTED});

  CHECK_FLOATING_COMPOUND(+, 0.75, 7)
  CHECK_FLOATING_COMPOUND(-, 0.75, 6)
  CHECK_FLOATING_COMPOUND(*, 1.5, 10)
  CHECK_FLOATING_COMPOUND(/, 2.5, 2)
#undef CHECK_FLOATING_COMPOUND

  volatile F zero = 0;
  value = I{7};
  value /= zero;
  EXPECT_EQ(value, std::numeric_limits<I>::max());
  value = I{7};
  value /= -zero;
  EXPECT_EQ(value, std::numeric_limits<I>::lowest());
  value = I{7};
  value += std::numeric_limits<F>::infinity();
  EXPECT_EQ(value, std::numeric_limits<I>::max());
  value = I{7};
  value -= std::numeric_limits<F>::infinity();
  EXPECT_EQ(value, std::numeric_limits<I>::lowest());
  value = I{7};
  value *= std::numeric_limits<F>::max();
  EXPECT_EQ(value, std::numeric_limits<I>::max());
  value = I{7};
  value += std::copysign(std::numeric_limits<F>::quiet_NaN(), F{1});
  EXPECT_EQ(value, std::numeric_limits<I>::max());
  value = I{7};
  value += std::copysign(std::numeric_limits<F>::quiet_NaN(), F{-1});
  EXPECT_EQ(value, std::numeric_limits<I>::lowest());

  // Native arithmetic is performed at F's precision before integer truncation.
  value = I{16'777'217};
  value += F{0};
  EXPECT_EQ(value, (I{std::is_same_v<F, float> ? 16'777'216 : 16'777'217}));
  static_assert(!requires(I a, F b) { a % b; });
  static_assert(!requires(I& a, F b) { a %= b; });
#define CHECK_REJECTED_FLOATING(OP)                    \
  static_assert(!requires(I a, F b) { a OP b; });      \
  static_assert(!requires(F a, I b) { a OP b; });      \
  static_assert(!requires(I& a, F b) { a OP## = b; }); \
  static_assert(!requires(F& a, I b) { a OP## = b; });

  CHECK_REJECTED_FLOATING(%)
  CHECK_REJECTED_FLOATING(&)
  CHECK_REJECTED_FLOATING(|)
  CHECK_REJECTED_FLOATING(^)
#undef CHECK_REJECTED_FLOATING
}

} // namespace

TEST(Int128Test, integralOperatorPromotion) {
  testIntegralSignedness<int8_t>();
  testIntegralSignedness<uint8_t>();
  testIntegralSignedness<int16_t>();
  testIntegralSignedness<uint16_t>();
  testIntegralSignedness<int32_t>();
  testIntegralSignedness<uint32_t>();
  testIntegralSignedness<int64_t>();
  testIntegralSignedness<uint64_t>();
  testIntegralSignedness<long>();
  testIntegralSignedness<unsigned long>();
  testBoolCompounds<int128_t>();
  testBoolCompounds<uint128_t>();

  static_assert(!HasFreeAddition<int, int>);
  static_assert(!HasFreeAddition<SmallOperand, SmallOperand>);
  static_assert(!HasFreeAddition<double, NativeConvertible>);
  static_assert(!HasFreeAddition<int128_t, NativeConvertible>);
  static_assert(!HasFreeAddition<NativeConvertible, uint128_t>);
  static_assert(!HasFreeAddition<Int128Convertible, int128_t>);
  static_assert(!HasFreeAddition<uint128_t, Int128Convertible>);
  static_assert(!HasFreeAddition<volatile int128_t, int>);
  static_assert(!HasFreeAddition<int, volatile uint128_t>);
  static_assert(!HasFreeAddition<int128_t, int128_t>);
  static_assert(!HasFreeAddition<uint128_t, uint128_t>);
  static_assert(HasFreeAddition<const int128_t, const uint128_t>);
  static_assert(HasFreeAddition<const int, const int128_t>);
  static_assert(NativeConvertible{} + int64_t{1} == 8);
  static_assert(std::is_same_v<decltype(NativeConvertible{} + 1), int64_t>);
#define CHECK_NATIVE_BINARY(OP) \
  static_assert(std::is_same_v<decltype(kOne OP kThree), int>);

  CHECK_NATIVE_BINARY(+)
  CHECK_NATIVE_BINARY(-)
  CHECK_NATIVE_BINARY(*)
  CHECK_NATIVE_BINARY(/)
  CHECK_NATIVE_BINARY(%)
  CHECK_NATIVE_BINARY(&)
  CHECK_NATIVE_BINARY(|)
  CHECK_NATIVE_BINARY(^)
#undef CHECK_NATIVE_BINARY
}

TEST(Int128Test, mixedSignednessArithmetic) {
  const int128_t signedValue{-1};
  const uint128_t unsignedValue{2};
  const auto maximum = std::numeric_limits<uint128_t>::max();
  const uint128_t maxMinusTwo(UINT64_MAX, UINT64_MAX - 2);
  const uint128_t maxMinusOne(UINT64_MAX, UINT64_MAX - 1);
  testIntegralComparisons(signedValue, unsignedValue, false, false);
  testIntegralComparisons(int128_t{2}, unsignedValue, false, true);
  testIntegralComparisons(int128_t{1}, unsignedValue, true, false);
  static_assert(int128_t{-1} == uint128_t(UINT64_MAX, UINT64_MAX));
  static_assert(uint128_t{1} != int128_t{-1});
  static_assert(int128_t{-1} >= uint128_t{1});
  static_assert(uint128_t{1} <= int128_t{-1});
#define CHECK_CROSS_BINARY(OP, FORWARD, REVERSE)                          \
  static_assert(                                                          \
      std::is_same_v<decltype(signedValue OP unsignedValue), uint128_t>); \
  static_assert(                                                          \
      std::is_same_v<decltype(unsignedValue OP signedValue), uint128_t>); \
  static_assert(std::is_same_v<                                           \
                decltype(std::declval<int128_t&>() OP## = unsignedValue), \
                int128_t&>);                                              \
  static_assert(std::is_same_v<                                           \
                decltype(std::declval<uint128_t&>() OP## = signedValue),  \
                uint128_t&>);                                             \
  EXPECT_EQ(signedValue OP unsignedValue, FORWARD);                       \
  EXPECT_EQ(unsignedValue OP signedValue, REVERSE);                       \
  {                                                                       \
    auto a = signedValue;                                                 \
    auto b = unsignedValue;                                               \
    EXPECT_EQ(&(a OP## = unsignedValue), &a);                             \
    EXPECT_EQ(a, int128_t(FORWARD));                                      \
    EXPECT_EQ(&(b OP## = signedValue), &b);                               \
    EXPECT_EQ(b, REVERSE);                                                \
  }

  CHECK_CROSS_BINARY(+, uint128_t{1}, uint128_t{1})
  CHECK_CROSS_BINARY(-, maxMinusTwo, uint128_t{3})
  CHECK_CROSS_BINARY(*, maxMinusOne, maxMinusOne)
  CHECK_CROSS_BINARY(
      /, uint128_t(std::numeric_limits<int128_t>::max()), uint128_t{0})
  CHECK_CROSS_BINARY(%, uint128_t{1}, uint128_t{2})
  CHECK_CROSS_BINARY(&, uint128_t{2}, uint128_t{2})
  CHECK_CROSS_BINARY(|, maximum, maximum)
  CHECK_CROSS_BINARY(^, maxMinusTwo, maxMinusTwo)
#undef CHECK_CROSS_BINARY

  int128_t signedLeft{-1};
  EXPECT_THROW(signedLeft /= uint128_t{0}, std::runtime_error);
  EXPECT_EQ(signedLeft, int128_t{-1});
  EXPECT_THROW(signedLeft %= uint128_t{0}, std::runtime_error);
  EXPECT_EQ(signedLeft, int128_t{-1});
  uint128_t unsignedLeft(1, 1);
  EXPECT_THROW(unsignedLeft /= int128_t{0}, std::runtime_error);
  EXPECT_EQ(unsignedLeft, uint128_t(1, 1));
  EXPECT_THROW(unsignedLeft %= int128_t{0}, std::runtime_error);
  EXPECT_EQ(unsignedLeft, uint128_t(1, 1));
  signedLeft = int128_t{12};
  signedLeft /= uint128_t(1, 0);
  EXPECT_EQ(signedLeft, int128_t{0});
  signedLeft = int128_t{12};
  signedLeft %= uint128_t(1, 1);
  EXPECT_EQ(signedLeft, int128_t{12});
}

TEST(Int128Test, boolAndEnumPromotion) {
  testPromotedOperand<int128_t>(true);
  testPromotedOperand<uint128_t>(true);
  testPromotedOperand<int128_t>(kOne);
  testPromotedOperand<uint128_t>(kOne);
  testRejectedEnumOperands<int128_t>();
  testRejectedEnumOperands<uint128_t>();
  const int128_t wide(1, 0);
  EXPECT_EQ(wide + false, wide);
  EXPECT_EQ(false + wide, wide);
  EXPECT_EQ(wide * false, int128_t{0});
  EXPECT_EQ(false * wide, int128_t{0});
  EXPECT_EQ(wide + kWideMax, int128_t(1, UINT64_MAX));
  EXPECT_EQ(kWideMax + wide, int128_t(1, UINT64_MAX));
  EXPECT_EQ(wide * kWideMax, int128_t(-1, 0));
  EXPECT_EQ(kWideMax * wide, int128_t(-1, 0));
  EXPECT_EQ(wide / kWideMax, int128_t{1});
  EXPECT_EQ(wide % kWideMax, int128_t{1});
  EXPECT_EQ(kWideMax / wide, int128_t{0});
  EXPECT_EQ(kWideMax % wide, int128_t(0, UINT64_MAX));
  EXPECT_LT(kWideMax, wide);
  EXPECT_GT(wide, kWideMax);
  EXPECT_EQ(int128_t{7} * kNegative, int128_t{-21});
  EXPECT_EQ(kNegative * int128_t{7}, int128_t{-21});
  EXPECT_EQ(int128_t{7} / kNegative, int128_t{-2});
  EXPECT_EQ(int128_t{7} % kNegative, int128_t{1});
  EXPECT_LT(kNegative, int128_t{0});
  EXPECT_GT(kNegative, uint128_t{0});
  auto product = wide;
  product *= kWideMax;
  EXPECT_EQ(product, int128_t(-1, 0));
  uint128_t unsignedProduct{7};
  unsignedProduct *= kThree;
  EXPECT_EQ(unsignedProduct, uint128_t{21});
  EXPECT_THROW(product /= false, std::runtime_error);
  EXPECT_EQ(product, int128_t(-1, 0));
  EXPECT_THROW(product %= false, std::runtime_error);
  EXPECT_EQ(product, int128_t(-1, 0));
}

TEST(Int128Test, unaryAndConstexprCompatibility) {
  constexpr int128_t signedWide(1, 0);
  constexpr uint128_t unsignedWide(1, 1);
  static_assert(std::is_same_v<decltype(+signedWide), int128_t>);
  static_assert(std::is_same_v<decltype(+unsignedWide), uint128_t>);
  static_assert(std::is_same_v<decltype(-signedWide), int128_t>);
  static_assert(std::is_same_v<decltype(-unsignedWide), uint128_t>);
  static_assert(+signedWide == signedWide);
  static_assert(+unsignedWide == unsignedWide);
  static_assert(-unsignedWide == uint128_t(UINT64_MAX - 1, UINT64_MAX));
  static_assert(-uint128_t{0} == uint128_t{0});
  static_assert(int128_t(unsignedWide) == int128_t(1, 1));
  static_assert(int128_t(uint128_t(UINT64_MAX, UINT64_MAX)) == int128_t{-1});
  static_assert((int128_t{1} | int128_t{2}) == int128_t{3});
  static_assert((int128_t{-1} & int128_t{3}) == int128_t{3});
  static_assert((signedWide ^ int128_t{3}) == int128_t(1, 3));
  static_assert(~signedWide == int128_t(-2, UINT64_MAX));
  static_assert(signedWide * int32_t{2} == int128_t(2, 0));
  static_assert(signedWide * int64_t{2} == int128_t(2, 0));
  static_assert(int32_t{2} * signedWide == int128_t(2, 0));
  static_assert(int64_t{2} * signedWide == int128_t(2, 0));
  static_assert(int128_t{-1} + uint128_t{1} == uint128_t{0});
  static_assert(
      int128_t{-1} * uint128_t{2} == uint128_t(UINT64_MAX, UINT64_MAX - 1));
  static_assert((int128_t{-1} & uint128_t{2}) == uint128_t{2});
  static_assert(signedWide + true == int128_t(1, 1));
  static_assert(kWideMax + signedWide == int128_t(1, UINT64_MAX));
  EXPECT_EQ(+signedWide, int128_t(1, 0));
  EXPECT_EQ(-unsignedWide, uint128_t(UINT64_MAX - 1, UINT64_MAX));

  static_assert(sizeof(int128_t) == 16);
  static_assert(sizeof(uint128_t) == 16);
  static_assert(alignof(int128_t) == alignof(uint64_t));
  static_assert(alignof(uint128_t) == alignof(uint64_t));
  static_assert(std::is_trivially_copyable_v<int128_t>);
  static_assert(std::is_trivially_copyable_v<uint128_t>);
  const auto limbs = std::bit_cast<std::array<uint64_t, 2>>(uint128_t(1, 2));
  EXPECT_EQ(limbs[0], 2);
  EXPECT_EQ(limbs[1], 1);
}

TEST(Int128Test, commonTypeCompatibility) {
  testCommonType<bool>();
  static_assert(
      std::is_same_v<std::common_type_t<int128_t, uint128_t>, uint128_t>);
  static_assert(
      std::is_same_v<std::common_type_t<uint128_t, int128_t>, uint128_t>);
  static_assert(std::is_same_v<
                std::common_type_t<const int128_t&, volatile uint128_t>,
                uint128_t>);
  static_assert(std::is_same_v<std::common_type_t<int128_t, float>, float>);
  static_assert(std::is_same_v<std::common_type_t<float, uint128_t>, float>);
  static_assert(std::is_same_v<std::common_type_t<double, int128_t>, double>);
  static_assert(std::is_same_v<std::common_type_t<uint128_t, double>, double>);
  static_assert(
      std::is_same_v<std::common_type_t<int128_t, long double>, long double>);
  static_assert(
      std::is_same_v<std::common_type_t<long double, uint128_t>, long double>);
}

TEST(Int128Test, floatingPointCompoundAndLongDouble) {
  testFloatingCompounds<int128_t, float>();
  testFloatingCompounds<int128_t, double>();
  testFloatingCompounds<int128_t, long double>();
  testFloatingCompounds<uint128_t, float>();
  testFloatingCompounds<uint128_t, double>();
  testFloatingCompounds<uint128_t, long double>();
  for (const auto value :
       {int128_t{0},
        int128_t{-1},
        int128_t(1, 1),
        std::numeric_limits<int128_t>::min(),
        std::numeric_limits<int128_t>::max()}) {
    EXPECT_EQ(
        std::bit_cast<uint64_t>(static_cast<long double>(value)),
        std::bit_cast<uint64_t>(static_cast<double>(value)));
  }
  EXPECT_EQ(
      std::bit_cast<uint64_t>(
          static_cast<long double>(std::numeric_limits<uint128_t>::max())),
      0x47f0000000000000ULL);
  int128_t negative{-7};
  volatile double zero = 0.0;
  negative /= zero;
  EXPECT_EQ(negative, std::numeric_limits<int128_t>::min());
}

TEST(Int128Test, constructionFromBoolAndFloatingPoint) {
  static_assert(std::is_constructible_v<int128_t, bool>);
  static_assert(!std::is_convertible_v<bool, int128_t>);
  static_assert(std::is_constructible_v<uint128_t, bool>);
  static_assert(!std::is_convertible_v<bool, uint128_t>);
  EXPECT_EQ(int128_t{false}, int128_t{0});
  EXPECT_EQ(int128_t{true}, int128_t{1});
  EXPECT_EQ(static_cast<int128_t>(42.75), int128_t{42});
  EXPECT_EQ(static_cast<int128_t>(-42.75), int128_t{-42});
  EXPECT_EQ(static_cast<uint128_t>(42.75f), uint128_t{42});
  EXPECT_EQ(static_cast<uint128_t>(42.75), uint128_t{42});
  EXPECT_EQ(static_cast<uint128_t>(-0.5f), uint128_t{0});
  EXPECT_EQ(static_cast<uint128_t>(-0.5), uint128_t{0});
  EXPECT_EQ(
      static_cast<uint128_t>(std::ldexp(1.0f, 100) + std::ldexp(1.0f, 77)),
      (uint128_t{1} << 100) + (uint128_t{1} << 77));
  EXPECT_EQ(
      static_cast<uint128_t>(std::ldexp(1.0, 100) + std::ldexp(1.0, 48)),
      (uint128_t{1} << 100) + (uint128_t{1} << 48));
  EXPECT_EQ(
      static_cast<int128_t>(-std::ldexp(1.0, 100) - std::ldexp(1.0, 48)),
      -((int128_t{1} << 100) + (int128_t{1} << 48)));
}

TEST(Int128Test, signednessTraitsAndMixedComparison) {
  static_assert(std::is_same_v<std::make_unsigned_t<int128_t>, uint128_t>);
  static_assert(std::is_same_v<std::make_unsigned_t<uint128_t>, uint128_t>);
  static_assert(std::is_same_v<std::make_signed_t<int128_t>, int128_t>);
  static_assert(std::is_same_v<std::make_signed_t<uint128_t>, int128_t>);

  EXPECT_LT(uint128_t{1}, int128_t{-1});
  EXPECT_GT(int128_t{-1}, uint128_t{1});
  EXPECT_EQ(uint128_t{42}, int128_t{42});
  EXPECT_GT((uint128_t{0, 1}), int128_t{0});
}

TEST(Int128Test, toChars) {
  char buffer[40];
  const auto value = std::numeric_limits<int128_t>::min();
  const auto [end, error] = value.toChars(buffer, buffer + sizeof(buffer));
  EXPECT_EQ(error, std::errc{});
  EXPECT_EQ(
      std::string(buffer, end), "-170141183460469231731687303715884105728");

  char shortBuffer[39];
  const auto [shortEnd, shortError] =
      value.toChars(shortBuffer, shortBuffer + sizeof(shortBuffer));
  EXPECT_EQ(shortEnd, shortBuffer + sizeof(shortBuffer));
  EXPECT_EQ(shortError, std::errc::value_too_large);

  const auto unsignedValue = std::numeric_limits<uint128_t>::max();
  const auto [unsignedEnd, unsignedError] =
      unsignedValue.toChars(buffer, buffer + sizeof(buffer));
  EXPECT_EQ(unsignedError, std::errc{});
  EXPECT_EQ(
      std::string(buffer, unsignedEnd),
      "340282366920938463463374607431768211455");
}

// A divisor in [2^63, 2^64) makes the partial remainder of the ARM64 128/64
// long division exceed 64 bits. Decimal overflow checks divide by such
// operands and toString() divides by 10^19, so both must stay exact.
TEST(Int128Test, divideByDivisorWithTopBitSet) {
  const int128_t tenPow19 = buildSigned(0, 10'000'000'000'000'000'000ULL);
  const int128_t dividend = buildSigned(0x1431e0d90ULL, 0x533a9e64caaa7200ULL);
  EXPECT_EQ(dividend / tenPow19, int128_t(9'999'998'999LL));
  EXPECT_EQ(dividend % tenPow19, int128_t{9'975'600'002'440'000'000ULL});
  EXPECT_EQ(dividend.toString(), "99999989999975600002440000000");

  const int128_t left = -6'935'504'817LL;
  const int128_t right = -int128_t{15'974'410'434'149'479'725ULL};
  const int128_t product = left * right;
  EXPECT_EQ(product, buildSigned(0x165fbd638ULL, 0x7470625946ea9f1dULL));
  EXPECT_EQ(product / right, left);
  const int128_t negative =
      buildSigned(0xffffffd859e0587fULL, 0x1a4663051a85a01cULL);
  const int128_t divisor{9'999'999'999'999'999'415ULL};
  EXPECT_EQ(negative / divisor, int128_t(-314'131'111'740LL));
  EXPECT_EQ(negative % divisor, int128_t(0));

  // 128/128 path with a divisor whose top bit is set.
  const auto unsignedMax = std::numeric_limits<uint128_t>::max();
  const uint128_t unsignedDivisor(0x8000000000000000ULL, 1);
  EXPECT_EQ(unsignedMax / unsignedDivisor, uint128_t(1));
  EXPECT_EQ(
      unsignedMax % unsignedDivisor,
      uint128_t(0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL));
}

// 128-bit dividends with 64-bit divisors exercise both the single 128/64
// divide (high < divisor) and the two-step divide (high >= divisor).
TEST(Int128Test, divideWideDividendBy64BitDivisor) {
  const auto unsignedMax = std::numeric_limits<uint128_t>::max();
  EXPECT_EQ(
      unsignedMax / uint128_t{3},
      uint128_t(0x5555555555555555ULL, 0x5555555555555555ULL));
  EXPECT_EQ(unsignedMax % uint128_t{3}, uint128_t{0});
  EXPECT_EQ(unsignedMax / uint128_t{~0ULL}, uint128_t(1, 1));
  EXPECT_EQ(unsignedMax % uint128_t{~0ULL}, uint128_t{0});
  EXPECT_EQ(
      uint128_t(0x7FFFFFFFFFFFFFFFULL, 0) / uint128_t{0x8000000000000000ULL},
      uint128_t(0, 0xFFFFFFFFFFFFFFFEULL));

  const auto signedMax = std::numeric_limits<int128_t>::max();
  const auto signedMin = std::numeric_limits<int128_t>::min();
  EXPECT_EQ(
      signedMax / int128_t{10},
      buildSigned(0x0cccccccccccccccULL, 0xccccccccccccccccULL));
  EXPECT_EQ(signedMax % int128_t{10}, int128_t{7});
  EXPECT_EQ(
      signedMin / int128_t{10},
      buildSigned(0xf333333333333333ULL, 0x3333333333333334ULL));
  EXPECT_EQ(signedMin % int128_t{10}, int128_t{-8});
  EXPECT_EQ(
      signedMin / int128_t{-10},
      buildSigned(0x0cccccccccccccccULL, 0xccccccccccccccccULL));
  EXPECT_EQ(signedMin % int128_t{-10}, int128_t{-8});
  EXPECT_EQ(
      signedMax / int128_t{std::numeric_limits<int64_t>::min()},
      buildSigned(0xffffffffffffffffULL, 1));
  EXPECT_EQ(
      signedMax % int128_t{std::numeric_limits<int64_t>::min()},
      int128_t{std::numeric_limits<int64_t>::max()});

  for (int shift = 64; shift < 127; shift += 7) {
    const int128_t dividend = (int128_t{1} << shift) + int128_t{12345};
    for (const int64_t divisor :
         {int64_t{3}, int64_t{1000000007}, int64_t{0x7FFFFFFFFFFFFFFF}}) {
      const int128_t quotient = dividend / int128_t{divisor};
      const int128_t remainder = dividend % int128_t{divisor};
      EXPECT_EQ(quotient * int128_t{divisor} + remainder, dividend);
      EXPECT_GE(remainder, int128_t{0});
      EXPECT_LT(remainder, int128_t{divisor});
    }
  }
}

TEST(Int128Test, divisionByZeroAndCompoundAssignment) {
  EXPECT_THROW(int128_t{1} / int128_t{0}, std::runtime_error);
  EXPECT_THROW(int128_t{1} % int128_t{0}, std::runtime_error);
  EXPECT_THROW(uint128_t{1} / uint128_t{0}, std::runtime_error);
  EXPECT_THROW(uint128_t{1} % uint128_t{0}, std::runtime_error);
  int128_t signedValue{1};
  EXPECT_THROW(signedValue /= int128_t{0}, std::runtime_error);
  EXPECT_THROW(signedValue %= int128_t{0}, std::runtime_error);
  uint128_t unsignedValue{1};
  EXPECT_THROW(unsignedValue /= uint128_t{0}, std::runtime_error);
  EXPECT_THROW(unsignedValue %= uint128_t{0}, std::runtime_error);

  signedValue = int128_t{-17};
  signedValue %= int128_t{5};
  EXPECT_EQ(signedValue, int128_t{-2});
  signedValue = int128_t{-17};
  signedValue /= int128_t{5};
  EXPECT_EQ(signedValue, int128_t{-3});
  // 2^64 + 6 == 8 (mod 7) because 2^64 == 2 (mod 7).
  unsignedValue = uint128_t(1, 6);
  unsignedValue %= uint128_t{7};
  EXPECT_EQ(unsignedValue, uint128_t{1});
}

TEST(Int128Test, carryingArithmeticAndShifts) {
  static_assert(int128_t{2} + int128_t{3} == int128_t{5});
  static_assert(uint128_t(0, ~0ULL) + uint128_t{1} == uint128_t(1, 0));
  static_assert(int128_t{-3} * int128_t{7} == int128_t{-21});

  EXPECT_EQ(uint128_t(0, ~0ULL) + uint128_t{1}, uint128_t(1, 0));
  EXPECT_EQ(uint128_t(1, 0) - uint128_t{1}, uint128_t(0, ~0ULL));
  EXPECT_EQ(uint128_t(0, ~0ULL) * uint128_t(0, ~0ULL), uint128_t(~0ULL - 1, 1));
  EXPECT_EQ(
      std::numeric_limits<int128_t>::max() + int128_t{1},
      std::numeric_limits<int128_t>::min());
  EXPECT_EQ(
      std::numeric_limits<int128_t>::min() - int128_t{1},
      std::numeric_limits<int128_t>::max());
  EXPECT_EQ(int128_t{-3} * int128_t{7}, int128_t{-21});
  EXPECT_EQ(int128_t{-1} * buildSigned(UINT64_MAX, 0), buildSigned(1, 0));

  const int128_t one{1};
  EXPECT_EQ(one << 0, one);
  EXPECT_EQ(one << 63, int128_t(0, 0x8000000000000000ULL));
  EXPECT_EQ(one << 64, int128_t(1, 0));
  EXPECT_EQ(one << 65, int128_t(2, 0));
  EXPECT_EQ(one << 127, std::numeric_limits<int128_t>::min());
  EXPECT_EQ(int128_t{-1} << 64, int128_t(-1, 0));
  EXPECT_EQ(std::numeric_limits<int128_t>::min() >> 127, int128_t{-1});
  EXPECT_EQ(
      std::numeric_limits<int128_t>::min() >> 64,
      int128_t(-1, 0x8000000000000000ULL));
  EXPECT_EQ(int128_t(1, 0) >> 1, int128_t(0, 0x8000000000000000ULL));
  EXPECT_EQ(std::numeric_limits<uint128_t>::max() >> 127, uint128_t{1});
  EXPECT_EQ(std::numeric_limits<uint128_t>::max() >> 64, uint128_t(0, ~0ULL));
  EXPECT_EQ(uint128_t{1} << 127, uint128_t(0x8000000000000000ULL, 0));
  EXPECT_EQ(uint128_t(3, 0) >> 65, uint128_t{1});
}

// Mixed floating-point arithmetic converts the 128-bit operand to the
// floating type, as native __int128 does, instead of narrowing it to an
// integer first.
TEST(Int128Test, floatingPointMixedOperators) {
  static_assert(std::is_same_v<decltype(int128_t{1} + 1.5), double>);
  static_assert(std::is_same_v<decltype(1.5 * uint128_t{1}), double>);
  static_assert(std::is_same_v<decltype(int128_t{1} - 1.5f), float>);

  const int128_t big = (int128_t{1} << 100) + int128_t{3};
  EXPECT_EQ(big + 1.5, std::ldexp(1.0, 100));
  EXPECT_EQ(1.5 + big, std::ldexp(1.0, 100));
  EXPECT_EQ(int128_t{7} - 0.5, 6.5);
  EXPECT_EQ(0.5 - int128_t{7}, -6.5);
  EXPECT_EQ(int128_t{-3} * 2.5, -7.5);
  EXPECT_EQ(1.0 / (int128_t{1} << 64), std::ldexp(1.0, -64));
  EXPECT_EQ((uint128_t{1} << 100) / 4.0, std::ldexp(1.0, 98));
  EXPECT_EQ(2.5f * uint128_t{4}, 10.0f);

  double accumulator = 0.0;
  accumulator += int128_t{1} << 63;
  EXPECT_EQ(accumulator, std::ldexp(1.0, 63));
  accumulator -= uint128_t{1} << 64;
  EXPECT_EQ(accumulator, -std::ldexp(1.0, 63));
  accumulator *= int128_t{-2};
  EXPECT_EQ(accumulator, std::ldexp(1.0, 64));
  accumulator /= uint128_t{1} << 64;
  EXPECT_EQ(accumulator, 1.0);

  EXPECT_TRUE((int128_t{1} << 100) > 1e30);
  EXPECT_TRUE(1e30 < (int128_t{1} << 100));
  EXPECT_TRUE(int128_t{-5} < -4.5);
  EXPECT_TRUE(int128_t{-5} <= -5.0);
  EXPECT_TRUE(0.5 < int128_t{1});
  EXPECT_TRUE(int128_t{3} == 3.0);
  EXPECT_TRUE(int128_t{3} != 3.5);
  EXPECT_TRUE(uint128_t{4} >= 4.0f);
  EXPECT_TRUE(std::numeric_limits<uint128_t>::max() == std::ldexp(1.0, 128));
}

// Out-of-range floating-point conversions saturate like libgcc/compiler-rt
// __fixdfti / __fixunsdfti.
TEST(Int128Test, floatingPointConversionSaturates) {
  const auto signedMax = std::numeric_limits<int128_t>::max();
  const auto signedMin = std::numeric_limits<int128_t>::min();
  const auto unsignedMax = std::numeric_limits<uint128_t>::max();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double two127 = std::ldexp(1.0, 127);
  const double two128 = std::ldexp(1.0, 128);

  EXPECT_EQ(static_cast<int128_t>(inf), signedMax);
  EXPECT_EQ(static_cast<int128_t>(-inf), signedMin);
  EXPECT_EQ(static_cast<int128_t>(std::copysign(nan, 1.0)), signedMax);
  EXPECT_EQ(static_cast<int128_t>(std::copysign(nan, -1.0)), signedMin);
  EXPECT_EQ(static_cast<int128_t>(two127), signedMax);
  EXPECT_EQ(static_cast<int128_t>(-two127), signedMin);
  EXPECT_EQ(static_cast<int128_t>(std::nextafter(-two127, -inf)), signedMin);
  EXPECT_EQ(
      static_cast<int128_t>(std::numeric_limits<double>::max()), signedMax);
  EXPECT_EQ(
      static_cast<int128_t>(std::numeric_limits<double>::lowest()), signedMin);
  EXPECT_EQ(
      static_cast<int128_t>(std::nextafter(two127, 0.0)),
      signedMax - ((int128_t{1} << 74) - int128_t{1}));
  EXPECT_EQ(
      static_cast<int128_t>(std::numeric_limits<float>::max()), signedMax);
  EXPECT_EQ(
      static_cast<int128_t>(-std::numeric_limits<float>::infinity()),
      signedMin);
  EXPECT_EQ(static_cast<int128_t>(0.999), int128_t{0});
  EXPECT_EQ(static_cast<int128_t>(-0.999), int128_t{0});
  EXPECT_EQ(
      static_cast<int128_t>(std::numeric_limits<double>::denorm_min()),
      int128_t{0});

  EXPECT_EQ(static_cast<uint128_t>(inf), unsignedMax);
  EXPECT_EQ(static_cast<uint128_t>(-inf), uint128_t{0});
  EXPECT_EQ(static_cast<uint128_t>(std::copysign(nan, 1.0)), unsignedMax);
  EXPECT_EQ(static_cast<uint128_t>(std::copysign(nan, -1.0)), uint128_t{0});
  EXPECT_EQ(static_cast<uint128_t>(two128), unsignedMax);
  EXPECT_EQ(static_cast<uint128_t>(-1.0), uint128_t{0});
  EXPECT_EQ(
      static_cast<uint128_t>(std::numeric_limits<double>::lowest()),
      uint128_t{0});
  EXPECT_EQ(
      static_cast<uint128_t>(std::nextafter(two128, 0.0)),
      unsignedMax - ((uint128_t{1} << 75) - uint128_t{1}));
  EXPECT_EQ(static_cast<uint128_t>(two127), uint128_t{1} << 127);
  EXPECT_EQ(
      static_cast<uint128_t>(std::numeric_limits<float>::max()),
      uint128_t(0xFFFFFF0000000000ULL, 0));
  EXPECT_EQ(
      static_cast<uint128_t>(std::numeric_limits<float>::infinity()),
      unsignedMax);
}

#endif

TEST(Int128Test, decimalTextOutput) {
  const auto signedMin = std::numeric_limits<int128_t>::min();
  const auto unsignedMax = std::numeric_limits<uint128_t>::max();
  EXPECT_EQ(
      folly::to<std::string>(signedMin),
      "-170141183460469231731687303715884105728");
  EXPECT_EQ(
      folly::to<std::string>(unsignedMax),
      "340282366920938463463374607431768211455");
  EXPECT_EQ(folly::to<std::string>(uint128_t{0}), "0");
  EXPECT_EQ(folly::to<std::string>("x=", int128_t{-42}), "x=-42");
  EXPECT_EQ(
      fmt::format("{}", unsignedMax),
      "340282366920938463463374607431768211455");
  EXPECT_EQ(fmt::format("{}", uint128_t{1} << 64), "18446744073709551616");
  EXPECT_EQ(
      fmt::format("{}", signedMin), "-170141183460469231731687303715884105728");
}

#if defined(_M_X64) && !defined(__SIZEOF_INT128__)
TEST(Int128Test, overflowWithAliasedResult) {
  int128_t value{5};
  EXPECT_FALSE(Int128::addOverflowX64(value, int128_t{7}, &value));
  EXPECT_EQ(value, int128_t{12});
  EXPECT_FALSE(Int128::subOverflowX64(int128_t{2}, value, &value));
  EXPECT_EQ(value, int128_t{-10});
  value = std::numeric_limits<int128_t>::max();
  EXPECT_TRUE(Int128::addOverflowX64(value, int128_t{1}, &value));
  EXPECT_EQ(value, std::numeric_limits<int128_t>::min());
  EXPECT_TRUE(Int128::subOverflowX64(value, int128_t{1}, &value));
  EXPECT_EQ(value, std::numeric_limits<int128_t>::max());
  uint128_t unsignedValue = std::numeric_limits<uint128_t>::max();
  EXPECT_TRUE(
      UInt128::addOverflowX64(unsignedValue, uint128_t{2}, &unsignedValue));
  EXPECT_EQ(unsignedValue, uint128_t{1});
  EXPECT_TRUE(
      UInt128::subOverflowX64(uint128_t{0}, unsignedValue, &unsignedValue));
  EXPECT_EQ(unsignedValue, std::numeric_limits<uint128_t>::max());
}
#endif

} // namespace facebook::velox
