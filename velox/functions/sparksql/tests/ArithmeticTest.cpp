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
#include <cstdint>
#include <limits>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class PmodTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> pmod(std::optional<T> a, std::optional<T> n) {
    return evaluateOnce<T>("pmod(c0, c1)", a, n);
  };
};

TEST_F(PmodTest, int8) {
  EXPECT_EQ(1, pmod<int8_t>(1, 3));
  EXPECT_EQ(2, pmod<int8_t>(-1, 3));
  EXPECT_EQ(1, pmod<int8_t>(3, -2));
  EXPECT_EQ(-1, pmod<int8_t>(-1, -3));
  EXPECT_EQ(std::nullopt, pmod<int8_t>(1, 0));
  EXPECT_EQ(std::nullopt, pmod<int8_t>(std::nullopt, 3));
  EXPECT_EQ(INT8_MAX, pmod<int8_t>(INT8_MAX, INT8_MIN));
  EXPECT_EQ(INT8_MAX - 1, pmod<int8_t>(INT8_MIN, INT8_MAX));
  EXPECT_EQ(0, pmod<int64_t>(INT8_MIN, -1));
}

TEST_F(PmodTest, int16) {
  EXPECT_EQ(0, pmod<int16_t>(23286, 3881));
  EXPECT_EQ(1026, pmod<int16_t>(-15892, 8459));
  EXPECT_EQ(127, pmod<int16_t>(7849, -297));
  EXPECT_EQ(-8, pmod<int16_t>(-1052, -12));
  EXPECT_EQ(INT16_MAX, pmod<int16_t>(INT16_MAX, INT16_MIN));
  EXPECT_EQ(INT16_MAX - 1, pmod<int16_t>(INT16_MIN, INT16_MAX));
  EXPECT_EQ(0, pmod<int64_t>(INT16_MIN, -1));
}

TEST_F(PmodTest, int32) {
  EXPECT_EQ(2095, pmod<int32_t>(391819, 8292));
  EXPECT_EQ(8102, pmod<int32_t>(-16848948, 48163));
  EXPECT_EQ(726, pmod<int32_t>(2145613151, -925));
  EXPECT_EQ(-11, pmod<int32_t>(-15181535, -12));
  EXPECT_EQ(INT32_MAX, pmod<int32_t>(INT32_MAX, INT32_MIN));
  EXPECT_EQ(INT32_MAX - 1, pmod<int32_t>(INT32_MIN, INT32_MAX));
  EXPECT_EQ(0, pmod<int64_t>(INT32_MIN, -1));
}

TEST_F(PmodTest, int64) {
  EXPECT_EQ(0, pmod<int64_t>(4611791058295013614, 2147532562));
  EXPECT_EQ(10807, pmod<int64_t>(-3828032596, 48163));
  EXPECT_EQ(673, pmod<int64_t>(4293096798, -925));
  EXPECT_EQ(-5, pmod<int64_t>(-15181561541535, -23));
  EXPECT_EQ(INT64_MAX, pmod<int64_t>(INT64_MAX, INT64_MIN));
  EXPECT_EQ(INT64_MAX - 1, pmod<int64_t>(INT64_MIN, INT64_MAX));
  EXPECT_EQ(0, pmod<int64_t>(INT64_MIN, -1));
}

class RemainderTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> remainder(std::optional<T> a, std::optional<T> n) {
    return evaluateOnce<T>("remainder(c0, c1)", a, n);
  };
};

TEST_F(RemainderTest, int8) {
  EXPECT_EQ(1, remainder<int8_t>(1, 3));
  EXPECT_EQ(-1, remainder<int8_t>(-1, 3));
  EXPECT_EQ(1, remainder<int8_t>(3, -2));
  EXPECT_EQ(-1, remainder<int8_t>(-1, -3));
  EXPECT_EQ(std::nullopt, remainder<int8_t>(1, 0));
  EXPECT_EQ(std::nullopt, remainder<int8_t>(std::nullopt, 3));
  EXPECT_EQ(INT8_MAX, remainder<int8_t>(INT8_MAX, INT8_MIN));
  EXPECT_EQ(-1, remainder<int8_t>(INT8_MIN, INT8_MAX));
}

TEST_F(RemainderTest, int16) {
  EXPECT_EQ(0, remainder<int16_t>(23286, 3881));
  EXPECT_EQ(-7433, remainder<int16_t>(-15892, 8459));
  EXPECT_EQ(127, remainder<int16_t>(7849, -297));
  EXPECT_EQ(-8, remainder<int16_t>(-1052, -12));
  EXPECT_EQ(INT16_MAX, remainder<int16_t>(INT16_MAX, INT16_MIN));
  EXPECT_EQ(-1, remainder<int16_t>(INT16_MIN, INT16_MAX));
}

TEST_F(RemainderTest, int32) {
  EXPECT_EQ(2095, remainder<int32_t>(391819, 8292));
  EXPECT_EQ(-40061, remainder<int32_t>(-16848948, 48163));
  EXPECT_EQ(726, remainder<int32_t>(2145613151, -925));
  EXPECT_EQ(-11, remainder<int32_t>(-15181535, -12));
  EXPECT_EQ(INT32_MAX, remainder<int32_t>(INT32_MAX, INT32_MIN));
  EXPECT_EQ(-1, remainder<int32_t>(INT32_MIN, INT32_MAX));
  EXPECT_EQ(0, remainder<int32_t>(-15181535, -1));
  EXPECT_EQ(0, remainder<int32_t>(INT32_MIN, -1));
}

TEST_F(RemainderTest, int64) {
  EXPECT_EQ(0, remainder<int64_t>(4611791058295013614, 2147532562));
  EXPECT_EQ(-37356, remainder<int64_t>(-3828032596, 48163));
  EXPECT_EQ(673, remainder<int64_t>(4293096798, -925));
  EXPECT_EQ(-5, remainder<int64_t>(-15181561541535, -23));
  EXPECT_EQ(INT64_MAX, remainder<int64_t>(INT64_MAX, INT64_MIN));
  EXPECT_EQ(-1, remainder<int64_t>(INT64_MIN, INT64_MAX));
}

class ArithmeticTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> unaryminus(std::optional<T> arg) {
    return evaluateOnce<T>("unaryminus(c0)", arg);
  }

  std::optional<double> divide(
      std::optional<double> numerator,
      std::optional<double> denominator) {
    return evaluateOnce<double>("divide(c0, c1)", numerator, denominator);
  }

  static constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  static constexpr float kInf = std::numeric_limits<float>::infinity();
};

TEST_F(ArithmeticTest, UnaryMinus) {
  EXPECT_EQ(unaryminus<int8_t>(1), -1);
  EXPECT_EQ(unaryminus<int16_t>(2), -2);
  EXPECT_EQ(unaryminus<int32_t>(3), -3);
  EXPECT_EQ(unaryminus<int64_t>(4), -4);
  EXPECT_EQ(unaryminus<float>(5), -5);
  EXPECT_EQ(unaryminus<double>(6), -6);
}

TEST_F(ArithmeticTest, UnaryMinusOverflow) {
  EXPECT_EQ(unaryminus<int8_t>(INT8_MIN), INT8_MIN);
  EXPECT_EQ(unaryminus<int16_t>(INT16_MIN), INT16_MIN);
  EXPECT_EQ(unaryminus<int32_t>(INT32_MIN), INT32_MIN);
  EXPECT_EQ(unaryminus<int64_t>(INT64_MIN), INT64_MIN);
  EXPECT_EQ(unaryminus<float>(-kInf), kInf);
  EXPECT_TRUE(std::isnan(unaryminus<float>(kNan).value_or(0)));
  EXPECT_EQ(unaryminus<double>(-kInf), kInf);
  EXPECT_TRUE(std::isnan(unaryminus<double>(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, Divide) {
  // Null cases.
  EXPECT_EQ(divide(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(divide(std::nullopt, 1), std::nullopt);
  EXPECT_EQ(divide(1, std::nullopt), std::nullopt);
  // Division by zero is always null.
  EXPECT_EQ(divide(1, 0), std::nullopt);
  EXPECT_EQ(divide(0, 0), std::nullopt);
  EXPECT_EQ(divide(kNan, 0), std::nullopt);
  EXPECT_EQ(divide(kInf, 0), std::nullopt);
  EXPECT_EQ(divide(-kInf, 0), std::nullopt);
  // Division by NaN is always NaN.
  EXPECT_TRUE(std::isnan(divide(1, kNan).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(0, kNan).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(kNan, kNan).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(kInf, kNan).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(-kInf, kNan).value_or(0)));
  // Division by infinity is zero except when dividing infinity, when it is
  // NaN.
  EXPECT_EQ(divide(std::numeric_limits<double>::max(), kInf), 0);
  EXPECT_EQ(divide(0, kInf), 0);
  EXPECT_EQ(divide(0, -kInf), 0); // Actually -0, but it should compare equal.
  EXPECT_TRUE(std::isnan(divide(kInf, kInf).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(-kInf, kInf).value_or(0)));
  EXPECT_TRUE(std::isnan(divide(kInf, -kInf).value_or(0)));
}

TEST_F(ArithmeticTest, acosh) {
  const auto acosh = [&](std::optional<double> a) {
    return evaluateOnce<double>("acosh(c0)", a);
  };

  EXPECT_EQ(acosh(1), 0);
  EXPECT_TRUE(std::isnan(acosh(0).value_or(0)));
  EXPECT_EQ(acosh(kInf), kInf);
  EXPECT_EQ(acosh(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(acosh(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, asinh) {
  const auto asinh = [&](std::optional<double> a) {
    return evaluateOnce<double>("asinh(c0)", a);
  };

  EXPECT_EQ(asinh(0), 0);
  EXPECT_EQ(asinh(kInf), kInf);
  EXPECT_EQ(asinh(-kInf), -kInf);
  EXPECT_EQ(asinh(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(asinh(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, atanh) {
  const auto atanh = [&](std::optional<double> a) {
    return evaluateOnce<double>("atanh(c0)", a);
  };

  EXPECT_EQ(atanh(0), 0);
  EXPECT_EQ(atanh(1), kInf);
  EXPECT_EQ(atanh(-1), -kInf);
  EXPECT_TRUE(std::isnan(atanh(1.1).value_or(0)));
  EXPECT_TRUE(std::isnan(atanh(-1.1).value_or(0)));
  EXPECT_EQ(atanh(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(atanh(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, sec) {
  const auto sec = [&](std::optional<double> a) {
    return evaluateOnce<double>("sec(c0)", a);
  };

  EXPECT_EQ(sec(0), 1);
  EXPECT_EQ(sec(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(sec(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, csc) {
  const auto csc = [&](std::optional<double> a) {
    return evaluateOnce<double>("csc(c0)", a);
  };

  EXPECT_EQ(csc(0), kInf);
  EXPECT_EQ(csc(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(csc(kNan).value_or(0)));
}

class CeilFloorTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<int64_t> ceil(std::optional<T> a) {
    return evaluateOnce<int64_t, T>("ceil(c0)", a);
  }
  template <typename T>
  std::optional<int64_t> floor(std::optional<T> a) {
    return evaluateOnce<int64_t, T>("floor(c0)", a);
  }
};

TEST_F(CeilFloorTest, Limits) {
  EXPECT_EQ(1, ceil<int64_t>(1));
  EXPECT_EQ(-1, ceil<int64_t>(-1));
  EXPECT_EQ(3, ceil<double>(2.878));
  EXPECT_EQ(2, floor<double>(2.878));
  EXPECT_EQ(1, floor<double>(1.5678));
  EXPECT_EQ(
      std::numeric_limits<int64_t>::max(),
      floor<int64_t>(std::numeric_limits<int64_t>::max()));
  EXPECT_EQ(
      std::numeric_limits<int64_t>::min(),
      floor<int64_t>(std::numeric_limits<int64_t>::min()));

  // Very large double values are truncated to int64_t::max/min.
  EXPECT_EQ(
      std::numeric_limits<int64_t>::max(),
      ceil<double>(std::numeric_limits<double>::infinity()));
  EXPECT_EQ(
      std::numeric_limits<int64_t>::max(),
      floor<double>(std::numeric_limits<double>::infinity()));

  EXPECT_EQ(
      std::numeric_limits<int64_t>::min(),
      ceil<double>(-std::numeric_limits<double>::infinity()));
  EXPECT_EQ(
      std::numeric_limits<int64_t>::min(),
      floor<double>(-std::numeric_limits<double>::infinity()));
}

TEST_F(ArithmeticTest, sinh) {
  const auto sinh = [&](std::optional<double> a) {
    return evaluateOnce<double>("sinh(c0)", a);
  };

  EXPECT_EQ(sinh(0), 0);
  EXPECT_EQ(sinh(kInf), kInf);
  EXPECT_EQ(sinh(-kInf), -kInf);
  EXPECT_EQ(sinh(std::nullopt), std::nullopt);
  EXPECT_TRUE(std::isnan(sinh(kNan).value_or(0)));
}

TEST_F(ArithmeticTest, log1p) {
  const double kE = std::exp(1);

  static const auto log1p = [&](std::optional<double> a) {
    return evaluateOnce<double>("log1p(c0)", a);
  };

  EXPECT_EQ(log1p(0), 0);
  EXPECT_EQ(log1p(kE - 1), 1);
  EXPECT_EQ(log1p(kInf), kInf);
  EXPECT_TRUE(std::isnan(log1p(kNan).value_or(0)));
}

class BinTest : public SparkFunctionBaseTest {
 protected:
  std::optional<std::string> bin(std::optional<std::int64_t> arg) {
    return evaluateOnce<std::string, int64_t>("bin(c0)", {arg}, {BIGINT()});
  }
};

TEST_F(BinTest, bin) {
  EXPECT_EQ(bin(std::nullopt), std::nullopt);
  EXPECT_EQ(bin(13), "1101");
  EXPECT_EQ(
      bin(-13),
      "1111111111111111111111111111111111111111111111111111111111110011");
  EXPECT_EQ(
      bin(std::numeric_limits<int64_t>::max()),
      "111111111111111111111111111111111111111111111111111111111111111");
  EXPECT_EQ(bin(0), "0");
}

TEST_F(ArithmeticTest, hypot) {
  const auto hypot = [&](std::optional<double> a, std::optional<double> b) {
    return evaluateOnce<double>("hypot(c0, c1)", a, b);
  };

  EXPECT_EQ(hypot(3, 4), 5);
  EXPECT_EQ(hypot(-3, -4), 5);
  EXPECT_EQ(hypot(3.0, -4.0), 5.0);
  EXPECT_DOUBLE_EQ(5.70087712549569, hypot(3.5, 4.5).value());
  EXPECT_DOUBLE_EQ(5.70087712549569, hypot(3.5, -4.5).value());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
