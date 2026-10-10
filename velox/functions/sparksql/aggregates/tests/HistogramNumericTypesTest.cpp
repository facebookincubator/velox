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

#include "velox/functions/sparksql/aggregates/HistogramNumericTypes.h"

#include <bit>
#include <cmath>
#include <limits>
#include <string>

#include <folly/String.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/sparksql/aggregates/SparkNumericHistogram.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

using Types = HistogramNumericTypes;
using Decimal = HistogramNumericDecimal;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Source-derived fixtures, NOT results of an executed oracle. See the public
// Spark expression at revision b84dc909a8856388faddc154c6a1d3aba271474e:
// https://github.com/apache/spark/blob/b84dc909a8856388faddc154c6a1d3aba271474e/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HistogramNumeric.scala
// Decimal expectations use Java 21 Double.toString and Scala 2.13.17 decimal.
// Binary expectations follow IEEE 754 and JLS 5.1.3.
uint64_t doubleBits(double value) {
  return std::bit_cast<uint64_t>(value);
}

uint32_t floatBits(float value) {
  return std::bit_cast<uint32_t>(value);
}

class HistogramNumericTypesTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(HistogramNumericTypesTest, primitiveNormalization) {
  EXPECT_EQ(Types::numericToDouble(int8_t{-128}), -128.0);
  EXPECT_EQ(Types::numericToDouble(int16_t{-32'768}), -32'768.0);
  EXPECT_EQ(Types::numericToDouble(INT32_MIN), -0x1p31);
  EXPECT_EQ(Types::numericToDouble(INT32_MAX), 2'147'483'647.0);
  EXPECT_EQ(Types::numericToDouble(INT64_MIN), -0x1p63);
  EXPECT_EQ(Types::numericToDouble(INT64_MAX), 0x1p63);
}

TEST_F(HistogramNumericTypesTest, longBinary64Collisions) {
  EXPECT_EQ(
      doubleBits(Types::numericToDouble(int64_t{9'007'199'254'740'993})),
      0x4340000000000000ULL);
  EXPECT_EQ(
      doubleBits(Types::numericToDouble(int64_t{9'007'199'254'740'995})),
      0x4340000000000002ULL);
  EXPECT_EQ(Types::numericToDouble(int64_t{-9'007'199'254'740'993}), -0x1p53);
}

TEST_F(HistogramNumericTypesTest, realWidening) {
  EXPECT_EQ(doubleBits(Types::numericToDouble(-0.0f)), 0x8000000000000000ULL);
  EXPECT_EQ(doubleBits(Types::numericToDouble(0.1f)), 0x3fb99999a0000000ULL);
  EXPECT_EQ(Types::numericToDouble(std::bit_cast<float>(1U)), 0x1p-149);
  EXPECT_EQ(
      Types::numericToDouble(std::numeric_limits<float>::max()),
      0x1.fffffep127);
  EXPECT_TRUE(
      std::isnan(Types::numericToDouble(std::bit_cast<float>(0x7fc12345U))));
}

TEST_F(HistogramNumericTypesTest, doubleIdentity) {
  for (uint64_t bits :
       {0ULL,
        0x8000000000000000ULL,
        0x7ff0000000000000ULL,
        0xfff0000000000000ULL,
        0x7ff8123456789abcULL,
        1ULL}) {
    EXPECT_EQ(
        doubleBits(Types::numericToDouble(std::bit_cast<double>(bits))), bits);
  }
}

TEST_F(HistogramNumericTypesTest, intTruncatesTowardZero) {
  EXPECT_EQ(Types::javaDoubleToInt(1.999), 1);
  EXPECT_EQ(Types::javaDoubleToInt(-1.999), -1);
  EXPECT_EQ(Types::javaDoubleToInt(-0.999), 0);
  EXPECT_EQ(Types::javaDoubleToInt(-0.0), 0);
}

TEST_F(HistogramNumericTypesTest, intLimitsAndNeighbors) {
  EXPECT_EQ(Types::javaDoubleToInt(0x1p31), INT32_MAX);
  EXPECT_EQ(Types::javaDoubleToInt(std::nextafter(0x1p31, 0.0)), INT32_MAX);
  EXPECT_EQ(
      Types::javaDoubleToInt(std::nextafter(0x1p31, kInfinity)), INT32_MAX);
  EXPECT_EQ(Types::javaDoubleToInt(-0x1p31), INT32_MIN);
  EXPECT_EQ(
      Types::javaDoubleToInt(std::nextafter(-0x1p31, 0.0)), INT32_MIN + 1);
  EXPECT_EQ(
      Types::javaDoubleToInt(std::nextafter(-0x1p31, -kInfinity)), INT32_MIN);
}

TEST_F(HistogramNumericTypesTest, intNonFinite) {
  EXPECT_EQ(Types::javaDoubleToInt(kNaN), 0);
  EXPECT_EQ(Types::javaDoubleToInt(kInfinity), INT32_MAX);
  EXPECT_EQ(Types::javaDoubleToInt(-kInfinity), INT32_MIN);
}

TEST_F(HistogramNumericTypesTest, longTruncatesTowardZero) {
  EXPECT_EQ(Types::javaDoubleToLong(3'000'000'000.75), 3'000'000'000LL);
  EXPECT_EQ(Types::javaDoubleToLong(-3'000'000'000.75), -3'000'000'000LL);
  EXPECT_EQ(Types::javaDoubleToLong(-0.999), 0);
  EXPECT_EQ(Types::javaDoubleToLong(-0.0), 0);
}

TEST_F(HistogramNumericTypesTest, longLimitsAndNeighbors) {
  EXPECT_EQ(Types::javaDoubleToLong(0x1p63), INT64_MAX);
  EXPECT_EQ(
      Types::javaDoubleToLong(std::nextafter(0x1p63, 0.0)), INT64_MAX - 1'023);
  EXPECT_EQ(
      Types::javaDoubleToLong(std::nextafter(0x1p63, kInfinity)), INT64_MAX);
  EXPECT_EQ(Types::javaDoubleToLong(-0x1p63), INT64_MIN);
  EXPECT_EQ(
      Types::javaDoubleToLong(std::nextafter(-0x1p63, 0.0)), INT64_MIN + 1'024);
  EXPECT_EQ(
      Types::javaDoubleToLong(std::nextafter(-0x1p63, -kInfinity)), INT64_MIN);
}

TEST_F(HistogramNumericTypesTest, longNonFinite) {
  EXPECT_EQ(Types::javaDoubleToLong(kNaN), 0);
  EXPECT_EQ(Types::javaDoubleToLong(kInfinity), INT64_MAX);
  EXPECT_EQ(Types::javaDoubleToLong(-kInfinity), INT64_MIN);
}

TEST_F(HistogramNumericTypesTest, byteUsesIntThenLowBits) {
  EXPECT_EQ(Types::javaDoubleToByte(127.99), 127);
  EXPECT_EQ(Types::javaDoubleToByte(128.0), -128);
  EXPECT_EQ(Types::javaDoubleToByte(-129.0), 127);
  EXPECT_EQ(Types::javaDoubleToByte(255.99), -1);
  EXPECT_EQ(Types::javaDoubleToByte(256.0), 0);
  EXPECT_EQ(Types::javaDoubleToByte(kInfinity), -1);
  EXPECT_EQ(Types::javaDoubleToByte(-kInfinity), 0);
  EXPECT_EQ(Types::javaDoubleToByte(kNaN), 0);
}

TEST_F(HistogramNumericTypesTest, shortUsesIntThenLowBits) {
  EXPECT_EQ(Types::javaDoubleToShort(32'767.99), 32'767);
  EXPECT_EQ(Types::javaDoubleToShort(32'768.0), -32'768);
  EXPECT_EQ(Types::javaDoubleToShort(-32'769.0), 32'767);
  EXPECT_EQ(Types::javaDoubleToShort(65'535.99), -1);
  EXPECT_EQ(Types::javaDoubleToShort(65'536.0), 0);
  EXPECT_EQ(Types::javaDoubleToShort(kInfinity), -1);
  EXPECT_EQ(Types::javaDoubleToShort(-kInfinity), 0);
  EXPECT_EQ(Types::javaDoubleToShort(kNaN), 0);
}

TEST_F(HistogramNumericTypesTest, floatTiesToEven) {
  const double evenTie = 1.0 + 0x1p-24;
  const double oddTie = 1.0 + 3 * 0x1p-24;
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(evenTie)), 0x3f800000U);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::nextafter(evenTie, kInfinity))),
      0x3f800001U);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::nextafter(evenTie, 0.0))),
      0x3f800000U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(oddTie)), 0x3f800002U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-oddTie)), 0xbf800002U);
}

TEST_F(HistogramNumericTypesTest, floatOverflow) {
  const double threshold = 0x1.ffffffp127;
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(0x1.fffffep127)), 0x7f7fffffU);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::nextafter(threshold, 0.0))),
      0x7f7fffffU);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(threshold)), 0x7f800000U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-threshold)), 0xff800000U);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::numeric_limits<double>::max())),
      0x7f800000U);
}

TEST_F(HistogramNumericTypesTest, floatNormalSubnormalBoundary) {
  const double threshold = 0x1p-126 - 0x1p-150;
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(0x1p-126)), 0x00800000U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(threshold)), 0x00800000U);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::nextafter(threshold, 0.0))),
      0x007fffffU);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(0x1p-149)), 1U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(3 * 0x1p-150)), 2U);
}

TEST_F(HistogramNumericTypesTest, floatUnderflow) {
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(0x1p-150)), 0U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-0x1p-150)), 0x80000000U);
  EXPECT_EQ(
      floatBits(Types::javaDoubleToFloat(std::nextafter(0x1p-150, kInfinity))),
      1U);
  EXPECT_EQ(
      floatBits(
          Types::javaDoubleToFloat(std::numeric_limits<double>::denorm_min())),
      0U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-0x1p-151)), 0x80000000U);
}

TEST_F(HistogramNumericTypesTest, floatZerosAndInfinities) {
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(0.0)), 0U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-0.0)), 0x80000000U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(kInfinity)), 0x7f800000U);
  EXPECT_EQ(floatBits(Types::javaDoubleToFloat(-kInfinity)), 0xff800000U);
}

TEST_F(HistogramNumericTypesTest, floatRoundTripExponentBoundaries) {
  // Each binary32 input is exactly representable in binary64. Expected bits
  // come from the input pattern, not from another narrowing implementation.
  for (uint32_t exponent = 0; exponent < 255; ++exponent) {
    for (uint32_t fraction : {0U, 1U, 0x003fffffU, 0x007ffffeU, 0x007fffffU}) {
      for (uint32_t sign : {0U, 0x80000000U}) {
        const uint32_t bits = sign | (exponent << 23) | fraction;
        const auto widened = Types::numericToDouble(std::bit_cast<float>(bits));
        EXPECT_EQ(floatBits(Types::javaDoubleToFloat(widened)), bits);
      }
    }
  }
}

TEST_F(HistogramNumericTypesTest, floatNaNs) {
  // Only classification is a portable Java contract. The native mapping keeps
  // sign/high payload bits and quiets NaNs; JVM payload parity needs the
  // oracle's recorded JDK/vendor/CPU profile, not a universal canonical-NaN
  // assertion.
  for (uint64_t bits :
       {0x7ff0000000000001ULL, 0x7ff8123456789abcULL, 0xfff8123456789abcULL}) {
    EXPECT_TRUE(
        std::isnan(Types::javaDoubleToFloat(std::bit_cast<double>(bits))));
  }
  EXPECT_EQ(
      floatBits(
          Types::javaDoubleToFloat(
              std::bit_cast<double>(0xfff8123456789abcULL))),
      0xffc091a2U);
}

TEST_F(HistogramNumericTypesTest, dateDaysAndYearMonthMonths) {
  for (int32_t value : {INT32_MIN, -12, -1, 0, 1, 12, INT32_MAX}) {
    const auto center = Types::numericToDouble(value);
    EXPECT_EQ(Types::javaDoubleToInt(center), value);
  }
  EXPECT_EQ(Types::javaDoubleToInt(-12.75), -12);
}

TEST_F(HistogramNumericTypesTest, timestampMicroseconds) {
  for (int64_t micros : {0, 1, 999, 1'001, 1'000'001}) {
    auto value = Types::javaDoubleToTimestamp(static_cast<double>(micros));
    EXPECT_EQ(value.toMicros(), micros);
    EXPECT_EQ(Types::timestampToDouble(value), static_cast<double>(micros));
    EXPECT_EQ(value.getNanos() % 1'000, 0);
  }
}

TEST_F(HistogramNumericTypesTest, timestampNegativeEpochs) {
  // TIMESTAMP and TIMESTAMP_UTC use this same no-time-zone conversion.
  EXPECT_EQ(Types::javaDoubleToTimestamp(-1.999), Timestamp(-1, 999'999'000));
  EXPECT_EQ(Types::javaDoubleToTimestamp(-999), Timestamp(-1, 999'001'000));
  EXPECT_EQ(Types::javaDoubleToTimestamp(-1'001), Timestamp(-1, 998'999'000));
  EXPECT_EQ(Types::javaDoubleToTimestamp(-1'000'000), Timestamp(-1, 0));
  EXPECT_EQ(Types::timestampToDouble(Timestamp(-2, 999'999'000)), -1'000'001.0);
}

TEST_F(HistogramNumericTypesTest, timestampInputLimits) {
  EXPECT_EQ(
      Types::timestampToDouble(Timestamp(-9'223'372'036'855, 224'192'000)),
      -0x1p63);
  EXPECT_EQ(
      Types::timestampToDouble(Timestamp(9'223'372'036'854, 775'807'000)),
      0x1p63);
}

TEST_F(HistogramNumericTypesTest, timestampRejectsSubMicrosecond) {
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(0, 1)), "microsecond precision");
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(-1, 999'999'999)),
      "microsecond precision");
}

TEST_F(HistogramNumericTypesTest, timestampRejectsMicrosecondOverflow) {
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(-9'223'372'036'855, 224'191'000)),
      "microseconds");
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(9'223'372'036'854, 775'808'000)),
      "microseconds");
}

TEST_F(HistogramNumericTypesTest, timestampOutputLimitsAndNonFinite) {
  EXPECT_EQ(
      Types::javaDoubleToTimestamp(-0x1p63),
      Timestamp(-9'223'372'036'855, 224'192'000));
  EXPECT_EQ(
      Types::javaDoubleToTimestamp(0x1p63),
      Timestamp(9'223'372'036'854, 775'807'000));
  EXPECT_EQ(Types::javaDoubleToTimestamp(-kInfinity).toMicros(), INT64_MIN);
  EXPECT_EQ(Types::javaDoubleToTimestamp(kInfinity).toMicros(), INT64_MAX);
  EXPECT_EQ(Types::javaDoubleToTimestamp(kNaN), Timestamp(0, 0));
}

TEST_F(HistogramNumericTypesTest, dayTimeIntervalMicrosCarrier) {
  // These are original Catalyst microseconds in BIGINT, never values loaded
  // from Velox's millisecond INTERVAL_DAY_TIME storage.
  for (int64_t micros : {-1'001, -999, -1, 0, 1, 999, 1'001}) {
    EXPECT_EQ(Types::javaDoubleToLong(Types::numericToDouble(micros)), micros);
  }
  EXPECT_EQ(
      Types::javaDoubleToLong(Types::numericToDouble(INT64_MIN)), INT64_MIN);
  EXPECT_EQ(
      Types::javaDoubleToLong(Types::numericToDouble(INT64_MAX)), INT64_MAX);
}

TEST_F(HistogramNumericTypesTest, decimalMetadataValidation) {
  VELOX_ASSERT_USER_THROW((Decimal(0, 0)), "precision");
  VELOX_ASSERT_USER_THROW((Decimal(39, 0)), "precision");
  VELOX_ASSERT_USER_THROW((Decimal(-1, 0)), "precision");
  VELOX_ASSERT_USER_THROW((Decimal(256, 0)), "precision");
  VELOX_ASSERT_USER_THROW((Decimal(3, -1)), "scale");
  VELOX_ASSERT_USER_THROW((Decimal(3, 4)), "scale");
}

TEST_F(HistogramNumericTypesTest, shortDecimalNormalization) {
  EXPECT_EQ(Decimal(18, 0).decimalToDouble(999'999'999'999'999'999LL), 1e18);
  EXPECT_EQ(Decimal(18, 18).decimalToDouble(999'999'999'999'999'999LL), 1.0);
  EXPECT_EQ(Decimal(18, 18).decimalToDouble(-1), -1e-18);
  EXPECT_EQ(
      doubleBits(Decimal(3, 2).decimalToDouble(10)), 0x3fb999999999999aULL);
  EXPECT_EQ(doubleBits(Decimal(1, 1).decimalToDouble(0)), 0ULL);
}

TEST_F(HistogramNumericTypesTest, longDecimalNormalization) {
  const auto p19 = HugeInt::parse("9999999999999999999");
  const auto p38 = HugeInt::parse("99999999999999999999999999999999999999");
  EXPECT_EQ(Decimal(19, 0).decimalToDouble(p19), 1e19);
  EXPECT_EQ(Decimal(19, 19).decimalToDouble(p19), 1.0);
  EXPECT_EQ(Decimal(38, 0).decimalToDouble(p38), 1e38);
  EXPECT_EQ(Decimal(38, 38).decimalToDouble(p38), 1.0);
  EXPECT_EQ(Decimal(38, 38).decimalToDouble(-p38), -1.0);
  EXPECT_EQ(Decimal(38, 38).decimalToDouble(-1), -1e-38);
}

TEST_F(HistogramNumericTypesTest, decimalNormalizationAvoidsDoubleRounding) {
  // The exact decimal lies just above the midpoint 1 + 2^-53. Dividing
  // separately rounded binary64 operands need not preserve that distinction.
  const auto unscaled = HugeInt::parse("100000000000000011103");
  EXPECT_EQ(
      doubleBits(Decimal(21, 20).decimalToDouble(unscaled)),
      0x3ff0000000000001ULL);
  EXPECT_EQ(
      doubleBits(Decimal(21, 20).decimalToDouble(-unscaled)),
      0xbff0000000000001ULL);
  EXPECT_EQ(
      doubleBits(Decimal(21, 20).decimalToDouble(unscaled - 1)),
      0x3ff0000000000000ULL);
}

TEST_F(HistogramNumericTypesTest, decimalNormalizationFastPathBoundaries) {
  constexpr int64_t kExactLimit = int64_t{1} << 53;
  EXPECT_EQ(
      Decimal(16, 0).decimalToDouble(kExactLimit),
      static_cast<double>(kExactLimit));
  EXPECT_EQ(
      Decimal(16, 0).decimalToDouble(-kExactLimit),
      static_cast<double>(-kExactLimit));
  EXPECT_EQ(
      doubleBits(Decimal(23, 22).decimalToDouble(kExactLimit)),
      doubleBits(9.007199254740992e-7));
  EXPECT_EQ(
      doubleBits(Decimal(24, 23).decimalToDouble(kExactLimit)),
      doubleBits(9.007199254740992e-8));
  EXPECT_EQ(
      doubleBits(Decimal(23, 22).decimalToDouble(kExactLimit + 1)),
      doubleBits(9.007199254740993e-7));
}

TEST_F(HistogramNumericTypesTest, decimalSignificantHalfUp) {
  EXPECT_EQ(Decimal(1, 0).reconstructDecimal(2.5), "3");
  EXPECT_EQ(Decimal(1, 0).reconstructDecimal(-2.5), "-3");
  EXPECT_EQ(Decimal(2, 1).reconstructDecimal(1.25), "1.3");
  EXPECT_EQ(Decimal(2, 1).reconstructDecimal(-1.25), "-1.3");
  EXPECT_EQ(Decimal(1, 0).reconstructDecimal(9.5), "10");
}

TEST_F(HistogramNumericTypesTest, decimalScaleHalfUp) {
  EXPECT_EQ(Decimal(38, 1).reconstructDecimal(1.25), "1.3");
  EXPECT_EQ(Decimal(38, 1).reconstructDecimal(-1.25), "-1.3");
  EXPECT_EQ(Decimal(38, 0).reconstructDecimal(0.5), "1");
  EXPECT_EQ(Decimal(38, 0).reconstructDecimal(-0.5), "-1");
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(5e-39),
      "0." + std::string(37, '0') + "1");
}

TEST_F(HistogramNumericTypesTest, decimalKeepsBothRounds) {
  EXPECT_EQ(Decimal(3, 1).reconstructDecimal(9.949), "10.0");
  EXPECT_EQ(Decimal(3, 1).reconstructDecimal(-9.949), "-10.0");
  EXPECT_EQ(Decimal(38, 1).reconstructDecimal(9.949), "9.9");
  EXPECT_EQ(Decimal(3, 1).tryReconstructNativeDecimal(9.949), 100);
}

TEST_F(HistogramNumericTypesTest, decimalPrecisionsAndFixedScale) {
  EXPECT_EQ(Decimal(1, 1).reconstructDecimal(0.0), "0.0");
  EXPECT_EQ(
      Decimal(18, 18).reconstructDecimal(-0.0), "0." + std::string(18, '0'));
  EXPECT_EQ(Decimal(19, 19).reconstructDecimal(0.1), "0.1000000000000000000");
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(-1e-39), "0." + std::string(38, '0'));
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(0.1), "0.1" + std::string(37, '0'));
}

TEST_F(HistogramNumericTypesTest, decimalJava21Selection) {
  // Java 17 can select 9.999999999999999E22 here; Java 21 selects 1.0E23.
  EXPECT_EQ(
      Decimal(38, 0).reconstructDecimal(1e23), "100000000000000000000000");
  EXPECT_EQ(
      Decimal(38, 0).reconstructDecimal(-1e23), "-100000000000000000000000");
  EXPECT_EQ(
      Decimal(38, 20).reconstructDecimal(1.2345678901234567),
      "1.23456789012345670000");
  EXPECT_EQ(Decimal(38, 20).reconstructDecimal(0.1), "0.10000000000000000000");
}

TEST_F(HistogramNumericTypesTest, decimalSelectionIntervalEndpoints) {
  // 1e23 is the upper midpoint of the even binary64 immediately below it.
  // The next (odd) binary64 has an OPEN lower endpoint: 1e23 cannot represent
  // that neighbor. Its shortest candidate is 10000000000000001 * 10^7.
  const Decimal decimal(38, 0);
  EXPECT_EQ(decimal.reconstructDecimal(1e23), "100000000000000000000000");
  EXPECT_EQ(
      decimal.reconstructDecimal(std::nextafter(1e23, kInfinity)),
      "100000000000000010000000");
  EXPECT_EQ(
      decimal.reconstructDecimal(std::nextafter(1e23, 0.0)),
      "99999999999999970000000");
}

TEST_F(HistogramNumericTypesTest, decimalSelectionNearestEvenSignificand) {
  // At 2^49 the binary64 spacing is 1/8. Both adjacent decimal tenths are
  // within the rounding interval around .25 and .75; each pair is equidistant.
  // Choose the even decimal significand BEFORE either HALF_UP operation.
  const Decimal decimal(38, 2);
  EXPECT_EQ(decimal.reconstructDecimal(0x1p49 + 0.25), "562949953421312.20");
  EXPECT_EQ(decimal.reconstructDecimal(0x1p49 + 0.75), "562949953421312.80");
  EXPECT_EQ(decimal.reconstructDecimal(-0x1p49 - 0.25), "-562949953421312.20");
}

TEST_F(HistogramNumericTypesTest, decimalExponentExtremes) {
  const double largest = std::numeric_limits<double>::max();
  EXPECT_EQ(
      Decimal(38, 0).reconstructDecimal(largest),
      "17976931348623157" + std::string(292, '0'));
  EXPECT_EQ(
      Decimal(1, 0).reconstructDecimal(largest), "2" + std::string(308, '0'));
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(-largest),
      "-17976931348623157" + std::string(292, '0') + "." +
          std::string(38, '0'));
  EXPECT_EQ(Decimal(38, 38).reconstructDecimal(-largest).size(), 349);
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(std::numeric_limits<double>::min()),
      "0." + std::string(38, '0'));
  EXPECT_EQ(
      Decimal(38, 38).reconstructDecimal(
          std::numeric_limits<double>::denorm_min()),
      "0." + std::string(38, '0'));
}

TEST_F(HistogramNumericTypesTest, decimalNativeAndCarrierAgreement) {
  EXPECT_EQ(Decimal(18, 2).tryReconstructNativeDecimal(12.345), 1'235);
  EXPECT_EQ(Decimal(18, 2).reconstructDecimal(12.345), "12.35");
  EXPECT_EQ(Decimal(19, 2).tryReconstructNativeDecimal(-12.345), -1'235);
  EXPECT_EQ(Decimal(19, 2).reconstructDecimal(-12.345), "-12.35");
  EXPECT_EQ(
      Decimal(38, 0).tryReconstructNativeDecimal(1e37),
      HugeInt::parse("10000000000000000000000000000000000000"));
}

TEST_F(HistogramNumericTypesTest, decimalNativeRepresentationBoundary) {
  const auto unscaled =
      HugeInt::parse("99999999999999999999999999999999999999");
  const Decimal decimal(38, 0);
  const double center = decimal.decimalToDouble(unscaled);
  EXPECT_EQ(
      decimal.reconstructDecimal(center),
      "100000000000000000000000000000000000000");
  EXPECT_EQ(
      decimal.reconstructDecimal(-center),
      "-100000000000000000000000000000000000000");
  EXPECT_EQ(decimal.tryReconstructNativeDecimal(center), std::nullopt);
  EXPECT_EQ(decimal.tryReconstructNativeDecimal(-center), std::nullopt);
  EXPECT_EQ(Decimal(1, 0).reconstructDecimal(-9.5), "-10");
  EXPECT_EQ(Decimal(1, 0).tryReconstructNativeDecimal(-9.5), std::nullopt);
}

TEST_F(HistogramNumericTypesTest, decimalNonFiniteErrors) {
  const Decimal decimal(38, 18);
  for (double value : {kNaN, kInfinity, -kInfinity}) {
    VELOX_ASSERT_USER_THROW(decimal.reconstructDecimal(value), "non-finite");
    VELOX_ASSERT_USER_THROW(
        decimal.tryReconstructNativeDecimal(value), "non-finite");
  }
}

TEST_F(HistogramNumericTypesTest, longCenterSurvivesKernelAndWire) {
  HashStringAllocator allocator(pool_.get());
  SparkNumericHistogram histogram(&allocator);
  histogram.initialize(3);
  histogram.add(Types::numericToDouble(int64_t{9'007'199'254'740'993}));
  std::string bytes(histogram.serializedSize(), '\0');
  histogram.serialize(bytes.data());
  EXPECT_EQ(
      folly::hexlify(bytes),
      "000000030000000143400000000000003ff0000000000000");
  SparkNumericHistogram restored(&allocator);
  restored.mergeSerialized(bytes);
  ASSERT_EQ(restored.bins().size(), 1);
  EXPECT_EQ(
      Types::javaDoubleToLong(restored.bins()[0].x), 9'007'199'254'740'992LL);
  EXPECT_EQ(doubleBits(restored.bins()[0].y), 0x3ff0000000000000ULL);
}

TEST_F(HistogramNumericTypesTest, decimalAndTimestampKernelCenters) {
  HashStringAllocator allocator(pool_.get());
  SparkNumericHistogram histogram(&allocator);
  histogram.initialize(3);
  histogram.add(Decimal(3, 1).decimalToDouble(125));
  histogram.add(Types::timestampToDouble(Timestamp(-1, 998'999'000)));
  ASSERT_EQ(histogram.bins().size(), 2);
  EXPECT_EQ(
      Types::javaDoubleToTimestamp(histogram.bins()[0].x).toMicros(), -1'001);
  EXPECT_EQ(
      Decimal(3, 1).tryReconstructNativeDecimal(histogram.bins()[1].x), 125);
  EXPECT_EQ(Decimal(3, 1).reconstructDecimal(histogram.bins()[1].x), "12.5");
  EXPECT_EQ(doubleBits(histogram.bins()[0].y), 0x3ff0000000000000ULL);
  EXPECT_EQ(doubleBits(histogram.bins()[1].y), 0x3ff0000000000000ULL);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
