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
// Binary expectations follow IEEE 754 and JLS 5.1.3.
uint64_t doubleBits(double value) {
  return std::bit_cast<uint64_t>(value);
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
      Types::timestampToDouble(Timestamp(0, 1)),
      "histogram_numeric timestamp requires microsecond precision");
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(-1, 999'999'999)),
      "histogram_numeric timestamp requires microsecond precision");
}

TEST_F(HistogramNumericTypesTest, timestampRejectsMicrosecondOverflow) {
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(-9'223'372'036'855, 224'191'000)),
      "Could not convert Timestamp(-9223372036855, 224191000) to microseconds");
  VELOX_ASSERT_USER_THROW(
      Types::timestampToDouble(Timestamp(9'223'372'036'854, 775'808'000)),
      "Could not convert Timestamp(9223372036854, 775808000) to microseconds");
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
  for (const auto precision : {0, 39, -1, 256}) {
    SCOPED_TRACE(precision);
    VELOX_ASSERT_USER_THROW(
        (Decimal(precision, 0)),
        "histogram_numeric Decimal precision must be in [1, 38]");
  }
  for (const auto scale : {-1, 4}) {
    SCOPED_TRACE(scale);
    VELOX_ASSERT_USER_THROW(
        (Decimal(3, scale)),
        "histogram_numeric Decimal scale must be in [0, precision]");
  }
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
  EXPECT_EQ(histogram.bins()[1].x, 12.5);
  EXPECT_EQ(doubleBits(histogram.bins()[0].y), 0x3ff0000000000000ULL);
  EXPECT_EQ(doubleBits(histogram.bins()[1].y), 0x3ff0000000000000ULL);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
