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

#include "velox/connectors/hive/PartitionValue.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Filter.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::connector::hive {
namespace {

using TimestampMode = PartitionValue::TimestampMode;
using DateMode = PartitionValue::DateMode;

Variant toVariant(
    std::string_view value,
    const TypePtr& type,
    TimestampMode timestampMode = TimestampMode::kUtc,
    DateMode dateMode = DateMode::kIsoString) {
  return PartitionValue::fromString(value, *type, timestampMode, dateMode);
}

int64_t packedTimestampWithTimeZone(std::string_view value) {
  return toVariant(value, TIMESTAMP_WITH_TIME_ZONE()).value<TypeKind::BIGINT>();
}

Timestamp parseTimestamp(std::string_view value) {
  return util::fromTimestampString(
             value.data(), value.size(), util::TimestampParseMode::kPrestoCast)
      .value();
}

TEST(PartitionValueTest, scalarTypes) {
  EXPECT_EQ(toVariant("true", BOOLEAN()), Variant(true));
  EXPECT_EQ(toVariant("-1", TINYINT()), Variant(int8_t{-1}));
  EXPECT_EQ(toVariant("-2", SMALLINT()), Variant(int16_t{-2}));
  EXPECT_EQ(toVariant("-3", INTEGER()), Variant(int32_t{-3}));
  EXPECT_EQ(toVariant("-4", BIGINT()), Variant(int64_t{-4}));
  EXPECT_EQ(toVariant("1.25", REAL()), Variant(1.25f));
  EXPECT_EQ(toVariant("2.5", DOUBLE()), Variant(2.5));
  EXPECT_EQ(toVariant("hello", VARCHAR()), Variant(std::string("hello")));
  EXPECT_EQ(toVariant("binary", VARBINARY()), Variant::binary("binary"));
}

TEST(PartitionValueTest, booleanAcceptedStrings) {
  EXPECT_EQ(toVariant("t", BOOLEAN()), Variant(true));
  EXPECT_EQ(toVariant("0", BOOLEAN()), Variant(false));
  EXPECT_EQ(toVariant("FALSE", BOOLEAN()), Variant(false));

  VELOX_ASSERT_USER_THROW(
      toVariant("yes", BOOLEAN()), "Cannot cast yes to BOOLEAN");
  VELOX_ASSERT_USER_THROW(
      toVariant("off", BOOLEAN()), "Cannot cast off to BOOLEAN");
}

// Each integer type is range-checked against its own native type.
TEST(PartitionValueTest, outOfRangeNarrowInteger) {
  VELOX_ASSERT_USER_THROW(
      toVariant("99999999999", INTEGER()), "Overflow during conversion");
  VELOX_ASSERT_USER_THROW(
      toVariant("40000", SMALLINT()), "Overflow during conversion");
  VELOX_ASSERT_USER_THROW(
      toVariant("300", TINYINT()), "Overflow during conversion");
}

TEST(PartitionValueTest, decimalTypes) {
  EXPECT_EQ(
      toVariant("12.34", DECIMAL(10, 2)).value<TypeKind::BIGINT>(), 1'234);
  EXPECT_EQ(
      toVariant("12345678901234567890.12", DECIMAL(25, 2))
          .value<TypeKind::HUGEINT>(),
      HugeInt::parse("1234567890123456789012"));
}

TEST(PartitionValueTest, dateEncodings) {
  EXPECT_EQ(toVariant("2020-01-02", DATE()).value<TypeKind::INTEGER>(), 18'263);
  EXPECT_EQ(
      toVariant("18263", DATE(), TimestampMode::kUtc, DateMode::kDaysSinceEpoch)
          .value<TypeKind::INTEGER>(),
      18'263);
}

TEST(PartitionValueTest, timestampModes) {
  const auto unshifted = parseTimestamp("2020-01-01 12:34:56");
  auto shifted = unshifted;
  shifted.toGMT(Timestamp::defaultTimezone());

  EXPECT_EQ(
      toVariant("2020-01-01 12:34:56", TIMESTAMP(), TimestampMode::kLocalTime)
          .value<TypeKind::TIMESTAMP>(),
      shifted);
  EXPECT_EQ(
      toVariant("2020-01-01 12:34:56", TIMESTAMP(), TimestampMode::kUtc)
          .value<TypeKind::TIMESTAMP>(),
      unshifted);
  EXPECT_EQ(
      toVariant(
          "2020-01-01 12:34:56", TIMESTAMP_UTC(), TimestampMode::kLocalTime)
          .value<TypeKind::TIMESTAMP>(),
      unshifted);
}

TEST(PartitionValueTest, filterOnConvertedValue) {
  const common::BigintRange bigintRange(10, 20, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(bigintRange, toVariant("15", BIGINT())));
  EXPECT_FALSE(applyFilter(bigintRange, toVariant("25", BIGINT())));

  // A REAL value is tested as a float, not as a double.
  const common::FloatRange floatRange(
      1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      2.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(floatRange, toVariant("1.25", REAL())));
  EXPECT_FALSE(applyFilter(floatRange, toVariant("2.5", REAL())));

  const common::BigintRange dateRange(18'263, 18'263, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(dateRange, toVariant("2020-01-02", DATE())));

  const common::BoolValue boolFilter(true, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(boolFilter, toVariant("true", BOOLEAN())));
  EXPECT_FALSE(applyFilter(boolFilter, toVariant("false", BOOLEAN())));
}

TEST(PartitionValueTest, filterOnDecimal) {
  const common::BigintRange shortRange(1'234, 1'234, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(shortRange, toVariant("12.34", DECIMAL(10, 2))));
  EXPECT_FALSE(applyFilter(shortRange, toVariant("12.35", DECIMAL(10, 2))));

  const auto longValue = HugeInt::parse("1234567890123456789012");
  const common::HugeintRange longRange(
      longValue, longValue, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(
      longRange, toVariant("12345678901234567890.12", DECIMAL(25, 2))));
}

// TIMESTAMP WITH TIME ZONE has kind BIGINT, so the dispatch would otherwise
// route it to the integer parse. Verify that fromString recognizes the type
// and packs the value.
TEST(PartitionValueTest, timestampWithTimeZoneNamedZone) {
  const auto utc = packedTimestampWithTimeZone("2021-01-01 00:00:00 UTC");
  EXPECT_EQ(unpackMillisUtc(utc), 1'609'459'200'000);
  EXPECT_EQ(unpackZoneKeyId(utc), tz::getTimeZoneID("UTC"));

  const auto newYork =
      packedTimestampWithTimeZone("2021-01-01 00:00:00 America/New_York");
  // 2021-01-01 00:00:00 in New York is 05:00:00 UTC.
  EXPECT_EQ(unpackMillisUtc(newYork), 1'609'459'200'000 + 5 * 3'600'000);
  EXPECT_EQ(unpackZoneKeyId(newYork), tz::getTimeZoneID("America/New_York"));
}

TEST(PartitionValueTest, timestampWithTimeZoneExplicitOffset) {
  const auto packed =
      packedTimestampWithTimeZone("2021-01-01 00:00:00 +03:00");
  EXPECT_EQ(unpackMillisUtc(packed), 1'609'459'200'000 - 3 * 3'600'000);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("+03:00"));
}

// A value with no zone is already UTC and is not shifted.
TEST(PartitionValueTest, timestampWithTimeZoneNoZoneIsUtc) {
  const auto packed = packedTimestampWithTimeZone("2021-01-01 00:00:00");
  EXPECT_EQ(unpackMillisUtc(packed), 1'609'459'200'000);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("UTC"));
}

// The packed layout holds milliseconds, so anything finer is truncated.
TEST(PartitionValueTest, timestampWithTimeZoneTruncatesSubMillis) {
  EXPECT_EQ(
      unpackMillisUtc(
          packedTimestampWithTimeZone("2021-01-01 00:00:00.123456")),
      1'609'459'200'123);
}

// A value that is not a timestamp string is parsed as an already packed
// integer, which is how a table can store pre-packed partition values.
TEST(PartitionValueTest, timestampWithTimeZoneAlreadyPacked) {
  const auto expected =
      pack(1'609'459'200'000, tz::getTimeZoneID("America/New_York"));
  EXPECT_EQ(packedTimestampWithTimeZone(fmt::format("{}", expected)), expected);
}

TEST(PartitionValueTest, timestampWithTimeZoneUnparseableValue) {
  VELOX_ASSERT_USER_THROW(
      packedTimestampWithTimeZone("not a timestamp"),
      "Cannot convert value to TIMESTAMP WITH TIME ZONE: not a timestamp");
}

// An offset with no corresponding zone key has nothing to pack with, so it is
// rejected rather than recorded under a different zone.
TEST(PartitionValueTest, timestampWithTimeZoneUnknownOffset) {
  VELOX_ASSERT_USER_THROW(
      packedTimestampWithTimeZone("2021-01-01 00:00:00 +19:00"),
      "Unknown timezone in TIMESTAMP WITH TIME ZONE value");
}

TEST(PartitionValueTest, filterUsesTimestampMode) {
  const auto unshifted = parseTimestamp("2020-01-01 12:34:56");
  auto shifted = unshifted;
  shifted.toGMT(Timestamp::defaultTimezone());

  const common::TimestampRange shiftedFilter(
      shifted, shifted, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(
      shiftedFilter,
      toVariant(
          "2020-01-01 12:34:56", TIMESTAMP(), TimestampMode::kLocalTime)));

  const common::TimestampRange unshiftedFilter(
      unshifted, unshifted, /*nullAllowed=*/false);
  EXPECT_TRUE(applyFilter(
      unshiftedFilter,
      toVariant("2020-01-01 12:34:56", TIMESTAMP(), TimestampMode::kUtc)));
}

} // namespace
} // namespace facebook::velox::connector::hive
