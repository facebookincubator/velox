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

#include "velox/connectors/hive/HivePartitionValue.h"

#include <gtest/gtest.h>

#include "velox/type/Filter.h"
#include "velox/type/TimestampConversion.h"

namespace facebook::velox::connector::hive {
namespace {

Variant toVariant(
    std::string_view value,
    const TypePtr& type,
    TimestampMode timestampMode,
    DateMode dateMode) {
  return partitionValueFromString(value, *type, timestampMode, dateMode);
}

Timestamp parseTimestamp(std::string_view value) {
  return util::fromTimestampString(
             value.data(), value.size(), util::TimestampParseMode::kPrestoCast)
      .value();
}

TEST(HivePartitionValueTest, scalarTypes) {
  const auto timestampMode = TimestampMode::kUtc;
  const auto dateMode = DateMode::kDateString;
  EXPECT_EQ(
      toVariant("true", BOOLEAN(), timestampMode, dateMode), Variant(true));
  EXPECT_EQ(
      toVariant("-1", TINYINT(), timestampMode, dateMode), Variant(int8_t{-1}));
  EXPECT_EQ(
      toVariant("-2", SMALLINT(), timestampMode, dateMode),
      Variant(int16_t{-2}));
  EXPECT_EQ(
      toVariant("-3", INTEGER(), timestampMode, dateMode),
      Variant(int32_t{-3}));
  EXPECT_EQ(
      toVariant("-4", BIGINT(), timestampMode, dateMode), Variant(int64_t{-4}));
  EXPECT_EQ(toVariant("1.25", REAL(), timestampMode, dateMode), Variant(1.25f));
  EXPECT_EQ(toVariant("2.5", DOUBLE(), timestampMode, dateMode), Variant(2.5));
  EXPECT_EQ(
      toVariant("hello", VARCHAR(), timestampMode, dateMode),
      Variant(std::string("hello")));
  EXPECT_EQ(
      toVariant("binary", VARBINARY(), timestampMode, dateMode),
      Variant::binary("binary"));
}

TEST(HivePartitionValueTest, decimalTypes) {
  const auto timestampMode = TimestampMode::kUtc;
  const auto dateMode = DateMode::kDateString;
  EXPECT_EQ(
      toVariant("12.34", DECIMAL(10, 2), timestampMode, dateMode)
          .value<TypeKind::BIGINT>(),
      1'234);
  EXPECT_EQ(
      toVariant(
          "12345678901234567890.12", DECIMAL(25, 2), timestampMode, dateMode)
          .value<TypeKind::HUGEINT>(),
      HugeInt::parse("1234567890123456789012"));
}

TEST(HivePartitionValueTest, dateEncodings) {
  EXPECT_EQ(
      toVariant(
          "2020-01-02", DATE(), TimestampMode::kUtc, DateMode::kDateString)
          .value<TypeKind::INTEGER>(),
      18'263);
  EXPECT_EQ(
      toVariant("18263", DATE(), TimestampMode::kUtc, DateMode::kDaysSinceEpoch)
          .value<TypeKind::INTEGER>(),
      18'263);
}

TEST(HivePartitionValueTest, timestampModes) {
  const auto unshifted = parseTimestamp("2020-01-01 12:34:56");
  auto shifted = unshifted;
  shifted.toGMT(Timestamp::defaultTimezone());

  EXPECT_EQ(
      toVariant(
          "2020-01-01 12:34:56",
          TIMESTAMP(),
          TimestampMode::kLocalTime,
          DateMode::kDateString)
          .value<TypeKind::TIMESTAMP>(),
      shifted);
  EXPECT_EQ(
      toVariant(
          "2020-01-01 12:34:56",
          TIMESTAMP(),
          TimestampMode::kUtc,
          DateMode::kDateString)
          .value<TypeKind::TIMESTAMP>(),
      unshifted);
  EXPECT_EQ(
      toVariant(
          "2020-01-01 12:34:56",
          TIMESTAMP_UTC(),
          TimestampMode::kLocalTime,
          DateMode::kDateString)
          .value<TypeKind::TIMESTAMP>(),
      unshifted);
}

TEST(HivePartitionValueTest, filterOnConvertedValue) {
  const auto timestampMode = TimestampMode::kUtc;
  const auto dateMode = DateMode::kDateString;

  const common::BigintRange bigintRange(10, 20, /*nullAllowed=*/false);
  EXPECT_TRUE(partitionValueMatchesFilter(
      "15", *BIGINT(), timestampMode, dateMode, bigintRange));
  EXPECT_FALSE(partitionValueMatchesFilter(
      "25", *BIGINT(), timestampMode, dateMode, bigintRange));

  // A REAL column tests the filter as a float, not as a double.
  const common::FloatRange floatRange(
      1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      2.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  EXPECT_TRUE(partitionValueMatchesFilter(
      "1.25", *REAL(), timestampMode, dateMode, floatRange));
  EXPECT_FALSE(partitionValueMatchesFilter(
      "2.5", *REAL(), timestampMode, dateMode, floatRange));

  EXPECT_TRUE(partitionValueMatchesFilter(
      "2020-01-02",
      *DATE(),
      timestampMode,
      dateMode,
      common::BigintRange(18'263, 18'263, /*nullAllowed=*/false)));
}

TEST(HivePartitionValueTest, filterUsesTimestampMode) {
  const auto unshifted = parseTimestamp("2020-01-01 12:34:56");
  auto shifted = unshifted;
  shifted.toGMT(Timestamp::defaultTimezone());

  EXPECT_TRUE(partitionValueMatchesFilter(
      "2020-01-01 12:34:56",
      *TIMESTAMP(),
      TimestampMode::kLocalTime,
      DateMode::kDateString,
      common::TimestampRange(shifted, shifted, /*nullAllowed=*/false)));
  EXPECT_TRUE(partitionValueMatchesFilter(
      "2020-01-01 12:34:56",
      *TIMESTAMP(),
      TimestampMode::kUtc,
      DateMode::kDateString,
      common::TimestampRange(unshifted, unshifted, /*nullAllowed=*/false)));
}

} // namespace
} // namespace facebook::velox::connector::hive
