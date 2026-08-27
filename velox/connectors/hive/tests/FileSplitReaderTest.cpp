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

#include "velox/connectors/hive/FileSplitReader.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::connector::hive {
namespace {

class FileSplitReaderTest : public testing::Test,
                            public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Returns the single packed value of the TIMESTAMP WITH TIME ZONE constant
  // built from 'value'.
  int64_t packedConstant(std::string_view value) {
    auto constant = newConstantFromString(
        TIMESTAMP_WITH_TIME_ZONE(),
        std::string(value),
        pool(),
        /*isLocalTimestamp=*/false,
        /*isDaysSinceEpoch=*/false);
    VELOX_CHECK(isTimestampWithTimeZoneType(constant->type()));
    VELOX_CHECK_EQ(constant->size(), 1);
    VELOX_CHECK(!constant->isNullAt(0));
    return constant->as<SimpleVector<int64_t>>()->valueAt(0);
  }
};

// newConstantFromString() handles TIMESTAMP WITH TIME ZONE itself rather than
// delegating to PartitionValue::fromString(), whose TypeKind dispatch would
// select the BIGINT parse.
TEST_F(FileSplitReaderTest, timestampWithTimeZoneNamedZone) {
  const auto utc = packedConstant("2021-01-01 00:00:00 UTC");
  EXPECT_EQ(unpackMillisUtc(utc), 1'609'459'200'000);
  EXPECT_EQ(unpackZoneKeyId(utc), tz::getTimeZoneID("UTC"));

  const auto newYork = packedConstant("2021-01-01 00:00:00 America/New_York");
  // 2021-01-01 00:00:00 in New York is 05:00:00 UTC.
  EXPECT_EQ(unpackMillisUtc(newYork), 1'609'459'200'000 + 5 * 3'600'000);
  EXPECT_EQ(unpackZoneKeyId(newYork), tz::getTimeZoneID("America/New_York"));
}

TEST_F(FileSplitReaderTest, timestampWithTimeZoneExplicitOffset) {
  const auto packed = packedConstant("2021-01-01 00:00:00 +03:00");
  EXPECT_EQ(unpackMillisUtc(packed), 1'609'459'200'000 - 3 * 3'600'000);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("+03:00"));
}

// A value with no zone is already UTC and is not shifted.
TEST_F(FileSplitReaderTest, timestampWithTimeZoneNoZoneIsUtc) {
  const auto packed = packedConstant("2021-01-01 00:00:00");
  EXPECT_EQ(unpackMillisUtc(packed), 1'609'459'200'000);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("UTC"));
}

// The packed layout holds milliseconds, so anything finer is truncated.
TEST_F(FileSplitReaderTest, timestampWithTimeZoneTruncatesSubMillis) {
  EXPECT_EQ(
      unpackMillisUtc(packedConstant("2021-01-01 00:00:00.123456")),
      1'609'459'200'123);
}

// A value that is not a timestamp string is parsed as an already packed
// integer, which is how a table can store pre-packed partition values.
TEST_F(FileSplitReaderTest, timestampWithTimeZoneAlreadyPacked) {
  const auto expected =
      pack(1'609'459'200'000, tz::getTimeZoneID("America/New_York"));
  EXPECT_EQ(packedConstant(fmt::format("{}", expected)), expected);
}

TEST_F(FileSplitReaderTest, timestampWithTimeZoneNullValue) {
  auto constant = newConstantFromString(
      TIMESTAMP_WITH_TIME_ZONE(),
      std::nullopt,
      pool(),
      /*isLocalTimestamp=*/false,
      /*isDaysSinceEpoch=*/false);
  ASSERT_TRUE(isTimestampWithTimeZoneType(constant->type()));
  ASSERT_EQ(constant->size(), 1);
  EXPECT_TRUE(constant->isNullAt(0));
}

TEST_F(FileSplitReaderTest, timestampWithTimeZoneUnparseableValue) {
  VELOX_ASSERT_USER_THROW(
      packedConstant("not a timestamp"),
      "Cannot convert value to TIMESTAMP WITH TIME ZONE: not a timestamp");
}

// An offset with no corresponding zone key has nothing to pack with, so it is
// rejected rather than recorded under a different zone.
TEST_F(FileSplitReaderTest, timestampWithTimeZoneUnknownOffset) {
  VELOX_ASSERT_USER_THROW(
      packedConstant("2021-01-01 00:00:00 +19:00"),
      "Unknown timezone in TIMESTAMP WITH TIME ZONE value");
}

} // namespace
} // namespace facebook::velox::connector::hive
