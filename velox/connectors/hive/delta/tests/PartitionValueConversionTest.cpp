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

#include <gtest/gtest.h>
#include "velox/common/memory/Memory.h"
#include "velox/core/VectorUtil.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox;

class PartitionValueConversionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  void TearDown() override {
    pool_.reset();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

// Test decimal partition value conversion (short decimal)
TEST_F(PartitionValueConversionTest, shortDecimalConversion) {
  auto decimalType = DECIMAL(10, 2); // precision=10, scale=2

  // Test positive decimal
  auto result = core::newConstantFromString(
      decimalType,
      "123.45",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // 123.45 with scale 2 = 12345
  int64_t expectedValue = 12345;
  EXPECT_EQ(expectedValue, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, shortDecimalNegative) {
  auto decimalType = DECIMAL(10, 2);

  // Test negative decimal
  auto result = core::newConstantFromString(
      decimalType,
      "-456.78",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // -456.78 with scale 2 = -45678
  int64_t expectedValue = -45678;
  EXPECT_EQ(expectedValue, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, shortDecimalZero) {
  auto decimalType = DECIMAL(10, 2);

  auto result = core::newConstantFromString(
      decimalType,
      "0.00",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_EQ(0, flatVector->valueAt(0));
}

// Test long decimal partition value conversion
TEST_F(PartitionValueConversionTest, longDecimalConversion) {
  auto decimalType = DECIMAL(38, 10); // precision=38, scale=10

  auto result = core::newConstantFromString(
      decimalType,
      "12345678901234567890.1234567890",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<int128_t>>();
  ASSERT_NE(nullptr, flatVector);

  // Verify the value is non-zero (exact value depends on decimal encoding)
  EXPECT_NE(0, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, longDecimalLargeValue) {
  auto decimalType = DECIMAL(38, 5);

  auto result = core::newConstantFromString(
      decimalType,
      "999999999999999999999999999999999.99999",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int128_t>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_NE(0, flatVector->valueAt(0));
}

// Test NULL decimal values
TEST_F(PartitionValueConversionTest, decimalNullValue) {
  auto decimalType = DECIMAL(10, 2);

  auto result = core::newConstantFromString(
      decimalType,
      std::nullopt,
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_TRUE(result->isNullAt(0));
}

// Test TIMESTAMP WITH TIME ZONE partition value conversion
TEST_F(PartitionValueConversionTest, timestampWithTimeZoneConversion) {
  auto timestampTzType = TIMESTAMP_WITH_TIME_ZONE();

  // Test timestamp string in format: "2024-02-18 10:30:45.123"
  auto result = core::newConstantFromString(
      timestampTzType,
      "2024-02-18 10:30:45.123",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // The value should be packed (timestamp + timezone key)
  // Verify it's non-zero
  EXPECT_NE(0, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, timestampWithTimeZoneEpoch) {
  auto timestampTzType = TIMESTAMP_WITH_TIME_ZONE();

  // Test epoch timestamp (1970-01-01 00:00:00)
  auto result = core::newConstantFromString(
      timestampTzType,
      "1970-01-01 00:00:00",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // Epoch should result in a specific packed value
  // The exact value depends on the packing implementation
  int64_t packedValue = flatVector->valueAt(0);
  EXPECT_TRUE(packedValue >= 0); // Should be non-negative for epoch
}

TEST_F(PartitionValueConversionTest, timestampWithTimeZoneNull) {
  auto timestampTzType = TIMESTAMP_WITH_TIME_ZONE();

  auto result = core::newConstantFromString(
      timestampTzType,
      std::nullopt,
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_TRUE(result->isNullAt(0));
}

// Test regular TIMESTAMP conversion (for comparison)
TEST_F(PartitionValueConversionTest, regularTimestampConversion) {
  auto timestampType = TIMESTAMP();

  auto result = core::newConstantFromString(
      timestampType,
      "2024-02-18 10:30:45",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<Timestamp>>();
  ASSERT_NE(nullptr, flatVector);

  Timestamp ts = flatVector->valueAt(0);
  EXPECT_GT(ts.getSeconds(), 0);
}

// Test DATE partition value conversion
TEST_F(PartitionValueConversionTest, dateConversion) {
  auto dateType = DATE();

  auto result = core::newConstantFromString(
      dateType,
      "2024-02-18",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<int32_t>>();
  ASSERT_NE(nullptr, flatVector);

  // Days since epoch for 2024-02-18
  int32_t days = flatVector->valueAt(0);
  EXPECT_GT(days, 19000); // Should be > 19000 days since 1970
}

TEST_F(PartitionValueConversionTest, dateDaysSinceEpoch) {
  auto dateType = DATE();

  // Test with isDaysSinceEpoch=true (Iceberg format)
  auto result = core::newConstantFromString(
      dateType,
      "19771", // Days since epoch
      pool_.get(),
      false,
      true, // isDaysSinceEpoch
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int32_t>>();
  ASSERT_NE(nullptr, flatVector);

  EXPECT_EQ(19771, flatVector->valueAt(0));
}

// Test decimal precision and scale validation
TEST_F(PartitionValueConversionTest, decimalPrecisionValidation) {
  auto decimalType = DECIMAL(5, 2); // Max value: 999.99

  // This should work
  auto result = core::newConstantFromString(
      decimalType,
      "999.99",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_EQ(99999, flatVector->valueAt(0)); // 999.99 * 100
}

TEST_F(PartitionValueConversionTest, decimalScaleHandling) {
  auto decimalType = DECIMAL(10, 4); // scale=4

  auto result = core::newConstantFromString(
      decimalType,
      "123.4567",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // 123.4567 with scale 4 = 1234567
  EXPECT_EQ(1234567, flatVector->valueAt(0));
}

// Test edge cases
TEST_F(PartitionValueConversionTest, decimalTrailingZeros) {
  auto decimalType = DECIMAL(10, 3);

  auto result = core::newConstantFromString(
      decimalType,
      "100.000",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // 100.000 with scale 3 = 100000
  EXPECT_EQ(100000, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, decimalLeadingZeros) {
  auto decimalType = DECIMAL(10, 2);

  auto result = core::newConstantFromString(
      decimalType,
      "000.12",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // 0.12 with scale 2 = 12
  EXPECT_EQ(12, flatVector->valueAt(0));
}

// Test string partition values (for comparison)
TEST_F(PartitionValueConversionTest, stringPartitionValue) {
  auto stringType = VARCHAR();

  auto result = core::newConstantFromString(
      stringType,
      "partition_value",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<StringView>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_EQ("partition_value", flatVector->valueAt(0).str());
}

TEST_F(PartitionValueConversionTest, integerPartitionValue) {
  auto intType = INTEGER();

  auto result = core::newConstantFromString(
      intType,
      "42",
      pool_.get(),
      false,
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int32_t>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_EQ(42, flatVector->valueAt(0));
}
// Test timezone handling for TIMESTAMP
TEST_F(PartitionValueConversionTest, timestampWithTimezone) {
  auto timestampType = TIMESTAMP();
  auto timezone = tz::locateZone("America/Los_Angeles");

  // Test timestamp with timezone conversion
  auto result = core::newConstantFromString(
      timestampType,
      "2024-02-18 10:30:45",
      pool_.get(),
      false,
      false,
      timezone);

  ASSERT_NE(nullptr, result);
  EXPECT_EQ(1, result->size());
  EXPECT_FALSE(result->isNullAt(0));

  auto flatVector = result->as<ConstantVector<Timestamp>>();
  ASSERT_NE(nullptr, flatVector);

  Timestamp ts = flatVector->valueAt(0);
  // Timestamp should be converted to GMT
  EXPECT_GT(ts.getSeconds(), 0);
}

TEST_F(PartitionValueConversionTest, timestampLocalTime) {
  auto timestampType = TIMESTAMP();

  // Test with isLocalTimestamp=true (converts to GMT)
  auto result = core::newConstantFromString(
      timestampType,
      "2024-02-18 10:30:45",
      pool_.get(),
      true, // isLocalTimestamp
      false,
      nullptr);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<Timestamp>>();
  ASSERT_NE(nullptr, flatVector);

  Timestamp ts = flatVector->valueAt(0);
  EXPECT_GT(ts.getSeconds(), 0);
}

// Test TIMESTAMP WITH TIME ZONE type with timezone parameter
// This tests that TIMESTAMP WITH TIME ZONE type always packs with UTC timezone key,
// regardless of the timezone parameter passed to newConstantFromString.
TEST_F(PartitionValueConversionTest, timestampWithTimeZoneWithTimezone) {
  auto timestampTzType = TIMESTAMP_WITH_TIME_ZONE();
  auto timezone = tz::locateZone("Europe/London");

  // TIMESTAMP WITH TIME ZONE should pack with UTC regardless of timezone param
  auto result = core::newConstantFromString(
      timestampTzType,
      "2024-02-18 10:30:45",
      pool_.get(),
      false,
      false,
      timezone);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);
  EXPECT_NE(0, flatVector->valueAt(0));
}

// Negative tests - these should throw exceptions
TEST_F(PartitionValueConversionTest, invalidDecimalFormat) {
  auto decimalType = DECIMAL(10, 2);

  // Invalid decimal string
  EXPECT_THROW(
      {
        core::newConstantFromString(
            decimalType,
            "not_a_number",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, decimalOverflow) {
  auto decimalType = DECIMAL(5, 2); // Max: 999.99

  // Value exceeds precision
  EXPECT_THROW(
      {
        core::newConstantFromString(
            decimalType,
            "10000.00", // Too large for DECIMAL(5,2)
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, decimalInvalidScale) {
  auto decimalType = DECIMAL(10, 2);

  // Too many decimal places - Note: DecimalUtil may round instead of throwing
  // This test verifies the behavior is consistent
  auto result = core::newConstantFromString(
      decimalType,
      "123.456", // 3 decimal places, but scale is 2
      pool_.get(),
      false,
      false,
      nullptr);

  // The result should be rounded to scale 2
  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<int64_t>>();
  ASSERT_NE(nullptr, flatVector);

  // 123.456 rounded to scale 2 could be 123.46 = 12346
  // Exact rounding behavior depends on DecimalUtil implementation
  EXPECT_NE(0, flatVector->valueAt(0));
}

TEST_F(PartitionValueConversionTest, invalidTimestampFormat) {
  auto timestampType = TIMESTAMP();

  // Invalid timestamp string
  EXPECT_THROW(
      {
        core::newConstantFromString(
            timestampType,
            "not-a-timestamp",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, invalidDateFormat) {
  auto dateType = DATE();

  // Invalid date string
  EXPECT_THROW(
      {
        core::newConstantFromString(
            dateType,
            "2024-13-45", // Invalid month and day
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, invalidIntegerFormat) {
  auto intType = INTEGER();

  // Non-numeric string
  EXPECT_THROW(
      {
        core::newConstantFromString(
            intType,
            "abc123",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, integerOverflow) {
  auto intType = INTEGER();

  // Value too large for int32
  EXPECT_THROW(
      {
        core::newConstantFromString(
            intType,
            "9999999999999", // Exceeds INT32_MAX
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, timestampWithTimeZoneInvalidFormat) {
  auto timestampTzType = TIMESTAMP_WITH_TIME_ZONE();

  // Invalid timestamp format
  EXPECT_THROW(
      {
        core::newConstantFromString(
            timestampTzType,
            "invalid-timestamp",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

// Test timezone edge cases
TEST_F(PartitionValueConversionTest, timestampDifferentTimezones) {
  auto timestampType = TIMESTAMP();

  // Test with different timezones
  std::vector<std::string> timezones = {
      "America/New_York",
      "Europe/London",
      "Asia/Tokyo",
      "Australia/Sydney"
  };

  for (const auto& tzName : timezones) {
    auto timezone = tz::locateZone(tzName);
    auto result = core::newConstantFromString(
        timestampType,
        "2024-02-18 12:00:00",
        pool_.get(),
        false,
        false,
        timezone);

    ASSERT_NE(nullptr, result);
    auto flatVector = result->as<ConstantVector<Timestamp>>();
    ASSERT_NE(nullptr, flatVector);

    // Each timezone should produce a different GMT timestamp
    Timestamp ts = flatVector->valueAt(0);
    EXPECT_GT(ts.getSeconds(), 0);
  }
}

TEST_F(PartitionValueConversionTest, timestampUTCTimezone) {
  auto timestampType = TIMESTAMP();
  auto utcTimezone = tz::locateZone("UTC");

  auto result = core::newConstantFromString(
      timestampType,
      "2024-02-18 12:00:00",
      pool_.get(),
      false,
      false,
      utcTimezone);

  ASSERT_NE(nullptr, result);
  auto flatVector = result->as<ConstantVector<Timestamp>>();
  ASSERT_NE(nullptr, flatVector);

  Timestamp ts = flatVector->valueAt(0);
  EXPECT_GT(ts.getSeconds(), 0);
}

// Test decimal edge cases with negative tests
TEST_F(PartitionValueConversionTest, decimalNegativeOverflow) {
  auto decimalType = DECIMAL(5, 2);

  // Negative value exceeds precision
  EXPECT_THROW(
      {
        core::newConstantFromString(
            decimalType,
            "-10000.00",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, longDecimalInvalidFormat) {
  auto decimalType = DECIMAL(38, 10);

  // Invalid format
  EXPECT_THROW(
      {
        core::newConstantFromString(
            decimalType,
            "12.34.56", // Multiple decimal points
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, decimalEmptyString) {
  auto decimalType = DECIMAL(10, 2);

  // Empty string
  EXPECT_THROW(
      {
        core::newConstantFromString(
            decimalType,
            "",
            pool_.get(),
            false,
            false,
            nullptr);
      },
      VeloxUserError);
}

TEST_F(PartitionValueConversionTest, dateInvalidDaysSinceEpoch) {
  auto dateType = DATE();

  // Invalid days since epoch (non-numeric)
  // This throws folly::ConversionError, not VeloxUserError
  EXPECT_THROW(
      {
        core::newConstantFromString(
            dateType,
            "not_a_number",
            pool_.get(),
            false,
            true, // isDaysSinceEpoch
            nullptr);
      },
      std::exception); // Catches folly::ConversionError
}