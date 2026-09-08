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

#include "velox/type/CalendarInterval.h"
#include "velox/type/Type.h"

namespace facebook::velox {

class CalendarIntervalTest : public ::testing::Test {};

// --- Pack/Unpack round-trip tests ---

TEST_F(CalendarIntervalTest, packUnpackZero) {
  CalendarInterval interval(0, 0, 0);
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
  EXPECT_EQ(packed, static_cast<int128_t>(0));
}

TEST_F(CalendarIntervalTest, packUnpackPositive) {
  CalendarInterval interval(14, 30, 3600000000L); // 1y2m, 30d, 1h
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
}

TEST_F(CalendarIntervalTest, packUnpackNegative) {
  CalendarInterval interval(-14, -30, -3600000000L);
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
}

TEST_F(CalendarIntervalTest, packUnpackMaxValues) {
  CalendarInterval interval(
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int64_t>::max());
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
}

TEST_F(CalendarIntervalTest, packUnpackMinValues) {
  CalendarInterval interval(
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int64_t>::min());
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
}

TEST_F(CalendarIntervalTest, packUnpackMixed) {
  CalendarInterval interval(-1, 5, -1000000L); // -1 month, 5 days, -1 second
  auto packed = interval.pack();
  auto unpacked = CalendarInterval::unpack(packed);
  EXPECT_EQ(interval, unpacked);
}

// --- toString tests (matching Spark CalendarIntervalSuite) ---

TEST_F(CalendarIntervalTest, toStringZero) {
  CalendarInterval interval(0, 0, 0);
  EXPECT_EQ(interval.toString(), "0 seconds");
}

TEST_F(CalendarIntervalTest, toStringYearsMonths) {
  CalendarInterval interval(14, 0, 0); // 1 year 2 months
  EXPECT_EQ(interval.toString(), "1 years 2 months");
}

TEST_F(CalendarIntervalTest, toStringDays) {
  CalendarInterval interval(0, 5, 0);
  EXPECT_EQ(interval.toString(), "5 days");
}

TEST_F(CalendarIntervalTest, toStringMicroseconds) {
  // 1 hour, 2 minutes, 3 seconds
  int64_t micros = 1L * 3600000000L + 2L * 60000000L + 3L * 1000000L;
  CalendarInterval interval(0, 0, micros);
  EXPECT_EQ(interval.toString(), "1 hours 2 minutes 3 seconds");
}

TEST_F(CalendarIntervalTest, toStringFull) {
  int64_t micros = 1L * 3600000000L + 2L * 60000000L + 3L * 1000000L;
  CalendarInterval interval(14, 5, micros); // 1y2m 5d 1h2m3s
  EXPECT_EQ(
      interval.toString(),
      "1 years 2 months 5 days 1 hours 2 minutes 3 seconds");
}

TEST_F(CalendarIntervalTest, toStringNegativeMonths) {
  CalendarInterval interval(-14, 0, 0); // -1 year -2 months
  EXPECT_EQ(interval.toString(), "-1 years -2 months");
}

TEST_F(CalendarIntervalTest, toStringFractionalSeconds) {
  CalendarInterval interval(0, 0, 1500000L); // 1.5 seconds
  EXPECT_EQ(interval.toString(), "1.5 seconds");
}

// --- Equality and comparison ---

TEST_F(CalendarIntervalTest, equality) {
  CalendarInterval a(1, 2, 3);
  CalendarInterval b(1, 2, 3);
  CalendarInterval c(1, 2, 4);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
}

TEST_F(CalendarIntervalTest, compare) {
  CalendarInterval a(1, 2, 3);
  CalendarInterval b(1, 2, 4);
  CalendarInterval c(1, 3, 0);
  CalendarInterval d(2, 0, 0);
  EXPECT_LT(a.compare(b), 0);
  EXPECT_GT(b.compare(a), 0);
  EXPECT_EQ(a.compare(a), 0);
  EXPECT_LT(a.compare(c), 0);
  EXPECT_LT(a.compare(d), 0);
}

// --- Type system tests ---

TEST_F(CalendarIntervalTest, typeIdentity) {
  auto type = CALENDAR_INTERVAL();
  EXPECT_EQ(type->name(), std::string("INTERVAL"));
  EXPECT_EQ(type->toString(), std::string("INTERVAL"));
  EXPECT_EQ(type->kind(), TypeKind::HUGEINT);
  EXPECT_TRUE(type->isCalendarInterval());
  EXPECT_FALSE(type->isLongDecimal());
  EXPECT_FALSE(type->isIntervalDayTime());
  // CalendarInterval is comparable (for grouping) but not orderable.
  EXPECT_TRUE(type->isComparable());
  EXPECT_FALSE(type->isOrderable());
}

TEST_F(CalendarIntervalTest, typeSingleton) {
  auto a = CALENDAR_INTERVAL();
  auto b = CALENDAR_INTERVAL();
  EXPECT_EQ(a.get(), b.get());
  EXPECT_TRUE(a->equivalent(*b));
}

TEST_F(CalendarIntervalTest, typeNotEquivalentToHugeint) {
  auto calendarType = CALENDAR_INTERVAL();
  auto hugeintType = HUGEINT();
  EXPECT_FALSE(calendarType->equivalent(*hugeintType));
}

TEST_F(CalendarIntervalTest, typeKindEqualsHugeint) {
  auto calendarType = CALENDAR_INTERVAL();
  EXPECT_TRUE(calendarType->kindEquals(HUGEINT()));
}

TEST_F(CalendarIntervalTest, typeSerde) {
  Type::registerSerDe();
  auto type = CALENDAR_INTERVAL();
  auto serialized = type->serialize();
  EXPECT_EQ(serialized["name"], "CalendarIntervalType");
  auto deserialized = CalendarIntervalType::deserialize(serialized);
  EXPECT_TRUE(type->equivalent(*deserialized));
}

TEST_F(CalendarIntervalTest, valueToString) {
  auto type = CALENDAR_INTERVAL();
  CalendarInterval interval(14, 5, 3723000000L);
  auto packed = interval.pack();
  auto str = type->valueToString(packed);
  EXPECT_EQ(str, "1 years 2 months 5 days 1 hours 2 minutes 3 seconds");
}

TEST_F(CalendarIntervalTest, singletonBuiltInTypeLookup) {
  auto type = getType("INTERVAL", {});
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->isCalendarInterval());
}

} // namespace facebook::velox
