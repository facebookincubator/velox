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
#include "velox/functions/prestosql/types/TimeWithTimezoneType.h"
#include "velox/functions/prestosql/types/TimeWithTimezoneRegistration.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"
#include "velox/type/Time.h"

namespace facebook::velox::test {

class TimeWithTimezoneTypeTest : public testing::Test, public TypeTestBase {
 public:
  TimeWithTimezoneTypeTest() {
    registerTimeWithTimezoneType();
  }
};

// Basic type properties test - similar to TimestampWithTimeZoneTypeTest
TEST_F(TimeWithTimezoneTypeTest, basic) {
  auto type = TIME_WITH_TIME_ZONE();

  // Test type name and properties
  ASSERT_STREQ(type->name(), "TIME WITH TIME ZONE");
  ASSERT_STREQ(type->kindName(), "BIGINT");
  ASSERT_EQ(type->toString(), "TIME WITH TIME ZONE");

  // Test that type is registered and can be retrieved
  ASSERT_TRUE(hasType("time with time zone"));
  auto retrievedType = getType("time with time zone", {});
  ASSERT_TRUE(retrievedType != nullptr);
}

// Test serialization/deserialization - similar to TimestampWithTimeZoneTypeTest
TEST_F(TimeWithTimezoneTypeTest, serde) {
  testTypeSerde(TIME_WITH_TIME_ZONE());
}

// Test type equivalence
TEST_F(TimeWithTimezoneTypeTest, equivalent) {
  auto type1 = TIME_WITH_TIME_ZONE();
  auto type2 = TIME_WITH_TIME_ZONE();

  // Since it's a singleton, they should be equivalent
  ASSERT_TRUE(type1->equivalent(*type2));
  ASSERT_TRUE(type2->equivalent(*type1));

  // Test with different type
  auto bigintType = BIGINT();
  ASSERT_FALSE(type1->equivalent(*bigintType));
}

// Test basic properties
TEST_F(TimeWithTimezoneTypeTest, properties) {
  auto type = TIME_WITH_TIME_ZONE();

  ASSERT_TRUE(type->isOrderable());
  ASSERT_TRUE(type->isComparable());
}

// Test value to string conversion
// Note: TIME WITH TIME ZONE stores time in UTC and displays it in local time
// by adding the timezone offset to the UTC value
TEST_F(TimeWithTimezoneTypeTest, valueToString) {
  auto type = TIME_WITH_TIME_ZONE();
  char buffer[TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize];
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);

  // Test midnight UTC (00:00:00.000 UTC) at UTC+00:00
  int64_t timeValue = 0;
  int16_t timeZone = util::biasEncode(0); // UTC
  auto value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "00:00:00.000+00:00");

  // Test 01:00:00.000 UTC at UTC+00:00
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  value = util::pack(3600000, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "01:00:00.000+00:00");

  // Test UTC 12:30:45.123 with UTC+00:00 → displays as 12:30:45.123+00:00
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeValue = 12 * 3600000 + 30 * 60000 + 45 * 1000 + 123;
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "12:30:45.123+00:00");

  // Test UTC 12:30:45.123 with UTC-08:00 (PST)
  // → displays as 04:30:45.123-08:00 (12:30 - 8h = 04:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(-480); // PST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "04:30:45.123-08:00");

  // Test UTC 12:30:45.123 with UTC-05:00 (EST)
  // → displays as 07:30:45.123-05:00 (12:30 - 5h = 07:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(-300); // EST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "07:30:45.123-05:00");

  // Test UTC 12:30:45.123 with UTC+00:00 (GMT)
  // → displays as 12:30:45.123+00:00
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(0); // GMT
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "12:30:45.123+00:00");

  // Test UTC 12:30:45.123 with UTC+01:00 (BST)
  // → displays as 13:30:45.123+01:00 (12:30 + 1h = 13:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(60); // BST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "13:30:45.123+01:00");

  // Test UTC 12:30:45.123 with UTC+09:00 (JST)
  // → displays as 21:30:45.123+09:00 (12:30 + 9h = 21:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(540); // JST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "21:30:45.123+09:00");

  // Test UTC 12:30:45.123 with UTC+10:00 (AEST)
  // → displays as 22:30:45.123+10:00 (12:30 + 10h = 22:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(600); // AEST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "22:30:45.123+10:00");

  // Test UTC 12:30:45.123 with UTC+11:00 (AEDT)
  // → displays as 23:30:45.123+11:00 (12:30 + 11h = 23:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(660); // AEDT
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "23:30:45.123+11:00");

  // Test UTC 12:30:45.123 with UTC+13:00 (NZST)
  // → displays as 01:30:45.123+13:00 (12:30 + 13h = 25:30 → wraps to 01:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(780); // NZST
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "01:30:45.123+13:00");

  // Test UTC 12:30:45.123 with UTC+14:00 (NZDT)
  // → displays as 02:30:45.123+14:00 (12:30 + 14h = 26:30 → wraps to 02:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(840); // NZDT
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "02:30:45.123+14:00");

  // Test UTC 12:30:45.123 with UTC-14:00
  // → displays as 22:30:45.123-14:00 (12:30 - 14h = -1:30 → wraps to 22:30)
  std::memset(
      buffer, 0, TimeWithTimezoneType::kTimeWithTimezoneToVarcharRowSize);
  timeZone = util::biasEncode(-840); // UTC-14:00
  value = util::pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value, buffer), "22:30:45.123-14:00");
}

TEST_F(TimeWithTimezoneTypeTest, compare) {
  auto compare = [](int32_t expected, int64_t left, int64_t right) {
    ASSERT_EQ(expected, TIME_WITH_TIME_ZONE()->compare(left, right));
  };

  // Same UTC time: "12:30:45.123+01:00" == "11:30:45.123+00:00"
  compare(
      0,
      util::pack(41445123, util::biasEncode(60)),
      util::pack(41445123, util::biasEncode(0)));

  // Different UTC times: "10:00:00.000+01:00" < "15:00:00.000+01:00"
  compare(
      -1,
      util::pack(32400000, util::biasEncode(60)),
      util::pack(50400000, util::biasEncode(60)));
  // "15:00:00.000+01:00" > "10:00:00.000+01:00"
  compare(
      1,
      util::pack(50400000, util::biasEncode(60)),
      util::pack(32400000, util::biasEncode(60)));
  // "10:00:00.000+01:00" < "15:00:00.000+03:00"
  compare(
      -1,
      util::pack(32400000, util::biasEncode(60)),
      util::pack(43200000, util::biasEncode(180)));

  // Wrap-around normalization: "00:00:00.000+01:00" < "23:59:59.999+03:00"
  compare(
      -1,
      util::pack(82800000, util::biasEncode(60)),
      util::pack(75599999, util::biasEncode(180)));
  // "15:00:00.000+03:00" > "10:00:00.000+01:00"
  compare(
      1,
      util::pack(43200000, util::biasEncode(180)),
      util::pack(32400000, util::biasEncode(60)));
  // "23:59:59.999+03:00" > "00:00:00.000+01:00"
  compare(
      1,
      util::pack(75599999, util::biasEncode(180)),
      util::pack(82800000, util::biasEncode(60)));
}

TEST_F(TimeWithTimezoneTypeTest, hash) {
  auto expectHashesEq = [](int64_t millis1,
                           int16_t tzOffsetMinutes1,
                           int64_t millis2,
                           int16_t tzOffsetMinutes2) {
    int64_t left = util::pack(millis1, util::biasEncode(tzOffsetMinutes1));
    int64_t right = util::pack(millis2, util::biasEncode(tzOffsetMinutes2));

    ASSERT_EQ(
        TIME_WITH_TIME_ZONE()->hash(left), TIME_WITH_TIME_ZONE()->hash(right));
  };

  auto expectHashesNeq = [](int64_t millis1,
                            int16_t tzOffsetMinutes1,
                            int64_t millis2,
                            int16_t tzOffsetMinutes2) {
    int64_t left = util::pack(millis1, util::biasEncode(tzOffsetMinutes1));
    int64_t right = util::pack(millis2, util::biasEncode(tzOffsetMinutes2));

    ASSERT_NE(
        TIME_WITH_TIME_ZONE()->hash(left), TIME_WITH_TIME_ZONE()->hash(right));
  };

  expectHashesEq(45045123, 60, 45045123, 180);
  expectHashesEq(45045123, 60, 45045123, -840);
  expectHashesEq(45045123, 180, 45045123, -840);
  expectHashesEq(0, 180, 0, -840);

  expectHashesNeq(36000000, 60, 54000000, 180);
  expectHashesNeq(36000000, 60, 54000000, -840);
  expectHashesNeq(36000000, 180, 54000000, -840);
  expectHashesNeq(0, 180, 86399999, -840);
}

} // namespace facebook::velox::test
