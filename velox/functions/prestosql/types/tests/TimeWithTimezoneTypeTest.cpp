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

// Test value to string conversion (if implemented)
TEST_F(TimeWithTimezoneTypeTest, valueToString) {
  auto type = TIME_WITH_TIME_ZONE();

  // Test basic time values - these should work based on the implementation
  // Test midnight (00:00:00.000)
  int64_t timeValue = 0;
  int16_t timeZone = TimeWithTimezoneType::biasEncode(0); // UTC
  auto value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "00:00:00.000+00:00");

  // Test 1 hour (01:00:00.000) at UTC
  value = pack(3600000, timeZone);
  ASSERT_EQ(type->valueToString(value), "01:00:00.000+00:00");

  // Test 12:30:45.123 at UTC
  timeValue = 12 * 3600000 + 30 * 60000 + 45 * 1000 + 123;
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+00:00");

  // Test 12:30:45.123 at PST
  timeZone = TimeWithTimezoneType::biasEncode(-480); // PST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123-08:00");

  // Test 12:30:45.123 at EST
  timeZone = TimeWithTimezoneType::biasEncode(-300); // EST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123-05:00");

  // Test 12:30:45.123 at GMT
  timeZone = TimeWithTimezoneType::biasEncode(0); // GMT
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+00:00");

  // Test 12:30:45.123 at BST
  timeZone = TimeWithTimezoneType::biasEncode(60); // BST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+01:00");

  // Test 12:30:45.123 at JST
  timeZone = TimeWithTimezoneType::biasEncode(540); // JST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+09:00");

  // Test 12:30:45.123 at AEST
  timeZone = TimeWithTimezoneType::biasEncode(600); // AEST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+10:00");

  // Test 12:30:45.123 at AEDT
  timeZone = TimeWithTimezoneType::biasEncode(660); // AEDT
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+11:00");

  // Test 12:30:45.123 at NZST
  timeZone = TimeWithTimezoneType::biasEncode(780); // NZST
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+13:00");

  // Test 12:30:45.123 at NZDT
  timeZone = TimeWithTimezoneType::biasEncode(840); // NZDT
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123+14:00");

  // Test 12:30:45.123 at UTC-14:00
  timeZone = TimeWithTimezoneType::biasEncode(-840); // UTC-14:00
  value = pack(timeValue, timeZone);
  ASSERT_EQ(type->valueToString(value), "12:30:45.123-14:00");
}

} // namespace facebook::velox::test
