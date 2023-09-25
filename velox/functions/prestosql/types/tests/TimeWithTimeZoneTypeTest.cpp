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
#include "velox/functions/prestosql/types/TimeWithTimeZoneType.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class TimeWithTimeZoneTypeTest : public testing::Test, public TypeTestBase {
 public:
  TimeWithTimeZoneTypeTest() {
    registerTimeWithTimeZoneType();
  }
};

TEST_F(TimeWithTimeZoneTypeTest, basic) {
  ASSERT_EQ(TIME_WITH_TIME_ZONE()->name(), "TIME WITH TIME ZONE");
  ASSERT_EQ(TIME_WITH_TIME_ZONE()->kindName(), "ROW");
  ASSERT_TRUE(TIME_WITH_TIME_ZONE()->parameters().empty());
  ASSERT_EQ(TIME_WITH_TIME_ZONE()->toString(), "TIME WITH TIME ZONE");

  ASSERT_TRUE(hasType("TIME WITH TIME ZONE"));
  ASSERT_EQ(*getType("TIME WITH TIME ZONE", {}), *TIME_WITH_TIME_ZONE());
}

TEST_F(TimeWithTimeZoneTypeTest, serde) {
  testTypeSerde(TIME_WITH_TIME_ZONE());
}
} // namespace facebook::velox::test
