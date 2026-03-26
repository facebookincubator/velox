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
#include "velox/functions/prestosql/types/UuidType.h"
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "velox/functions/prestosql/types/UuidRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::test {

class UuidTypeTest : public testing::Test, public TypeTestBase {
 public:
  UuidTypeTest() {
    registerUuidType();
  }
};

TEST_F(UuidTypeTest, basic) {
  ASSERT_STREQ(UUID()->name(), "UUID");
  ASSERT_STREQ(UUID()->kindName(), "HUGEINT");
  ASSERT_TRUE(UUID()->parameters().empty());
  ASSERT_EQ(UUID()->toString(), "UUID");

  ASSERT_TRUE(hasType("UUID"));
  ASSERT_EQ(*getType("UUID", {}), *UUID());
}

TEST_F(UuidTypeTest, serde) {
  testTypeSerde(UUID());
}

TEST_F(UuidTypeTest, valueToString) {
  // Round-trip: parse UUID string via the same path as CAST(varchar AS uuid),
  // then format back via valueToString.
  auto roundTrip = [](std::string_view uuidStr) {
    auto uuid = boost::lexical_cast<boost::uuids::uuid>(uuidStr);
    int128_t value;
    memcpy(&value, &uuid, 16);
    // boost::uuid is big-endian, convert to Velox's little-endian storage.
    value = DecimalUtil::bigEndian(value);
    char buffer[UuidType::kStringSize];
    return std::string(UUID()->valueToString(value, buffer));
  };

  EXPECT_EQ(
      roundTrip("12151fd2-7586-11e9-8f9e-2a86e4085a59"),
      "12151fd2-7586-11e9-8f9e-2a86e4085a59");
  EXPECT_EQ(
      roundTrip("00000000-0000-0000-0000-000000000000"),
      "00000000-0000-0000-0000-000000000000");
  EXPECT_EQ(
      roundTrip("ffffffff-ffff-ffff-ffff-ffffffffffff"),
      "ffffffff-ffff-ffff-ffff-ffffffffffff");
}

} // namespace facebook::velox::test
