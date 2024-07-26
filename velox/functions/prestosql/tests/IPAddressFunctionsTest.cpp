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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions::prestosql {

namespace {

class IPAddressTest : public functions::test::FunctionBaseTest {
 protected:
  std::optional<std::string> castIPAddressVarcharCycle(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as ipaddress) as varchar)", input);
    return result;
  }

  std::optional<std::string> castIPAddressVarbinary(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(from_hex(c0) as ipaddress) as varchar)", input);
    return result;
  }

  std::optional<std::string> castIPAddressVarbinaryVarchar(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(cast(cast(c0 as ipaddress) as varbinary) as ipaddress) as varchar)", input);
    return result;
  }
};

TEST_F(IPAddressTest, testVarcharIpAddressCast) {
  EXPECT_EQ(castIPAddressVarcharCycle("::ffff:1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarcharCycle("1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarcharCycle("192.168.0.0"), "192.168.0.0");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:0db8:0000:0000:0000:ff00:0042:8329"), "2001:db8::ff00:42:8329");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:db8::ff00:42:8329"), "2001:db8::ff00:42:8329");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:db8:0:0:1:0:0:1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:db8:0:0:1::1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:db8::1:0:0:1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(castIPAddressVarcharCycle("2001:DB8::FF00:ABCD:12EF"), "2001:db8::ff00:abcd:12ef");
  EXPECT_THROW(castIPAddressVarcharCycle("facebook.com"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("localhost"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("2001:db8::1::1"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("2001:zxy::1::1"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("789.1.1.1"), VeloxUserError);
}

TEST_F(IPAddressTest, testVarbinaryIpAddressCast) {
  EXPECT_EQ(castIPAddressVarbinary("00000000000000000000ffff01020304"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarbinary("01020304"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarbinary("c0a80000"), "192.168.0.0");
  EXPECT_EQ(castIPAddressVarbinary("20010db8000000000000ff0000428329"), "2001:db8::ff00:42:8329");
  EXPECT_THROW(castIPAddressVarbinary("f000001100"), VeloxUserError);
}

TEST_F(IPAddressTest, testVarbinaryIpAddressVarchar) {
  EXPECT_EQ(castIPAddressVarbinaryVarchar("::ffff:1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarbinaryVarchar("2001:0db8:0000:0000:0000:ff00:0042:8329"), "2001:db8::ff00:42:8329");
  EXPECT_EQ(castIPAddressVarbinaryVarchar("2001:db8::ff00:42:8329"), "2001:db8::ff00:42:8329");
}

TEST_F(IPAddressTest, nullTest) {
  EXPECT_EQ(castIPAddressVarcharCycle(std::nullopt), std::nullopt);
  EXPECT_EQ(castIPAddressVarbinary(std::nullopt), std::nullopt);
}

TEST_F(IPAddressTest, castRoundTrip) {
  auto strings = makeFlatVector<std::string>(
      {"87a0:ce14:8989:44c9:826e:b4d8:73f9:1542",
       "7cd6:bcec:1216:5c20:4b67:b1bd:173:ced",
       "192.128.0.0"});

  auto ipaddresses =
      evaluate("cast(c0 as ipaddress)", makeRowVector({strings}));
  auto stringsCopy =
      evaluate("cast(c0 as varchar)", makeRowVector({ipaddresses}));
  auto ipaddressesCopy =
      evaluate("cast(c0 as ipaddress)", makeRowVector({stringsCopy}));

  velox::test::assertEqualVectors(strings, stringsCopy);
  velox::test::assertEqualVectors(ipaddresses, ipaddressesCopy);
}
} // namespace

} // namespace facebook::velox::functions::prestosql

