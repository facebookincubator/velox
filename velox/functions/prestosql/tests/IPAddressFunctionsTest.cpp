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
        "cast(cast(cast(cast(c0 as ipaddress) as varbinary) as ipaddress) as varchar)",
        input);
    return result;
  }

  std::optional<std::string> castIPPrefixVarcharCycle(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as ipprefix) as varchar)", input);
    return result;
  }

  std::optional<std::string> castIPAddressIPPrefixVarchar(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(cast(c0 as ipaddress) as ipprefix) as varchar)", input);
    return result;
  }

  std::optional<std::string> castIPPrefixIPAddressVarchar(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(cast(c0 as ipprefix) as ipaddress) as varchar)", input);
    return result;
  }
};

TEST_F(IPAddressTest, testVarcharIpAddressCast) {
  EXPECT_EQ(castIPAddressVarcharCycle("::ffff:1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarcharCycle("1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarcharCycle("192.168.0.0"), "192.168.0.0");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:0db8:0000:0000:0000:ff00:0042:8329"),
      "2001:db8::ff00:42:8329");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:db8::ff00:42:8329"),
      "2001:db8::ff00:42:8329");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:db8:0:0:1:0:0:1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:db8:0:0:1::1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:db8::1:0:0:1"), "2001:db8::1:0:0:1");
  EXPECT_EQ(
      castIPAddressVarcharCycle("2001:DB8::FF00:ABCD:12EF"),
      "2001:db8::ff00:abcd:12ef");
  EXPECT_THROW(castIPAddressVarcharCycle("facebook.com"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("localhost"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("2001:db8::1::1"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("2001:zxy::1::1"), VeloxUserError);
  EXPECT_THROW(castIPAddressVarcharCycle("789.1.1.1"), VeloxUserError);
}

TEST_F(IPAddressTest, testVarbinaryIpAddressCast) {
  EXPECT_EQ(
      castIPAddressVarbinary("00000000000000000000ffff01020304"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarbinary("01020304"), "1.2.3.4");
  EXPECT_EQ(castIPAddressVarbinary("c0a80000"), "192.168.0.0");
  EXPECT_EQ(
      castIPAddressVarbinary("20010db8000000000000ff0000428329"),
      "2001:db8::ff00:42:8329");
  EXPECT_THROW(castIPAddressVarbinary("f000001100"), VeloxUserError);
}

TEST_F(IPAddressTest, testVarbinaryIpAddressVarchar) {
  EXPECT_EQ(castIPAddressVarbinaryVarchar("::ffff:1.2.3.4"), "1.2.3.4");
  EXPECT_EQ(
      castIPAddressVarbinaryVarchar("2001:0db8:0000:0000:0000:ff00:0042:8329"),
      "2001:db8::ff00:42:8329");
  EXPECT_EQ(
      castIPAddressVarbinaryVarchar("2001:db8::ff00:42:8329"),
      "2001:db8::ff00:42:8329");
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

TEST_F(IPAddressTest, testVarcharIpPrefixCast) {
  EXPECT_EQ(castIPPrefixVarcharCycle("::ffff:1.2.3.4/24"), "1.2.3.0/24");
  EXPECT_EQ(castIPPrefixVarcharCycle("192.168.0.0/24"), "192.168.0.0/24");
  EXPECT_EQ(castIPPrefixVarcharCycle("255.2.3.4/0"), "0.0.0.0/0");
  EXPECT_EQ(castIPPrefixVarcharCycle("255.2.3.4/1"), "128.0.0.0/1");
  EXPECT_EQ(castIPPrefixVarcharCycle("255.2.3.4/2"), "192.0.0.0/2");
  EXPECT_EQ(castIPPrefixVarcharCycle("255.2.3.4/4"), "240.0.0.0/4");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.4/8"), "1.0.0.0/8");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.4/16"), "1.2.0.0/16");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.4/24"), "1.2.3.0/24");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.255/25"), "1.2.3.128/25");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.255/26"), "1.2.3.192/26");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.255/28"), "1.2.3.240/28");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.255/30"), "1.2.3.252/30");
  EXPECT_EQ(castIPPrefixVarcharCycle("1.2.3.255/32"), "1.2.3.255/32");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:0db8:0000:0000:0000:ff00:0042:8329/128"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:db8::ff00:42:8329/128"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:db8:0:0:1:0:0:1/128"),
      "2001:db8::1:0:0:1/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:db8:0:0:1::1/128"),
      "2001:db8::1:0:0:1/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:db8::1:0:0:1/128"),
      "2001:db8::1:0:0:1/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("2001:DB8::FF00:ABCD:12EF/128"),
      "2001:db8::ff00:abcd:12ef/128");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/0"),
      "::/0");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/1"),
      "8000::/1");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/2"),
      "c000::/2");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/4"),
      "f000::/4");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/8"),
      "ff00::/8");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/16"),
      "ffff::/16");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/32"),
      "ffff:ffff::/32");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/48"),
      "ffff:ffff:ffff::/48");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/64"),
      "ffff:ffff:ffff:ffff::/64");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/80"),
      "ffff:ffff:ffff:ffff:ffff::/80");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/96"),
      "ffff:ffff:ffff:ffff:ffff:ffff::/96");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/112"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:0/112");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/120"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ff00/120");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/124"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0/124");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/126"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffc/126");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/127"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe/127");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128");
  EXPECT_EQ(castIPPrefixVarcharCycle("10.0.0.0/32"), "10.0.0.0/32");
  EXPECT_EQ(
      castIPPrefixVarcharCycle("64:ff9b::10.0.0.0/128"), "64:ff9b::a00:0/128");
  EXPECT_THROW(castIPPrefixVarcharCycle("facebook.com/32"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("localhost/32"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("2001:db8::1::1/128"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("2001:zxy::1::1/128"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("789.1.1.1/32"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("192.1.1.1"), VeloxUserError);
  EXPECT_THROW(castIPPrefixVarcharCycle("192.1.1.1/128"), VeloxUserError);
}

TEST_F(IPAddressTest, testCastIPAddressIPPrefixVarchar) {
  EXPECT_EQ(castIPAddressIPPrefixVarchar("1.2.3.4"), "1.2.3.4/32");
  EXPECT_EQ(castIPAddressIPPrefixVarchar("::ffff:102:304"), "1.2.3.4/32");
  EXPECT_EQ(castIPAddressIPPrefixVarchar("::1"), "::1/128");
  EXPECT_EQ(
      castIPAddressIPPrefixVarchar("2001:db8::ff00:42:8329"),
      "2001:db8::ff00:42:8329/128");
}

TEST_F(IPAddressTest, testCastIPPrefixIPAddressVarchar) {
  EXPECT_EQ(castIPPrefixIPAddressVarchar("1.2.3.4/32"), "1.2.3.4");
  EXPECT_EQ(castIPPrefixIPAddressVarchar("1.2.3.4/24"), "1.2.3.0");
  EXPECT_EQ(castIPPrefixIPAddressVarchar("::1/128"), "::1");
  EXPECT_EQ(
      castIPPrefixIPAddressVarchar("2001:db8::ff00:42:8329/128"),
      "2001:db8::ff00:42:8329");
  EXPECT_EQ(
      castIPPrefixIPAddressVarchar("2001:db8::ff00:42:8329/64"), "2001:db8::");
}

} // namespace

} // namespace facebook::velox::functions::prestosql
