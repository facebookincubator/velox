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
  std::optional<std::string> castIPAddress(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as ipaddress) as varchar)", input);
    return result;
  }

  std::optional<std::string> castIPPrefix(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as ipprefix) as varchar)", input);
    return result;
  }

  std::optional<std::string> getIPPrefix(
      const std::optional<std::string> input,
      std::optional<int8_t> mask) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_prefix(cast(c0 as ipaddress), c1) as varchar)", input, mask);
    return result;
  }

  std::optional<std::string> getIPPrefixUsingVarchar(
      const std::optional<std::string> input,
      std::optional<int8_t> mask) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_prefix(c0, c1) as varchar)", input, mask);
    return result;
  }

  std::optional<std::string> getIPSubnetMin(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_min(cast(c0 as ipprefix)) as varchar)", input);
    return result;
  }

  std::optional<std::string> getIPSubnetMax(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_max(cast(c0 as ipprefix)) as varchar)", input);
    return result;
  }

  std::optional<std::string> getIPSubnetRangeMin(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[1] as varchar)", input);
    return result;
  }

  std::optional<std::string> getIPSubnetRangeMax(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[2] as varchar)", input);
    return result;
  }

  std::optional<bool> getIsSubnetOfIP(
      const std::optional<std::string> prefix,
      const std::optional<std::string> ip) {
    auto result = evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipaddress))",
        prefix,
        ip);
    return result;
  }

  std::optional<bool> getIsSubnetOfIPPrefix(
      const std::optional<std::string> prefix,
      const std::optional<std::string> prefix2) {
    auto result = evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipprefix))",
        prefix,
        prefix2);
    return result;
  }
};

TEST_F(IPAddressTest, castFail) {
  EXPECT_THROW(castIPAddress("12.483.09.1"), VeloxUserError);
  EXPECT_THROW(castIPAddress("10.135.23.12.12"), VeloxUserError);
  EXPECT_THROW(castIPAddress("10.135.23"), VeloxUserError);
  EXPECT_THROW(
      castIPAddress("q001:0db8:85a3:0001:0001:8a2e:0370:7334"), VeloxUserError);
  EXPECT_THROW(
      castIPAddress("2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334"),
      VeloxUserError);
  EXPECT_THROW(
      castIPAddress("2001:0db8:85a3:0001:0001:8a2e:0370"), VeloxUserError);

  EXPECT_THROW(castIPPrefix("12.135.23.12/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("10.135.23.12/33"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:1.2.3.4/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:1.2.3.4/33"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64:ff9b::10/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64:ff9b::10/129"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:1.2.3.4/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:1.2.3.4/33"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:909:909/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("::ffff:909:909/33"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64:ff9b::10/-1"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64:ff9b::10/129"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("localhost/24"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64::ff9b::10/24"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("64:face:book::10/24"), VeloxUserError);
  EXPECT_THROW(castIPPrefix("123.456.789.012/24"), VeloxUserError);
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

TEST_F(IPAddressTest, IPPrefixv4) {
  // EXPECT_EQ("10.0.0.0/8", getIPPrefixUsingVarchar("10.135.23.12", 8));

  EXPECT_EQ("10.0.0.0/8", getIPPrefix("10.135.23.12", 8));
  EXPECT_EQ("192.128.0.0/9", getIPPrefix("192.168.255.255", 9));
  EXPECT_EQ("192.168.255.255/32", getIPPrefix("192.168.255.255", 32));
  EXPECT_EQ("0.0.0.0/0", getIPPrefix("192.168.255.255", 0));

  EXPECT_THROW(getIPPrefix("12.483.09.1", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefix("10.135.23.12.12", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefix("10.135.23", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefix("12.135.23.12", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefix("10.135.23.12", 33), VeloxUserError);
}

TEST_F(IPAddressTest, IPPrefixv4UsingVarchar) {
  EXPECT_EQ("10.0.0.0/8", getIPPrefixUsingVarchar("10.135.23.12", 8));
  EXPECT_EQ("192.128.0.0/9", getIPPrefixUsingVarchar("192.168.255.255", 9));
  EXPECT_EQ(
      "192.168.255.255/32", getIPPrefixUsingVarchar("192.168.255.255", 32));
  EXPECT_EQ("0.0.0.0/0", getIPPrefixUsingVarchar("192.168.255.255", 0));

  EXPECT_THROW(getIPPrefixUsingVarchar("12.483.09.1", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("10.135.23.12.12", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("10.135.23", 8), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("12.135.23.12", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("10.135.23.12", 33), VeloxUserError);
}

TEST_F(IPAddressTest, IPPrefixv6) {
  EXPECT_EQ(
      "2001:db8:85a3::/48",
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48));
  EXPECT_EQ(
      "2001:db8:85a3::/52",
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334/128",
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128));
  EXPECT_EQ("::/0", getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0));

  EXPECT_THROW(
      getIPPrefix("q001:0db8:85a3:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefix("2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370", 8), VeloxUserError);
  EXPECT_THROW(
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", -1),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 140),
      VeloxUserError);
}

TEST_F(IPAddressTest, IPPrefixv6UsingVarchar) {
  EXPECT_EQ(
      "2001:db8:85a3::/48",
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48));
  EXPECT_EQ(
      "2001:db8:85a3::/52",
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334/128",
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128));
  EXPECT_EQ(
      "::/0",
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0));

  EXPECT_THROW(
      getIPPrefixUsingVarchar("q001:0db8:85a3:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefixUsingVarchar(
          "2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370", 8),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", -1),
      VeloxUserError);
  EXPECT_THROW(
      getIPPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 140),
      VeloxUserError);
}

TEST_F(IPAddressTest, IPPrefixPrestoTests) {
  EXPECT_EQ(getIPPrefix("1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(getIPPrefix("1.2.3.4", 32), "1.2.3.4/32");
  EXPECT_EQ(getIPPrefix("1.2.3.4", 0), "0.0.0.0/0");
  EXPECT_EQ(getIPPrefix("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(getIPPrefix("64:ff9b::17", 64), "64:ff9b::/64");
  EXPECT_EQ(getIPPrefix("64:ff9b::17", 127), "64:ff9b::16/127");
  EXPECT_EQ(getIPPrefix("64:ff9b::17", 128), "64:ff9b::17/128");
  EXPECT_EQ(getIPPrefix("64:ff9b::17", 0), "::/0");
  EXPECT_THROW(getIPPrefix("::ffff:1.2.3.4", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefix("::ffff:1.2.3.4", 33), VeloxUserError);
  EXPECT_THROW(getIPPrefix("64:ff9b::10", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefix("64:ff9b::10", 129), VeloxUserError);
}

TEST_F(IPAddressTest, IPPrefixVarcharPrestoTests) {
  EXPECT_EQ(getIPPrefixUsingVarchar("1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(getIPPrefixUsingVarchar("1.2.3.4", 32), "1.2.3.4/32");
  EXPECT_EQ(getIPPrefixUsingVarchar("1.2.3.4", 0), "0.0.0.0/0");
  EXPECT_EQ(getIPPrefixUsingVarchar("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(getIPPrefixUsingVarchar("64:ff9b::17", 64), "64:ff9b::/64");
  EXPECT_EQ(getIPPrefixUsingVarchar("64:ff9b::17", 127), "64:ff9b::16/127");
  EXPECT_EQ(getIPPrefixUsingVarchar("64:ff9b::17", 128), "64:ff9b::17/128");
  EXPECT_EQ(getIPPrefixUsingVarchar("64:ff9b::17", 0), "::/0");
  EXPECT_THROW(getIPPrefixUsingVarchar("::ffff:1.2.3.4", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("::ffff:1.2.3.4", 33), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("64:ff9b::10", -1), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("64:ff9b::10", 129), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("localhost", 24), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("64::ff9b::10", 24), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("64:face:book::10", 24), VeloxUserError);
  EXPECT_THROW(getIPPrefixUsingVarchar("123.456.789.012", 24), VeloxUserError);
}

TEST_F(IPAddressTest, castRoundTripPrefix) {
  auto strings = makeFlatVector<std::string>(
      {"87a0:ce14:8989::/48", "7800::/5", "192.0.0.0/5"});

  auto ipprefixes = evaluate("cast(c0 as ipprefix)", makeRowVector({strings}));
  auto stringsCopy =
      evaluate("cast(c0 as varchar)", makeRowVector({ipprefixes}));
  auto ipprefixesCopy =
      evaluate("cast(c0 as ipprefix)", makeRowVector({stringsCopy}));

  velox::test::assertEqualVectors(strings, stringsCopy);
}

TEST_F(IPAddressTest, IPSubnetMin) {
  EXPECT_EQ("192.0.0.0", getIPSubnetMin("192.64.1.1/9"));
  EXPECT_EQ("0.0.0.0", getIPSubnetMin("192.64.1.1/0"));
  EXPECT_EQ("128.0.0.0", getIPSubnetMin("192.64.1.1/1"));
  EXPECT_EQ("192.64.1.0", getIPSubnetMin("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", getIPSubnetMin("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3::",
      getIPSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ("::", getIPSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ("::", getIPSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressTest, IPSubnetMinPrestoTests) {
  EXPECT_EQ(getIPSubnetMin("1.2.3.4/24"), "1.2.3.0");
  EXPECT_EQ(getIPSubnetMin("1.2.3.4/32"), "1.2.3.4");
  EXPECT_EQ(getIPSubnetMin("64:ff9b::17/64"), "64:ff9b::");
  EXPECT_EQ(getIPSubnetMin("64:ff9b::17/127"), "64:ff9b::16");
  EXPECT_EQ(getIPSubnetMin("64:ff9b::17/128"), "64:ff9b::17");
  EXPECT_EQ(getIPSubnetMin("64:ff9b::17/0"), "::");
}

TEST_F(IPAddressTest, IPSubnetMax) {
  EXPECT_EQ("192.127.255.255", getIPSubnetMax("192.64.1.1/9"));
  EXPECT_EQ("255.255.255.255", getIPSubnetMax("192.64.1.1/0"));
  EXPECT_EQ("255.255.255.255", getIPSubnetMax("192.64.1.1/1"));
  EXPECT_EQ("192.64.1.1", getIPSubnetMax("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", getIPSubnetMax("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ(
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7335",
      getIPSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressTest, IPSubnetMaxPrestoTests) {
  EXPECT_EQ(getIPSubnetMax("1.2.3.128/26"), "1.2.3.191");
  EXPECT_EQ(getIPSubnetMax("192.168.128.4/32"), "192.168.128.4");
  EXPECT_EQ(getIPSubnetMax("10.1.16.3/9"), "10.127.255.255");
  EXPECT_EQ(getIPSubnetMax("2001:db8::16/127"), "2001:db8::17");
  EXPECT_EQ(getIPSubnetMax("2001:db8::16/128"), "2001:db8::16");
  EXPECT_EQ(getIPSubnetMax("64:ff9b::17/64"), "64:ff9b::ffff:ffff:ffff:ffff");
  EXPECT_EQ(getIPSubnetMax("64:ff9b::17/72"), "64:ff9b::ff:ffff:ffff:ffff");
  EXPECT_EQ(
      getIPSubnetMax("64:ff9b::17/0"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff");
}

TEST_F(IPAddressTest, IPSubnetRange) {
  EXPECT_EQ("192.0.0.0", getIPSubnetRangeMin("192.64.1.1/9"));
  EXPECT_EQ("192.127.255.255", getIPSubnetRangeMax("192.64.1.1/9"));

  EXPECT_EQ("0.0.0.0", getIPSubnetRangeMin("192.64.1.1/0"));
  EXPECT_EQ("255.255.255.255", getIPSubnetRangeMax("192.64.1.1/0"));

  EXPECT_EQ("128.0.0.0", getIPSubnetRangeMin("192.64.1.1/1"));
  EXPECT_EQ("255.255.255.255", getIPSubnetRangeMax("192.64.1.1/1"));

  EXPECT_EQ("192.64.1.0", getIPSubnetRangeMin("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", getIPSubnetRangeMax("192.64.1.1/31"));

  EXPECT_EQ("192.64.1.1", getIPSubnetRangeMin("192.64.1.1/32"));
  EXPECT_EQ("192.64.1.1", getIPSubnetRangeMax("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3::",
      getIPSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ(
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));

  EXPECT_EQ(
      "::", getIPSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "::", getIPSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));

  EXPECT_EQ(
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      getIPSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7335",
      getIPSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      getIPSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressTest, IPSubnetRangePrestoTests) {
  EXPECT_EQ(getIPSubnetRangeMin("1.2.3.160/24"), "1.2.3.0");
  EXPECT_EQ(getIPSubnetRangeMin("1.2.3.128/31"), "1.2.3.128");
  EXPECT_EQ(getIPSubnetRangeMin("10.1.6.46/32"), "10.1.6.46");
  EXPECT_EQ(getIPSubnetRangeMin("10.1.6.46/0"), "0.0.0.0");
  EXPECT_EQ(getIPSubnetRangeMin("64:ff9b::17/64"), "64:ff9b::");
  EXPECT_EQ(getIPSubnetRangeMin("64:ff9b::52f4/120"), "64:ff9b::5200");
  EXPECT_EQ(getIPSubnetRangeMin("64:ff9b::17/128"), "64:ff9b::17");

  EXPECT_EQ(getIPSubnetRangeMax("1.2.3.160/24"), "1.2.3.255");
  EXPECT_EQ(getIPSubnetRangeMax("1.2.3.128/31"), "1.2.3.129");
  EXPECT_EQ(getIPSubnetRangeMax("10.1.6.46/32"), "10.1.6.46");
  EXPECT_EQ(getIPSubnetRangeMax("10.1.6.46/0"), "255.255.255.255");
  EXPECT_EQ(
      getIPSubnetRangeMax("64:ff9b::17/64"), "64:ff9b::ffff:ffff:ffff:ffff");
  EXPECT_EQ(getIPSubnetRangeMax("64:ff9b::52f4/120"), "64:ff9b::52ff");
  EXPECT_EQ(getIPSubnetRangeMax("64:ff9b::17/128"), "64:ff9b::17");
}

TEST_F(IPAddressTest, IPSubnetOfIPAddress) {
  EXPECT_EQ(getIsSubnetOfIP("1.2.3.128/26", "1.2.3.129"), true);
  EXPECT_EQ(getIsSubnetOfIP("64:fa9b::17/64", "64:ffff::17"), false);
}

TEST_F(IPAddressTest, IPSubnetOfIPPrefix) {
  EXPECT_EQ(
      getIsSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.144/30"), true);
  EXPECT_EQ(getIsSubnetOfIPPrefix("64:ff9b::17/64", "64:ffff::17/64"), false);
  EXPECT_EQ(getIsSubnetOfIPPrefix("64:ff9b::17/32", "64:ffff::17/24"), false);
  EXPECT_EQ(getIsSubnetOfIPPrefix("64:ffff::17/24", "64:ff9b::17/32"), true);
  EXPECT_EQ(
      getIsSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.131/26"), true);
}

TEST_F(IPAddressTest, IPSubnetOfPrestoTests) {
  EXPECT_EQ(getIsSubnetOfIP("1.2.3.128/26", "1.2.3.129"), true);
  EXPECT_EQ(getIsSubnetOfIP("1.2.3.128/26", "1.2.5.1"), false);
  EXPECT_EQ(getIsSubnetOfIP("1.2.3.128/32", "1.2.3.128"), true);
  EXPECT_EQ(getIsSubnetOfIP("1.2.3.128/0", "192.168.5.1"), true);
  EXPECT_EQ(getIsSubnetOfIP("64:ff9b::17/64", "64:ff9b::ffff:ff"), true);
  EXPECT_EQ(getIsSubnetOfIP("64:ff9b::17/64", "64:ffff::17"), false);

  EXPECT_EQ(
      getIsSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.144/30"), true);
  EXPECT_EQ(getIsSubnetOfIPPrefix("1.2.3.128/26", "1.2.5.1/30"), false);
  EXPECT_EQ(getIsSubnetOfIPPrefix("1.2.3.128/26", "1.2.3.128/26"), true);
  EXPECT_EQ(getIsSubnetOfIPPrefix("64:ff9b::17/64", "64:ff9b::ff:25/80"), true);
  EXPECT_EQ(getIsSubnetOfIPPrefix("64:ff9b::17/64", "64:ffff::17/64"), false);
  EXPECT_EQ(
      getIsSubnetOfIPPrefix("2804:431:b000::/37", "2804:431:b000::/38"), true);
  EXPECT_EQ(
      getIsSubnetOfIPPrefix("2804:431:b000::/38", "2804:431:b000::/37"), false);
  EXPECT_EQ(getIsSubnetOfIPPrefix("170.0.52.0/22", "170.0.52.0/24"), true);
  EXPECT_EQ(getIsSubnetOfIPPrefix("170.0.52.0/24", "170.0.52.0/22"), false);
}

} // namespace

} // namespace facebook::velox::functions::prestosql
                                                    