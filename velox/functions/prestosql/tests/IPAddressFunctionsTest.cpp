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

class IPAddressFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  std::optional<std::string> ipPrefix(
      const std::optional<std::string> input,
      std::optional<int64_t> mask) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_prefix(cast(c0 as ipaddress), c1) as varchar)", input, mask);
    return result;
  }

  std::optional<std::string> ipPrefixUsingVarchar(
      const std::optional<std::string> input,
      std::optional<int64_t> mask) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_prefix(c0, c1) as varchar)", input, mask);
    return result;
  }

  std::optional<std::string> ipSubnetMin(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_min(cast(c0 as ipprefix)) as varchar)", input);
    return result;
  }

  std::optional<std::string> ipSubnetMax(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_max(cast(c0 as ipprefix)) as varchar)", input);
    return result;
  }

  std::optional<std::string> ipSubnetRangeMin(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[1] as varchar)", input);
    return result;
  }

  std::optional<std::string> ipSubnetRangeMax(
      const std::optional<std::string> input) {
    auto result = evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[2] as varchar)", input);
    return result;
  }

  std::optional<bool> isSubnetOfIP(
      const std::optional<std::string> prefix,
      const std::optional<std::string> ip) {
    auto result = evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipaddress))",
        prefix,
        ip);
    return result;
  }

  std::optional<bool> isSubnetOfIPPrefix(
      const std::optional<std::string> prefix,
      const std::optional<std::string> prefix2) {
    auto result = evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipprefix))",
        prefix,
        prefix2);
    return result;
  }
};

/*
TEST_F(IPAddressFunctionsTest, nullTest) {
  EXPECT_EQ(ipPrefix(std::nullopt, 8), std::nullopt);
  EXPECT_EQ(ipPrefix(std::nullopt, 8), std::nullopt);
  EXPECT_EQ(ipPrefixUsingVarchar(std::nullopt, 48), std::nullopt);
  EXPECT_EQ(ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334",
std::nullopt), std::nullopt); EXPECT_EQ(ipSubnetMin(std::nullopt),
std::nullopt); EXPECT_EQ(ipSubnetMax(std::nullopt), std::nullopt);
  EXPECT_EQ(ipSubnetRangeMin(std::nullopt), std::nullopt);
  EXPECT_EQ(ipSubnetRangeMax(std::nullopt), std::nullopt);
  EXPECT_EQ(isSubnetOfIPPrefix(std::nullopt, "64:ff9b::17/32"),
std::nullopt); EXPECT_EQ(isSubnetOfIPPrefix("64:ffff::17/24", std::nullopt),
std::nullopt); EXPECT_EQ(isSubnetOfIP(std::nullopt, "1.2.3.129"),
std::nullopt); EXPECT_EQ(isSubnetOfIP("1.2.3.128/26", std::nullopt),
std::nullopt);
}

TEST_F(IPAddressFunctionsTest, IPPrefixv4) {
  EXPECT_EQ("10.0.0.0/8", ipPrefix("10.135.23.12", 8));
  EXPECT_EQ("192.128.0.0/9", ipPrefix("192.168.255.255", 9));
  EXPECT_EQ("192.168.255.255/32", ipPrefix("192.168.255.255", 32));
  EXPECT_EQ("0.0.0.0/0", ipPrefix("192.168.255.255", 0));

  EXPECT_THROW(ipPrefix("12.483.09.1", 8), VeloxUserError);
  EXPECT_THROW(ipPrefix("10.135.23.12.12", 8), VeloxUserError);
  EXPECT_THROW(ipPrefix("10.135.23", 8), VeloxUserError);
  EXPECT_THROW(ipPrefix("12.135.23.12", -1), VeloxUserError);
  EXPECT_THROW(ipPrefix("10.135.23.12", 33), VeloxUserError);
}


TEST_F(IPAddressFunctionsTest, IPPrefixv4UsingVarchar) {
  EXPECT_EQ("10.0.0.0/8", ipPrefixUsingVarchar("10.135.23.12", 8));
  EXPECT_EQ("192.128.0.0/9", ipPrefixUsingVarchar("192.168.255.255", 9));
  EXPECT_EQ(
      "192.168.255.255/32", ipPrefixUsingVarchar("192.168.255.255", 32));
  EXPECT_EQ("0.0.0.0/0", ipPrefixUsingVarchar("192.168.255.255", 0));

  EXPECT_THROW(ipPrefixUsingVarchar("12.483.09.1", 8), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("10.135.23.12.12", 8), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("10.135.23", 8), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("12.135.23.12", -1), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("10.135.23.12", 33), VeloxUserError);
}

TEST_F(IPAddressFunctionsTest, IPPrefixv6) {
  EXPECT_EQ(
      "2001:db8:85a3::/48",
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48));
  EXPECT_EQ(
      "2001:db8:85a3::/52",
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334/128",
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128));

  EXPECT_EQ("::/0", ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0));

  EXPECT_THROW(
      ipPrefix("q001:0db8:85a3:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefix("2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370", 8), VeloxUserError);
  EXPECT_THROW(
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", -1),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 140),
      VeloxUserError);
}

TEST_F(IPAddressFunctionsTest, IPPrefixv6UsingVarchar) {
  EXPECT_EQ(
      "2001:db8:85a3::/48",
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48));
  EXPECT_EQ(
      "2001:db8:85a3::/52",
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334/128",
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128));
  EXPECT_EQ(
      "::/0",
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0));

  EXPECT_THROW(
      ipPrefixUsingVarchar("q001:0db8:85a3:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefixUsingVarchar(
          "2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334", 8),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370", 8),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", -1),
      VeloxUserError);
  EXPECT_THROW(
      ipPrefixUsingVarchar("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 140),
      VeloxUserError);
}

TEST_F(IPAddressFunctionsTest, IPPrefixPrestoTests) {
  EXPECT_EQ(ipPrefix("1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(ipPrefix("1.2.3.4", 32), "1.2.3.4/32");
  EXPECT_EQ(ipPrefix("1.2.3.4", 0), "0.0.0.0/0");
  EXPECT_EQ(ipPrefix("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(ipPrefix("64:ff9b::17", 64), "64:ff9b::/64");
  EXPECT_EQ(ipPrefix("64:ff9b::17", 127), "64:ff9b::16/127");
  EXPECT_EQ(ipPrefix("64:ff9b::17", 128), "64:ff9b::17/128");
  EXPECT_EQ(ipPrefix("64:ff9b::17", 0), "::/0");
  EXPECT_THROW(ipPrefix("::ffff:1.2.3.4", -1), VeloxUserError);
  EXPECT_THROW(ipPrefix("::ffff:1.2.3.4", 33), VeloxUserError);
  EXPECT_THROW(ipPrefix("64:ff9b::10", -1), VeloxUserError);
  EXPECT_THROW(ipPrefix("64:ff9b::10", 129), VeloxUserError);
}

TEST_F(IPAddressFunctionsTest, IPPrefixVarcharPrestoTests) {
  EXPECT_EQ(ipPrefixUsingVarchar("1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(ipPrefixUsingVarchar("1.2.3.4", 32), "1.2.3.4/32");
  EXPECT_EQ(ipPrefixUsingVarchar("1.2.3.4", 0), "0.0.0.0/0");
  EXPECT_EQ(ipPrefixUsingVarchar("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  EXPECT_EQ(ipPrefixUsingVarchar("64:ff9b::17", 64), "64:ff9b::/64");
  EXPECT_EQ(ipPrefixUsingVarchar("64:ff9b::17", 127), "64:ff9b::16/127");
  EXPECT_EQ(ipPrefixUsingVarchar("64:ff9b::17", 128), "64:ff9b::17/128");
  EXPECT_EQ(ipPrefixUsingVarchar("64:ff9b::17", 0), "::/0");
  EXPECT_THROW(ipPrefixUsingVarchar("::ffff:1.2.3.4", -1), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("::ffff:1.2.3.4", 33), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("64:ff9b::10", -1), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("64:ff9b::10", 129), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("localhost", 24), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("64::ff9b::10", 24), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("64:face:book::10", 24), VeloxUserError);
  EXPECT_THROW(ipPrefixUsingVarchar("123.456.789.012", 24), VeloxUserError);
}

TEST_F(IPAddressFunctionsTest, ipSubnetMin) {
  EXPECT_EQ("192.0.0.0", ipSubnetMin("192.64.1.1/9"));
  EXPECT_EQ("0.0.0.0", ipSubnetMin("192.64.1.1/0"));
  EXPECT_EQ("128.0.0.0", ipSubnetMin("192.64.1.1/1"));
  EXPECT_EQ("192.64.1.0", ipSubnetMin("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", ipSubnetMin("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3::",
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ("::", ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ("::", ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressFunctionsTest, IPSubnetMinPrestoTests) {
  EXPECT_EQ(ipSubnetMin("1.2.3.4/24"), "1.2.3.0");
  EXPECT_EQ(ipSubnetMin("1.2.3.4/32"), "1.2.3.4");
  EXPECT_EQ(ipSubnetMin("64:ff9b::17/64"), "64:ff9b::");
  EXPECT_EQ(ipSubnetMin("64:ff9b::17/127"), "64:ff9b::16");
  EXPECT_EQ(ipSubnetMin("64:ff9b::17/128"), "64:ff9b::17");
  EXPECT_EQ(ipSubnetMin("64:ff9b::17/0"), "::");
}

TEST_F(IPAddressFunctionsTest, ipSubnetMax) {
  EXPECT_EQ("192.127.255.255", ipSubnetMax("192.64.1.1/9"));
  EXPECT_EQ("255.255.255.255", ipSubnetMax("192.64.1.1/0"));
  EXPECT_EQ("255.255.255.255", ipSubnetMax("192.64.1.1/1"));
  EXPECT_EQ("192.64.1.1", ipSubnetMax("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", ipSubnetMax("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff",
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ(
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7335",
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressFunctionsTest, IPSubnetMaxPrestoTests) {
  EXPECT_EQ(ipSubnetMax("1.2.3.128/26"), "1.2.3.191");
  EXPECT_EQ(ipSubnetMax("192.168.128.4/32"), "192.168.128.4");
  EXPECT_EQ(ipSubnetMax("10.1.16.3/9"), "10.127.255.255");
  EXPECT_EQ(ipSubnetMax("2001:db8::16/127"), "2001:db8::17");
  EXPECT_EQ(ipSubnetMax("2001:db8::16/128"), "2001:db8::16");
  EXPECT_EQ(ipSubnetMax("64:ff9b::17/64"), "64:ff9b::ffff:ffff:ffff:ffff");
  EXPECT_EQ(ipSubnetMax("64:ff9b::17/72"), "64:ff9b::ff:ffff:ffff:ffff");
  EXPECT_EQ(
      ipSubnetMax("64:ff9b::17/0"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff");
}

TEST_F(IPAddressFunctionsTest, IPSubnetRange) {
  EXPECT_EQ("192.0.0.0", ipSubnetRangeMin("192.64.1.1/9"));
  EXPECT_EQ("192.127.255.255", ipSubnetRangeMax("192.64.1.1/9"));

  EXPECT_EQ("0.0.0.0", ipSubnetRangeMin("192.64.1.1/0"));
  EXPECT_EQ("255.255.255.255", ipSubnetRangeMax("192.64.1.1/0"));

  EXPECT_EQ("128.0.0.0", ipSubnetRangeMin("192.64.1.1/1"));
  EXPECT_EQ("255.255.255.255", ipSubnetRangeMax("192.64.1.1/1"));

  EXPECT_EQ("192.64.1.0", ipSubnetRangeMin("192.64.1.1/31"));
  EXPECT_EQ("192.64.1.1", ipSubnetRangeMax("192.64.1.1/31"));

  EXPECT_EQ("192.64.1.1", ipSubnetRangeMin("192.64.1.1/32"));
  EXPECT_EQ("192.64.1.1", ipSubnetRangeMax("192.64.1.1/32"));

  EXPECT_EQ(
      "2001:db8:85a3::",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  EXPECT_EQ(
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));

  EXPECT_EQ(
      "::", ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "::", ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));

  EXPECT_EQ(
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  EXPECT_EQ(
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7335",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  EXPECT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
}

TEST_F(IPAddressFunctionsTest, IPSubnetRangePrestoTests) {
  EXPECT_EQ(ipSubnetRangeMin("1.2.3.160/24"), "1.2.3.0");
  EXPECT_EQ(ipSubnetRangeMin("1.2.3.128/31"), "1.2.3.128");
  EXPECT_EQ(ipSubnetRangeMin("10.1.6.46/32"), "10.1.6.46");
  EXPECT_EQ(ipSubnetRangeMin("10.1.6.46/0"), "0.0.0.0");
  EXPECT_EQ(ipSubnetRangeMin("64:ff9b::17/64"), "64:ff9b::");
  EXPECT_EQ(ipSubnetRangeMin("64:ff9b::52f4/120"), "64:ff9b::5200");
  EXPECT_EQ(ipSubnetRangeMin("64:ff9b::17/128"), "64:ff9b::17");

  EXPECT_EQ(ipSubnetRangeMax("1.2.3.160/24"), "1.2.3.255");
  EXPECT_EQ(ipSubnetRangeMax("1.2.3.128/31"), "1.2.3.129");
  EXPECT_EQ(ipSubnetRangeMax("10.1.6.46/32"), "10.1.6.46");
  EXPECT_EQ(ipSubnetRangeMax("10.1.6.46/0"), "255.255.255.255");
  EXPECT_EQ(
      ipSubnetRangeMax("64:ff9b::17/64"), "64:ff9b::ffff:ffff:ffff:ffff");
  EXPECT_EQ(ipSubnetRangeMax("64:ff9b::52f4/120"), "64:ff9b::52ff");
  EXPECT_EQ(ipSubnetRangeMax("64:ff9b::17/128"), "64:ff9b::17");
}

TEST_F(IPAddressFunctionsTest, IPSubnetOfIPAddress) {
  EXPECT_EQ(isSubnetOfIP("1.2.3.128/26", "1.2.3.129"), true);
  EXPECT_EQ(isSubnetOfIP("64:fa9b::17/64", "64:ffff::17"), false);
}*/

TEST_F(IPAddressFunctionsTest, IPSubnetOfIPPrefix) {
  EXPECT_EQ(isSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.144/30"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/64", "64:ffff::17/64"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/32", "64:ffff::17/24"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ffff::17/24", "64:ff9b::17/32"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.131/26"), true);
}

TEST_F(IPAddressFunctionsTest, IPSubnetOfPrestoTests) {
  EXPECT_EQ(isSubnetOfIP("1.2.3.128/26", "1.2.3.129"), true);
  EXPECT_EQ(isSubnetOfIP("1.2.3.128/26", "1.2.5.1"), false);
  EXPECT_EQ(isSubnetOfIP("1.2.3.128/32", "1.2.3.128"), true);
  EXPECT_EQ(isSubnetOfIP("1.2.3.128/0", "192.168.5.1"), true);
  EXPECT_EQ(isSubnetOfIP("64:ff9b::17/64", "64:ff9b::ffff:ff"), true);
  EXPECT_EQ(isSubnetOfIP("64:ff9b::17/64", "64:ffff::17"), false);

  EXPECT_EQ(isSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.144/30"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("1.2.3.128/26", "1.2.5.1/30"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("1.2.3.128/26", "1.2.3.128/26"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/64", "64:ff9b::ff:25/80"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/64", "64:ffff::17/64"), false);
  EXPECT_EQ(
      isSubnetOfIPPrefix("2804:431:b000::/37", "2804:431:b000::/38"), true);
  EXPECT_EQ(
      isSubnetOfIPPrefix("2804:431:b000::/38", "2804:431:b000::/37"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("170.0.52.0/22", "170.0.52.0/24"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("170.0.52.0/24", "170.0.52.0/22"), false);
}
} // namespace

} // namespace facebook::velox::functions::prestosql
                                                    