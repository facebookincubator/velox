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
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions::prestosql {
class IPAddressFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  std::optional<std::string> ipPrefixFunctionFromIpAddress(
      const std::optional<std::string>& input,
      const std::optional<int64_t>& mask) {
    return evaluateOnce<std::string>(
        "cast(ip_prefix(cast(c0 as ipaddress), c1) as varchar)", input, mask);
  }

  std::optional<std::string> ipPrefixFromVarChar(
      const std::optional<std::string>& input,
      const std::optional<int64_t>& mask) {
    return evaluateOnce<std::string>(
        "cast(ip_prefix(c0, c1) as varchar)", input, mask);
  }

  std::optional<std::string> ipSubnetMin(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(ip_subnet_min(cast(c0 as ipprefix)) as varchar)", input);
  }

  std::optional<std::string> ipSubnetMax(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(ip_subnet_max(cast(c0 as ipprefix)) as varchar)", input);
  }

  std::optional<std::string> ipSubnetRangeMin(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[1] as varchar)", input);
  }

  std::optional<std::string> ipSubnetRangeMax(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(ip_subnet_range(cast(c0 as ipprefix))[2] as varchar)", input);
  }

  std::optional<bool> isSubnetOfIP(
      const std::optional<std::string>& prefix,
      const std::optional<std::string>& ip) {
    return evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipaddress))",
        prefix,
        ip);
  }

  std::optional<bool> isSubnetOfIPPrefix(
      const std::optional<std::string>& prefix,
      const std::optional<std::string>& prefix2) {
    return evaluateOnce<bool>(
        "is_subnet_of(cast(c0 as ipprefix), cast(c1 as ipprefix))",
        prefix,
        prefix2);
  }
};

TEST_F(IPAddressFunctionsTest, ipPrefixFromIpAddress) {
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("1.2.3.4", 24), "1.2.3.0/24");
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("1.2.3.4", 32), "1.2.3.4/32");
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("1.2.3.4", 0), "0.0.0.0/0");
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("64:ff9b::17", 64), "64:ff9b::/64");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress("64:ff9b::17", 127), "64:ff9b::16/127");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress("64:ff9b::17", 128), "64:ff9b::17/128");
  ASSERT_EQ(ipPrefixFunctionFromIpAddress("64:ff9b::17", 0), "::/0");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress(
          "2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48),
      "2001:db8:85a3::/48");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress(
          "2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52),
      "2001:db8:85a3::/52");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress(
          "2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128),
      "2001:db8:85a3:1:1:8a2e:370:7334/128");
  ASSERT_EQ(
      ipPrefixFunctionFromIpAddress(
          "2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0),
      "::/0");
  VELOX_ASSERT_THROW(
      ipPrefixFunctionFromIpAddress("::ffff:1.2.3.4", -1),
      "IPv4 subnet size must be in range [0, 32]");
  VELOX_ASSERT_THROW(
      ipPrefixFunctionFromIpAddress("::ffff:1.2.3.4", 33),
      "IPv4 subnet size must be in range [0, 32]");
  VELOX_ASSERT_THROW(
      ipPrefixFunctionFromIpAddress("64:ff9b::10", -1),
      "IPv6 subnet size must be in range [0, 128]");
  VELOX_ASSERT_THROW(
      ipPrefixFunctionFromIpAddress("64:ff9b::10", 129),
      "IPv6 subnet size must be in range [0, 128]");
}

TEST_F(IPAddressFunctionsTest, ipPrefixFromVarChar) {
  ASSERT_EQ(ipPrefixFromVarChar("1.2.3.4", 24), "1.2.3.0/24");
  ASSERT_EQ(ipPrefixFromVarChar("1.2.3.4", 32), "1.2.3.4/32");
  ASSERT_EQ(ipPrefixFromVarChar("1.2.3.4", 0), "0.0.0.0/0");
  ASSERT_EQ(ipPrefixFromVarChar("::ffff:1.2.3.4", 24), "1.2.3.0/24");
  ASSERT_EQ(ipPrefixFromVarChar("64:ff9b::17", 64), "64:ff9b::/64");
  ASSERT_EQ(ipPrefixFromVarChar("64:ff9b::17", 127), "64:ff9b::16/127");
  ASSERT_EQ(ipPrefixFromVarChar("64:ff9b::17", 128), "64:ff9b::17/128");
  ASSERT_EQ(ipPrefixFromVarChar("64:ff9b::17", 0), "::/0");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("::ffff:1.2.3.4", -1),
      "IPv4 subnet size must be in range [0, 32]");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("::ffff:1.2.3.4", 33),
      "IPv4 subnet size must be in range [0, 32]");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("64:ff9b::10", -1),
      "IPv6 subnet size must be in range [0, 128]");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("64:ff9b::10", 129),
      "IPv6 subnet size must be in range [0, 128]");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("localhost", 24),
      "Cannot cast value to IPADDRESS: localhost");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("64::ff9b::10", 24),
      "Cannot cast value to IPADDRESS: 64::ff9b::10");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("64:face:book::10", 24),
      "Cannot cast value to IPADDRESS: 64:face:book::10");
  VELOX_ASSERT_THROW(
      ipPrefixFromVarChar("123.456.789.012", 24),
      "Cannot cast value to IPADDRESS: 123.456.789.012");
}

TEST_F(IPAddressFunctionsTest, ipSubnetMin) {
  ASSERT_EQ(ipSubnetMin("1.2.3.4/24"), "1.2.3.0");
  ASSERT_EQ(ipSubnetMin("1.2.3.4/32"), "1.2.3.4");
  ASSERT_EQ(ipSubnetMin("64:ff9b::17/64"), "64:ff9b::");
  ASSERT_EQ(ipSubnetMin("64:ff9b::17/127"), "64:ff9b::16");
  ASSERT_EQ(ipSubnetMin("64:ff9b::17/128"), "64:ff9b::17");
  ASSERT_EQ(ipSubnetMin("64:ff9b::17/0"), "::");
  ASSERT_EQ(ipSubnetMin("192.64.1.1/9"), "192.0.0.0");
  ASSERT_EQ(ipSubnetMin("192.64.1.1/0"), "0.0.0.0");
  ASSERT_EQ(ipSubnetMin("192.64.1.1/1"), "128.0.0.0");
  ASSERT_EQ(ipSubnetMin("192.64.1.1/31"), "192.64.1.0");
  ASSERT_EQ(ipSubnetMin("192.64.1.1/32"), "192.64.1.1");
  ASSERT_EQ(
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"),
      "2001:db8:85a3::");
  ASSERT_EQ(ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"), "::");
  ASSERT_EQ(ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"), "::");
  ASSERT_EQ(
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"),
      "2001:db8:85a3:1:1:8a2e:370:7334");
  ASSERT_EQ(
      ipSubnetMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"),
      "2001:db8:85a3:1:1:8a2e:370:7334");
}

TEST_F(IPAddressFunctionsTest, ipSubnetMax) {
  ASSERT_EQ(ipSubnetMax("1.2.3.128/26"), "1.2.3.191");
  ASSERT_EQ(ipSubnetMax("192.168.128.4/32"), "192.168.128.4");
  ASSERT_EQ(ipSubnetMax("10.1.16.3/9"), "10.127.255.255");
  ASSERT_EQ(ipSubnetMax("2001:db8::16/127"), "2001:db8::17");
  ASSERT_EQ(ipSubnetMax("2001:db8::16/128"), "2001:db8::16");
  ASSERT_EQ(ipSubnetMax("64:ff9b::17/64"), "64:ff9b::ffff:ffff:ffff:ffff");
  ASSERT_EQ(ipSubnetMax("64:ff9b::17/72"), "64:ff9b::ff:ffff:ffff:ffff");
  ASSERT_EQ(
      ipSubnetMax("64:ff9b::17/0"), "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff");
  ASSERT_EQ(ipSubnetMax("192.64.1.1/9"), "192.127.255.255");
  ASSERT_EQ(ipSubnetMax("192.64.1.1/0"), "255.255.255.255");
  ASSERT_EQ(ipSubnetMax("192.64.1.1/1"), "255.255.255.255");
  ASSERT_EQ(ipSubnetMax("192.64.1.1/31"), "192.64.1.1");
  ASSERT_EQ(ipSubnetMax("192.64.1.1/32"), "192.64.1.1");
  ASSERT_EQ(
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"),
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff");
  ASSERT_EQ(
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff");
  ASSERT_EQ(
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"),
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff");
  ASSERT_EQ(
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"),
      "2001:db8:85a3:1:1:8a2e:370:7335");
  ASSERT_EQ(
      ipSubnetMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"),
      "2001:db8:85a3:1:1:8a2e:370:7334");
}

TEST_F(IPAddressFunctionsTest, IPSubnetRange) {
  ASSERT_EQ("192.0.0.0", ipSubnetRangeMin("192.64.1.1/9"));
  ASSERT_EQ("192.127.255.255", ipSubnetRangeMax("192.64.1.1/9"));
  ASSERT_EQ("0.0.0.0", ipSubnetRangeMin("192.64.1.1/0"));
  ASSERT_EQ("255.255.255.255", ipSubnetRangeMax("192.64.1.1/0"));
  ASSERT_EQ("128.0.0.0", ipSubnetRangeMin("192.64.1.1/1"));
  ASSERT_EQ("255.255.255.255", ipSubnetRangeMax("192.64.1.1/1"));
  ASSERT_EQ("192.64.1.0", ipSubnetRangeMin("192.64.1.1/31"));
  ASSERT_EQ("192.64.1.1", ipSubnetRangeMax("192.64.1.1/31"));
  ASSERT_EQ("192.64.1.1", ipSubnetRangeMin("192.64.1.1/32"));
  ASSERT_EQ("192.64.1.1", ipSubnetRangeMax("192.64.1.1/32"));
  ASSERT_EQ(
      "2001:db8:85a3::",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  ASSERT_EQ(
      "2001:db8:85a3:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/48"));
  ASSERT_EQ(
      "::", ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  ASSERT_EQ(
      "::", ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  ASSERT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  ASSERT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMin("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
  ASSERT_EQ(
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/0"));
  ASSERT_EQ(
      "7fff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/1"));
  ASSERT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7335",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/127"));
  ASSERT_EQ(
      "2001:db8:85a3:1:1:8a2e:370:7334",
      ipSubnetRangeMax("2001:0db8:85a3:0001:0001:8a2e:0370:7334/128"));
  ASSERT_EQ("1.2.3.0", ipSubnetRangeMin("1.2.3.160/24"));
  ASSERT_EQ("1.2.3.128", ipSubnetRangeMin("1.2.3.128/31"));
  ASSERT_EQ("10.1.6.46", ipSubnetRangeMin("10.1.6.46/32"));
  ASSERT_EQ("0.0.0.0", ipSubnetRangeMin("10.1.6.46/0"));
  ASSERT_EQ("64:ff9b::", ipSubnetRangeMin("64:ff9b::17/64"));
  ASSERT_EQ("64:ff9b::5200", ipSubnetRangeMin("64:ff9b::52f4/120"));
  ASSERT_EQ("64:ff9b::17", ipSubnetRangeMin("64:ff9b::17/128"));
  ASSERT_EQ("1.2.3.255", ipSubnetRangeMax("1.2.3.160/24"));
  ASSERT_EQ("1.2.3.129", ipSubnetRangeMax("1.2.3.128/31"));
  ASSERT_EQ("10.1.6.46", ipSubnetRangeMax("10.1.6.46/32"));
  ASSERT_EQ("255.255.255.255", ipSubnetRangeMax("10.1.6.46/0"));
  ASSERT_EQ("64:ff9b::ffff:ffff:ffff:ffff", ipSubnetRangeMax("64:ff9b::17/64"));
  ASSERT_EQ("64:ff9b::52ff", ipSubnetRangeMax("64:ff9b::52f4/120"));
  ASSERT_EQ("64:ff9b::17", ipSubnetRangeMax("64:ff9b::17/128"));
}

TEST_F(IPAddressFunctionsTest, IPSubnetOfIPPrefix) {
  EXPECT_EQ(isSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.144/30"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/64", "64:ffff::17/64"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ff9b::17/32", "64:ffff::17/24"), false);
  EXPECT_EQ(isSubnetOfIPPrefix("64:ffff::17/24", "64:ff9b::17/32"), true);
  EXPECT_EQ(isSubnetOfIPPrefix("192.168.3.131/26", "192.168.3.131/26"), true);

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

TEST_F(IPAddressFunctionsTest, IPPrefixCollapseTest) {
  auto makeIPPrefixFunc =
      [](const std::string& ipprefix) -> std::tuple<int128_t, int8_t> {
    auto ret = ipaddress::tryParseIpPrefixString(ipprefix);
    return std::make_tuple(ret->first, ret->second);
  };

  // Test cannot order by a single null element
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{std::nullopt}};
    VELOX_ASSERT_THROW(
        evaluate(
            "ip_prefix_collapse(c0)",
            makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())})),
        "ip_prefix_collapse does not support null elements");
  }

  // Test edge case where we have ipv4 between two ipv6 addresses after sorting
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("2200::/64"),
          makeIPPrefixFunc("1.1.1.1/32"),
          makeIPPrefixFunc("::1/32")}};
    VELOX_ASSERT_THROW(
        evaluate(
            "ip_prefix_collapse(c0)",
            makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())})),
        "All IPPREFIX elements must be the same IP version.");
  }

  // Test that single element case
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.1.0/24")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(data, IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {{makeIPPrefixFunc("2804:431:b000::/38")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(data, IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test a single null element with non-null elements
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.1.0/24"), std::nullopt}};
    VELOX_ASSERT_THROW(
        evaluate(
            "ip_prefix_collapse(c0)",
            makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())})),
        "ip_prefix_collapse does not support null elements");
  }

  // Test empty case
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(data, IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test different IP version case
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.0.0/22"),
          makeIPPrefixFunc("2409:4043:251a:d200::/56")}};
    VELOX_ASSERT_THROW(
        evaluate(
            "ip_prefix_collapse(c0)",
            makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())})),
        "All IPPREFIX elements must be the same IP version.");
  }

  // Basic IPV6 test
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("2620:10d:c090::/48"),
          makeIPPrefixFunc("2620:10d:c091::/48")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("2620:10d:c090::/47")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("2804:13c:4d6:e200::/56"),
         makeIPPrefixFunc("2804:13c:4d6:dd00::/56"),
         makeIPPrefixFunc("2804:13c:4d6:dc00::/56"),
         makeIPPrefixFunc("2804:13c:4d6:de00::/56")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("2804:13c:4d6:dc00::/55"),
             makeIPPrefixFunc("2804:13c:4d6:de00::/56"),
             makeIPPrefixFunc("2804:13c:4d6:e200::/56")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test collapse single IPv4s
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.0.1/32"),
          makeIPPrefixFunc("192.168.33.1/32")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.1/32"),
             makeIPPrefixFunc("192.168.33.1/32")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test collapse single IPv6s
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("2620:10d:c090:400::5:a869/128"),
          makeIPPrefixFunc("2620:10d:c091:400::5:a869/128")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("2620:10d:c090:400::5:a869/128"),
             makeIPPrefixFunc("2620:10d:c091:400::5:a869/128")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test same ipprefix
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.0.0/22")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/22")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }

  // Test overlapping prefixes
  {
    std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>> data =
        {{makeIPPrefixFunc("192.168.0.0/22"),
          makeIPPrefixFunc("192.168.0.0/24")}};
    auto ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    auto expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/22")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("192.168.0.0/22"),
         makeIPPrefixFunc("192.168.2.0/24")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/22")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("192.168.0.0/22"),
         makeIPPrefixFunc("192.168.3.0/24")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/22")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("10.0.64.0/18"),
         makeIPPrefixFunc("10.2.0.0/15"),
         makeIPPrefixFunc("10.0.0.0/8"),
         makeIPPrefixFunc("11.0.0.0/8"),
         makeIPPrefixFunc("172.168.32.0/20"),
         makeIPPrefixFunc("172.168.0.0/18")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("10.0.0.0/7"),
             makeIPPrefixFunc("172.168.0.0/18")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {{makeIPPrefixFunc("10.0.0.0/8"), makeIPPrefixFunc("10.0.0.0/7")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("10.0.0.0/7")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("192.168.0.0/24"),
         makeIPPrefixFunc("192.168.1.0/24")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/23")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);

    data = {
        {makeIPPrefixFunc("192.168.1.0/24"),
         makeIPPrefixFunc("192.168.0.0/24"),
         makeIPPrefixFunc("192.168.2.0/24"),
         makeIPPrefixFunc("192.168.9.0/24")}};
    ret = evaluate(
        "ip_prefix_collapse(c0)",
        makeRowVector({vectorMaker_.arrayOfRowVector(data, IPPREFIX())}));
    expected = vectorMaker_.arrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int128_t, int8_t>>>>{
            {makeIPPrefixFunc("192.168.0.0/23"),
             makeIPPrefixFunc("192.168.2.0/24"),
             makeIPPrefixFunc("192.168.9.0/24")}},
        IPPREFIX());
    ::facebook::velox::test::assertEqualVectors(ret, expected);
  }
}

} // namespace facebook::velox::functions::prestosql
