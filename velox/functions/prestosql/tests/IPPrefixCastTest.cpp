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

class IPPrefixTypeTest : public functions::test::FunctionBaseTest {
 protected:
  std::optional<std::string> castToVarchar(
      const std::optional<std::string>& input) {
    auto result = evaluateOnce<std::string>(
        "cast(cast(c0 as ipprefix) as varchar)", input);
    return result;
  }

  std::optional<std::string> castToIpAddress(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(cast(cast(c0 as ipprefix) as ipaddress) as varchar)", input);
  }

  std::optional<std::string> castFromIPAddress(
      const std::optional<std::string>& input) {
    return evaluateOnce<std::string>(
        "cast(cast(cast(c0 as ipaddress) as ipprefix) as varchar)", input);
  }
};

TEST_F(IPPrefixTypeTest, invalidIPPrefix) {
  VELOX_ASSERT_THROW(
      castToVarchar("facebook.com/32"),
      "Cannot cast value to IPPREFIX: facebook.com");
  VELOX_ASSERT_THROW(
      castToVarchar("localhost/32"),
      "Cannot cast value to IPPREFIX: localhost");
  VELOX_ASSERT_THROW(
      castToVarchar("2001:db8::1::1/128"),
      "Cannot cast value to IPPREFIX: 2001:db8::1::1");
  VELOX_ASSERT_THROW(
      castToVarchar("2001:zxy::1::1/128"),
      "Cannot cast value to IPPREFIX: 2001:zxy::1::1");
  VELOX_ASSERT_THROW(
      castToVarchar("789.1.1.1/32"),
      "Cannot cast value to IPPREFIX: 789.1.1.1");
  VELOX_ASSERT_THROW(
      castToVarchar("192.1.1.1"), "Cannot cast value to IPPREFIX: 192.1.1.1");
  VELOX_ASSERT_THROW(
      castToVarchar("192.1.1.1/128"),
      "Cannot cast value to IPPREFIX: 192.1.1.1/128");
  VELOX_ASSERT_THROW(
      castToVarchar("192.1.1.1/-1"),
      "Cannot cast value to IPPREFIX: 192.1.1.1/-1");
  VELOX_ASSERT_THROW(
      castToVarchar("::ffff:ffff:ffff/33"),
      "Cannot cast value to IPPREFIX: ::ffff:ffff:ffff/33");
  VELOX_ASSERT_THROW(
      castToVarchar("::ffff:ffff:ffff/-1"),
      "Cannot cast value to IPPREFIX: ::ffff:ffff:ffff/-1");
  VELOX_ASSERT_THROW(
      castToVarchar("::/129"), "Cannot cast value to IPPREFIX: ::/129");
  VELOX_ASSERT_THROW(
      castToVarchar("::/-1"), "Cannot cast value to IPPREFIX: ::/-1");
}

TEST_F(IPPrefixTypeTest, castFromIpAddress) {
  EXPECT_EQ(castFromIPAddress(std::nullopt), std::nullopt);
  EXPECT_EQ(castFromIPAddress("1.2.3.4"), "1.2.3.4/32");
  EXPECT_EQ(castFromIPAddress("::ffff:1.2.3.4"), "1.2.3.4/32");
  EXPECT_EQ(castFromIPAddress("::ffff:102:304"), "1.2.3.4/32");
  EXPECT_EQ(castFromIPAddress("192.168.0.0"), "192.168.0.0/32");
  EXPECT_EQ(
      castFromIPAddress("2001:0db8:0000:0000:0000:ff00:0042:8329"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(castFromIPAddress("2001:db8:0:0:1:0:0:1"), "2001:db8::1:0:0:1/128");
  EXPECT_EQ(castFromIPAddress("::1"), "::1/128");
  EXPECT_EQ(
      castFromIPAddress("2001:db8::ff00:42:8329"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(castFromIPAddress("2001:db8::"), "2001:db8::/128");
}

TEST_F(IPPrefixTypeTest, castToIpAddress) {
  EXPECT_EQ(castToIpAddress(std::nullopt), std::nullopt);
  EXPECT_EQ(castToIpAddress("1.2.3.4/32"), "1.2.3.4");
  EXPECT_EQ(castToIpAddress("1.2.3.4/24"), "1.2.3.0");
  EXPECT_EQ(castToIpAddress("::1/128"), "::1");
  EXPECT_EQ(
      castToIpAddress("2001:db8::ff00:42:8329/128"), "2001:db8::ff00:42:8329");
  EXPECT_EQ(castToIpAddress("2001:db8::ff00:42:8329/64"), "2001:db8::");
}

TEST_F(IPPrefixTypeTest, castToVarchar) {
  EXPECT_EQ(castToVarchar("::ffff:1.2.3.4/24"), "1.2.3.0/24");
  EXPECT_EQ(castToVarchar("192.168.0.0/24"), "192.168.0.0/24");
  EXPECT_EQ(castToVarchar("255.2.3.4/0"), "0.0.0.0/0");
  EXPECT_EQ(castToVarchar("255.2.3.4/1"), "128.0.0.0/1");
  EXPECT_EQ(castToVarchar("255.2.3.4/2"), "192.0.0.0/2");
  EXPECT_EQ(castToVarchar("255.2.3.4/4"), "240.0.0.0/4");
  EXPECT_EQ(castToVarchar("1.2.3.4/8"), "1.0.0.0/8");
  EXPECT_EQ(castToVarchar("1.2.3.4/16"), "1.2.0.0/16");
  EXPECT_EQ(castToVarchar("1.2.3.4/24"), "1.2.3.0/24");
  EXPECT_EQ(castToVarchar("1.2.3.255/25"), "1.2.3.128/25");
  EXPECT_EQ(castToVarchar("1.2.3.255/26"), "1.2.3.192/26");
  EXPECT_EQ(castToVarchar("1.2.3.255/28"), "1.2.3.240/28");
  EXPECT_EQ(castToVarchar("1.2.3.255/30"), "1.2.3.252/30");
  EXPECT_EQ(castToVarchar("1.2.3.255/32"), "1.2.3.255/32");
  EXPECT_EQ(
      castToVarchar("2001:0db8:0000:0000:0000:ff00:0042:8329/128"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(
      castToVarchar("2001:db8::ff00:42:8329/128"),
      "2001:db8::ff00:42:8329/128");
  EXPECT_EQ(castToVarchar("2001:db8:0:0:1:0:0:1/128"), "2001:db8::1:0:0:1/128");
  EXPECT_EQ(castToVarchar("2001:db8:0:0:1::1/128"), "2001:db8::1:0:0:1/128");
  EXPECT_EQ(castToVarchar("2001:db8::1:0:0:1/128"), "2001:db8::1:0:0:1/128");
  EXPECT_EQ(
      castToVarchar("2001:DB8::FF00:ABCD:12EF/128"),
      "2001:db8::ff00:abcd:12ef/128");
  EXPECT_EQ(castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/0"), "::/0");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/1"), "8000::/1");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/2"), "c000::/2");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/4"), "f000::/4");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/8"), "ff00::/8");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/16"), "ffff::/16");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/32"),
      "ffff:ffff::/32");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/48"),
      "ffff:ffff:ffff::/48");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/64"),
      "ffff:ffff:ffff:ffff::/64");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/80"),
      "ffff:ffff:ffff:ffff:ffff::/80");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/96"),
      "ffff:ffff:ffff:ffff:ffff:ffff::/96");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/112"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:0/112");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/120"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ff00/120");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/124"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0/124");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/126"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffc/126");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/127"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe/127");
  EXPECT_EQ(
      castToVarchar("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128"),
      "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128");
  EXPECT_THROW(castToVarchar("facebook.com/32"), VeloxUserError);
  EXPECT_THROW(castToVarchar("localhost/32"), VeloxUserError);
  EXPECT_THROW(castToVarchar("2001:db8::1::1/128"), VeloxUserError);
  EXPECT_THROW(castToVarchar("2001:zxy::1::1/128"), VeloxUserError);
  EXPECT_THROW(castToVarchar("789.1.1.1/32"), VeloxUserError);
  EXPECT_THROW(castToVarchar("192.1.1.1"), VeloxUserError);
  EXPECT_THROW(castToVarchar("192.1.1.1/128"), VeloxUserError);
}

TEST_F(IPPrefixTypeTest, ipv6CastEncodingBug) {
  // Regression for IPv6 CAST encoding bug where castToString used SimpleVector
  // for child vectors, breaking constant and dictionary encodings.
  auto ipv6Prefixes = makeFlatVector<std::string>({
      "2607:f0d0:1000::/38",
      "2001:db8::/32",
      "2001:db8::/48",
      "2804:431:b000::/38",
      "::/0",
      "2001:db8::ff00:42:8329/128",
  });
  auto rowVector = makeRowVector({ipv6Prefixes});
  auto expected = evaluate("cast(c0 as ipprefix)", rowVector);
  auto expectedAsVarchar =
      evaluate("cast(cast(c0 as ipprefix) as varchar)", rowVector);

  // Test that CAST VARCHAR -> IPPREFIX -> VARCHAR round-trips correctly
  // even with dictionary and constant encodings.
  auto typedExpr = makeTypedExpr(
      "cast(cast(c0 as ipprefix) as varchar)", asRowType(rowVector->type()));
  testEncodings(typedExpr, {ipv6Prefixes}, expectedAsVarchar);

  // Test IPPREFIX -> VARCHAR directly with encodings.
  // Need to create a row vector wrapping the IPPREFIX vector
  auto ipPrefixRow = makeRowVector({expected});
  auto typedExpr2 =
      makeTypedExpr("cast(c0 as varchar)", asRowType(ipPrefixRow->type()));
  auto flatVarcharResult = evaluate("cast(c0 as varchar)", ipPrefixRow);
  testEncodings(typedExpr2, {expected}, flatVarcharResult);

  // Test constant encoding explicitly: 2607:f0d0:1000::/38 should not become
  // ::/38
  auto constantInput = BaseVector::wrapInConstant(3, 0, ipv6Prefixes);
  auto constantRow = makeRowVector({constantInput});
  auto constResult =
      evaluate("cast(cast(c0 as ipprefix) as varchar)", constantRow);
  auto expectedConst = BaseVector::wrapInConstant(3, 0, expectedAsVarchar);
  velox::test::assertEqualVectors(expectedConst, constResult);
}

} // namespace facebook::velox::functions::prestosql
