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
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/functions/prestosql/types/IPPrefixRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class IPPrefixTypeTest : public testing::Test, public TypeTestBase {
 public:
  IPPrefixTypeTest() {
    registerIPPrefixType();
  }
};

TEST_F(IPPrefixTypeTest, basic) {
  ASSERT_STREQ(IPPREFIX()->name(), "IPPREFIX");
  ASSERT_STREQ(IPPREFIX()->kindName(), "ROW");
  ASSERT_TRUE(IPPREFIX()->parameters().empty());

  ASSERT_TRUE(hasType("IPPREFIX"));
  ASSERT_EQ(*getType("IPPREFIX", {}), *IPPREFIX());
}

TEST_F(IPPrefixTypeTest, serde) {
  testTypeSerde(IPPREFIX());
}

TEST_F(IPPrefixTypeTest, valueToString) {
  auto toString = [](std::string_view ip, int8_t prefixLength) {
    auto ipValue = ipaddress::tryGetIPv6asInt128FromString(ip).value();
    char buffer[IPPrefixType::kMaxStringSize];
    return std::string(
        IPPREFIX()->valueToString(ipValue, prefixLength, buffer));
  };

  EXPECT_EQ(toString("192.128.0.0", 9), "192.128.0.0/9");
  EXPECT_EQ(toString("192.168.255.255", 32), "192.168.255.255/32");
  EXPECT_EQ(toString("2001:db8::1", 32), "2001:db8::1/32");
  EXPECT_EQ(toString("::ffff:1.2.3.4", -128), "1.2.3.4/128");
  // Longest possible: full IPv6 + /128. Prefix length 128 is stored as -128
  // in int8_t (TINYINT); valueToString casts to uint8_t for formatting.
  EXPECT_EQ(
      toString("1234:5678:90ab:cdef:1234:5678:90ab:cdef", -128),
      "1234:5678:90ab:cdef:1234:5678:90ab:cdef/128");
}

} // namespace facebook::velox::test
