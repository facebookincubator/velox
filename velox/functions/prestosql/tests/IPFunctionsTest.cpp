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


#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "folly/IPAddress.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace {
class IPFunctionsTest : public FunctionBaseTest {
 protected:

  std::optional<std::string> getIPPrefix(const std::optional<std::string> input, std::optional<int> mask) {
    auto result = evaluateOnce<std::string>("ip_prefix(c0, c1)", input, mask);
    return result;
  }
};

TEST_F(IPFunctionsTest, IPv4) {

  EXPECT_EQ("10.135.23.0", getIPPrefix("10.135.23.12",8));
  EXPECT_EQ("10.132.0.0", getIPPrefix("10.135.23.12",18));
  EXPECT_EQ("0.0.0.0", getIPPrefix("10.135.23.12",32));
  EXPECT_EQ("10.135.23.12", getIPPrefix("10.135.23.12",0));

  EXPECT_ANY_THROW(getIPPrefix("12.483.09.1",8));
  EXPECT_ANY_THROW(getIPPrefix("10.135.23.12.12",8));
  EXPECT_ANY_THROW(getIPPrefix("10.135.23",8));
  EXPECT_ANY_THROW(getIPPrefix("12.135.23.12",-1));
  EXPECT_ANY_THROW(getIPPrefix("10.135.23.12",33));

}

TEST_F(IPFunctionsTest, IPv6) {

  EXPECT_EQ("2001:db8:85a3::", getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 48));
  EXPECT_EQ("2001:db8:85a3::", getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 52));
  EXPECT_EQ("::", getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 128));
  EXPECT_EQ("2001:db8:85a3:1:1:8a2e:370:7334", getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334", 0));

  EXPECT_ANY_THROW(getIPPrefix("q001:0db8:85a3:0001:0001:8a2e:0370:7334",8));
  EXPECT_ANY_THROW(getIPPrefix("2001:0db8:85a3:542e:0001:0001:8a2e:0370:7334",8));
  EXPECT_ANY_THROW(getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370",8));
  EXPECT_ANY_THROW(getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334",-1));
  EXPECT_ANY_THROW(getIPPrefix("2001:0db8:85a3:0001:0001:8a2e:0370:7334",140));
}



} // namespace
