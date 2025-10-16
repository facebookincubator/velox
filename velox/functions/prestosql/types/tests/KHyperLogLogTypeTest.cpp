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

#include "velox/functions/prestosql/types/KHyperLogLogType.h"
#include "velox/functions/prestosql/types/KHyperLogLogRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class KHyperLogLogTypeTest : public testing::Test, public TypeTestBase {
 public:
  KHyperLogLogTypeTest() {
    registerKHyperLogLogType();
  }
};

TEST_F(KHyperLogLogTypeTest, basic) {
  ASSERT_STREQ(KHYPERLOGLOG()->name(), "KHYPERLOGLOG");
  ASSERT_EQ(KHYPERLOGLOG()->toString(), "KHYPERLOGLOG");
  ASSERT_FALSE(KHYPERLOGLOG()->isOrderable());
  ASSERT_TRUE(KHYPERLOGLOG()->equivalent(*KHYPERLOGLOG()));
  ASSERT_FALSE(KHYPERLOGLOG()->equivalent(*BIGINT()));

  ASSERT_TRUE(hasType("KHYPERLOGLOG"));
  ASSERT_EQ(*getType("KHYPERLOGLOG", {}), *KHYPERLOGLOG());
}

TEST_F(KHyperLogLogTypeTest, isKHyperLogLogType) {
  ASSERT_TRUE(isKHyperLogLogType(KHYPERLOGLOG()));
  ASSERT_FALSE(isKHyperLogLogType(BIGINT()));
  ASSERT_FALSE(isKHyperLogLogType(VARBINARY()));
}

TEST_F(KHyperLogLogTypeTest, serde) {
  testTypeSerde(KHYPERLOGLOG());
}

} // namespace facebook::velox::test
