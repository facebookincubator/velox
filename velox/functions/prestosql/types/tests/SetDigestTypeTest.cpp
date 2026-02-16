/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/prestosql/types/SetDigestType.h"
#include "velox/functions/prestosql/types/SetDigestRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class SetDigestTypeTest : public testing::Test, public TypeTestBase {
 public:
  SetDigestTypeTest() {
    registerSetDigestType();
  }
};

TEST_F(SetDigestTypeTest, basic) {
  ASSERT_STREQ(SETDIGEST()->name(), "SETDIGEST");
  ASSERT_STREQ(SETDIGEST()->kindName(), "VARBINARY");
  ASSERT_TRUE(SETDIGEST()->parameters().empty());
  ASSERT_EQ(SETDIGEST()->toString(), "SETDIGEST");

  ASSERT_TRUE(hasType("SETDIGEST"));
  ASSERT_EQ(*getType("SETDIGEST", {}), *SETDIGEST());

  ASSERT_FALSE(SETDIGEST()->isOrderable());
}

TEST_F(SetDigestTypeTest, isSetDigestType) {
  ASSERT_TRUE(isSetDigestType(SETDIGEST()));
  ASSERT_FALSE(isSetDigestType(BIGINT()));
  ASSERT_FALSE(isSetDigestType(VARBINARY()));
}

TEST_F(SetDigestTypeTest, serde) {
  testTypeSerde(SETDIGEST());
}
} // namespace facebook::velox::test
