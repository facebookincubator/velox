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
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class TDigestTypeTest : public testing::Test, public TypeTestBase {
 public:
  TDigestTypeTest() {
    registerTDigestType();
  }
};

TEST_F(TDigestTypeTest, basic) {
  TypePtr doubleType = DOUBLE();
  ASSERT_STREQ(TDIGEST(doubleType)->name(), "TDIGEST");
  ASSERT_STREQ(TDIGEST(DOUBLE())->kindName(), "VARBINARY");
  ASSERT_TRUE(TDIGEST(DOUBLE())->parameters().empty());
  ASSERT_EQ(TDIGEST(DOUBLE())->toString(), "TDIGEST(DOUBLE)");

  ASSERT_TRUE(hasType("TDIGEST"));
  ASSERT_EQ(*getType("TDIGEST", {}), *TDIGEST(DOUBLE()));
}

TEST_F(TDigestTypeTest, serde) {
  testTypeSerde(TDIGEST(DOUBLE()));
}
} // namespace facebook::velox::test
