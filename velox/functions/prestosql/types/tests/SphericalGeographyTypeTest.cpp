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

#include "velox/functions/prestosql/types/SphericalGeographyType.h"
#include "velox/functions/prestosql/types/SphericalGeographyRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class SphericalGeographyTypeTest : public testing::Test, public TypeTestBase {
 public:
  SphericalGeographyTypeTest() {
    registerSphericalGeographyType();
  }
};

TEST_F(SphericalGeographyTypeTest, basic) {
  ASSERT_EQ(SPHERICAL_GEOGRAPHY()->name(), "SPHERICALGEOGRAPHY");
  ASSERT_STREQ(SPHERICAL_GEOGRAPHY()->kindName(), "VARBINARY");
  ASSERT_TRUE(SPHERICAL_GEOGRAPHY()->parameters().empty());
  ASSERT_EQ(SPHERICAL_GEOGRAPHY()->toString(), "SPHERICALGEOGRAPHY");

  ASSERT_TRUE(hasType("SPHERICALGEOGRAPHY"));
  ASSERT_EQ(*getType("SPHERICALGEOGRAPHY", {}), *SPHERICAL_GEOGRAPHY());

  ASSERT_FALSE(SPHERICAL_GEOGRAPHY()->isOrderable());
}

TEST_F(SphericalGeographyTypeTest, serde) {
  testTypeSerde(SPHERICAL_GEOGRAPHY());
}
} // namespace facebook::velox::test
