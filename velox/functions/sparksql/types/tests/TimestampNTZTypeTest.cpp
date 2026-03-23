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
#include "velox/functions/sparksql/types/TimestampNTZType.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"

namespace facebook::velox::functions::sparksql::test {

class TimestampNTZTypeTest : public testing::Test {
 public:
  TimestampNTZTypeTest() {
    registerTimestampNTZType();
  }
};

TEST_F(TimestampNTZTypeTest, basic) {
  ASSERT_STREQ(TIMESTAMP_NTZ()->name(), "TIMESTAMP_NTZ");
  ASSERT_STREQ(TIMESTAMP_NTZ()->kindName(), "BIGINT");
  ASSERT_TRUE(TIMESTAMP_NTZ()->parameters().empty());
  ASSERT_EQ(TIMESTAMP_NTZ()->toString(), "TIMESTAMP_NTZ");

  ASSERT_TRUE(hasType("TIMESTAMP_NTZ"));
  ASSERT_EQ(*getType("TIMESTAMP_NTZ", {}), *TIMESTAMP_NTZ());
}

TEST_F(TimestampNTZTypeTest, serde) {
  Type::registerSerDe();
  auto type = TIMESTAMP_NTZ();

  auto copy = velox::ISerializable::deserialize<Type>(
      velox::ISerializable::serialize(type));

  ASSERT_EQ(type->toString(), copy->toString());
  ASSERT_EQ(*type, *copy);
}

} // namespace facebook::velox::functions::sparksql::test
