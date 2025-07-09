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
#include "velox/functions/prestosql/types/BigintEnumType.h"
#include "velox/functions/prestosql/types/BigintEnumRegistration.h"
#include "velox/functions/prestosql/types/tests/TypeTestBase.h"

namespace facebook::velox::test {

class BigintEnumTypeTest : public testing::Test, public TypeTestBase {
 protected:
  std::string enumName = "TEST.ENUM.MOOD";
  std::string enumMoodString =
      "test.enum.mood:BigintEnum(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0})";

  std::string otherEnumString =
      "someEnumType:BigintEnum(someEnumType{“CURIOUS”:-2, “HAPPY”:0})";

  std::string varcharEnumTypeString =
      "test.enum.mood:VarcharEnum(test.enum.mood{“CURIOUS”:“CURIOUS”, “HAPPY”:“HAPPY”})";
  std::string invalidFormatEnumString =
      "testNoColon(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0})";
  std::string invalidParenthesisEnumString =
      "test.enum.mood:BigintEnum(test.enum.moodCURIOUS”:-2, “HAPPY”:0})";

  BigintEnumTypeTest() {
    registerBigintEnumType(enumMoodString);
  }
};

TEST_F(BigintEnumTypeTest, basic) {
  ASSERT_STREQ(BIGINT_ENUM(enumName)->name(), "TEST.ENUM.MOOD");
  ASSERT_STREQ(BIGINT_ENUM(enumName)->kindName(), "BIGINT");
  ASSERT_TRUE(BIGINT_ENUM(enumName)->parameters().empty());
  ASSERT_EQ(BIGINT_ENUM(enumName)->toString(), "TEST.ENUM.MOOD");

  ASSERT_TRUE(hasType("TEST.ENUM.MOOD"));
  ASSERT_EQ(getType("TEST.ENUM.MOOD", {}), BIGINT_ENUM(enumName));

  VELOX_ASSERT_THROW(
      BIGINT_ENUM("TEST.ENUM.NOT_REGISTERED"),
      "Unregistered type: TEST.ENUM.NOT_REGISTERED");
}

TEST_F(BigintEnumTypeTest, serde) {
  testTypeSerde(BIGINT_ENUM(enumName));
}

TEST_F(BigintEnumTypeTest, parsing) {
  auto [enumName, enumMap] = BigintEnumType::parseTypeInfo(enumMoodString);
  ASSERT_EQ(enumName, "TEST.ENUM.MOOD");
  std::unordered_map<std::string, int64_t> expectedMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  ASSERT_EQ(enumMap, expectedMap);

  auto [otherName, otherMap] = BigintEnumType::parseTypeInfo(otherEnumString);
  ASSERT_EQ(otherName, "SOMEENUMTYPE");
  expectedMap = {{"CURIOUS", -2}, {"HAPPY", 0}};
  ASSERT_EQ(otherMap, expectedMap);

  // invalid types
  ASSERT_THROW(
      BigintEnumType::parseTypeInfo(varcharEnumTypeString),
      std::invalid_argument);
  ASSERT_THROW(
      BigintEnumType::parseTypeInfo(invalidFormatEnumString),
      std::invalid_argument);
  ASSERT_THROW(
      BigintEnumType::parseTypeInfo(invalidParenthesisEnumString),
      std::invalid_argument);

  // register invalid types
  VELOX_ASSERT_THROW(
      registerBigintEnumType(varcharEnumTypeString),
      "Failed to parse type test.enum.mood:VarcharEnum(test.enum.mood{“CURIOUS”:“CURIOUS”, “HAPPY”:“HAPPY”}), invalid type: VarcharEnum");
  VELOX_ASSERT_THROW(
      registerBigintEnumType(invalidFormatEnumString),
      "Failed to parse type testNoColon(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0}), malformed enum type");
  VELOX_ASSERT_THROW(
      registerBigintEnumType(invalidParenthesisEnumString),
      "Failed to parse type test.enum.mood:BigintEnum(test.enum.moodCURIOUS”:-2, “HAPPY”:0}), malformed enum type");
}
} // namespace facebook::velox::test
