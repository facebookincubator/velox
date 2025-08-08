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
  BigintEnumTypeTest() {
    registerBigintEnumType();
  }
};

TEST_F(BigintEnumTypeTest, basic) {
  std::string enumName = "test.enum.mood";
  std::unordered_map<std::string, int64_t> enumMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  LongEnumParameter longEnumParameter(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(longEnumParameter)};
  ASSERT_STREQ(BIGINT_ENUM(typeParameters)->name(), "BIGINT_ENUM");
  ASSERT_STREQ(BIGINT_ENUM(typeParameters)->kindName(), "BIGINT");
  ASSERT_EQ(BIGINT_ENUM(typeParameters)->enumName(), "test.enum.mood");
  ASSERT_EQ(BIGINT_ENUM(typeParameters)->parameters().size(), 1);
  ASSERT_EQ(
      BIGINT_ENUM(typeParameters)->toString(),
      "test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0})");

  ASSERT_TRUE(hasType("BIGINT_ENUM"));
  ASSERT_EQ(
      getType("BIGINT_ENUM", typeParameters), BIGINT_ENUM(typeParameters));
}

TEST_F(BigintEnumTypeTest, serde) {
  std::string enumName = "test.enum.mood";
  std::unordered_map<std::string, int64_t> enumMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  LongEnumParameter longEnumParameter(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(longEnumParameter)};
  testTypeSerde(BIGINT_ENUM(typeParameters));
}

} // namespace facebook::velox::test
