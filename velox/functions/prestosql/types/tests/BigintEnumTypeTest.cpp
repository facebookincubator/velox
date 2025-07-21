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
  std::string enumMap = "\"CURIOUSs\": -2, \"HAPPY\": 0";
  std::string enumName2 = "someEnumType";
  std::string enumMap2 = "\"CURIOUSs\": -2, \"HAPPY\": 3";
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(enumName), TypeParameter(enumMap)};
  const std::vector<TypeParameter>& typeParameters2 = {
      TypeParameter(enumName2), TypeParameter(enumMap)};
  ASSERT_STREQ(BIGINT_ENUM(typeParameters)->name(), "BIGINT_ENUM");
  ASSERT_STREQ(BIGINT_ENUM(typeParameters)->kindName(), "BIGINT");
  ASSERT_EQ(BIGINT_ENUM(typeParameters)->enumName(), "test.enum.mood");
  ASSERT_EQ(BIGINT_ENUM(typeParameters)->parameters().size(), 2);
  ASSERT_EQ(
      BIGINT_ENUM(typeParameters)->toString(),
      "test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 0})");

  ASSERT_TRUE(hasType("BIGINT_ENUM"));
  ASSERT_EQ(
      getType("BIGINT_ENUM", {TypeParameter(enumName), TypeParameter(enumMap)}),
      BIGINT_ENUM(typeParameters));
}

TEST_F(BigintEnumTypeTest, serde) {
  std::string enumName = "test.enum.mood";
  std::string enumMap = "\"CURIOUSs\": -2, \"HAPPY\": 0";
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(enumName), TypeParameter(enumMap)};
  testTypeSerde(BIGINT_ENUM(typeParameters));
}

TEST_F(BigintEnumTypeTest, parse) {
  // Missing quotations around key
  std::string enumName = "test.enum.mood";
  std::string invalidJsonMap = "CURIOUS: -2, HAPPY: 0";
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(enumName), TypeParameter(invalidJsonMap)};
  VELOX_ASSERT_THROW(
      BIGINT_ENUM(typeParameters),
      "Failed to parse enum values CURIOUS: -2, HAPPY: 0, json parse error on line 0 near `CURIOUS: -2, HAP': expected json value");

  // Missing value
  std::string invalidJsonMap2 = "\"CURIOUSs\": -2, \"HAPPY\"";
  const std::vector<TypeParameter>& typeParameters2 = {
      TypeParameter(enumName), TypeParameter(invalidJsonMap2)};
  VELOX_ASSERT_THROW(
      BIGINT_ENUM(typeParameters2),
      "Failed to parse enum values \"CURIOUSs\": -2, \"HAPPY\", json parse error on line 0 near `}': expected ':'");

  // Trailing comma
  std::string invalidJsonMap3 = "\"CURIOUSs\": -2, \"HAPPY\": 0,";
  const std::vector<TypeParameter>& typeParameters3 = {
      TypeParameter(enumName), TypeParameter(invalidJsonMap3)};
  VELOX_ASSERT_THROW(
      BIGINT_ENUM(typeParameters3),
      "Failed to parse enum values \"CURIOUSs\": -2, \"HAPPY\": 0,, json parse error on line 0 near `}': expected json value");
}

} // namespace facebook::velox::test
