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

  // Different TypeParameters with same enumName and enumMap
  LongEnumParameter longEnumParameter2(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters2 = {
      TypeParameter(longEnumParameter2)};
  ASSERT_EQ(
      getType("BIGINT_ENUM", typeParameters),
      getType("BIGINT_ENUM", typeParameters2));
  EXPECT_TRUE(getType("BIGINT_ENUM", typeParameters)
                  ->equivalent(*getType("BIGINT_ENUM", typeParameters2)));

  // Different TypeParameters with different enumName, same enumMap
  std::string enumName3 = "test.enum.mood2";
  LongEnumParameter longEnumParameter3(enumName3, enumMap);
  const std::vector<TypeParameter>& typeParameters3 = {
      TypeParameter(longEnumParameter3)};
  ASSERT_NE(
      getType("BIGINT_ENUM", typeParameters),
      getType("BIGINT_ENUM", typeParameters3));
  EXPECT_FALSE(getType("BIGINT_ENUM", typeParameters)
                   ->equivalent(*getType("BIGINT_ENUM", typeParameters3)));

  // Different TypeParameters with same enumName, different enumMap
  std::unordered_map<std::string, int64_t> enumMap4 = {
      {"CURIOUS", -2}, {"HAPPY", 0}, {"ANGRY", 1}};
  LongEnumParameter longEnumParameter4(enumName, enumMap4);
  const std::vector<TypeParameter>& typeParameters4 = {
      TypeParameter(longEnumParameter4)};
  ASSERT_NE(
      getType("BIGINT_ENUM", typeParameters),
      getType("BIGINT_ENUM", typeParameters4));
  EXPECT_FALSE(getType("BIGINT_ENUM", typeParameters)
                   ->equivalent(*getType("BIGINT_ENUM", typeParameters4)));

  // Type Parameter with duplicate value in the enum map.
  std::string enumName5 = "DuplicateValues";
  std::unordered_map<std::string, int64_t> enumMap5 = {
      {"HAPPY", 0}, {"SAD", 0}, {"ANGRY", 0}};
  LongEnumParameter longEnumParameter5(enumName5, enumMap5);
  const std::vector<TypeParameter>& typeParameters5 = {
      TypeParameter(longEnumParameter5)};
  VELOX_ASSERT_THROW(
      getType("BIGINT_ENUM", typeParameters5),
      "Invalid enum type DuplicateValues, contains duplicate value 0");

  // Different TypeParameters with same enumName and enumMap but in different
  // order
  std::unordered_map<std::string, int64_t> enumMapDifferentOrder = {
      {"HAPPY", 0}, {"CURIOUS", -2}};
  LongEnumParameter longEnumParameter6(enumName, enumMapDifferentOrder);
  const std::vector<TypeParameter>& typeParameters6 = {
      TypeParameter(longEnumParameter6)};
  ASSERT_EQ(
      getType("BIGINT_ENUM", typeParameters),
      getType("BIGINT_ENUM", typeParameters6));
  EXPECT_TRUE(getType("BIGINT_ENUM", typeParameters)
                  ->equivalent(*getType("BIGINT_ENUM", typeParameters2)));
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
