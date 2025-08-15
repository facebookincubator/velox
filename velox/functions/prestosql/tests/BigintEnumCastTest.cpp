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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/prestosql/types/BigintEnumRegistration.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;

namespace {

class BigintEnumCastTest : public functions::test::CastBaseTest {
 protected:
  BigintEnumCastTest() {
    registerBigintEnumType();
  }
};

TEST_F(BigintEnumCastTest, toBigintEnum) {
  auto enumName = "test.enum.mood";
  std::unordered_map<std::string, int64_t> enumMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  LongEnumParameter longEnumParameter(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(longEnumParameter)};

  // Cast base type to enum type
  testCast<int64_t, int64_t>(
      BIGINT(),
      BIGINT_ENUM(typeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast TINYINT to enum type
  testCast<int8_t, int64_t>(
      TINYINT(),
      BIGINT_ENUM(typeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast SMALLINT to enum type
  testCast<int16_t, int64_t>(
      SMALLINT(),
      BIGINT_ENUM(typeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast INTEGER to enum type
  testCast<int32_t, int64_t>(
      INTEGER(),
      BIGINT_ENUM(typeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to same enum type
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(typeParameters),
      BIGINT_ENUM(typeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to same enum type where the values map is in a different
  // order
  std::unordered_map<std::string, int64_t> sameEnumMap = {
      {"HAPPY", 0}, {"CURIOUS", -2}};
  LongEnumParameter sameLongEnumParameter(enumName, sameEnumMap);
  const std::vector<TypeParameter>& sameTypeParameters = {
      TypeParameter(sameLongEnumParameter)};
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(typeParameters),
      BIGINT_ENUM(sameTypeParameters),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast base type to enum type where the value does not exist in the enum
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT(),
          BIGINT_ENUM(typeParameters),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "No value '1' in test.enum.mood");

  // Cast enum type to different enum type
  std::string enumName2 = "someEnumType";
  std::unordered_map<std::string, int64_t> enumMap2 = {
      {"CURIOUS", -2}, {"HAPPY", 3}};
  LongEnumParameter longEnumParameter2(enumName2, enumMap2);
  auto typeParameters2 = {TypeParameter(longEnumParameter2)};
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          BIGINT_ENUM(typeParameters2),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to someEnumType:BigintEnum(someEnumType{\"CURIOUS\": -2, \"HAPPY\": 3}).");

  // Cast enum type to different enum type with same name
  LongEnumParameter longEnumParameter3(enumName, enumMap2);
  auto typeParameters3 = {TypeParameter(longEnumParameter3)};
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          BIGINT_ENUM(typeParameters3),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 3})");

  // Cast varchar type to enum type
  VELOX_ASSERT_THROW(
      evaluateCast(
          VARCHAR(),
          BIGINT_ENUM(typeParameters),
          makeRowVector({makeNullableFlatVector<StringView>(
              {"a"_sv, "b"_sv, std::nullopt})})),
      "Cannot cast VARCHAR to test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}).");

  // Cast boolean type to enum type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BOOLEAN(),
          BIGINT_ENUM(typeParameters),
          makeRowVector(
              {makeNullableFlatVector<bool>({true, false, std::nullopt})})),
      "Cannot cast BOOLEAN to test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}).");
}

TEST_F(BigintEnumCastTest, fromBigintEnum) {
  auto enumName = "test.enum.mood";
  std::unordered_map<std::string, int64_t> enumMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  LongEnumParameter longEnumParameter(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(longEnumParameter)};
  // Cast enum type to base type
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(typeParameters),
      BIGINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to TINYINT type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          TINYINT(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to TINYINT.");

  // Cast enum type to SMALLINT type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          SMALLINT(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to SMALLINT.");

  // Cast enum type to INTEGER type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          INTEGER(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to INTEGER.");

  // Cast enum type to VARCHAR type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          VARCHAR(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to VARCHAR.");

  // Cast enum type to BOOLEAN type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          BOOLEAN(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}) to BOOLEAN.");
}
} // namespace
