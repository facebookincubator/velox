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

class BigintEnumCastTest : public functions::test::CastBaseTest {
 protected:
  BigintEnumCastTest() {
    registerBigintEnumType();
  }
};

TEST_F(BigintEnumCastTest, ToBigintEnum) {
  auto enumName = "test.enum.mood";
  auto enumMap = "\"CURIOUSs\": -2, \"HAPPY\": 0";
  auto typeParameters = {TypeParameter(enumName), TypeParameter(enumMap)};

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

  // Cast base type to enum type where the value does not exist in the enum
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT(),
          BIGINT_ENUM(typeParameters),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "No value '1' in test.enum.mood");

  // Cast enum type to different enum type
  auto enumName2 = "someEnumType";
  auto enumMap2 = "\"CURIOUSs\": -2, \"HAPPY\": 3";
  auto typeParameters2 = {TypeParameter(enumName2), TypeParameter(enumMap2)};
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          BIGINT_ENUM(typeParameters2),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 0}) to someEnumType:BigintEnum(someEnumType{\"CURIOUSs\": -2, \"HAPPY\": 3}).");

  // Cast enum type to different enum type with same name
  auto typeParameters3 = {TypeParameter(enumName), TypeParameter(enumMap2)};
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          BIGINT_ENUM(typeParameters3),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT_ENUM(typeParameters))})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 0}) to test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 3})");

  // Cast varchar type to enum type
  VELOX_ASSERT_THROW(
      evaluateCast(
          VARCHAR(),
          BIGINT_ENUM(typeParameters),
          makeRowVector({makeNullableFlatVector<StringView>(
              {"a"_sv, "b"_sv, std::nullopt})})),
      "Cannot cast VARCHAR to test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 0}).");
}

TEST_F(BigintEnumCastTest, FromBigintEnum) {
  auto enumName = "test.enum.mood";
  auto enumMap = "\"CURIOUSs\": -2, \"HAPPY\": 0";
  auto typeParameters = {TypeParameter(enumName), TypeParameter(enumMap)};
  // Cast enum type to base type
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(typeParameters),
      BIGINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to TINYINT
  testCast<int64_t, int8_t>(
      BIGINT_ENUM(typeParameters),
      TINYINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to SMALLINT
  testCast<int64_t, int8_t>(
      BIGINT_ENUM(typeParameters),
      TINYINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to INTEGER
  testCast<int64_t, int8_t>(
      BIGINT_ENUM(typeParameters),
      TINYINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to varchar type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(typeParameters),
          VARCHAR(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "Cannot cast test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUSs\": -2, \"HAPPY\": 0}) to VARCHAR.");
}
