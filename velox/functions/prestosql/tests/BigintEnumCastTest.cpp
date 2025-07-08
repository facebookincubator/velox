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
    std::string enumMoodString =
        "test.enum.mood:BigintEnum(test.enum.mood{“CURIOUS”:-2, “HAPPY”:0})";
    std::string otherEnumString =
        "someEnumType:BigintEnum(someEnumType{“CURIOUS”:-2, “HAPPY”:0})";
    registerBigintEnumType(enumMoodString);
    registerBigintEnumType(otherEnumString);
  }
};

TEST_F(BigintEnumCastTest, ToBigintEnum) {
  std::string enumName = "TEST.ENUM.MOOD";
  std::string enumName2 = "SOMEENUMTYPE";

  // Cast base type to enum type
  testCast<int64_t, int64_t>(
      BIGINT(),
      BIGINT_ENUM(enumName),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to same enum type
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(enumName),
      BIGINT_ENUM(enumName),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast base type to enum type where the value does not exist in the enumn
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT(),
          BIGINT_ENUM(enumName),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "No value '1' in TEST.ENUM.MOOD");

  // Cast enum type to different enum type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(enumName),
          BIGINT_ENUM(enumName2),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "Cannot cast TEST.ENUM.MOOD to SOMEENUMTYPE.");

  // Cast varchar type to enum type
  VELOX_ASSERT_THROW(
      evaluateCast(
          VARCHAR(),
          BIGINT_ENUM(enumName),
          makeRowVector({makeNullableFlatVector<StringView>(
              {"a"_sv, "b"_sv, std::nullopt})})),
      "Cannot cast VARCHAR to TEST.ENUM.MOOD.");
}

TEST_F(BigintEnumCastTest, FromBigintEnum) {
  std::string enumName = "TEST.ENUM.MOOD";

  // Cast enum type to base type
  testCast<int64_t, int64_t>(
      BIGINT_ENUM(enumName),
      BIGINT(),
      {0, -2, std::nullopt},
      {0, -2, std::nullopt});

  // Cast enum type to varchar type
  VELOX_ASSERT_THROW(
      evaluateCast(
          BIGINT_ENUM(enumName),
          VARCHAR(),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {0, 1, std::nullopt}, BIGINT())})),
      "Cannot cast TEST.ENUM.MOOD to VARCHAR.");
}
