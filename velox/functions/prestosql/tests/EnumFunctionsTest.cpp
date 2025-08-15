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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/BigintEnumRegistration.h"
#include "velox/functions/prestosql/types/BigintEnumType.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::functions::test;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class EnumFunctionsTest : public FunctionBaseTest {
 protected:
  void SetUp() override {
    FunctionBaseTest::SetUp();
    registerBigintEnumType();
  }
};

TEST_F(EnumFunctionsTest, enum_key) {
  std::string enumName = "test.enum.mood";
  std::unordered_map<std::string, int64_t> enumMap = {
      {"CURIOUS", -2}, {"HAPPY", 0}};
  LongEnumParameter longEnumParameter(enumName, enumMap);
  const std::vector<TypeParameter>& typeParameters = {
      TypeParameter(longEnumParameter)};
  auto bigintEnum = BIGINT_ENUM(typeParameters);

  auto expected =
      makeNullableFlatVector<StringView>({"HAPPY", "CURIOUS", std::nullopt});
  auto result = evaluate(
      "enum_key(c0)",
      makeRowVector({makeNullableFlatVector<int64_t>(
          {0, -2, std::nullopt}, bigintEnum)}));
  assertEqualVectors(expected, result);

  VELOX_ASSERT_THROW(
      evaluateOnce<std::string>(
          "enum_key(c0)",
          makeRowVector({makeNullableFlatVector<int64_t>({1}, bigintEnum)})),
      "Value '1' not in enum 'BigintEnum'");

  VELOX_ASSERT_THROW(
      evaluateOnce<std::string>(
          "enum_key(c0)",
          makeRowVector({makeNullableFlatVector<int64_t>({1}, BIGINT())})),
      "Scalar function signature is not supported: enum_key(BIGINT). Supported signatures: (bigint_enum(enumParameters)) -> varchar.");

  VELOX_ASSERT_THROW(
      evaluateOnce<std::string>(
          "enum_key(c0, 2)",
          makeRowVector({makeNullableFlatVector<int64_t>({1}, bigintEnum)})),
      "Scalar function signature is not supported: enum_key(test.enum.mood:BigintEnum(test.enum.mood{\"CURIOUS\": -2, \"HAPPY\": 0}), BIGINT). Supported signatures: (bigint_enum(enumParameters)) -> varchar.");
}
} // namespace
} // namespace facebook::velox::functions
