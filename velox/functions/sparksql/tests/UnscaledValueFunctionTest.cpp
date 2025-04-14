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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class UnscaledValueFunctionTest : public SparkFunctionBaseTest {};

TEST_F(UnscaledValueFunctionTest, unscaledValue) {
  auto testUnscaledValue = [&](const VectorPtr& input,
                               const VectorPtr& expected) {
    auto result = evaluate("unscaled_value(c0)", makeRowVector({input}));
    assertEqualVectors(expected, result);
  };

  auto flatInput =
      makeFlatVector<int64_t>({1000, 2000, -3000, -4000}, DECIMAL(18, 3));
  auto flatExpected = makeFlatVector<int64_t>({1000, 2000, -3000, -4000});
  auto nullableFlatInput =
      makeNullableFlatVector<int64_t>({0, std::nullopt}, DECIMAL(18, 3));
  auto nullableFlatExpected =
      makeNullableFlatVector<int64_t>({0, std::nullopt});

  auto constInput = makeConstant<int64_t>(1000, 4, DECIMAL(18, 3));
  auto constExpected = makeConstant<int64_t>(1000, 4);
  auto constNullInput = makeConstant<int64_t>(std::nullopt, 4, DECIMAL(18, 3));
  auto constNullExpected = makeConstant<int64_t>(std::nullopt, 4);

  auto indices = makeIndices(8, [](auto row) { return row % 4; });
  auto dictInput = wrapInDictionary(indices, 8, flatInput);
  auto dictExpected = wrapInDictionary(indices, 8, flatExpected);
  auto dictConstInput = wrapInDictionary(indices, 8, constNullInput);
  auto dictConstExpected = wrapInDictionary(indices, 8, constNullExpected);

  auto invalidInput = makeFlatVector<int64_t>({0, 0, 0, 0}, DECIMAL(20, 3));

  testUnscaledValue(flatInput, flatExpected);
  testUnscaledValue(nullableFlatInput, nullableFlatExpected);

  testUnscaledValue(constInput, constExpected);
  testUnscaledValue(constNullInput, constNullExpected);

  testUnscaledValue(dictInput, dictExpected);
  testUnscaledValue(dictConstInput, dictConstExpected);

  VELOX_ASSERT_USER_THROW(
      testUnscaledValue(invalidInput, flatExpected),
      "Expect short decimal type, but got: DECIMAL(20, 3)");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
