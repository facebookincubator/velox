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

namespace facebook::velox::functions::sparksql::test {
namespace {

class AssertNotNullTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  void testAssertNotNull(
      const std::vector<std::optional<T>>& input,
      const std::vector<T>& expected) {
    auto inputVector = makeNullableFlatVector<T>(input);
    auto result = evaluate("assert_not_null(c0)", makeRowVector({inputVector}));
    auto expectedVector = makeFlatVector<T>(expected);
    velox::test::assertEqualVectors(expectedVector, result);
  }
};

TEST_F(AssertNotNullTest, integer) {
  testAssertNotNull<int32_t>({1, 2, 3}, {1, 2, 3});
}

TEST_F(AssertNotNullTest, bigint) {
  testAssertNotNull<int64_t>({10, 20, 30}, {10, 20, 30});
}

TEST_F(AssertNotNullTest, varchar) {
  testAssertNotNull<StringView>(
      {StringView("hello"), StringView("world")},
      {StringView("hello"), StringView("world")});
}

TEST_F(AssertNotNullTest, boolean) {
  testAssertNotNull<bool>({true, false, true}, {true, false, true});
}

TEST_F(AssertNotNullTest, nullInput) {
  auto input = makeNullableFlatVector<int32_t>({1, std::nullopt, 3});
  VELOX_ASSERT_THROW(
      evaluate("assert_not_null(c0)", makeRowVector({input})),
      "Null value appeared in non-nullable field");
}

TEST_F(AssertNotNullTest, allNulls) {
  auto input = makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  VELOX_ASSERT_THROW(
      evaluate("assert_not_null(c0)", makeRowVector({input})),
      "Null value appeared in non-nullable field");
}

TEST_F(AssertNotNullTest, complexType) {
  auto arrayVector = makeNullableArrayVector<int32_t>(
      std::vector<std::vector<std::optional<int32_t>>>{
          {{1, 2, 3}}, {{4, 5}}, {{6}}});
  auto row = makeRowVector(std::vector<VectorPtr>{arrayVector});
  auto result = evaluate("assert_not_null(c0)", row);
  velox::test::assertEqualVectors(arrayVector, result);
}

TEST_F(AssertNotNullTest, complexTypeWithNull) {
  auto arrayVector = makeNullableArrayVector<int32_t>(
      std::vector<std::optional<std::vector<std::optional<int32_t>>>>{
          {{{1, 2, 3}}}, std::nullopt, {{{6}}}});
  auto row = makeRowVector(std::vector<VectorPtr>{arrayVector});
  VELOX_ASSERT_THROW(
      evaluate("assert_not_null(c0)", row),
      "Null value appeared in non-nullable field");
}

TEST_F(AssertNotNullTest, customErrorMessage) {
  auto input = makeNullableFlatVector<int32_t>({1, std::nullopt, 3});
  VELOX_ASSERT_THROW(
      evaluate(
          "assert_not_null(c0, 'custom error message')",
          makeRowVector({input})),
      "custom error message");
}

TEST_F(AssertNotNullTest, nullErrorMessage) {
  // When errMsg constant is NULL, use default error message.
  auto input = makeNullableFlatVector<int32_t>({1, std::nullopt, 3});
  VELOX_ASSERT_THROW(
      evaluate(
          "assert_not_null(c0, cast(null as varchar))", makeRowVector({input})),
      "Null value appeared in non-nullable field");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
