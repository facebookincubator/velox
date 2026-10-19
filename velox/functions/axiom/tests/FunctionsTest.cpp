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
#include "velox/functions/axiom/Register.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions::axiom {
namespace {

class FunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    FunctionBaseTest::SetUpTestCase();
    registerFunctions("axiom_");
  }
};

TEST_F(FunctionsTest, bigintAdd) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 5}),
      makeNullableFlatVector<int64_t>({2, 3, std::nullopt}),
  });
  velox::test::assertEqualVectors(
      makeNullableFlatVector<int64_t>({3, std::nullopt, std::nullopt}),
      evaluate("axiom_bigint_add(c0, c1)", data));

  VELOX_ASSERT_THROW(
      evaluate(
          "axiom_bigint_add(c0, 1)",
          makeRowVector({makeFlatVector<int64_t>(
              {std::numeric_limits<int64_t>::max()})})),
      "integer overflow: 9223372036854775807 + 1");
}

TEST_F(FunctionsTest, arrayAppend) {
  auto data = makeRowVector({
      makeArrayVectorFromJson<int64_t>({"[1, 2]", "[]", "null", "[3]"}),
      makeNullableFlatVector<int64_t>({3, 1, 5, std::nullopt}),
  });
  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[1, 2, 3]", "[1]", "null", "[3, null]"});
  velox::test::assertEqualVectors(
      expected, evaluate("axiom_array_append(c0, c1)", data));
}

TEST_F(FunctionsTest, arrayPrepend) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({0, 1, 5, std::nullopt}),
      makeArrayVectorFromJson<int64_t>({"[1, 2]", "[]", "null", "[3]"}),
  });
  auto expected = makeArrayVectorFromJson<int64_t>(
      {"[0, 1, 2]", "[1]", "null", "[null, 3]"});
  velox::test::assertEqualVectors(
      expected, evaluate("axiom_array_prepend(c0, c1)", data));
}

} // namespace
} // namespace facebook::velox::functions::axiom
