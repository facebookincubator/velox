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
#include <optional>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/lib/CheckDuplicateKeys.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/tests/TestingDictionaryArrayElementsFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {
class MultimapFromEntriesTest : public FunctionBaseTest {
 protected:
  void testExpression(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  void testExpressionWithError(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const std::string& expectedError) {
    VELOX_ASSERT_THROW(
        evaluate(expression, makeRowVector(input)), expectedError);
  }
};
} // namespace

TEST_F(MultimapFromEntriesTest, intKeyAndVarcharValue) {
  const auto inputRowVector = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {
          {variant::row({1, "red"}),
           variant::row({2, "yellow"}),
           variant::row({1, "blue"})},
          {},
          {variant::row({3, "green"}), variant::row({3, "blue"})},
      });
  const auto keyVector = makeFlatVector<int32_t>({1, 2, 3});
  const auto valueVector = makeArrayVector<StringView>(
      {{"red", "blue"}, {"yellow"}, {"green", "blue"}});
  const auto expected = makeMapVector({0, 2, 2}, keyVector, valueVector);

  testExpression("multimap_from_entries(C0)", {inputRowVector}, expected);
  testExpression("try(multimap_from_entries(C0))", {inputRowVector}, expected);
}

TEST_F(MultimapFromEntriesTest, nullMapEntries) {
  const auto inputRowVector = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {
          {variant::row({3, "green"}),
           variant(TypeKind::ROW) /*null*/,
           variant::row({3, "blue"})},
          {variant::row({1, "red"}),
           variant::row({2, "yellow"}),
           variant::row({1, "blue"})},
          {variant(TypeKind::ROW) /*null*/},
      });
  const auto keyVector = makeFlatVector<int32_t>({1, 2});
  const auto valueVector =
      makeArrayVector<StringView>({{"red", "blue"}, {"yellow"}});
  const auto expected =
      makeMapVector({0, 0, 2}, keyVector, valueVector, {0, 2});

  testExpressionWithError(
      "multimap_from_entries(C0)",
      {inputRowVector},
      "map entry cannot be null");
  testExpression("try(multimap_from_entries(C0))", {inputRowVector}, expected);
}

TEST_F(MultimapFromEntriesTest, nullKeys) {
  const auto inputRowVector = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {
          {variant::row({3, "green"}),
           variant::row({variant::null(TypeKind::INTEGER), "blue"})},
          {variant::row({1, "red"}),
           variant::row({2, "yellow"}),
           variant::row({1, "blue"})},
          {variant::row({variant::null(TypeKind::INTEGER), "yellow"})},
      });
  const auto keyVector = makeFlatVector<int32_t>({1, 2});
  const auto valueVector =
      makeArrayVector<StringView>({{"red", "blue"}, {"yellow"}});
  const auto expected =
      makeMapVector({0, 0, 2}, keyVector, valueVector, {0, 2});

  testExpressionWithError(
      "multimap_from_entries(C0)", {inputRowVector}, "map key cannot be null");
  testExpression("try(multimap_from_entries(C0))", {inputRowVector}, expected);
}

TEST_F(MultimapFromEntriesTest, nullValues) {
  const auto inputRowVector = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {
          {variant::row({1, "red"}),
           variant::row({2, "yellow"}),
           variant::row({1, variant::null(TypeKind::VARCHAR)})},
          {},
          {variant::row({3, variant::null(TypeKind::VARCHAR)}),
           variant::row({3, "blue"})},
      });
  const auto keyVector = makeFlatVector<int32_t>({1, 2, 3});
  const auto valueVector = makeNullableArrayVector<StringView>(
      {{"red", std::nullopt}, {"yellow"}, {std::nullopt, "blue"}});
  const auto expected = makeMapVector({0, 2, 2}, keyVector, valueVector);

  testExpression("multimap_from_entries(C0)", {inputRowVector}, expected);
  testExpression("try(multimap_from_entries(C0))", {inputRowVector}, expected);
}

TEST_F(MultimapFromEntriesTest, constant) {
  const vector_size_t kConstantSize = 1'000;
  const auto singleRowVector = makeArrayOfRowVector(
      ROW({INTEGER(), VARCHAR()}),
      {
          {variant::row({1, "red"}),
           variant::row({2, "yellow"}),
           variant::row({1, "blue"}),
           variant::row({3, "green"}),
           variant::row({3, "blue"})},
      });
  const auto inputRowVector =
      BaseVector::wrapInConstant(kConstantSize, 0, singleRowVector);
  const auto keyVector = makeFlatVector<int32_t>({1, 2, 3});
  const auto valueVector = makeArrayVector<StringView>(
      {{"red", "blue"}, {"yellow"}, {"green", "blue"}});
  const auto singleRowMapVector = makeMapVector({0}, keyVector, valueVector);
  const auto expected =
      BaseVector::wrapInConstant(kConstantSize, 0, singleRowMapVector);

  testExpression("multimap_from_entries(C0)", {inputRowVector}, expected);
  testExpression("try(multimap_from_entries(C0))", {inputRowVector}, expected);
}
