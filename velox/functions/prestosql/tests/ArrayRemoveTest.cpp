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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

namespace {

class ArrayRemoveTest : public FunctionBaseTest {
 protected:
  void testExpr(
      const VectorPtr& expected,
      const std::vector<VectorPtr>& input) {
    auto result =
        evaluate<ArrayVector>("array_remove(C0, C1)", makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(ArrayRemoveTest, integer) {
  auto input = makeNullableArrayVector<int32_t>({
      {}, // No values.
      {1}, // Only one value.
      {1, 2, std::nullopt}, // Trim left.
      {2, std::nullopt, 1}, // Trim right.
      {1, 1, 1, 1, 1}, // Only search elements.
      {2, 3, 4, 5}, // No search element.
      {2, 3, 1, 1, 4, 5}, // Search element is in the middle.
      {1, 2, 1, 2, 1, 2}, // Alternating values.
      {std::nullopt, std::nullopt} // Nulls.
  });
  auto search = makeConstant(1, input->size());
  auto expected = makeNullableArrayVector<int32_t>(
      {{},
       {},
       {2, std::nullopt},
       {2, std::nullopt},
       {},
       {2, 3, 4, 5},
       {2, 3, 4, 5},
       {2, 2, 2},
       {std::nullopt, std::nullopt}});
  testExpr(expected, {input, search});
}

TEST_F(ArrayRemoveTest, boolean) {
  auto input = makeNullableArrayVector<bool>({
      {}, // No values.
      {true}, // Only one value.
      {true, false, std::nullopt}, // Trim left.
      {false, std::nullopt, true}, // Trim right.
      {true, true, true, true}, // Only search elements.
      {false, false}, // No search element.
      {false, false, true, false, false}, // Search element is in the middle.
      {false, true, false, true}, // Alternating values.
      {std::nullopt, std::nullopt} // Nulls.
  });
  auto search = makeConstant(true, input->size());
  auto expected = makeNullableArrayVector<bool>(
      {{},
       {},
       {false, std::nullopt},
       {false, std::nullopt},
       {},
       {false, false},
       {false, false, false, false},
       {false, false},
       {std::nullopt, std::nullopt}});
  testExpr(expected, {input, search});
}

TEST_F(ArrayRemoveTest, varchar) {
  using S = StringView;

  auto input = makeNullableArrayVector<S>({
      {}, // No values.
      {S("a")}, // Only one value.
      {S("a"), S("b"), std::nullopt}, // Trim left.
      {S("b"), std::nullopt, S("a")}, // Trim right.
      {S("a"), S("a"), S("a"), S("a")}, // Only search elements.
      {S("b"), S("b"), S("b")}, // No search element.
      {S("b"), S("c"), S("a"), S("d"), S("e")}, // Search element is in the
                                                // middle.
      {S("a"), S("b"), S("a"), S("b")}, // Alternating values.
      {std::nullopt, std::nullopt} // Nulls.
  });
  auto search = makeConstant("a", input->size());
  auto expected = makeNullableArrayVector<S>(
      {{},
       {},
       {S("b"), std::nullopt},
       {S("b"), std::nullopt},
       {},
       {S("b"), S("b"), S("b")},
       {S("b"), S("c"), S("d"), S("e")},
       {S("b"), S("b")},
       {std::nullopt, std::nullopt}});
  testExpr(expected, {input, search});
}

TEST_F(ArrayRemoveTest, array) {
  std::vector<std::optional<int32_t>> a{1, 2, 3};
  std::vector<std::optional<int32_t>> b{1};
  std::vector<std::optional<int32_t>> c{2, 3};

  auto input = makeNullableNestedArrayVector<int32_t>(
      {{{{a}, {b}, {c}}},
       {{{b}}},
       {{{b}, {b}, {b}}},
       {{{a}, {c}}},
       {{}},
       {{std::nullopt}}});
  auto search = makeConstantArray<int32_t>(input->size(), {1});
  auto expected = makeNullableNestedArrayVector<int32_t>({
      {{{a}, {c}}},
      {{}},
      {{}},
      {{{a}, {c}}},
      {{}},
      {{std::nullopt}},
  });

  testExpr(expected, {input, search});
}

TEST_F(ArrayRemoveTest, map) {
  using P = std::pair<int32_t, std::optional<int32_t>>;

  std::vector<P> a{P{1, 100}, P{2, 200}, P{3, 300}};
  std::vector<P> b{P{1, 100}};
  std::vector<P> c{P{2, 200}, P{3, 300}};
  std::vector<P> d{P{4, 400}};

  auto input = makeArrayOfMapVector<int32_t, int32_t>(
      {{},
       {{}},
       {a, b, c},
       {b, b, b},
       {a, c, d},
       {b, std::nullopt, b},
       {std::nullopt}});
  auto search = makeMapVector<int32_t, int32_t>({b, b, b, b, b, b, b});
  auto expected = makeArrayOfMapVector<int32_t, int32_t>(
      {{}, {{}}, {a, c}, {}, {a, c, d}, {std::nullopt}, {std::nullopt}});

  testExpr(expected, {input, search});
}

TEST_F(ArrayRemoveTest, row) {
  auto rowType = ROW({"col"}, {INTEGER()});

  auto input = makeArrayOfRowVector(
      rowType,
      {{},
       {variant(TypeKind::ROW), variant(TypeKind::ROW)},
       {variant(TypeKind::ROW), variant::row({1})},
       {variant::row({1}), variant::row({2}), variant::row({3})},
       {variant::row({1}), variant::row({1})},
       {variant::row({2})}});

  auto search = makeConstantRow(rowType, variant::row({1}), input->size());
  auto expected = makeArrayOfRowVector(
      rowType,
      {{},
       {variant(TypeKind::ROW), variant(TypeKind::ROW)},
       {variant(TypeKind::ROW)},
       {variant::row({2}), variant::row({3})},
       {},
       {variant::row({2})}});

  testExpr(expected, {input, search});
}

} // namespace
