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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {
class MapFromEntriesTest : public FunctionBaseTest {
 protected:
  // Evaluate an expression.
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result = evaluate<MapVector>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  // Evaluate an expression only, usually expect error thrown.
  void evaluateExprOnly(
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    evaluate(expression, makeRowVector(input));
  }
};
} // namespace

TEST_F(MapFromEntriesTest, intKeyAndIntValue) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, 11}), variant::row({2, 22}), variant::row({3, 33})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected =
      makeMapVector<int32_t, int32_t>({{{1, 11}, {2, 22}, {3, 33}}});
  testExpr(expected, "map_from_entries(C0)", {input});
}

TEST_F(MapFromEntriesTest, intKeyAndVarcharValue) {
  auto rowType = ROW({INTEGER(), VARCHAR()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, "red"}),
       variant::row({2, "blue"}),
       variant::row({3, "green"})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected = makeMapVector<int32_t, StringView>(
      {{{1, "red"_sv}, {2, "blue"_sv}, {3, "green"_sv}}});
  testExpr(expected, "map_from_entries(C0)", {input});
}

TEST_F(MapFromEntriesTest, varcharKeyAndIntValue) {
  auto rowType = ROW({VARCHAR(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({"red shiny car ahead", 1}),
       variant::row({"blue clear sky above", 2}),
       variant::row({"yellow rose flowers", 3})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected = makeMapVector<StringView, int32_t>(
      {{{"red shiny car ahead"_sv, 1},
        {"blue clear sky above"_sv, 2},
        {"yellow rose flowers"_sv, 3}}});
  testExpr(expected, "map_from_entries(C0)", {input});
}

TEST_F(MapFromEntriesTest, varcharKeyAndVarcharValue) {
  auto rowType = ROW({VARCHAR(), VARCHAR()});
  std::vector<std::vector<variant>> data = {
      {variant::row({"red", "red shiny car ahead"}),
       variant::row({"blue", "blue clear sky above"}),
       variant::row({"yellow", "yellow rose flowers"})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected = makeMapVector<StringView, StringView>(
      {{{"red"_sv, "red shiny car ahead"_sv},
        {"blue"_sv, "blue clear sky above"_sv},
        {"yellow"_sv, "yellow rose flowers"_sv}}});
  testExpr(expected, "map_from_entries(C0)", {input});
}

TEST_F(MapFromEntriesTest, valueIsNullToPass) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, variant::null(TypeKind::INTEGER)}),
       variant::row({2, 22}),
       variant::row({3, 33})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected =
      makeMapVector<int32_t, int32_t>({{{1, std::nullopt}, {2, 22}, {3, 33}}});
  testExpr(expected, "map_from_entries(C0)", {input});
}

TEST_F(MapFromEntriesTest, mapEntryIsNullToFail) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant(TypeKind::ROW), variant::row({1, 11})}};
  auto input = makeArrayOfRowVector(rowType, data);
  EXPECT_THROW(
      evaluateExprOnly("map_from_entries(C0)", {input}), VeloxUserError);
}

TEST_F(MapFromEntriesTest, keyIsNullToFail) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({variant::null(TypeKind::INTEGER), 0}),
       variant::row({1, 11})}};
  auto input = makeArrayOfRowVector(rowType, data);
  EXPECT_THROW(
      evaluateExprOnly("map_from_entries(C0)", {input}), VeloxUserError);
}

TEST_F(MapFromEntriesTest, duplicateIntKeyToFail) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, 10}), variant::row({1, 11}), variant::row({2, 22})}};
  auto input = makeArrayOfRowVector(rowType, data);
  EXPECT_THROW(
      evaluateExprOnly("map_from_entries(C0)", {input}), VeloxUserError);
}

TEST_F(MapFromEntriesTest, duplicateVarcharKeyToFail) {
  auto rowType = ROW({VARCHAR(), VARCHAR()});
  std::vector<std::vector<variant>> data = {
      {variant::row({"blue clear sky above", "blue"}),
       variant::row({"red shiny car ahead", "red"}),
       variant::row({"red shiny car ahead", "shiny"})}};
  auto input = makeArrayOfRowVector(rowType, data);
  EXPECT_THROW(
      evaluateExprOnly("map_from_entries(C0)", {input}), VeloxUserError);
}
