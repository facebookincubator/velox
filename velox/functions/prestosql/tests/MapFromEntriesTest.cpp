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
#include "velox/vector/tests/TestingDictionaryArrayElementsFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {
std::optional<std::vector<std::pair<int32_t, std::optional<int32_t>>>> O(
    const std::vector<std::pair<int32_t, std::optional<int32_t>>>& vector) {
  return std::make_optional(vector);
}
} // namespace

namespace {
class MapFromEntriesTest : public FunctionBaseTest {
 protected:
  /// Create an MAP vector of size 1 using specified 'keys' and 'values' vector.
  VectorPtr makeSingleRowMapVector(
      const VectorPtr& keys,
      const VectorPtr& values) {
    BufferPtr offsets = allocateOffsets(1, pool());
    BufferPtr sizes = allocateSizes(1, pool());
    sizes->asMutable<vector_size_t>()[0] = keys->size();

    return std::make_shared<MapVector>(
        pool(),
        MAP(keys->type(), values->type()),
        nullptr,
        1,
        offsets,
        sizes,
        keys,
        values);
  }

  void verifyMapFromEntries(
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected,
      const std::string& funcArg = "C0",
      bool wrappedWithTry = false) {
    const std::string expr = wrappedWithTry
        ? fmt::format("try(map_from_entries({}))", funcArg)
        : fmt::format("map_from_entries({})", funcArg);
    auto result = evaluate<MapVector>(expr, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  // Evaluate an expression only, usually expect error thrown.
  void evaluateExpr(
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    evaluate(expression, makeRowVector(input));
  }
};
} // namespace

TEST_F(MapFromEntriesTest, intKeyAndVarcharValue) {
  auto rowType = ROW({INTEGER(), VARCHAR()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, "red"}),
       variant::row({2, "blue"}),
       variant::row({3, "green"})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected = makeMapVector<int32_t, StringView>(
      {{{1, "red"_sv}, {2, "blue"_sv}, {3, "green"_sv}}});
  verifyMapFromEntries({input}, expected);
}

TEST_F(MapFromEntriesTest, nullMapEntries) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  {
    std::vector<std::vector<variant>> data = {
        {variant(TypeKind::ROW) /*null*/}, {variant::row({1, 11})}};
    auto input = makeArrayOfRowVector(rowType, data);
    VELOX_ASSERT_THROW(
        evaluateExpr("map_from_entries(C0)", {input}),
        "map entry cannot be null");
    auto expected =
        makeNullableMapVector<int32_t, int32_t>({std::nullopt, O({{1, 11}})});
    verifyMapFromEntries({input}, expected, "C0", true);
  }
  {
    // Create array(row(a,b)) where a, b sizes are 0 because all row(a, b)
    // values are null.
    std::vector<std::vector<variant>> data = {
        {variant(TypeKind::ROW),
         variant(TypeKind::ROW),
         variant(TypeKind::ROW) /*nulls*/},
        {variant(TypeKind::ROW) /*null*/}};
    auto input = makeArrayOfRowVector(rowType, data);
    auto rowInput = input->as<ArrayVector>();
    rowInput->elements()->as<RowVector>()->childAt(0)->resize(0);
    rowInput->elements()->as<RowVector>()->childAt(1)->resize(0);

    VELOX_ASSERT_THROW(
        evaluateExpr("map_from_entries(C0)", {input}),
        "map entry cannot be null");
    auto expected =
        makeNullableMapVector<int32_t, int32_t>({std::nullopt, std::nullopt});
    verifyMapFromEntries({input}, expected, "C0", true);
  }
}

TEST_F(MapFromEntriesTest, nullKeys) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({variant::null(TypeKind::INTEGER), 0})},
      {variant::row({1, 11})}};
  auto input = makeArrayOfRowVector(rowType, data);
  VELOX_ASSERT_THROW(
      evaluateExpr("map_from_entries(C0)", {input}), "map key cannot be null");
  auto expected =
      makeNullableMapVector<int32_t, int32_t>({std::nullopt, O({{1, 11}})});
  verifyMapFromEntries({input}, expected, "C0", true);
}

TEST_F(MapFromEntriesTest, duplicateKeys) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, 10}), variant::row({1, 11})}, {variant::row({2, 22})}};
  auto input = makeArrayOfRowVector(rowType, data);
  VELOX_ASSERT_THROW(
      evaluateExpr("map_from_entries(C0)", {input}),
      "Duplicate map keys (1) are not allowed");
  auto expected =
      makeNullableMapVector<int32_t, int32_t>({std::nullopt, O({{2, 22}})});
  verifyMapFromEntries({input}, expected, "C0", true);
}

TEST_F(MapFromEntriesTest, nullValues) {
  auto rowType = ROW({INTEGER(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, variant::null(TypeKind::INTEGER)}),
       variant::row({2, 22}),
       variant::row({3, 33})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected =
      makeMapVector<int32_t, int32_t>({{{1, std::nullopt}, {2, 22}, {3, 33}}});
  verifyMapFromEntries({input}, expected);
}

TEST_F(MapFromEntriesTest, constant) {
  const vector_size_t kConstantSize = 1'000;
  auto rowType = ROW({VARCHAR(), INTEGER()});
  std::vector<std::vector<variant>> data = {
      {variant::row({"red", 1}),
       variant::row({"blue", 2}),
       variant::row({"green", 3})},
      {variant::row({"red shiny car ahead", 4}),
       variant::row({"blue clear sky above", 5})},
      {variant::row({"r", 11}),
       variant::row({"g", 22}),
       variant::row({"b", 33})}};
  auto input = makeArrayOfRowVector(rowType, data);

  auto evaluateConstant = [&](vector_size_t row, const VectorPtr& vector) {
    return evaluate(
        "map_from_entries(C0)",
        makeRowVector(
            {BaseVector::wrapInConstant(kConstantSize, row, vector)}));
  };

  auto result = evaluateConstant(0, input);
  auto expected = BaseVector::wrapInConstant(
      kConstantSize,
      0,
      makeSingleRowMapVector(
          makeFlatVector<StringView>({"red"_sv, "blue"_sv, "green"_sv}),
          makeFlatVector<int32_t>({1, 2, 3})));
  test::assertEqualVectors(expected, result);

  result = evaluateConstant(1, input);
  expected = BaseVector::wrapInConstant(
      kConstantSize,
      0,
      makeSingleRowMapVector(
          makeFlatVector<StringView>(
              {"red shiny car ahead"_sv, "blue clear sky above"_sv}),
          makeFlatVector<int32_t>({4, 5})));
  test::assertEqualVectors(expected, result);

  result = evaluateConstant(2, input);
  expected = BaseVector::wrapInConstant(
      kConstantSize,
      0,
      makeSingleRowMapVector(
          makeFlatVector<StringView>({"r"_sv, "g"_sv, "b"_sv}),
          makeFlatVector<int32_t>({11, 22, 33})));
  test::assertEqualVectors(expected, result);
}

TEST_F(MapFromEntriesTest, dictionaryEncodedElementsInFlat) {
  exec::registerVectorFunction(
      "testing_dictionary_array_elements",
      test::TestingDictionaryArrayElementsFunction::signatures(),
      std::make_unique<test::TestingDictionaryArrayElementsFunction>());

  auto rowType = ROW({INTEGER(), VARCHAR()});
  std::vector<std::vector<variant>> data = {
      {variant::row({1, "red"}),
       variant::row({2, "blue"}),
       variant::row({3, "green"})}};
  auto input = makeArrayOfRowVector(rowType, data);
  auto expected = makeMapVector<int32_t, StringView>(
      {{{1, "red"_sv}, {2, "blue"_sv}, {3, "green"_sv}}});
  verifyMapFromEntries(
      {input}, expected, "testing_dictionary_array_elements(C0)");
}

TEST_F(MapFromEntriesTest, outputSizeIsBoundBySelectedRows) {
  // This test makes sure that map_from_entries output vector size is
  // `rows.end()` instead of `rows.size()`.

  auto rowType = ROW({INTEGER(), INTEGER()});
  core::QueryConfig config({});
  auto function =
      exec::getVectorFunction("map_from_entries", {ARRAY(rowType)}, {}, config);

  std::vector<std::vector<variant>> data = {
      {variant::row({1, 11}), variant::row({2, 22}), variant::row({3, 33})},
      {variant::row({4, 44}), variant::row({5, 55})},
      {variant::row({6, 66})}};
  auto array = makeArrayOfRowVector(rowType, data);

  auto rowVector = makeRowVector({array});

  // Only the first 2 rows selected.
  SelectivityVector rows(2);
  // This is larger than input array size but rows beyond the input vector size
  // are not selected.
  rows.resize(1000, false);

  ASSERT_EQ(rows.size(), 1000);
  ASSERT_EQ(rows.end(), 2);
  ASSERT_EQ(array->size(), 3);

  auto typedExpr =
      makeTypedExpr("map_from_entries(c0)", asRowType(rowVector->type()));
  std::vector<VectorPtr> results(1);

  exec::ExprSet exprSet({typedExpr}, &execCtx_);
  exec::EvalCtx evalCtx(&execCtx_, &exprSet, rowVector.get());
  exprSet.eval(rows, evalCtx, results);

  ASSERT_EQ(results[0]->size(), 2);
}
