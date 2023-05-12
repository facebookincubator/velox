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
#include "velox/expression/Expr.h"
#include "velox/functions/lib/SubscriptUtil.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;
using facebook::velox::functions::SubscriptImpl;

namespace {

class ElementAtTest : public FunctionBaseTest {
 protected:
  static const vector_size_t kVectorSize{1'000};

  template <typename T>
  void testElementAt(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters,
      std::function<T(vector_size_t /* row */)> expectedValueAt,
      std::function<bool(vector_size_t /* row */)> expectedNullAt = nullptr) {
    auto result =
        evaluate<SimpleVector<T>>(expression, makeRowVector(parameters));
    for (vector_size_t i = 0; i < parameters[0]->size(); ++i) {
      auto isNull =
          expectedNullAt ? expectedNullAt(i) : parameters[0]->isNullAt(i);
      EXPECT_EQ(isNull, result->isNullAt(i)) << "at " << i;
      if (!isNull) {
        EXPECT_EQ(expectedValueAt(i), result->valueAt(i)) << "at " << i;
      }
    }
  }

  template <typename T>
  void testVariableInputMap() {
    auto indicesVector = makeFlatVector<T>(
        kVectorSize, [](vector_size_t row) { return row % 7; });

    auto sizeAt = [](vector_size_t row) { return 7; };
    auto keyAt = [](vector_size_t idx) { return idx % 7; };
    auto valueAt = [](vector_size_t idx) { return idx / 7; };
    auto expectedValueAt = [](vector_size_t row) { return row; };
    auto mapVector = makeMapVector<T, T>(kVectorSize, sizeAt, keyAt, valueAt);
    testElementAt<T>(
        "element_at(C0, C1)", {mapVector, indicesVector}, expectedValueAt);
  }

  template <typename T = int64_t>
  std::optional<T> elementAtSimple(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters) {
    auto result =
        evaluate<SimpleVector<T>>(expression, makeRowVector(parameters));
    if (result->size() != 1) {
      throw std::invalid_argument(
          "elementAtSimple expects a single output row.");
    }
    if (result->isNullAt(0)) {
      return std::nullopt;
    }
    return result->valueAt(0);
  }

  // Create a simple vector containing a single map ([10=>10, 11=>11, 12=>12]).
  MapVectorPtr getSimpleMapVector() {
    auto keyAt = [](auto idx) { return idx + 10; };
    auto sizeAt = [](auto) { return 3; };
    auto mapValueAt = [](auto idx) { return idx + 10; };
    return makeMapVector<int64_t, int64_t>(1, sizeAt, keyAt, mapValueAt);
  }
};

template <>
void ElementAtTest::testVariableInputMap<StringView>() {
  auto toStr = [](size_t input) {
    return StringView(folly::sformat("str{}", input).c_str());
  };

  auto indicesVector = makeFlatVector<StringView>(
      kVectorSize, [&](vector_size_t row) { return toStr(row % 7); });

  auto sizeAt = [](vector_size_t row) { return 7; };
  auto keyAt = [&](vector_size_t idx) { return toStr(idx % 7); };
  auto valueAt = [&](vector_size_t idx) { return toStr(idx / 7); };
  auto expectedValueAt = [&](vector_size_t row) { return toStr(row); };
  auto mapVector = makeMapVector<StringView, StringView>(
      kVectorSize, sizeAt, keyAt, valueAt);
  testElementAt<StringView>(
      "element_at(C0, C1)", {mapVector, indicesVector}, expectedValueAt);
}

} // namespace

TEST_F(ElementAtTest, mapWithDictionaryKeys) {
  {
    auto keyIndices = makeIndices({6, 5, 4, 3, 2, 1, 0});
    auto keys = wrapInDictionary(
        keyIndices, makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6}));

    // values vector is [100, 200, 300, 400, 500, 600, 0].
    auto valuesIndices = makeIndices({1, 2, 3, 4, 5, 6, 0});
    auto values = wrapInDictionary(
        valuesIndices,
        makeFlatVector<int64_t>({0, 100, 200, 300, 400, 500, 600}));

    // map vector is [{6->100, 5->200, 4->300}, {3->400, 2->500, 1->600},
    // {0->0}].
    auto inputMap = makeMapVector({0, 3, 6}, keys, values);

    auto inputIndices = makeFlatVector<int64_t>({5, 2, 3});
    auto expected = makeNullableFlatVector<int64_t>({200, 500, std::nullopt});

    auto result =
        evaluate("element_at(c0, c1)", makeRowVector({inputMap, inputIndices}));
    test::assertEqualVectors(expected, result);
  }

  {
    auto result = evaluateOnce<int64_t>(
        "element_at(map(array_constructor(85,22,79,76,10,80,57,31),array_constructor(14,10,16,15,12,11,17,13)),85)",
        makeRowVector({}));
    ASSERT_EQ(result, 14);
  }
}

TEST_F(ElementAtTest, arrayWithDictionaryElements) {
  {
    auto elementsIndices = makeIndices({6, 5, 4, 3, 2, 1, 0});
    auto elements = wrapInDictionary(
        elementsIndices, makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6}));

    // array vector is [[6, 5, 4], [3, 2, 1], [0]].
    auto inputArray = makeArrayVector({0, 3, 6}, elements);
    auto inputIndices = makeFlatVector<int32_t>({3, -3, 1});
    auto expected = makeFlatVector<int64_t>({4, 3, 0});

    auto result = evaluate(
        "element_at(c0, c1)", makeRowVector({inputArray, inputIndices}));
    test::assertEqualVectors(expected, result);
  }

  {
    auto result = evaluateOnce<int64_t>(
        "element_at(array_constructor(14,10,16,15,12,11,17,13), -3)",
        makeRowVector({}));
    ASSERT_EQ(result, 11);
  }
}

TEST_F(ElementAtTest, constantInputArray) {
  {
    auto arrayVector = makeArrayVector<int64_t>({
        {1, 2, 3, 4, 5, 6, 7},
        {1, 2, 7},
        {1, 2, 3, 5, 6, 7},
    });
    auto expectedValueAt = [](vector_size_t row) { return 2; };

    testElementAt<int64_t>("element_at(C0, 2)", {arrayVector}, expectedValueAt);
    testElementAt<int64_t>("C0[2]", {arrayVector}, expectedValueAt);

    // negative index
    auto expectedValueAt2 = [](vector_size_t row) { return 7; };
    testElementAt<int64_t>(
        "element_at(C0, -1)", {arrayVector}, expectedValueAt2);
  }

  {
    // test variable size arrays + negative index
    // the last element in each array is always "row number".
    auto arrayVector = makeArrayVector<int64_t>({
        {1, 2, 6, 0},
        {1},
        {9, 100, 10043, 44, 22, 2},
        {-1, -10, 3},
    });
    auto expectedValueAt = [](vector_size_t row) { return row; };
    testElementAt<int64_t>(
        "element_at(C0, -1)", {arrayVector}, expectedValueAt);
  }
}

// First flavor - the regular element_at() behavior:
// #1 - start indices at 1.
// #2 - allow out of bounds access for array and map.
// #3 - allow negative indices.
TEST_F(ElementAtTest, allFlavors1) {
  // Case 1: Simple arrays and maps
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});
  auto mapVector = getSimpleMapVector();

  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("element_at(C0, 0)", {arrayVector}),
      "SQL array indices start at 1");

  EXPECT_EQ(
      elementAtSimple("try(element_at(C0, 0))", {arrayVector}), std::nullopt);

  EXPECT_EQ(elementAtSimple("element_at(C0, 1)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("element_at(C0, 2)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("element_at(C0, 3)", {arrayVector}), 12);

  // #2
  EXPECT_EQ(elementAtSimple("element_at(C0, 4)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("element_at(C0, 5)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("element_at(C0, 1001)", {mapVector}), std::nullopt);

  // #3
  EXPECT_EQ(elementAtSimple("element_at(C0, -1)", {arrayVector}), 12);
  EXPECT_EQ(elementAtSimple("element_at(C0, -2)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("element_at(C0, -3)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("element_at(C0, -4)", {arrayVector}), std::nullopt);

  // Case 2: Empty values vector
  auto emptyValues = makeFlatVector<int64_t>({});
  auto emptyKeys = makeFlatVector<int64_t>(std::vector<int64_t>{1});
  auto offsets = allocateOffsets(1, pool());
  auto sizes = allocateSizes(1, pool());
  auto emptyValuesArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 1, offsets, sizes, emptyValues);
  auto emptyValuesMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      1,
      offsets,
      sizes,
      emptyKeys,
      emptyValues);
  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("element_at(C0, 0)", {emptyValuesArray}),
      "SQL array indices start at 1");

  // #2
  EXPECT_EQ(
      elementAtSimple("element_at(C0, 4)", {emptyValuesArray}), std::nullopt);
  EXPECT_EQ(
      elementAtSimple("element_at(C0, 1001)", {emptyValuesMap}), std::nullopt);

  // #3
  EXPECT_EQ(
      elementAtSimple("element_at(C0, -1)", {emptyValuesArray}), std::nullopt);

  // Case 3: Empty individual arrays/maps and non-empty values vector
  auto nonEmptyValues = makeFlatVector<int64_t>(std::vector<int64_t>{2});
  auto nonEmptyKeys = makeFlatVector<int64_t>(std::vector<int64_t>{1});
  auto emptyContainerNonEmptyValuesArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 1, offsets, sizes, nonEmptyValues);
  auto emptyContainerNonEmptyValuesMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      1,
      offsets,
      sizes,
      nonEmptyKeys,
      nonEmptyValues);

  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("element_at(C0, 0)", {emptyContainerNonEmptyValuesArray}),
      "SQL array indices start at 1");

  // #2
  EXPECT_EQ(
      elementAtSimple("element_at(C0, 4)", {emptyContainerNonEmptyValuesArray}),
      std::nullopt);
  EXPECT_EQ(
      elementAtSimple(
          "element_at(C0, 1001)", {emptyContainerNonEmptyValuesMap}),
      std::nullopt);

  // #3
  EXPECT_EQ(
      elementAtSimple("element_at(C0, -1)", {emptyContainerNonEmptyValuesMap}),
      std::nullopt);

  // Case 4: Intermittently empty individual arrays/maps and non-empty values
  // vector
  offsets = allocateOffsets(2, pool());
  sizes = allocateSizes(2, pool());
  auto rawSizes = sizes->asMutable<vector_size_t>();
  rawSizes[0] = 0;
  rawSizes[1] = 1;

  auto partiallyEmptyArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 2, offsets, sizes, nonEmptyValues);
  auto partiallyEmptyMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      2,
      offsets,
      sizes,
      nonEmptyKeys,
      nonEmptyValues);
  // Input vector has 2 elements, but we only evaluate on the first, this way
  // the elements aren't empty though the collection we're evaluating on is.
  SelectivityVector rows(1);
  VectorPtr result;

  // #1
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int64_t>>(
          "element_at(c0, 0)",
          makeRowVector({partiallyEmptyArray}),
          rows,
          result),
      "SQL array indices start at 1");

  // #2
  auto expected = makeNullConstant(TypeKind::BIGINT, 1);
  evaluate<SimpleVector<int64_t>>(
      "element_at(C0, 4)", makeRowVector({partiallyEmptyArray}), rows, result);
  test::assertEqualVectors(expected, result);

  evaluate<SimpleVector<int64_t>>(
      "element_at(C0, 1001)", makeRowVector({partiallyEmptyMap}), rows, result);
  test::assertEqualVectors(expected, result);

  // #3
  evaluate<SimpleVector<int64_t>>(
      "element_at(C0, -1)", makeRowVector({partiallyEmptyArray}), rows, result);
  test::assertEqualVectors(expected, result);
}

// Second flavor - the regular subscript ("a[1]") behavior:
// #1 - start indices at 1.
// #2 - do not allow out of bounds access for arrays.
// #3 - do not allow negative indices.
TEST_F(ElementAtTest, allFlavors2) {
  // Case 1: Simple arrays and maps
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});
  auto mapVector = getSimpleMapVector();

  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[0]", {arrayVector}), "SQL array indices start at 1");

  EXPECT_EQ(elementAtSimple("try(C0[0])", {arrayVector}), std::nullopt);

  EXPECT_EQ(elementAtSimple("C0[1]", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("C0[2]", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("C0[3]", {arrayVector}), 12);

  // #2
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[4]", {arrayVector}),
      "Array subscript out of bounds.");
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[5]", {arrayVector}),
      "Array subscript out of bounds.");

  EXPECT_EQ(elementAtSimple("try(C0[4])", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("try(C0[5])", {arrayVector}), std::nullopt);

  // Maps are ok.
  EXPECT_EQ(elementAtSimple("C0[1001]", {mapVector}), std::nullopt);

  // #3
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[-1]", {arrayVector}), "Array subscript is negative.");
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[-4]", {arrayVector}), "Array subscript is negative.");

  EXPECT_EQ(elementAtSimple("try(C0[-1])", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("try(C0[-4])", {arrayVector}), std::nullopt);

  // Case 2: Empty values vector
  auto emptyValues = makeFlatVector<int64_t>({});
  auto emptyKeys = makeFlatVector<int64_t>(std::vector<int64_t>{1});
  auto offsets = allocateOffsets(1, pool());
  auto sizes = allocateSizes(1, pool());
  auto emptyValuesArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 1, offsets, sizes, emptyValues);
  auto emptyValuesMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      1,
      offsets,
      sizes,
      emptyKeys,
      emptyValues);
  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[0]", {emptyValuesArray}),
      "SQL array indices start at 1");

  // #2
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[4]", {arrayVector}),
      "Array subscript out of bounds.");
  EXPECT_EQ(elementAtSimple("try(C0[4])", {emptyValuesArray}), std::nullopt);
  // Maps are ok.
  EXPECT_EQ(elementAtSimple("C0[1001]", {emptyValuesMap}), std::nullopt);

  // #3
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[-1]", {arrayVector}), "Array subscript is negative.");

  // Case 3: Empty individual arrays/maps and non-empty values vector
  auto nonEmptyValues = makeFlatVector<int64_t>(std::vector<int64_t>{2});
  auto nonEmptyKeys = makeFlatVector<int64_t>(std::vector<int64_t>{1});
  auto emptyContainerNonEmptyValuesArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 1, offsets, sizes, nonEmptyValues);
  auto emptyContainerNonEmptyValuesMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      1,
      offsets,
      sizes,
      nonEmptyKeys,
      nonEmptyValues);

  // #1
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[0]", {emptyContainerNonEmptyValuesArray}),
      "SQL array indices start at 1");

  // #2
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[4]", {emptyContainerNonEmptyValuesArray}),
      "Array subscript out of bounds.");
  EXPECT_EQ(
      elementAtSimple("try(C0[4])", {emptyContainerNonEmptyValuesArray}),
      std::nullopt);
  // Maps are ok.
  EXPECT_EQ(
      elementAtSimple("C0[1001]", {emptyContainerNonEmptyValuesMap}),
      std::nullopt);

  // #3
  VELOX_ASSERT_THROW(
      elementAtSimple("C0[-1]", {emptyContainerNonEmptyValuesArray}),
      "Array subscript is negative.");

  // Case 4: Intermittently empty individual arrays/maps and non-empty values
  // vector
  offsets = allocateOffsets(2, pool());
  sizes = allocateSizes(2, pool());
  auto rawSizes = sizes->asMutable<vector_size_t>();
  rawSizes[0] = 0;
  rawSizes[1] = 1;

  auto partiallyEmptyArray = std::make_shared<ArrayVector>(
      pool(), ARRAY(BIGINT()), nullptr, 2, offsets, sizes, nonEmptyValues);
  auto partiallyEmptyMap = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      2,
      offsets,
      sizes,
      nonEmptyKeys,
      nonEmptyValues);
  // Input vector has 2 elements, but we only evaluate on the first, this way
  // the elements aren't empty though the collection we're evaluating on is.
  SelectivityVector rows(1);
  VectorPtr result;

  // #1
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int64_t>>(
          "C0[0]", makeRowVector({partiallyEmptyArray}), rows, result),
      "SQL array indices start at 1");

  // #2
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int64_t>>(
          "C0[4]", makeRowVector({partiallyEmptyArray}), rows, result),
      "Array subscript out of bounds.");
  auto expected = makeNullConstant(TypeKind::BIGINT, 1);
  evaluate<SimpleVector<int64_t>>(
      "try(C0[4])", makeRowVector({partiallyEmptyArray}), rows, result);
  test::assertEqualVectors(expected, result);
  // Maps are ok.
  evaluate<SimpleVector<int64_t>>(
      "C0[1001]", makeRowVector({partiallyEmptyMap}), rows, result);
  test::assertEqualVectors(expected, result);

  // #3
  VELOX_ASSERT_THROW(
      evaluate<SimpleVector<int64_t>>(
          "C0[-1]", makeRowVector({partiallyEmptyArray}), rows, result),
      "Array subscript is negative.");
}

// Third flavor:
// #1 - indices start at 0.
// #2 - do not allow out of bound access for arrays (throw).
// #3 - null on negative indices.
TEST_F(ElementAtTest, allFlavors3) {
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});
  auto mapVector = getSimpleMapVector();

  exec::registerVectorFunction(
      "__f2",
      SubscriptImpl<false, true, false, false>::signatures(),
      std::make_unique<SubscriptImpl<false, true, false, false>>());

  // #1
  EXPECT_EQ(elementAtSimple("__f2(C0, 0)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("__f2(C0, 1)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("__f2(C0, 2)", {arrayVector}), 12);

  // #2
  VELOX_ASSERT_THROW(
      elementAtSimple("__f2(C0, 3)", {arrayVector}),
      "Array subscript out of bounds.");
  VELOX_ASSERT_THROW(
      elementAtSimple("__f2(C0, 4)", {arrayVector}),
      "Array subscript out of bounds.");
  VELOX_ASSERT_THROW(
      elementAtSimple("__f2(C0, 100)", {arrayVector}),
      "Array subscript out of bounds.");

  EXPECT_EQ(elementAtSimple("try(__f2(C0, 3))", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("try(__f2(C0, 4))", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("try(__f2(C0, 100))", {arrayVector}), std::nullopt);

  // We still allow non-existent map keys, even if out of bounds is disabled for
  // arrays.
  EXPECT_EQ(elementAtSimple("__f2(C0, 1001)", {mapVector}), std::nullopt);

  // #3
  EXPECT_EQ(elementAtSimple("__f2(C0, -1)", {mapVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("__f2(C0, -5)", {mapVector}), std::nullopt);
}

// Fourth flavor:
// #1 - indices start at 0.
// #2 - allow out of bound access (throw).
// #3 - allow negative indices.
TEST_F(ElementAtTest, allFlavors4) {
  auto arrayVector = makeArrayVector<int64_t>({{10, 11, 12}});
  auto mapVector = getSimpleMapVector();

  exec::registerVectorFunction(
      "__f3",
      SubscriptImpl<true, false, true, false>::signatures(),
      std::make_unique<SubscriptImpl<true, false, true, false>>());

  EXPECT_EQ(elementAtSimple("__f3(C0, 0)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("__f3(C0, 1)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("__f3(C0, 2)", {arrayVector}), 12);
  EXPECT_EQ(elementAtSimple("__f3(C0, 3)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("__f3(C0, 4)", {arrayVector}), std::nullopt);
  EXPECT_EQ(elementAtSimple("__f3(C0, -1)", {arrayVector}), 12);
  EXPECT_EQ(elementAtSimple("__f3(C0, -2)", {arrayVector}), 11);
  EXPECT_EQ(elementAtSimple("__f3(C0, -3)", {arrayVector}), 10);
  EXPECT_EQ(elementAtSimple("__f3(C0, -4)", {arrayVector}), std::nullopt);

  EXPECT_EQ(elementAtSimple("__f3(C0, 1001)", {mapVector}), std::nullopt);
}

TEST_F(ElementAtTest, constantInputMap) {
  {
    // Matches a key on every single row.
    auto sizeAt = [](vector_size_t /* row */) { return 5; };
    auto keyAt = [](vector_size_t idx) { return idx % 5; };
    auto valueAt = [](vector_size_t idx) { return (idx % 5) + 10; };
    auto expectedValueAt = [](vector_size_t /* row */) { return 13; };
    auto mapVector =
        makeMapVector<int64_t, int64_t>(kVectorSize, sizeAt, keyAt, valueAt);
    testElementAt<int64_t>("element_at(C0, 3)", {mapVector}, expectedValueAt);
    testElementAt<int64_t>("C0[3]", {mapVector}, expectedValueAt);
  }

  {
    // Match only on row 1, everything else is null.
    auto sizeAt = [](vector_size_t /* row */) { return 5; };
    auto keyAt = [](vector_size_t idx) { return idx + 1; };
    auto valueAt = [](vector_size_t idx) { return idx + 1; };
    auto expectedValueAt = [](vector_size_t /* row */) { return 7; };
    auto expectedNullAt = [](vector_size_t row) { return row != 1; };
    auto mapVector =
        makeMapVector<int64_t, int64_t>(kVectorSize, sizeAt, keyAt, valueAt);
    testElementAt<int64_t>(
        "element_at(C0, 7)", {mapVector}, expectedValueAt, expectedNullAt);

    // Matches nothing.
    auto expectedNullAt2 = [](vector_size_t /* row */) { return true; };
    testElementAt<int64_t>(
        "element_at(C0, 10000)", {mapVector}, expectedValueAt, expectedNullAt2);
    testElementAt<int64_t>(
        "element_at(C0, -1)", {mapVector}, expectedValueAt, expectedNullAt2);
  }

  {
    // Test with IF filtering first.
    auto sizeAt = [](vector_size_t /* row */) { return 5; };
    auto keyAt = [](vector_size_t idx) { return idx % 5; };
    auto valueAt = [](vector_size_t idx) { return (idx % 5) + 10; };
    auto expectedValueAt = [](vector_size_t /* row */) { return 13; };
    auto mapVector =
        makeMapVector<int64_t, int64_t>(kVectorSize, sizeAt, keyAt, valueAt);

    // Boolean columns used for filtering.
    auto boolVector = makeFlatVector<bool>(
        kVectorSize, [](vector_size_t row) { return (row % 3) == 0; });
    auto expectedNullAt = [](vector_size_t row) { return (row % 3) != 0; };

    testElementAt<int64_t>(
        "if(C1, element_at(C0, 3), element_at(C0, 10000))",
        {mapVector, boolVector},
        expectedValueAt,
        expectedNullAt);
  }

  {
    // Vary map value types.
    auto sizeAt = [](vector_size_t /* row */) { return 5; };
    auto keyAt = [](vector_size_t idx) { return idx % 5; };
    auto valueAt = [](vector_size_t idx) { return (idx % 5) + 10; };
    auto expectedValueAt = [](vector_size_t /* row */) { return 13; };
    auto mapVector =
        makeMapVector<int64_t, int16_t>(kVectorSize, sizeAt, keyAt, valueAt);
    testElementAt<int16_t>("element_at(C0, 3)", {mapVector}, expectedValueAt);

    // String map value type.
    auto valueAt2 = [](vector_size_t idx) {
      return StringView(folly::sformat("str{}", idx % 5).c_str());
    };
    auto expectedValueAt2 = [](vector_size_t /* row */) {
      return StringView("str3");
    };
    auto mapVector2 = makeMapVector<int64_t, StringView>(
        kVectorSize, sizeAt, keyAt, valueAt2);
    testElementAt<StringView>(
        "element_at(C0, 3)", {mapVector2}, expectedValueAt2);
  }
}

TEST_F(ElementAtTest, variableInputMap) {
  testVariableInputMap<int64_t>(); // BIGINT
  testVariableInputMap<int32_t>(); // INTEGER
  testVariableInputMap<int16_t>(); // SMALLINT
  testVariableInputMap<int8_t>(); // TINYINT

  testVariableInputMap<StringView>(); // VARCHAR
}

TEST_F(ElementAtTest, variableInputArray) {
  {
    auto indicesVector = makeFlatVector<int64_t>(
        kVectorSize, [](vector_size_t row) { return 1 + row % 7; });

    auto sizeAt = [](vector_size_t row) { return 7; };
    auto valueAt = [](vector_size_t row, vector_size_t idx) { return 1 + idx; };
    auto expectedValueAt = [](vector_size_t row) { return 1 + row % 7; };
    auto arrayVector = makeArrayVector<int64_t>(kVectorSize, sizeAt, valueAt);
    testElementAt<int64_t>(
        "element_at(C0, C1)", {arrayVector, indicesVector}, expectedValueAt);
  }

  {
    auto indicesVector =
        makeFlatVector<int64_t>(kVectorSize, [](vector_size_t row) {
          if (row % 7 == 0) {
            // every first row should be null
            return 8;
          }
          if (row % 7 == 4) {
            // every fourth out of seven rows should be a negative access
            return -3;
          }
          return 1 + row % 7; // access the last element
        });

    auto sizeAt = [](vector_size_t row) { return row % 7 + 1; };
    auto valueAt = [](vector_size_t row, vector_size_t idx) { return idx; };
    auto expectedValueAt = [](vector_size_t row) {
      if (row % 7 == 4) {
        return row % 7 - 2; // negative access
      }
      return row % 7;
    };
    auto expectedNullAt = [](vector_size_t row) { return row % 7 == 0; };
    auto arrayVector = makeArrayVector<int64_t>(kVectorSize, sizeAt, valueAt);

    testElementAt<int64_t>(
        "element_at(C0, C1)",
        {arrayVector, indicesVector},
        expectedValueAt,
        expectedNullAt);
  }
}

TEST_F(ElementAtTest, varcharVariableInput) {
  auto indicesVector = makeFlatVector<int64_t>(
      kVectorSize, [](vector_size_t row) { return 1 + row % 7; });

  auto sizeAt = [](vector_size_t row) { return 7; };
  auto valueAt = [](vector_size_t row, vector_size_t idx) {
    return StringView(folly::to<std::string>(idx + 1).c_str());
  };
  auto expectedValueAt = [](vector_size_t row) {
    return StringView(folly::to<std::string>(row % 7 + 1).c_str());
  };
  auto arrayVector = makeArrayVector<StringView>(kVectorSize, sizeAt, valueAt);

  testElementAt<StringView>(
      "element_at(C0, C1)", {arrayVector, indicesVector}, expectedValueAt);
}

TEST_F(ElementAtTest, allIndicesGreaterThanArraySize) {
  auto indices = makeFlatVector<int64_t>(
      kVectorSize, [](vector_size_t /*row*/) { return 2; });

  auto sizeAt = [](vector_size_t row) { return 1; };
  auto expectedValueAt = [](vector_size_t row) { return 1; };
  auto valueAt = [](vector_size_t row, vector_size_t idx) { return 1; };
  auto expectedNullAt = [](vector_size_t row) { return true; };

  auto arrayVector = makeArrayVector<int64_t>(kVectorSize, sizeAt, valueAt);

  testElementAt<int64_t>(
      "element_at(C0, C1)",
      {arrayVector, indices},
      expectedValueAt,
      expectedNullAt);
}

TEST_F(ElementAtTest, errorStatesArray) {
  auto indicesVector =
      makeFlatVector<int64_t>(kVectorSize, [](vector_size_t row) {
        // zero out index -> should throw an error
        if (row == 40) {
          return 0;
        }
        return 1 + row % 7;
      });

  // Make arrays of 1-based sequences of variying lengths: [1, 2], [1, 2, 3],
  // [1, 2, 3, 4], etc.
  auto sizeAt = [](vector_size_t row) { return 2 + row % 7; };
  auto expectedValueAt = [](vector_size_t row) { return 1 + row % 7; };
  auto valueAt = [](vector_size_t /*row*/, vector_size_t idx) {
    return 1 + idx;
  };
  auto arrayVector = makeArrayVector<int64_t>(kVectorSize, sizeAt, valueAt);

  // Trying to use index = 0.
  VELOX_ASSERT_THROW(
      testElementAt<int64_t>(
          "element_at(C0, C1)", {arrayVector, indicesVector}, expectedValueAt),
      "SQL array indices start at 1");

  testElementAt<int64_t>(
      "try(element_at(C0, C1))",
      {arrayVector, indicesVector},
      expectedValueAt,
      [](auto row) { return row == 40; });
}
