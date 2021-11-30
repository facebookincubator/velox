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
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

namespace {

class ArrayPositionTest : public FunctionBaseTest {};

TEST_F(ArrayPositionTest, integer) {
  auto arrayVector = makeNullableArrayVector<int64_t>(
      {{1, std::nullopt, 3, 4},
       {3, 4, 5},
       {},
       {std::nullopt},
       {5, 6, std::nullopt, 8, 9},
       {7},
       {3, 9, 8, 7}});

  auto testPosition = [&](std::optional<int64_t> search,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1)",
        makeRowVector({
            arrayVector,
            makeConstant(search, arrayVector->size()),
        }));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(3, {3, 1, 0, 0, 0, 0, 1});
  testPosition(4, {4, 2, 0, 0, 0, 0, 0});
  testPosition(5, {0, 3, 0, 0, 1, 0, 0});
  testPosition(7, {0, 0, 0, 0, 0, 1, 4});
  testPosition(
      std::nullopt,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, integerDictionaryEncoding) {
  auto arrayVector = makeNullableArrayVector<int64_t>(
      {{1, std::nullopt, 3, 4},
       {3, 4, 5},
       {},
       {std::nullopt},
       {5, 6, std::nullopt, 8, 9},
       {7},
       {3, 9, 8, 7}});
  auto newSize = arrayVector->size() * 2;
  auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
  auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

  auto testPosition = [&](std::optional<int64_t> search,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1)",
        makeRowVector({
            dictArrayVector,
            makeConstant(search, dictArrayVector->size()),
        }));

    auto expectedVector = makeNullableFlatVector<int64_t>(expected);
    auto dictExpectedVector =
        wrapInDictionary(indices, newSize, expectedVector);
    assertEqualVectors(dictExpectedVector, result);
  };

  testPosition(3, {3, 1, 0, 0, 0, 0, 1});
  testPosition(4, {4, 2, 0, 0, 0, 0, 0});
  testPosition(5, {0, 3, 0, 0, 1, 0, 0});
  testPosition(7, {0, 0, 0, 0, 0, 1, 4});
  testPosition(
      std::nullopt,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, varchar) {
  std::vector<std::string> colors = {
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;

  auto arrayVector = makeNullableArrayVector<StringView>({
      {S("red"), S("blue")},
      {std::nullopt, S("yellow"), S("orange")},
      {},
      {std::nullopt},
      {S("red"), S("purple"), S("green")},
  });

  auto testPosition = [&](std::optional<const char*> search,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1)",
        makeRowVector({
            arrayVector,
            makeConstant(search, arrayVector->size()),
        }));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition("red", {1, 0, 0, 0, 1});
  testPosition("blue", {2, 0, 0, 0, 0});
  testPosition("yellow", {0, 2, 0, 0, 0});
  testPosition("green", {0, 0, 0, 0, 3});
  testPosition("crimson red", {0, 0, 0, 0, 0});
  testPosition(
      std::nullopt,
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

TEST_F(ArrayPositionTest, boolean) {
  auto arrayVector = makeNullableArrayVector<bool>({
      {true, false},
      {true},
      {false},
      {},
      {std::nullopt},
      {true, std::nullopt, true},
      {false, false, false},
  });

  auto testPosition = [&](std::optional<bool> search,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1)",
        makeRowVector({
            arrayVector,
            makeConstant(search, arrayVector->size()),
        }));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(true, {1, 1, 0, 0, 0, 1, 0});
  testPosition(false, {2, 0, 1, 0, 0, 0, 1});
  testPosition(
      std::nullopt,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, row) {
  std::vector<std::vector<variant>> data = {
      {
          variant::row({1, "red"}),
          variant::row({2, "blue"}),
          variant::row({3, "green"}),
      },
      {
          variant::row({2, "blue"}),
          variant(TypeKind::ROW), // null
          variant::row({5, "green"}),
      },
      {},
      {
          variant(TypeKind::ROW), // null
      },
      {
          variant::row({1, "yellow"}),
          variant::row({2, "blue"}),
          variant::row({4, "green"}),
          variant::row({5, "purple"}),
      },
  };

  auto rowType = ROW({INTEGER(), VARCHAR()});
  auto arrayVector = makeArrayOfRowVector(rowType, data);

  auto testPosition = [&](int32_t n,
                          const char* color,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto search =
        makeConstantRow(rowType, variant::row({n, color}), arrayVector->size());

    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1)", makeRowVector({arrayVector, search}));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(1, "red", {1, 0, 0, 0, 0});
  testPosition(2, "blue", {2, 1, 0, 0, 2});
  testPosition(4, "green", {0, 0, 0, 0, 3});
  testPosition(5, "green", {0, 3, 0, 0, 0});
  testPosition(1, "purple", {0, 0, 0, 0, 0});
}

TEST_F(ArrayPositionTest, integerWithInstance) {
  auto arrayVector = makeNullableArrayVector<int64_t>(
      {{1, 2, std::nullopt, 4},
       {3, 4, 4},
       {},
       {std::nullopt},
       {5, 2, 4, 2, 9},
       {7},
       {10, 4, 8, 4}});

  auto testPosition = [&](std::optional<int64_t> search,
                          std::optional<int64_t> instance,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1, c2)",
        makeRowVector(
            {arrayVector,
             makeConstant(search, arrayVector->size()),
             makeConstant(instance, arrayVector->size())}));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(4, 2, {0, 3, 0, 0, 0, 0, 4});
  testPosition(4, -1, {4, 3, 0, 0, 3, 0, 4});
  testPosition(2, 2, {0, 0, 0, 0, 4, 0, 0});
  testPosition(2, 3, {0, 0, 0, 0, 0, 0, 0});
  testPosition(2, -1, {2, 0, 0, 0, 4, 0, 0});
  testPosition(
      std::nullopt,
      1,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, integerDictionaryEncodingWithInstance) {
  auto arrayVector = makeNullableArrayVector<int64_t>(
      {{1, 2, std::nullopt, 4},
       {3, 4, 4},
       {},
       {std::nullopt},
       {5, 2, 4, 2, 9},
       {7},
       {10, 4, 8, 4}});
  auto newSize = arrayVector->size() * 2;
  auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
  auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

  auto testPosition = [&](std::optional<int64_t> search,
                          std::optional<int64_t> instance,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1, c2)",
        makeRowVector(
            {dictArrayVector,
             makeConstant(search, dictArrayVector->size()),
             makeConstant(instance, dictArrayVector->size())}));

    auto expectedVector = makeNullableFlatVector<int64_t>(expected);
    auto dictExpectedVector =
        wrapInDictionary(indices, newSize, expectedVector);
    assertEqualVectors(dictExpectedVector, result);
  };

  testPosition(4, 2, {0, 3, 0, 0, 0, 0, 4});
  testPosition(4, -1, {4, 3, 0, 0, 3, 0, 4});
  testPosition(2, 2, {0, 0, 0, 0, 4, 0, 0});
  testPosition(2, 3, {0, 0, 0, 0, 0, 0, 0});
  testPosition(2, -1, {2, 0, 0, 0, 4, 0, 0});
  testPosition(
      std::nullopt,
      1,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, varcharWithInstance) {
  std::vector<std::string> colors = {
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;

  auto arrayVector = makeNullableArrayVector<StringView>({
      {S("red"), S("blue"), S("red")},
      {std::nullopt, S("yellow"), S("yellow")},
      {},
      {std::nullopt},
      {S("red"), S("green"), S("purple"), S("green")},
  });

  auto testPosition = [&](std::optional<const char*> search,
                          std::optional<int64_t> instance,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1, c2)",
        makeRowVector({
            arrayVector,
            makeConstant(search, arrayVector->size()),
            makeConstant(instance, arrayVector->size()),
        }));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition("red", 2, {3, 0, 0, 0, 0});
  testPosition("red", -1, {3, 0, 0, 0, 1});
  testPosition("yellow", 2, {0, 3, 0, 0, 0});
  testPosition("green", -2, {0, 0, 0, 0, 2});
  testPosition("green", 3, {0, 0, 0, 0, 0});
  testPosition("crimson red", 1, {0, 0, 0, 0, 0});
  testPosition(
      std::nullopt,
      1,
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

TEST_F(ArrayPositionTest, booleanWithInstance) {
  auto arrayVector = makeNullableArrayVector<bool>({
      {true, false},
      {true},
      {false},
      {},
      {std::nullopt},
      {true, std::nullopt, true},
      {false, false, false},
  });

  auto testPosition = [&](std::optional<bool> search,
                          std::optional<int64_t> instance,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1, c2)",
        makeRowVector({
            arrayVector,
            makeConstant(search, arrayVector->size()),
            makeConstant(instance, arrayVector->size()),
        }));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(true, 2, {0, 0, 0, 0, 0, 3, 0});
  testPosition(true, -1, {1, 1, 0, 0, 0, 3, 0});
  testPosition(false, 3, {0, 0, 0, 0, 0, 0, 3});
  testPosition(false, 4, {0, 0, 0, 0, 0, 0, 0});
  testPosition(false, -2, {0, 0, 0, 0, 0, 0, 2});
  testPosition(
      std::nullopt,
      1,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, rowWithInstance) {
  std::vector<std::vector<variant>> data = {
      {variant::row({1, "red"}),
       variant::row({2, "blue"}),
       variant::row({3, "green"}),
       variant::row({1, "red"})},
      {variant::row({2, "blue"}),
       variant(TypeKind::ROW), // null
       variant::row({5, "green"}),
       variant::row({2, "blue"})},
      {},
      {
          variant(TypeKind::ROW), // null
      },
      {
          variant::row({1, "yellow"}),
          variant::row({2, "blue"}),
          variant::row({4, "green"}),
          variant::row({2, "blue"}),
          variant::row({2, "blue"}),
          variant::row({5, "purple"}),
      },
  };

  auto rowType = ROW({INTEGER(), VARCHAR()});
  auto arrayVector = makeArrayOfRowVector(rowType, data);

  auto testPosition = [&](int32_t n,
                          const char* color,
                          int64_t instance,
                          const std::vector<std::optional<int64_t>>& expected) {
    auto search =
        makeConstantRow(rowType, variant::row({n, color}), arrayVector->size());
    auto instances = makeConstant(instance, arrayVector->size());

    auto result = evaluate<SimpleVector<int64_t>>(
        "array_position(c0, c1, c2)",
        makeRowVector({arrayVector, search, instances}));

    assertEqualVectors(makeNullableFlatVector<int64_t>(expected), result);
  };

  testPosition(1, "red", 1, {1, 0, 0, 0, 0});
  testPosition(1, "red", -1, {4, 0, 0, 0, 0});
  testPosition(2, "blue", 3, {0, 0, 0, 0, 5});
  testPosition(2, "blue", -2, {0, 1, 0, 0, 4});
}
} // namespace
