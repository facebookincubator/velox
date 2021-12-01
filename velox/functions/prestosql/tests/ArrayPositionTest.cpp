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

template <typename T>
using TwoDimVector = std::vector<std::vector<std::optional<T>>>;

class ArrayPositionTest : public FunctionBaseTest {
 protected:
  void evalExpr(
      const std::vector<VectorPtr>& input,
      const std::string& expression,
      const VectorPtr& expected) {
    auto result =
        evaluate<SimpleVector<int64_t>>(expression, makeRowVector(input));

    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testPosition(
      const TwoDimVector<T>& array,
      const std::optional<T>& search,
      const std::vector<std::optional<int64_t>>& expected) {
    evalExpr(
        {makeNullableArrayVector<T>(array), makeConstant(search, array.size())},
        "array_position(c0, c1)",
        makeNullableFlatVector<int64_t>(expected));
  }

  template <typename T>
  void testPositionDictEncoding(
      const TwoDimVector<T>& array,
      const std::optional<T>& search,
      const std::vector<std::optional<int64_t>>& expected) {
    auto arrayVector = makeNullableArrayVector<T>(array);

    auto newSize = arrayVector->size() * 2;
    auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
    auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

    auto expectedVector = makeNullableFlatVector<int64_t>(expected);
    auto dictExpectedVector =
        wrapInDictionary(indices, newSize, expectedVector);

    auto searchVector = makeConstant(search, array.size());
    auto searchIndices = makeIndices(newSize, [](auto row) { return 0; });
    auto dictSearchVector =
        wrapInDictionary(searchIndices, newSize, searchVector);

    evalExpr(
        {dictArrayVector, dictSearchVector},
        "array_position(c0, c1)",
        dictExpectedVector);
  }

  template <typename T>
  void testPositionWithInstance(
      const TwoDimVector<T>& array,
      const std::optional<T>& search,
      const int64_t instance,
      const std::vector<std::optional<int64_t>>& expected) {
    evalExpr(
        {makeNullableArrayVector<T>(array),
         makeConstant(search, array.size()),
         makeConstant(instance, array.size())},
        "array_position(c0, c1, c2)",
        makeNullableFlatVector<int64_t>(expected));
  }

  template <typename T>
  void testPositionDictEncodingWithInstance(
      const TwoDimVector<T>& array,
      const std::optional<T>& search,
      const int64_t instance,
      const std::vector<std::optional<int64_t>>& expected) {
    auto arrayVector = makeNullableArrayVector<T>(array);

    auto newSize = arrayVector->size() * 2;
    auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
    auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

    auto expectedVector = makeNullableFlatVector<int64_t>(expected);
    auto dictExpectedVector =
        wrapInDictionary(indices, newSize, expectedVector);

    auto searchVector = makeConstant(search, array.size());
    auto constIndices = makeIndices(newSize, [](auto row) { return 0; });
    auto dictSearchVector =
        wrapInDictionary(constIndices, newSize, searchVector);

    auto instanceVector = makeConstant(instance, array.size());
    auto dictInstanceVector =
        wrapInDictionary(constIndices, newSize, instanceVector);

    evalExpr(
        {dictArrayVector, dictSearchVector, dictInstanceVector},
        "array_position(c0, c1, c2)",
        dictExpectedVector);
  }
};

TEST_F(ArrayPositionTest, integer) {
  TwoDimVector<int64_t> arrayVector{
      {1, std::nullopt, 3, 4},
      {3, 4, 5},
      {},
      {std::nullopt},
      {5, 6, std::nullopt, 8, 9},
      {7},
      {3, 9, 8, 7}};

  testPosition<int64_t>(arrayVector, 3, {3, 1, 0, 0, 0, 0, 1});
  testPosition<int64_t>(arrayVector, 4, {4, 2, 0, 0, 0, 0, 0});
  testPosition<int64_t>(arrayVector, 5, {0, 3, 0, 0, 1, 0, 0});
  testPosition<int64_t>(arrayVector, 7, {0, 0, 0, 0, 0, 1, 4});
  testPosition<int64_t>(
      arrayVector,
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
  TwoDimVector<int64_t> arrayVector{
      {1, std::nullopt, 3, 4},
      {3, 4, 5},
      {},
      {std::nullopt},
      {5, 6, std::nullopt, 8, 9},
      {7},
      {3, 9, 8, 7}};

  testPositionDictEncoding<int64_t>(arrayVector, 3, {3, 1, 0, 0, 0, 0, 1});
  testPositionDictEncoding<int64_t>(arrayVector, 4, {4, 2, 0, 0, 0, 0, 0});
  testPositionDictEncoding<int64_t>(arrayVector, 5, {0, 3, 0, 0, 1, 0, 0});
  testPositionDictEncoding<int64_t>(arrayVector, 7, {0, 0, 0, 0, 0, 1, 4});
  testPositionDictEncoding<int64_t>(
      arrayVector,
      std::nullopt,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, varcharDictionaryEncoding) {
  std::vector<std::string> colors{
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;

  TwoDimVector<S> arrayVector{
      {S{"red"}, S{"blue"}},
      {std::nullopt, S{"yellow"}, S{"orange"}},
      {},
      {std::nullopt},
      {S{"red"}, S{"purple"}, S{"green"}},
  };

  testPositionDictEncoding<S>(arrayVector, S{"red"}, {1, 0, 0, 0, 1});
  testPositionDictEncoding<S>(arrayVector, S{"blue"}, {2, 0, 0, 0, 0});
  testPositionDictEncoding<S>(arrayVector, S{"yellow"}, {0, 2, 0, 0, 0});
  testPositionDictEncoding<S>(arrayVector, S{"green"}, {0, 0, 0, 0, 3});
  testPositionDictEncoding<S>(arrayVector, S{"crimson red"}, {0, 0, 0, 0, 0});
  testPositionDictEncoding<S>(
      arrayVector,
      std::nullopt,
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

TEST_F(ArrayPositionTest, booleanDictionaryEncoding) {
  TwoDimVector<bool> arrayVector{
      {true, false},
      {true},
      {false},
      {},
      {std::nullopt},
      {true, std::nullopt, true},
      {false, false, false},
  };

  testPositionDictEncoding<bool>(arrayVector, true, {1, 1, 0, 0, 0, 1, 0});
  testPositionDictEncoding<bool>(arrayVector, false, {2, 0, 1, 0, 0, 0, 1});
  testPositionDictEncoding<bool>(
      arrayVector,
      std::nullopt,
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

TEST_F(ArrayPositionTest, rowDictionaryEncoding) {
  std::vector<std::vector<variant>> data{
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

  auto newSize = arrayVector->size() * 2;
  auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
  auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

  auto testPositionOfRow =
      [&](int32_t n,
          const char* color,
          const std::vector<std::optional<int64_t>>& expected) {
        auto expectedVector = makeNullableFlatVector<int64_t>(expected);
        auto dictExpectedVector =
            wrapInDictionary(indices, newSize, expectedVector);

        auto searchVector = makeConstantRow(
            rowType, variant::row({n, color}), arrayVector->size());
        auto searchIndices = makeIndices(newSize, [](auto row) { return 0; });
        auto dictSearchVector =
            wrapInDictionary(searchIndices, newSize, searchVector);

        evalExpr(
            {dictArrayVector, dictSearchVector},
            "array_position(c0, c1)",
            dictExpectedVector);
      };

  testPositionOfRow(1, "red", {1, 0, 0, 0, 0});
  testPositionOfRow(2, "blue", {2, 1, 0, 0, 2});
  testPositionOfRow(4, "green", {0, 0, 0, 0, 3});
  testPositionOfRow(5, "green", {0, 3, 0, 0, 0});
  testPositionOfRow(1, "purple", {0, 0, 0, 0, 0});
}

TEST_F(ArrayPositionTest, array) {
  auto O = [](const std::vector<std::optional<int64_t>>& data) {
    return std::make_optional(data);
  };

  std::vector<std::optional<int64_t>> a{1, 2, 3};
  std::vector<std::optional<int64_t>> b{4, 5};
  std::vector<std::optional<int64_t>> c{6, 7, 8};
  std::vector<std::optional<int64_t>> d{1, 2};
  std::vector<std::optional<int64_t>> e{4, 3};
  auto arrayVector = makeNestedArrayVector<int64_t>(
      {{O(a), O(b), O(c)},
       {O(d), O(e)},
       {O(d), O(a), O(b)},
       {O(c), O(e)},
       {O({})},
       {O({std::nullopt})}});

  auto testPositionOfArray =
      [&](const std::vector<std::vector<std::optional<int64_t>>>& search,
          const std::vector<std::optional<int64_t>>& expected) {
        evalExpr(
            {arrayVector, makeNullableArrayVector<int64_t>(search)},
            "array_position(c0, c1)",
            makeNullableFlatVector<int64_t>(expected));
      };

  testPositionOfArray({a, a, a, a, a, a}, {1, 0, 2, 0, 0, 0});
  testPositionOfArray({b, e, a, e, e, e}, {2, 2, 2, 2, 0, 0});
  testPositionOfArray(
      {b, {std::nullopt}, a, {}, {}, {std::nullopt}}, {2, 0, 2, 0, 1, 1});
}

TEST_F(ArrayPositionTest, map) {
  std::vector<std::string> colors{
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;
  using P = std::pair<int64_t, std::optional<S>>;

  std::vector<P> a{P{1, S{"red"}}, P{2, S{"blue"}}, P{3, S{"green"}}};
  std::vector<P> b{P{1, S{"yellow"}}, P{2, S{"orange"}}};
  std::vector<P> c{P{1, S{"red"}}, P{2, S{"yellow"}}, P{3, S{"purple"}}};
  std::vector<P> d{P{0, std::nullopt}};
  std::vector<std::vector<std::vector<P>>> data{{a, b, c}, {b}, {c, a}};
  auto arrayVector = makeArrayOfMapVector<int64_t, S>(data);

  auto testPositionOfMap =
      [&](const std::vector<std::vector<P>>& search,
          const std::vector<std::optional<int64_t>>& expected) {
        evalExpr(
            {arrayVector, makeMapVector<int64_t, S>(search)},
            "array_position(c0, c1)",
            makeNullableFlatVector<int64_t>(expected));
      };

  testPositionOfMap({a, a, a}, {1, 0, 2});
  testPositionOfMap({b, b, a}, {2, 1, 2});
  testPositionOfMap({d, d, d}, {0, 0, 0});
}

TEST_F(ArrayPositionTest, integerWithInstance) {
  TwoDimVector<int64_t> arrayVector{
      {1, 2, std::nullopt, 4},
      {3, 4, 4},
      {},
      {std::nullopt},
      {5, 2, 4, 2, 9},
      {7},
      {10, 4, 8, 4}};

  testPositionWithInstance<int64_t>(arrayVector, 4, 2, {0, 3, 0, 0, 0, 0, 4});
  testPositionWithInstance<int64_t>(arrayVector, 4, -1, {4, 3, 0, 0, 3, 0, 4});
  testPositionWithInstance<int64_t>(arrayVector, 2, 2, {0, 0, 0, 0, 4, 0, 0});
  testPositionWithInstance<int64_t>(arrayVector, 2, 3, {0, 0, 0, 0, 0, 0, 0});
  testPositionWithInstance<int64_t>(arrayVector, 2, -1, {2, 0, 0, 0, 4, 0, 0});
  testPositionWithInstance<int64_t>(
      arrayVector,
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
  TwoDimVector<int64_t> arrayVector{
      {1, 2, std::nullopt, 4},
      {3, 4, 4},
      {},
      {std::nullopt},
      {5, 2, 4, 2, 9},
      {7},
      {10, 4, 8, 4}};

  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector, 4, 2, {0, 3, 0, 0, 0, 0, 4});
  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector, 4, -1, {4, 3, 0, 0, 3, 0, 4});
  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector, 2, 2, {0, 0, 0, 0, 4, 0, 0});
  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector, 2, 3, {0, 0, 0, 0, 0, 0, 0});
  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector, 2, -1, {2, 0, 0, 0, 4, 0, 0});
  testPositionDictEncodingWithInstance<int64_t>(
      arrayVector,
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

TEST_F(ArrayPositionTest, varcharDictionaryEncodingWithInstance) {
  std::vector<std::string> colors{
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;

  TwoDimVector<S> arrayVector{
      {S{"red"}, S{"blue"}, S{"red"}},
      {std::nullopt, S{"yellow"}, S{"yellow"}},
      {},
      {std::nullopt},
      {S{"red"}, S{"green"}, S{"purple"}, S{"green"}},
  };

  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"red"}, 2, {3, 0, 0, 0, 0});
  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"red"}, -1, {3, 0, 0, 0, 1});
  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"yellow"}, 2, {0, 3, 0, 0, 0});
  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"green"}, -2, {0, 0, 0, 0, 2});
  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"green"}, 3, {0, 0, 0, 0, 0});
  testPositionDictEncodingWithInstance<S>(
      arrayVector, S{"crimson red"}, 1, {0, 0, 0, 0, 0});
  testPositionDictEncodingWithInstance<S>(
      arrayVector,
      std::nullopt,
      1,
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

TEST_F(ArrayPositionTest, booleanDictionaryEncodingWithInstance) {
  TwoDimVector<bool> arrayVector{
      {true, false},
      {true},
      {false},
      {},
      {std::nullopt},
      {true, std::nullopt, true},
      {false, false, false},
  };

  testPositionDictEncodingWithInstance<bool>(
      arrayVector, true, 2, {0, 0, 0, 0, 0, 3, 0});
  testPositionDictEncodingWithInstance<bool>(
      arrayVector, true, -1, {1, 1, 0, 0, 0, 3, 0});
  testPositionDictEncodingWithInstance<bool>(
      arrayVector, false, 3, {0, 0, 0, 0, 0, 0, 3});
  testPositionDictEncodingWithInstance<bool>(
      arrayVector, false, 4, {0, 0, 0, 0, 0, 0, 0});
  testPositionDictEncodingWithInstance<bool>(
      arrayVector, false, -2, {0, 0, 0, 0, 0, 0, 2});
  testPositionDictEncodingWithInstance<bool>(
      arrayVector,
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

TEST_F(ArrayPositionTest, rowDictionaryEncodingWithInstance) {
  std::vector<std::vector<variant>> data{
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

  auto newSize = arrayVector->size() * 2;
  auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
  auto dictArrayVector = wrapInDictionary(indices, newSize, arrayVector);

  auto testPositionOfRow =
      [&](int32_t n,
          const char* color,
          int64_t instance,
          const std::vector<std::optional<int64_t>>& expected) {
        auto expectedVector = makeNullableFlatVector<int64_t>(expected);
        auto dictExpectedVector =
            wrapInDictionary(indices, newSize, expectedVector);

        auto searchVector = makeConstantRow(
            rowType, variant::row({n, color}), arrayVector->size());
        auto constIndices = makeIndices(newSize, [](auto row) { return 0; });
        auto dictSearchVector =
            wrapInDictionary(constIndices, newSize, searchVector);

        auto instanceVector = makeConstant(instance, arrayVector->size());
        auto dictInstanceVector =
            wrapInDictionary(constIndices, newSize, instanceVector);

        evalExpr(
            {dictArrayVector, dictSearchVector, dictInstanceVector},
            "array_position(c0, c1, c2)",
            dictExpectedVector);
      };

  testPositionOfRow(1, "red", 1, {1, 0, 0, 0, 0});
  testPositionOfRow(1, "red", -1, {4, 0, 0, 0, 0});
  testPositionOfRow(2, "blue", 3, {0, 0, 0, 0, 5});
  testPositionOfRow(2, "blue", -2, {0, 1, 0, 0, 4});
}

TEST_F(ArrayPositionTest, arrayWithInstance) {
  auto O = [](const std::vector<std::optional<int64_t>>& data) {
    return std::make_optional(data);
  };

  std::vector<std::optional<int64_t>> a{1, 2, 3};
  std::vector<std::optional<int64_t>> b{4, 5};
  std::vector<std::optional<int64_t>> c{6, 7, 8};
  std::vector<std::optional<int64_t>> d{1, 2};
  std::vector<std::optional<int64_t>> e{4, 3};
  auto arrayVector = makeNestedArrayVector<int64_t>(
      {{O(a), O(b), O(a)},
       {O(d), O(e)},
       {O(d), O(a), O(a)},
       {O(c), O(c)},
       {O({})},
       {O({std::nullopt})}});

  auto testPositionOfArray =
      [&](const std::vector<std::vector<std::optional<int64_t>>>& search,
          int64_t instance,
          const std::vector<std::optional<int64_t>>& expected) {
        evalExpr(
            {arrayVector,
             makeNullableArrayVector<int64_t>(search),
             makeConstant(instance, arrayVector->size())},
            "array_position(c0, c1, c2)",
            makeNullableFlatVector<int64_t>(expected));
      };

  testPositionOfArray({a, a, a, a, a, a}, 1, {1, 0, 2, 0, 0, 0});
  testPositionOfArray({a, a, a, a, a, a}, -1, {3, 0, 3, 0, 0, 0});
  testPositionOfArray({b, e, a, e, e, e}, -1, {2, 2, 3, 0, 0, 0});
  testPositionOfArray(
      {b, {std::nullopt}, a, {}, {}, {std::nullopt}}, 1, {2, 0, 2, 0, 1, 1});
}

TEST_F(ArrayPositionTest, mapWithInstance) {
  std::vector<std::string> colors{
      "red", "green", "blue", "yellow", "orange", "purple"};

  using S = StringView;
  using P = std::pair<int64_t, std::optional<S>>;

  std::vector<P> a{P{1, S{"red"}}, P{2, S{"blue"}}, P{3, S{"green"}}};
  std::vector<P> b{P{1, S{"yellow"}}, P{2, S{"orange"}}};
  std::vector<P> c{P{1, S{"red"}}, P{2, S{"yellow"}}, P{3, S{"purple"}}};
  std::vector<P> d{P{0, std::nullopt}};
  std::vector<std::vector<std::vector<P>>> data{{a, b, b}, {b, c}, {c, a, c}};
  auto arrayVector = makeArrayOfMapVector<int64_t, S>(data);

  auto testPositionOfMap =
      [&](const std::vector<std::vector<P>>& search,
          int64_t instance,
          const std::vector<std::optional<int64_t>>& expected) {
        evalExpr(
            {arrayVector,
             makeMapVector<int64_t, S>(search),
             makeConstant(instance, arrayVector->size())},
            "array_position(c0, c1, c2)",
            makeNullableFlatVector<int64_t>(expected));
      };

  testPositionOfMap({a, a, a}, 1, {1, 0, 2});
  testPositionOfMap({a, a, a}, 2, {0, 0, 0});
  testPositionOfMap({b, b, c}, -1, {3, 1, 3});
  testPositionOfMap({d, d, d}, 1, {0, 0, 0});
}
} // namespace
