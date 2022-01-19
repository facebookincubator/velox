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

class ArrayUnionTest : public FunctionBaseTest {
 protected:
  void evalExpr(
      const std::vector<VectorPtr>& input,
      const std::string& expression,
      const VectorPtr& expected) {
    auto result = evaluate<ArrayVector>(expression, makeRowVector(input));

    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testUnion(
      const TwoDimVector<T>& left,
      const TwoDimVector<T>& right,
      const TwoDimVector<T>& expected) {
    evalExpr(
        {makeNullableArrayVector<T>(left), makeNullableArrayVector<T>(right)},
        "array_union(c0, c1)",
        makeNullableArrayVector<T>(expected));
  }

  void testUnionDictionaryEncoding(
      const ArrayVectorPtr& left,
      const ArrayVectorPtr& right,
      const ArrayVectorPtr& expected) {
    auto newSize = left->size() * 2;
    auto indices = makeIndices(newSize, [](auto row) { return row / 2; });
    auto dictLeftVector = wrapInDictionary(indices, newSize, left);

    auto dictExpectedVector = wrapInDictionary(indices, newSize, expected);

    // Encode the right vector using a different instance of indices
    // to prevent peeling.
    auto rightIndices = makeIndices(newSize, [](auto row) { return row / 2; });
    auto dictRightVector = wrapInDictionary(rightIndices, newSize, right);

    evalExpr(
        {dictLeftVector, dictRightVector},
        "array_union(c0, c1)",
        dictExpectedVector);
  }

  template <typename T>
  void testUnionDictionaryEncoding(
      const TwoDimVector<T>& left,
      const TwoDimVector<T>& right,
      const TwoDimVector<T>& expected) {
    auto leftVector = makeNullableArrayVector<T>(left);
    auto rightVector = makeNullableArrayVector<T>(right);
    auto expectedVector = makeNullableArrayVector<T>(expected);

    testUnionDictionaryEncoding(leftVector, rightVector, expectedVector);
  }
};

TEST_F(ArrayUnionTest, integer) {
  TwoDimVector<int64_t> left{
      {1, std::nullopt, 3}, {3, 4, 5}, {}, {std::nullopt, std::nullopt}, {}};

  TwoDimVector<int64_t> right{
      {1, 4, std::nullopt},
      {2, 2},
      {std::nullopt},
      {1},
      {5, 6, std::nullopt, 8, 9}};

  TwoDimVector<int64_t> expected{
      {1, std::nullopt, 3, 4},
      {3, 4, 5, 2},
      {std::nullopt},
      {std::nullopt, 1},
      {5, 6, std::nullopt, 8, 9}};

  testUnion<int64_t>(left, right, expected);
  testUnion<int64_t>(
      TwoDimVector<int64_t>{{}},
      TwoDimVector<int64_t>{{}},
      TwoDimVector<int64_t>{{}});
  testUnion<int64_t>(
      TwoDimVector<int64_t>{{std::nullopt}},
      TwoDimVector<int64_t>{{std::nullopt}},
      TwoDimVector<int64_t>{{std::nullopt}});

  testUnionDictionaryEncoding<int64_t>(left, right, expected);
  testUnionDictionaryEncoding<int64_t>(
      TwoDimVector<int64_t>{{}, {std::nullopt}},
      TwoDimVector<int64_t>{{std::nullopt}, {}},
      TwoDimVector<int64_t>{{std::nullopt}, {std::nullopt}});
}

TEST_F(ArrayUnionTest, varchar) {
  TwoDimVector<StringView> left{
      {"red"_sv},
      {std::nullopt, "yellow"_sv},
      {},
      {std::nullopt},
      {"red"_sv, "purple"_sv}};

  TwoDimVector<StringView> right{
      {"blue"_sv, "blue"_sv},
      {"orange"_sv, std::nullopt},
      {std::nullopt},
      {"blue"_sv},
      {"green"_sv, "purple"_sv}};

  TwoDimVector<StringView> expected{
      {"red"_sv, "blue"_sv},
      {std::nullopt, "yellow"_sv, "orange"_sv},
      {std::nullopt},
      {std::nullopt, "blue"_sv},
      {"red"_sv, "purple"_sv, "green"_sv}};

  testUnion<StringView>(left, right, expected);
  testUnion<StringView>(
      TwoDimVector<StringView>{{}},
      TwoDimVector<StringView>{{}},
      TwoDimVector<StringView>{{}});
  testUnion<StringView>(
      TwoDimVector<StringView>{{std::nullopt}},
      TwoDimVector<StringView>{{std::nullopt}},
      TwoDimVector<StringView>{{std::nullopt}});

  testUnionDictionaryEncoding<StringView>(left, right, expected);
  testUnionDictionaryEncoding<StringView>(
      TwoDimVector<StringView>{{}, {std::nullopt}},
      TwoDimVector<StringView>{{std::nullopt}, {}},
      TwoDimVector<StringView>{{std::nullopt}, {std::nullopt}});
}

TEST_F(ArrayUnionTest, boolean) {
  TwoDimVector<bool> left{
      {true, false},
      {true},
      {false},
      {},
      {true},
      {true, std::nullopt},
      {false, false, false}};

  TwoDimVector<bool> right{
      {false},
      {false},
      {},
      {std::nullopt},
      {std::nullopt},
      {std::nullopt, false},
      {false, false}};

  TwoDimVector<bool> expected{
      {true, false},
      {true, false},
      {false},
      {std::nullopt},
      {true, std::nullopt},
      {true, std::nullopt, false},
      {false}};

  testUnion<bool>(left, right, expected);
  testUnion<bool>(
      TwoDimVector<bool>{{}}, TwoDimVector<bool>{{}}, TwoDimVector<bool>{{}});
  testUnion<bool>(
      TwoDimVector<bool>{{std::nullopt}},
      TwoDimVector<bool>{{std::nullopt}},
      TwoDimVector<bool>{{std::nullopt}});

  testUnionDictionaryEncoding<bool>(left, right, expected);
  testUnionDictionaryEncoding<bool>(
      TwoDimVector<bool>{{}, {std::nullopt}},
      TwoDimVector<bool>{{std::nullopt}, {}},
      TwoDimVector<bool>{{std::nullopt}, {std::nullopt}});
}

TEST_F(ArrayUnionTest, timestamp) {
  using T = Timestamp;

  TwoDimVector<Timestamp> left{
      {T{0, 0}},
      {std::nullopt, T{1000, 1000}},
      {},
      {std::nullopt, std::nullopt},
      {T{0, 0}, T{1000, 1000}}};

  TwoDimVector<Timestamp> right{
      {T{1000, 1000}, T{1000, 1000}},
      {T{2000, 2000}, std::nullopt},
      {std::nullopt},
      {T{1000, 1000}},
      {T{3000, 3000}, T{1000, 1000}}};

  TwoDimVector<Timestamp> expected{
      {T{0, 0}, T{1000, 1000}},
      {std::nullopt, T{1000, 1000}, T{2000, 2000}},
      {std::nullopt},
      {std::nullopt, T{1000, 1000}},
      {T{0, 0}, T{1000, 1000}, T{3000, 3000}}};

  testUnion<Timestamp>(left, right, expected);
  testUnion<Timestamp>(
      TwoDimVector<Timestamp>{{}},
      TwoDimVector<Timestamp>{{}},
      TwoDimVector<Timestamp>{{}});
  testUnion<Timestamp>(
      TwoDimVector<Timestamp>{{std::nullopt}},
      TwoDimVector<Timestamp>{{std::nullopt}},
      TwoDimVector<Timestamp>{{std::nullopt}});

  testUnionDictionaryEncoding<Timestamp>(left, right, expected);
  testUnionDictionaryEncoding<Timestamp>(
      TwoDimVector<Timestamp>{{}, {std::nullopt}},
      TwoDimVector<Timestamp>{{std::nullopt}, {}},
      TwoDimVector<Timestamp>{{std::nullopt}, {std::nullopt}});
}

TEST_F(ArrayUnionTest, row) {
  using TwoDimVariant = std::vector<std::vector<variant>>;

  TwoDimVariant left{
      {
          variant::row({1, "red"}),
          variant::row({2, "blue"}),
          variant::row({2, "blue"}),
      },
      {
          variant::row({2, "blue"}),
          variant(TypeKind::ROW), // null
          variant::row({5, "green"}),
      },
      {},
      {
          variant(TypeKind::ROW), // null
      }};

  TwoDimVariant right{
      {
          variant::row({4, "blue"}),
          variant::row({3, "green"}),
          variant::row({3, "green"}),
      },
      {
          variant(TypeKind::ROW), // null
          variant::row({5, "red"}),
      },
      {
          variant(TypeKind::ROW), // null
      },
      {
          variant::row({5, "red"}),
      }};

  TwoDimVariant expected{
      {
          variant::row({1, "red"}),
          variant::row({2, "blue"}),
          variant::row({4, "blue"}),
          variant::row({3, "green"}),
      },
      {
          variant::row({2, "blue"}),
          variant(TypeKind::ROW), // null
          variant::row({5, "green"}),
          variant::row({5, "red"}),
      },
      {
          variant(TypeKind::ROW), // null
      },
      {
          variant(TypeKind::ROW), // null
          variant::row({5, "red"}),
      }};

  auto testUnionOfRow = [&](const TwoDimVariant& left,
                            const TwoDimVariant& right,
                            const TwoDimVariant& expected) {
    auto rowType = ROW({INTEGER(), VARCHAR()});
    auto leftVector = makeArrayOfRowVector(rowType, left);
    auto rightVector = makeArrayOfRowVector(rowType, right);
    auto expectedVector = makeArrayOfRowVector(rowType, expected);

    evalExpr({leftVector, rightVector}, "array_union(c0, c1)", expectedVector);
  };

  testUnionOfRow(left, right, expected);
  testUnionOfRow(TwoDimVariant{{}}, TwoDimVariant{{}}, TwoDimVariant{{}});
  testUnionOfRow(
      TwoDimVariant{{variant(TypeKind::ROW)}},
      TwoDimVariant{{variant(TypeKind::ROW)}},
      TwoDimVariant{{variant(TypeKind::ROW)}});

  auto testUnionOfRowDictionaryEncoding = [&](const TwoDimVariant& left,
                                              const TwoDimVariant& right,
                                              const TwoDimVariant& expected) {
    auto rowType = ROW({INTEGER(), VARCHAR()});
    auto leftVector = makeArrayOfRowVector(rowType, left);
    auto rightVector = makeArrayOfRowVector(rowType, right);
    auto expectedVector = makeArrayOfRowVector(rowType, expected);

    testUnionDictionaryEncoding(leftVector, rightVector, expectedVector);
  };

  testUnionOfRowDictionaryEncoding(left, right, expected);
  testUnionOfRowDictionaryEncoding(
      TwoDimVariant{{}, {variant(TypeKind::ROW)}},
      TwoDimVariant{{variant(TypeKind::ROW)}, {}},
      TwoDimVariant{{variant(TypeKind::ROW)}, {variant(TypeKind::ROW)}});
}

TEST_F(ArrayUnionTest, array) {
  using ThreeDimVector = std::vector<
      std::vector<std::optional<std::vector<std::optional<int64_t>>>>>;
  using OneDimVector = std::vector<std::optional<int64_t>>;

  auto O = [](const std::vector<std::optional<int64_t>>& data) {
    return std::make_optional(data);
  };

  OneDimVector a{1, 2, 3};
  OneDimVector b{4, 5};
  OneDimVector c{6, 7, 8};
  OneDimVector d{1, 2};
  OneDimVector e{4, 3};

  ThreeDimVector left{
      {O(a), O(b)}, {O(d)}, {}, {O({std::nullopt})}, {O(a), O({std::nullopt})}};
  ThreeDimVector right{
      {O(a), O(e)},
      {O(a)},
      {O({std::nullopt})},
      {O(c)},
      {O(b), O({std::nullopt})}};
  ThreeDimVector expected{
      {O(a), O(b), O(e)},
      {O(d), O(a)},
      {O({std::nullopt})},
      {O({std::nullopt}), O(c)},
      {O(a), O({std::nullopt}), O(b)}};

  auto testUnionOfArray = [&](const ThreeDimVector& left,
                              const ThreeDimVector& right,
                              const ThreeDimVector& expected) {
    auto leftVector = makeNestedArrayVector<int64_t>(left);
    auto rightVector = makeNestedArrayVector<int64_t>(right);
    auto expectedVector = makeNestedArrayVector<int64_t>(expected);

    evalExpr({leftVector, rightVector}, "array_union(c0, c1)", expectedVector);
  };

  testUnionOfArray(left, right, expected);
  testUnionOfArray(
      ThreeDimVector{{OneDimVector{}}},
      ThreeDimVector{{OneDimVector{}}},
      ThreeDimVector{{OneDimVector{}}});
  testUnionOfArray(
      ThreeDimVector{{OneDimVector{std::nullopt}}},
      ThreeDimVector{{OneDimVector{std::nullopt}}},
      ThreeDimVector{{OneDimVector{std::nullopt}}});
  testUnionOfArray(
      ThreeDimVector{{std::nullopt}},
      ThreeDimVector{{std::nullopt}},
      ThreeDimVector{{std::nullopt}});

  auto testUnionOfArrayDictionaryEncoding =
      [&](const ThreeDimVector& left,
          const ThreeDimVector& right,
          const ThreeDimVector& expected) {
        auto leftVector = makeNestedArrayVector<int64_t>(left);
        auto rightVector = makeNestedArrayVector<int64_t>(right);
        auto expectedVector = makeNestedArrayVector<int64_t>(expected);

        testUnionDictionaryEncoding(leftVector, rightVector, expectedVector);
      };

  testUnionOfArrayDictionaryEncoding(left, right, expected);
  testUnionOfArrayDictionaryEncoding(
      ThreeDimVector{
          {OneDimVector{std::nullopt}}, {OneDimVector{}}, {std::nullopt}},
      ThreeDimVector{{OneDimVector{}}, {OneDimVector{}}, {std::nullopt}},
      ThreeDimVector{
          {OneDimVector{std::nullopt}, OneDimVector{}},
          {OneDimVector{}},
          {std::nullopt}});
}

TEST_F(ArrayUnionTest, map) {
  using P = std::pair<int64_t, std::optional<StringView>>;
  using ThreeDimVector = std::vector<std::vector<std::vector<P>>>;

  std::vector<P> a{P{1, "red"_sv}, P{2, "blue"_sv}, P{3, "green"_sv}};
  std::vector<P> b{P{1, "yellow"_sv}, P{2, "orange"_sv}};
  std::vector<P> c{P{1, "red"_sv}, P{2, "yellow"_sv}, P{3, "purple"_sv}};
  std::vector<P> d{P{0, std::nullopt}};

  ThreeDimVector left{{a, b}, {}, {c}, {d}};
  ThreeDimVector right{{c, a}, {b}, {}, {d}};
  ThreeDimVector expected{{a, b, c}, {b}, {c}, {d}};

  auto testUnionOfMap = [&](const ThreeDimVector& left,
                            const ThreeDimVector& right,
                            const ThreeDimVector& expected) {
    auto leftVector = makeArrayOfMapVector<int64_t, StringView>(left);
    auto rightVector = makeArrayOfMapVector<int64_t, StringView>(right);
    auto expectedVector = makeArrayOfMapVector<int64_t, StringView>(expected);

    evalExpr({leftVector, rightVector}, "array_union(c0, c1)", expectedVector);
  };

  testUnionOfMap(left, right, expected);
  testUnionOfMap(ThreeDimVector{{}}, ThreeDimVector{{}}, ThreeDimVector{{}});
  testUnionOfMap(
      ThreeDimVector{{{}}}, ThreeDimVector{{}}, ThreeDimVector{{{}}});

  auto testUnionOfMapDictionaryEncoding = [&](const ThreeDimVector& left,
                                              const ThreeDimVector& right,
                                              const ThreeDimVector& expected) {
    auto leftVector = makeArrayOfMapVector<int64_t, StringView>(left);
    auto rightVector = makeArrayOfMapVector<int64_t, StringView>(right);
    auto expectedVector = makeArrayOfMapVector<int64_t, StringView>(expected);

    testUnionDictionaryEncoding(leftVector, rightVector, expectedVector);
  };

  testUnionOfMapDictionaryEncoding(left, right, expected);
  testUnionOfMapDictionaryEncoding(
      ThreeDimVector{{}, {{}}},
      ThreeDimVector{{}, {}},
      ThreeDimVector{{}, {{}}});
}

} // namespace
