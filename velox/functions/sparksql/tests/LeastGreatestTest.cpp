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

#include <cmath>
#include <limits>
#include <type_traits>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using facebook::velox::test::assertEqualVectors;

class LeastTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> least(
      std::optional<T> arg0,
      std::optional<T> arg1,
      std::optional<T> arg2,
      const TypePtr& type = CppToType<T>::create()) {
    return evaluateOnce<T>(
        "least(c0, c1, c2)", {type, type, type}, arg0, arg1, arg2);
  }

  template <typename T>
  std::optional<T> least(
      std::optional<T> arg0,
      std::optional<T> arg1,
      std::optional<T> arg2,
      std::optional<T> arg3,
      const TypePtr& type = CppToType<T>::create()) {
    return evaluateOnce<T>(
        "least(c0, c1, c2, c3)",
        {type, type, type, type},
        arg0,
        arg1,
        arg2,
        arg3);
  }

  template <typename T>
  void flat(const TypePtr& type = CppToType<T>::create()) {
    vector_size_t size = 20;

    // {0, null, null, 3, null, null, 6, null, null, ...}.
    auto first = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row; },
        [](vector_size_t row) { return row % 3 != 0; },
        type);

    // {0, 10, null, 30, 40, null, 60, 70, null, ...}.
    auto second = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row * 10; },
        nullEvery(3, 2),
        type);

    // {0, 100, 200, 300, 400, 500, 600, 700, 800, ...}.
    auto third = makeFlatVector<T>(
        size, [](vector_size_t row) { return row * 100; }, nullptr, type);

    auto data = makeRowVector({first, second, third});

    // Expect {0, 10, 200, 3, 40, 500, 6, 70, 800, ...}.
    auto expected = [](vector_size_t row) {
      return std::min<T>(
          {row % 3 == 0 ? T(row) : std::numeric_limits<T>::max(),
           row % 3 == 2 ? std::numeric_limits<T>::max() : T(10 * row),
           T(100 * row)});
    };

    auto result = evaluate<FlatVector<T>>("least(c0, c1, c2)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }

    result = evaluate<FlatVector<T>>("least(c2, c1, c0)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }

    result = evaluate<FlatVector<T>>("least(c1, c0, c2)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }
  }

  template <typename T>
  void constant(const TypePtr& type = CppToType<T>::create()) {
    vector_size_t size = 20;

    // {0, null, null, 3, null, null, 6, null, null, ...}.
    auto first = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row; },
        [](vector_size_t row) { return row % 3 != 0; },
        type);

    // {9, 9, 9, ...}.
    auto second = makeConstant<T>(9, size, type);

    auto data = makeRowVector({first, second});

    // Expect {0, 9, 9, 3, 9, 9, 6, 9, 9, ...}.
    auto expected = [](vector_size_t row) {
      return std::min<T>(9, row % 3 == 0 ? row : std::numeric_limits<T>::max());
    };

    auto result = evaluate<FlatVector<T>>("least(c0, c1)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }
  }
};

TEST_F(LeastTest, integral) {
  flat<int16_t>();
  constant<int16_t>();

  flat<int32_t>();
  constant<int32_t>();

  flat<int64_t>();
  constant<int64_t>();

  flat<int32_t>(DATE());
  constant<int32_t>(DATE());
}

TEST_F(LeastTest, floating) {
  flat<float>();
  constant<float>();

  flat<double>();
  constant<double>();
}

TEST_F(LeastTest, floatingPointTiesAndFirstNaN) {
  auto verify = [&](auto value) {
    using T = decltype(value);
    const auto nan = std::numeric_limits<T>::quiet_NaN();
    const auto inf = std::numeric_limits<T>::infinity();
    for (const std::string name : {"least", "greatest"}) {
      SCOPED_TRACE(fmt::format("{} bytes={}", name, sizeof(T)));
      for (const T first : {T{0}, -T{0}}) {
        const T second = std::signbit(first) ? T{0} : -T{0};
        const auto result = evaluateOnce<T>(
            name + "(c0, c1)",
            std::optional<T>{first},
            std::optional<T>{second});
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, T{0});
        EXPECT_EQ(std::signbit(*result), std::signbit(first));
      }
      for (const T second : {T{-1}, T{1}, -inf, inf, nan}) {
        const auto result = evaluateOnce<T>(
            name + "(c0, c1)", std::optional<T>{nan}, std::optional<T>{second});
        ASSERT_TRUE(result.has_value());
        if (name == "greatest" || std::isnan(second)) {
          EXPECT_TRUE(std::isnan(*result));
        } else {
          EXPECT_EQ(*result, second);
        }
      }
      const auto result = evaluateOnce<T>(
          name + "(c0, c1)", std::optional<T>{nan}, std::optional<T>{});
      ASSERT_TRUE(result.has_value());
      EXPECT_TRUE(std::isnan(*result));
    }
  };
  verify(float{});
  verify(double{});
}

TEST_F(LeastTest, edgecases) {
  // All inputs are null.
  std::optional<int32_t> null;
  EXPECT_EQ(least<int32_t>(null, null, null), null);

  // IEEE-floating point exceptions.
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();

  EXPECT_EQ(least<float>(inf, nan, -inf, 0.f), -inf);
  EXPECT_EQ(least<float>(nan, inf, 0.f), 0.f);
  EXPECT_EQ(least<float>(nan, nan, inf), inf);
  EXPECT_TRUE(std::isnan(least<float>(nan, nan, nan).value()));
}

TEST_F(LeastTest, boolean) {
  EXPECT_EQ(least<bool>(true, false, false), false);
}

TEST_F(LeastTest, string) {
  EXPECT_EQ(least<std::string>("b", "abcde", "abcdefg"), "abcde");
}

TEST_F(LeastTest, timestamp) {
  EXPECT_EQ(
      least<Timestamp>(
          Timestamp(1569, 25), Timestamp(4859, 482), Timestamp(581, 1651)),
      Timestamp(581, 1651));
}

TEST_F(LeastTest, date) {
  EXPECT_EQ(least<int32_t>(100, 1000, 10000, DATE()), 100);
}

TEST_F(LeastTest, decimal) {
  flat<int64_t>(DECIMAL(6, 2));
  constant<int64_t>(DECIMAL(6, 2));

  flat<int128_t>(DECIMAL(28, 12));
  constant<int128_t>(DECIMAL(28, 12));
}

TEST_F(LeastTest, noArtificialNulls) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({0, 16, 17, 18}),
      makeConstant<int64_t>(17, 4),
  });
  for (const std::string name : {"least", "greatest"}) {
    SCOPED_TRACE(name);
    auto result = evaluate(name + "(c0, c1)", input);
    assertEqualVectors(
        name == "least" ? makeFlatVector<int64_t>({0, 16, 17, 17})
                        : makeFlatVector<int64_t>({17, 17, 17, 18}),
        result);
    EXPECT_EQ(result->rawNulls(), nullptr);
  }
}

TEST_F(LeastTest, firstArgumentEncodingsAndNulls) {
  auto first = wrapInDictionary(
      makeIndices({1, 0, 3, 2, 5, 4}),
      makeNullableFlatVector<int64_t>(
          {0, std::nullopt, 30, std::nullopt, 50, 60}));
  auto second = makeNullableFlatVector<int64_t>(
      {std::nullopt, 7, 8, std::nullopt, 40, 70});
  for (const std::string name : {"least", "greatest"}) {
    SCOPED_TRACE(name);
    assertEqualVectors(
        name == "least"
            ? makeNullableFlatVector<int64_t>({std::nullopt, 0, 8, 30, 40, 50})
            : makeNullableFlatVector<int64_t>({std::nullopt, 7, 8, 30, 60, 70}),
        evaluate(name + "(c0, c1)", makeRowVector({first, second})));
    assertEqualVectors(
        second,
        evaluate(name + "(cast(null as bigint), c0)", makeRowVector({second})));
    assertEqualVectors(
        name == "least" ? makeFlatVector<int64_t>({17, 7, 8, 17, 17, 17})
                        : makeFlatVector<int64_t>({17, 17, 17, 17, 40, 70}),
        evaluate(name + "(cast(17 as bigint), c0)", makeRowVector({second})));
  }
}

TEST_F(LeastTest, selectedRowsAndReusedResult) {
  auto input = makeRowVector({
      makeNullableFlatVector<int64_t>(
          {std::nullopt, 0, std::nullopt, 30, 60, 50}),
      makeNullableFlatVector<int64_t>(
          {std::nullopt, 7, 8, std::nullopt, 40, 70}),
  });
  SelectivityVector rows(6, false);
  for (const auto row : {0, 1, 2, 4}) {
    rows.setValid(row, true);
  }
  rows.updateBounds();
  auto original = makeNullableFlatVector<int64_t>(
      {101, std::nullopt, 103, 104, std::nullopt, 106});
  for (const std::string name : {"least", "greatest"}) {
    for (const bool shared : {false, true}) {
      SCOPED_TRACE(fmt::format("{} shared={}", name, shared));
      VectorPtr result = BaseVector::copy(*original, pool());
      VectorPtr alias = shared ? result : nullptr;
      evaluate<FlatVector<int64_t>>(name + "(c0, c1)", input, rows, result);
      assertEqualVectors(
          name == "least" ? makeNullableFlatVector<int64_t>(
                                {std::nullopt, 0, 8, 104, 40, 106})
                          : makeNullableFlatVector<int64_t>(
                                {std::nullopt, 7, 8, 104, 60, 106}),
          result);
      if (alias) {
        assertEqualVectors(original, alias);
      }
    }
  }
}

TEST_F(LeastTest, booleanBitmapBoundariesAndReusedResult) {
  constexpr vector_size_t size = 130;
  std::vector<std::optional<bool>> left(size), right(size), sentinels(size);
  for (vector_size_t row = 0; row < size; ++row) {
    if (row % 3 != 0 && row != 64 && row != 128) {
      left[row] = row % 4 < 2;
    }
    if (row % 7 != 0 && row != 64 && row != 128) {
      right[row] = row % 5 < 3;
    }
    if (row % 13 != 2) {
      sentinels[row] = row % 2 == 0;
    }
  }
  auto original = makeNullableFlatVector<bool>(sentinels);
  auto second = makeNullableFlatVector<bool>(right);
  for (const std::string encoding :
       {"flat",
        "constant-false",
        "constant-true",
        "constant-null",
        "dictionary"}) {
    VectorPtr first;
    auto firstValues = left;
    if (encoding == "flat") {
      first = makeNullableFlatVector<bool>(left);
    } else if (encoding == "dictionary") {
      first = BaseVector::wrapInDictionary(
          makeNulls(
              size,
              [](auto row) {
                return row % 11 == 0 || row == 64 || row == 128;
              }),
          makeIndicesInReverse(size),
          size,
          makeNullableFlatVector<bool>(left));
      for (vector_size_t row = 0; row < size; ++row) {
        firstValues[row] = row % 11 == 0 || row == 64 || row == 128
            ? std::nullopt
            : left[size - row - 1];
      }
    } else {
      const std::optional<bool> value = encoding == "constant-null"
          ? std::nullopt
          : std::optional<bool>(encoding == "constant-true");
      first = value.has_value()
          ? makeConstant<bool>(*value, size)
          : BaseVector::createNullConstant(BOOLEAN(), size, pool());
      std::fill(firstValues.begin(), firstValues.end(), value);
    }
    auto input = makeRowVector({first, second});
    for (const bool sparse : {false, true}) {
      SelectivityVector rows(size, false);
      for (vector_size_t row = 0; row < size; ++row) {
        rows.setValid(
            row,
            sparse ? row % 3 == 1 || row == 63 || row == 128
                   : row >= 1 && row <= 128);
      }
      rows.updateBounds();
      for (const std::string name : {"least", "greatest"}) {
        auto expected = sentinels;
        rows.applyToSelected([&](auto row) {
          const auto a = firstValues[row];
          const auto b = right[row];
          if (!a.has_value()) {
            expected[row] = b;
          } else if (!b.has_value()) {
            expected[row] = a;
          } else {
            expected[row] = name == "least" ? *a && *b : *a || *b;
          }
        });
        for (const bool shared : {false, true}) {
          SCOPED_TRACE(
              fmt::format(
                  "{} {} sparse={} shared={}", name, encoding, sparse, shared));
          VectorPtr result = BaseVector::copy(*original, pool());
          VectorPtr alias = shared ? result : nullptr;
          evaluate<FlatVector<bool>>(name + "(c0, c1)", input, rows, result);
          ASSERT_EQ(result->size(), size);
          assertEqualVectors(makeNullableFlatVector<bool>(expected), result);
          const auto* values =
              result->asFlatVector<bool>()->rawValues<uint64_t>();
          const auto* originalValues = original->rawValues<uint64_t>();
          for (vector_size_t row = 0; row < size; ++row) {
            if (!rows.isValid(row)) {
              EXPECT_EQ(result->isNullAt(row), original->isNullAt(row)) << row;
              EXPECT_EQ(
                  bits::isBitSet(values, row),
                  bits::isBitSet(originalValues, row))
                  << row;
            }
          }
          if (alias) {
            assertEqualVectors(original, alias);
          }
          assertEqualVectors(makeNullableFlatVector<bool>(firstValues), first);
          assertEqualVectors(makeNullableFlatVector<bool>(right), second);
        }
      }
    }
  }
}

TEST_F(LeastTest, lazyFirstArgumentSelectedRowsAndLifetime) {
  constexpr vector_size_t size = 6;
  const std::vector<vector_size_t> selected{0, 2, 4};
  SelectivityVector rows(size, false);
  for (const auto row : selected) {
    rows.setValid(row, true);
  }
  rows.updateBounds();

  auto verify = [&](auto value) {
    using T = decltype(value);
    auto firstValue = [](vector_size_t row) -> T {
      if constexpr (std::is_same_v<T, std::string>) {
        return std::string(64, 'm') + std::to_string(row);
      } else {
        return 40 + row * 10;
      }
    };
    auto secondValue = [](vector_size_t row) -> T {
      if constexpr (std::is_same_v<T, std::string>) {
        return std::string(64, row == 4 ? 'z' : 'a') + std::to_string(row);
      } else {
        return row == 4 ? 100 : 10;
      }
    };
    const T sentinel = []() -> T {
      if constexpr (std::is_same_v<T, std::string>) {
        return "unchanged";
      } else {
        return -1;
      }
    }();
    for (const std::string name : {"least", "greatest"}) {
      for (const bool shared : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "{} {} shared={}",
                name,
                CppToType<T>::create()->toString(),
                shared));
        std::vector<vector_size_t> loadedRows;
        auto first = vectorMaker_.lazyFlatVector<T>(
            size, firstValue, [&](vector_size_t row) {
              loadedRows.push_back(row);
              return row == 2;
            });
        auto second = makeFlatVector<T>(
            size, secondValue, [](vector_size_t row) { return row == 2; });
        auto input = makeRowVector({first, second});
        VectorPtr result = makeFlatVector<T>(std::vector<T>(size, sentinel));
        VectorPtr alias = shared ? result : nullptr;
        ASSERT_FALSE(first->isLoaded());
        evaluate<BaseVector>(name + "(c0, c1)", input, rows, result);
        EXPECT_TRUE(first->isLoaded());
        EXPECT_EQ(loadedRows, selected);
        input.reset();
        first.reset();
        second.reset();

        std::vector<std::optional<T>> expected(size, sentinel);
        expected[2] = std::nullopt;
        for (const auto row : {0, 4}) {
          expected[row] = name == "least"
              ? std::min(firstValue(row), secondValue(row))
              : std::max(firstValue(row), secondValue(row));
        }
        assertEqualVectors(makeNullableFlatVector<T>(expected), result);
        if (alias) {
          assertEqualVectors(
              makeFlatVector<T>(std::vector<T>(size, sentinel)), alias);
        }
      }
    }
  };
  verify(int64_t{});
  verify(std::string{});
}

TEST_F(LeastTest, copiedStringMetadataAndLifetime) {
  const SelectivityVector rows(2);
  for (const std::string name : {"least", "greatest"}) {
    SCOPED_TRACE(name);
    auto first =
        makeFlatVector<std::string>({"m-first-argument", "n-first-argument"});
    first->setIsAscii(true, rows);
    const std::string replacement =
        name == "least" ? "a\xC3\xA9-long-result" : "z\xC3\xA9-long-result";
    auto second =
        makeFlatVector<std::string>({replacement, name == "least" ? "z" : "a"});
    auto result = evaluate<FlatVector<StringView>>(
        name + "(c0, c1)", makeRowVector({first, second}));
    first.reset();
    second.reset();
    assertEqualVectors(
        makeFlatVector<std::string>({replacement, "n-first-argument"}), result);
    EXPECT_FALSE(result->isAscii(rows).value_or(false));
    assertEqualVectors(
        makeFlatVector<int32_t>({14, 16}),
        evaluate("length(c0)", makeRowVector({result})));
  }
}

TEST_F(LeastTest, selectedStringMetadataAndReusedResult) {
  const SelectivityVector allRows(6);
  SelectivityVector selected(6, false);
  selected.setValid(1, true);
  selected.setValid(4, true);
  selected.updateBounds();
  auto unselected = allRows;
  unselected.deselect(selected);
  for (const std::string name : {"least", "greatest"}) {
    auto first =
        makeFlatVector<std::string>({"m0", "m1", "m2", "m3", "m4", "m5"});
    first->setIsAscii(true, allRows);
    const std::string replacement = name == "least" ? "a\xC3\xA9" : "z\xC3\xA9";
    const std::string asciiReplacement = name == "least" ? "b4" : "y4";
    auto input = makeRowVector({
        first,
        makeFlatVector<std::string>(
            {"unused",
             replacement,
             "unused",
             "unused",
             asciiReplacement,
             "unused"}),
    });
    auto original = makeFlatVector<std::string>(
        {"keep-0", "keep-1", "keep-2", "keep-3", "keep-4", "keep-5"});
    original->setIsAscii(true, allRows);
    for (const bool shared : {false, true}) {
      SCOPED_TRACE(fmt::format("{} shared={}", name, shared));
      VectorPtr result = BaseVector::copy(*original, pool());
      result->asFlatVector<StringView>()->setIsAscii(true, allRows);
      VectorPtr alias = shared ? result : nullptr;
      ASSERT_EQ(result->asFlatVector<StringView>()->isAscii(allRows), true);
      evaluate<FlatVector<StringView>>(
          name + "(c0, c1)", input, selected, result);
      EXPECT_FALSE(
          result->asFlatVector<StringView>()->isAscii(selected).has_value());
      EXPECT_EQ(result->asFlatVector<StringView>()->isAscii(unselected), true);
      assertEqualVectors(
          makeFlatVector<std::string>(
              {"keep-0",
               replacement,
               "keep-2",
               "keep-3",
               asciiReplacement,
               "keep-5"}),
          result);
      assertEqualVectors(
          makeFlatVector<int32_t>({6, 2, 6, 6, 2, 6}),
          evaluate("length(c0)", makeRowVector({result})));
      if (alias) {
        assertEqualVectors(original, alias);
        EXPECT_EQ(alias->asFlatVector<StringView>()->isAscii(allRows), true);
      }
    }
  }
}

TEST_F(LeastTest, stringNullPolicy) {
  const std::string longA(64, 'a');
  const std::string longC(64, 'c');
  const std::string longZ(64, 'z');
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    auto first = makeNullableFlatVector<std::string>(
        {std::nullopt, "b", "b", "", std::nullopt, std::nullopt, longZ, "tie"},
        type);
    auto input = makeRowVector({
        first,
        makeNullableFlatVector<std::string>(
            {"c", std::nullopt, "a", "b", std::nullopt, "x", longA, "tie"},
            type),
        makeNullableFlatVector<std::string>(
            {"b",
             "a",
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             "z"},
            type),
        makeNullableFlatVector<std::string>(
            {"d",
             "d",
             "c",
             "d",
             std::nullopt,
             std::nullopt,
             longC,
             std::nullopt},
            type),
    });
    for (const std::string name : {"least", "greatest"}) {
      SCOPED_TRACE(fmt::format("{} {}", name, type->toString()));
      assertEqualVectors(
          name == "least"
              ? makeNullableFlatVector<std::string>(
                    {"b", "a", "a", "", std::nullopt, "x", longA, "tie"}, type)
              : makeNullableFlatVector<std::string>(
                    {"d", "d", "c", "d", std::nullopt, "x", longZ, "z"}, type),
          evaluate(name + "(c0, c1, c2, c3)", input));
      for (const std::string args : {"(c0, c1)", "(c1, c0)"}) {
        assertEqualVectors(
            name == "least"
                ? makeNullableFlatVector<std::string>(
                      {"c", "b", "a", "", std::nullopt, "x", longA, "tie"},
                      type)
                : makeNullableFlatVector<std::string>(
                      {"c", "b", "b", "b", std::nullopt, "x", longZ, "tie"},
                      type),
            evaluate(name + args, input));
      }
      const auto castType =
          type->kind() == TypeKind::VARCHAR ? "varchar" : "varbinary";
      assertEqualVectors(
          first,
          evaluate(
              fmt::format("{}(cast(null as {}), c0)", name, castType), input));
      auto allNull = evaluate(
          fmt::format(
              "{}(cast(null as {}), cast(null as {}))",
              name,
              castType,
              castType),
          input);
      for (vector_size_t row = 0; row < input->size(); ++row) {
        EXPECT_TRUE(allNull->isNullAt(row));
      }
    }
  }
}

TEST_F(LeastTest, stringAndBinaryOrdering) {
  const std::string nul("\0", 1);
  const std::string high("\x80", 1);
  const std::string highest("\xFF", 1);
  const std::string nulTail("a\0b", 3);
  for (const std::string name : {"least", "greatest"}) {
    assertEqualVectors(
        name == "least"
            ? makeFlatVector<std::string>({"z", "", "same"})
            : makeFlatVector<std::string>({"\xC3\xA9", "\xE4\xB8\xAD", "same"}),
        evaluate(
            name + "(c0, c1)",
            makeRowVector({
                makeFlatVector<std::string>({"\xC3\xA9", "", "same"}),
                makeFlatVector<std::string>({"z", "\xE4\xB8\xAD", "same"}),
            })));
    auto result = evaluate(
        name + "(c0, c1)",
        makeRowVector({
            makeFlatVector<std::string>(
                {high, nul, nulTail, highest, ""}, VARBINARY()),
            makeFlatVector<std::string>(
                {highest, high, std::string("a\0c", 3), high, nul},
                VARBINARY()),
        }));
    assertEqualVectors(
        name == "least"
            ? makeFlatVector<std::string>(
                  {high, nul, nulTail, high, ""}, VARBINARY())
            : makeFlatVector<std::string>(
                  {highest, high, std::string("a\0c", 3), highest, nul},
                  VARBINARY()),
        result);
  }
}

TEST_F(LeastTest, stringDictionarySelectionAndLifetime) {
  constexpr vector_size_t size = 130;
  auto otherPool = rootPool_->addLeafChild("least-destination");
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    std::vector<std::optional<std::string>> left(size), right(size);
    for (vector_size_t row = 0; row < size; ++row) {
      if (row % 3 != 0) {
        left[row] = fmt::format("z-long-first-argument-{}", row);
      }
      if (row % 5 != 0) {
        right[row] = fmt::format("a-long-second-argument-{}", row);
      }
    }
    SelectivityVector rows(size, false);
    for (vector_size_t row = 0; row < size; row += 2) {
      rows.setValid(row, true);
    }
    rows.updateBounds();
    for (const std::string name : {"least", "greatest"}) {
      std::vector<std::optional<std::string>> expected(size, "keep");
      for (vector_size_t row = 0; row < size; row += 2) {
        const auto first = row % 7 == 0 ? std::nullopt : left[size - row - 1];
        expected[row] = name == "least"
            ? (right[row].has_value() ? right[row] : first)
            : (first.has_value() ? first : right[row]);
      }
      for (const bool shared : {false, true}) {
        auto first = BaseVector::wrapInDictionary(
            makeNulls(size, [](auto row) { return row % 7 == 0; }),
            makeIndicesInReverse(size),
            size,
            makeNullableFlatVector<std::string>(left, type));
        auto second = makeNullableFlatVector<std::string>(right, type);
        auto input = makeRowVector({first, second});
        auto original = makeFlatVector<std::string>(
            std::vector<std::string>(size, "keep"), type);
        VectorPtr result = BaseVector::copy(*original, otherPool.get());
        VectorPtr alias = shared ? result : nullptr;
        evaluate<FlatVector<StringView>>(
            name + "(c0, c1)", input, rows, result);
        input.reset();
        first.reset();
        second.reset();
        assertEqualVectors(
            makeNullableFlatVector<std::string>(expected, type), result);
        if (alias) {
          assertEqualVectors(original, alias);
        }
      }
    }
  }
}

class GreatestTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  std::optional<T> greatest(
      std::optional<T> arg0,
      std::optional<T> arg1,
      std::optional<T> arg2,
      const TypePtr& type = CppToType<T>::create()) {
    return evaluateOnce<T>(
        "greatest(c0, c1, c2)", {type, type, type}, arg0, arg1, arg2);
  }

  template <typename T>
  std::optional<T> greatest(
      std::optional<T> arg0,
      std::optional<T> arg1,
      std::optional<T> arg2,
      std::optional<T> arg3,
      const TypePtr& type = CppToType<T>::create()) {
    return evaluateOnce<T>(
        "greatest(c0, c1, c2, c3)",
        {type, type, type, type},
        arg0,
        arg1,
        arg2,
        arg3);
  }

  template <typename T>
  void flat(const TypePtr& type = CppToType<T>::create()) {
    vector_size_t size = 20;

    // {0, null, null, 300, null, null, 600, null, null, ...}.
    auto first = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row * 100; },
        [](vector_size_t row) { return row % 3 != 0; },
        type);

    // {0, 10, null, 30, 40, null, 60, 70, null, ...}.
    auto second = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row * 10; },
        nullEvery(3, 2),
        type);

    // {0, 1, 2, 3, 4, 5, 6, 7, 8, ...}.
    auto third = makeFlatVector<T>(
        size, [](vector_size_t row) { return row; }, nullptr, type);

    auto data = makeRowVector({first, second, third});

    // Expect {0, 10, 2, 300, 40, 5, 600, 70, 8, ...}.
    auto expected = [](vector_size_t row) {
      return std::max<T>(
          {row % 3 == 0 ? T(100 * row) : std::numeric_limits<T>::min(),
           row % 3 == 2 ? std::numeric_limits<T>::min() : T(10 * row),
           T(row)});
    };

    auto result = evaluate<FlatVector<T>>("greatest(c0, c1, c2)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }

    result = evaluate<FlatVector<T>>("greatest(c2, c1, c0)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }

    result = evaluate<FlatVector<T>>("greatest(c1, c0, c2)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }
  }

  template <typename T>
  void constant(const TypePtr& type = CppToType<T>::create()) {
    vector_size_t size = 20;

    // {0, null, null, 3, null, null, 6, null, null, ...}.
    auto first = makeFlatVector<T>(
        size,
        [](vector_size_t row) { return row; },
        [](vector_size_t row) { return row % 3 != 0; },
        type);

    // {9, 9, 9, ...}.
    auto second = makeConstant<T>(9, size, type);

    auto data = makeRowVector({first, second});

    // Expect {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 12, 9, 9, 15...}.
    auto expected = [](vector_size_t row) {
      return std::max(
          T(9), row % 3 == 0 ? T(row) : std::numeric_limits<T>::min());
    };

    auto result = evaluate<FlatVector<T>>("greatest(c0, c1)", data);
    for (vector_size_t i = 0; i < size; i++) {
      EXPECT_EQ(result->valueAt(i), expected(i)) << "at " << i;
    }
  }
};

TEST_F(GreatestTest, integral) {
  flat<int16_t>();
  constant<int16_t>();

  flat<int32_t>();
  constant<int32_t>();

  flat<int64_t>();
  constant<int64_t>();

  flat<int32_t>(DATE());
  constant<int32_t>(DATE());
}

TEST_F(GreatestTest, floating) {
  flat<float>();
  constant<float>();

  flat<double>();
  constant<double>();
}

TEST_F(GreatestTest, edgecases) {
  // All inputs are null.
  std::optional<int32_t> null;
  EXPECT_EQ(greatest<int32_t>(null, null, null), null);

  // IEEE-floating point exceptions.
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();

  EXPECT_TRUE(std::isnan(greatest<float>(inf, nan, -inf, 0.f).value()));
  EXPECT_EQ(greatest<float>(0.f, -inf, inf), inf);
  EXPECT_EQ(greatest<float>(-inf, -inf, 0.f), 0.f);
  EXPECT_EQ(greatest<float>(-inf, -inf, -inf), -inf);
}

TEST_F(GreatestTest, boolean) {
  EXPECT_EQ(greatest<bool>(true, false, false), true);
}

TEST_F(GreatestTest, string) {
  EXPECT_EQ(greatest<std::string>("b", "abcde", "abcdefg"), "b");
}

TEST_F(GreatestTest, timestamp) {
  EXPECT_EQ(
      greatest<Timestamp>(
          Timestamp(1569, 25), Timestamp(4859, 482), Timestamp(581, 1651)),
      Timestamp(4859, 482));
}

TEST_F(GreatestTest, date) {
  EXPECT_EQ(greatest<int32_t>(100, 1000, 10000, DATE()), 10000);
}

TEST_F(GreatestTest, decimal) {
  flat<int64_t>(DECIMAL(6, 2));
  constant<int64_t>(DECIMAL(6, 2));

  flat<int128_t>(DECIMAL(28, 12));
  constant<int128_t>(DECIMAL(28, 12));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
