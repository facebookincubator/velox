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
#include "velox/functions/prestosql/tests/utils/LambdaParameterizedBaseTest.h"

namespace facebook::velox::functions {
namespace {

class MapValuesMatchTest : public functions::test::LambdaParameterizedBaseTest {
 protected:
  void match(
      const std::string& functionName,
      const VectorPtr& input,
      const std::string& lambda,
      const std::vector<std::optional<bool>>& expected) {
    const std::string expr =
        fmt::format("{}(c0, x -> ({}))", functionName, lambda);
    SCOPED_TRACE(expr);
    auto result = evaluateParameterized(expr, makeRowVector({input}));
    velox::test::assertEqualVectors(
        makeNullableFlatVector<bool>(expected), result);
  }

  void allValuesMatch(
      const VectorPtr& input,
      const std::string& lambda,
      const std::vector<std::optional<bool>>& expected) {
    match("map_values_all_match", input, lambda, expected);
  }

  void anyValuesMatch(
      const VectorPtr& input,
      const std::string& lambda,
      const std::vector<std::optional<bool>>& expected) {
    match("map_values_any_match", input, lambda, expected);
  }

  void noneValuesMatch(
      const VectorPtr& input,
      const std::string& lambda,
      const std::vector<std::optional<bool>>& expected) {
    match("map_values_none_match", input, lambda, expected);
  }
};

// =============================================================================
// map_values_all_match tests
// =============================================================================

TEST_P(MapValuesMatchTest, allMatchBasic) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  allValuesMatch(data, "x > 0", {true, true});
  allValuesMatch(data, "x > 15", {false, false});
  allValuesMatch(data, "x > 25", {false, false});
  allValuesMatch(data, "x <= 30", {true, true});
  allValuesMatch(data, "x <= 100", {true, true});
  allValuesMatch(data, "x >= 10", {true, true});
  allValuesMatch(data, "x >= 20", {false, false});
}

TEST_P(MapValuesMatchTest, allMatchNullPredicate) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  // Predicate returns NULL for x=20 and false otherwise -> false (early exit).
  allValuesMatch(data, "if(x = 20, null::boolean, false)", {false, false});
  // Predicate returns NULL for x=20 and true otherwise -> NULL for row 0,
  // true for row 1 (no 20 in second map).
  allValuesMatch(data, "if(x = 20, null::boolean, true)", {std::nullopt, true});
}

TEST_P(MapValuesMatchTest, allMatchEmptyMap) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{}",
  });
  // Vacuously true.
  allValuesMatch(data, "x > 0", {true});
  allValuesMatch(data, "x < 0", {true});
}

TEST_P(MapValuesMatchTest, allMatchStringValues) {
  auto data = makeMapVectorFromJson<int32_t, std::string>({
      R"({1: "apple", 2: "banana", 3: "cherry"})",
      R"({1: "dog", 2: "cat"})",
  });

  allValuesMatch(data, "length(x) > 2", {true, true});
  // apple=5, banana=6, cherry=6 all > 4; dog=3, cat=3 not > 4
  allValuesMatch(data, "length(x) > 4", {true, false});
}

TEST_P(MapValuesMatchTest, allMatchDoubleValues) {
  auto data = makeMapVectorFromJson<int32_t, double>({
      "{1: 1.5, 2: 2.5, 3: 3.5}",
      "{1: 0.1, 2: 0.2}",
  });

  allValuesMatch(data, "x > 1.0", {true, false});
  allValuesMatch(data, "x > 0.0", {true, true});
  allValuesMatch(data, "x < 4.0", {true, true});
}

TEST_P(MapValuesMatchTest, allMatchVarcharKeys) {
  auto data = makeMapVectorFromJson<std::string, int64_t>({
      R"({"a": 10, "b": 20, "c": 30})",
      R"({"x": 5, "y": 15})",
  });

  allValuesMatch(data, "x > 5", {true, false});
  allValuesMatch(data, "x > 0", {true, true});
}

TEST_P(MapValuesMatchTest, allMatchNullMapValues) {
  auto data = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 10}, {2, std::nullopt}, {3, 30}}},
      {{{1, std::nullopt}, {2, std::nullopt}}},
  });

  // Row 0: 10>0=true, null->null, 30>0=true => NULL
  // Row 1: null->null, null->null => NULL
  allValuesMatch(data, "x > 0", {std::nullopt, std::nullopt});
}

TEST_P(MapValuesMatchTest, allMatchEquivalenceWithMapValues) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
      "{}",
      "{1: 5}",
  });

  auto lambdas = {"x > 0", "x > 15", "x <= 30", "x = 10", "x % 2 = 0"};
  for (const auto& lambda : lambdas) {
    SCOPED_TRACE(lambda);
    auto direct = evaluateParameterized(
        fmt::format("map_values_all_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    auto viaMapValues = evaluateParameterized(
        fmt::format("all_match(map_values(c0), x -> ({}))", lambda),
        makeRowVector({data}));
    velox::test::assertEqualVectors(viaMapValues, direct);
  }
}

TEST_P(MapValuesMatchTest, allMatchErrorPropagation) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 0, 3: 20}",
      "{1: 10, 2: 20}",
  });

  // Row 0: all non-error true, error propagates -> try wraps to NULL.
  // Row 1: 10>0=true, 20>0=true -> true.
  auto result = evaluateParameterized(
      "try(map_values_all_match(c0, x -> (if(x = 0, 1/0 > 0, true))))",
      makeRowVector({data}));
  auto expected = makeNullableFlatVector<bool>({std::nullopt, true});
  velox::test::assertEqualVectors(expected, result);
}

// =============================================================================
// map_values_any_match tests
// =============================================================================

TEST_P(MapValuesMatchTest, anyMatchBasic) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  anyValuesMatch(data, "x = 10", {true, false});
  anyValuesMatch(data, "x = 22", {false, true});
  anyValuesMatch(data, "x < 15", {true, true});
  anyValuesMatch(data, "x < 0", {false, false});
  anyValuesMatch(data, "x IN (20, 11)", {true, true});
  anyValuesMatch(data, "x > 25", {true, false});
}

TEST_P(MapValuesMatchTest, anyMatchNullPredicate) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  // Predicate returns NULL for x=20 and false otherwise.
  // Row 0: false, null, false -> NULL (no true, has null)
  // Row 1: false, false -> false
  anyValuesMatch(
      data, "if(x = 20, null::boolean, false)", {std::nullopt, false});
  // Predicate returns NULL for x=20 and true otherwise.
  // Row 0: true (early return on x=10)
  // Row 1: true (early return on x=11)
  anyValuesMatch(data, "if(x = 20, null::boolean, true)", {true, true});
}

TEST_P(MapValuesMatchTest, anyMatchEmptyMap) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{}",
  });
  anyValuesMatch(data, "x > 0", {false});
  anyValuesMatch(data, "true", {false});
}

TEST_P(MapValuesMatchTest, anyMatchStringValues) {
  auto data = makeMapVectorFromJson<int32_t, std::string>({
      R"({1: "apple", 2: "banana", 3: "cherry"})",
      R"({1: "dog", 2: "cat"})",
  });

  anyValuesMatch(data, "x = 'banana'", {true, false});
  anyValuesMatch(data, "x = 'elephant'", {false, false});
  anyValuesMatch(data, "length(x) > 5", {true, false});
}

TEST_P(MapValuesMatchTest, anyMatchDoubleValues) {
  auto data = makeMapVectorFromJson<int32_t, double>({
      "{1: 1.5, 2: 2.5, 3: 3.5}",
      "{1: 0.1, 2: 0.2}",
  });

  anyValuesMatch(data, "x < 1.0", {false, true});
  anyValuesMatch(data, "x > 3.0", {true, false});
  anyValuesMatch(data, "x > 0.0", {true, true});
}

TEST_P(MapValuesMatchTest, anyMatchVarcharKeys) {
  auto data = makeMapVectorFromJson<std::string, int64_t>({
      R"({"a": 10, "b": 20, "c": 30})",
      R"({"x": 5, "y": 15})",
  });

  anyValuesMatch(data, "x > 25", {true, false});
  anyValuesMatch(data, "x > 0", {true, true});
}

TEST_P(MapValuesMatchTest, anyMatchNullMapValues) {
  auto data = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 10}, {2, std::nullopt}, {3, 30}}},
      {{{1, std::nullopt}, {2, std::nullopt}}},
  });

  // Row 0: 10>25=false, null->null, 30>25=true => true (early return)
  // Row 1: null->null, null->null => NULL
  anyValuesMatch(data, "x > 25", {true, std::nullopt});
}

TEST_P(MapValuesMatchTest, anyMatchConsistencyWithAnyValuesMatch) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
      "{}",
      "{1: 5}",
  });

  auto lambdas = {"x > 15", "x = 10", "x < 0", "x IN (20, 11)", "x % 2 = 0"};
  for (const auto& lambda : lambdas) {
    SCOPED_TRACE(lambda);
    auto newFn = evaluateParameterized(
        fmt::format("map_values_any_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    auto oldFn = evaluateParameterized(
        fmt::format("any_values_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    velox::test::assertEqualVectors(oldFn, newFn);
  }
}

TEST_P(MapValuesMatchTest, anyMatchEquivalenceWithMapValues) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
      "{}",
      "{1: 5}",
  });

  auto lambdas = {"x > 0", "x > 15", "x <= 30", "x = 10", "x % 2 = 0"};
  for (const auto& lambda : lambdas) {
    SCOPED_TRACE(lambda);
    auto direct = evaluateParameterized(
        fmt::format("map_values_any_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    auto viaMapValues = evaluateParameterized(
        fmt::format("any_match(map_values(c0), x -> ({}))", lambda),
        makeRowVector({data}));
    velox::test::assertEqualVectors(viaMapValues, direct);
  }
}

TEST_P(MapValuesMatchTest, anyMatchErrorPropagation) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 0, 3: 20}",
      "{1: 0, 2: 0}",
  });

  // Row 0: x=10 -> true -> early return true.
  // Row 1: x=0 -> error, x=0 -> error -> try wraps to NULL.
  auto result = evaluateParameterized(
      "try(map_values_any_match(c0, x -> (if(x = 0, 1/0 > 0, x > 5))))",
      makeRowVector({data}));
  auto expected = makeNullableFlatVector<bool>({true, std::nullopt});
  velox::test::assertEqualVectors(expected, result);
}

// =============================================================================
// map_values_none_match tests
// =============================================================================

TEST_P(MapValuesMatchTest, noneMatchBasic) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  noneValuesMatch(data, "x = 7", {true, true});
  noneValuesMatch(data, "x > 15", {false, false});
  noneValuesMatch(data, "x > 25", {false, true});
  noneValuesMatch(data, "x % 11 = 0", {true, false});
  noneValuesMatch(data, "x < 0", {true, true});
  noneValuesMatch(data, "x > 0", {false, false});
}

TEST_P(MapValuesMatchTest, noneMatchNullPredicate) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
  });

  // Predicate returns NULL for x=20 and false otherwise.
  // Row 0: false, null, false -> NULL (no true/earlyReturn, has null)
  // Row 1: false, false -> true (initialValue preserved)
  noneValuesMatch(
      data, "if(x = 20, null::boolean, false)", {std::nullopt, true});
  // Predicate returns NULL for x=20 and true otherwise.
  // Row 0: true -> earlyReturn -> false
  // Row 1: true -> earlyReturn -> false
  noneValuesMatch(data, "if(x = 20, null::boolean, true)", {false, false});
}

TEST_P(MapValuesMatchTest, noneMatchEmptyMap) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{}",
  });
  // Vacuously true.
  noneValuesMatch(data, "x > 0", {true});
  noneValuesMatch(data, "true", {true});
}

TEST_P(MapValuesMatchTest, noneMatchStringValues) {
  auto data = makeMapVectorFromJson<int32_t, std::string>({
      R"({1: "apple", 2: "banana", 3: "cherry"})",
      R"({1: "dog", 2: "cat"})",
  });

  noneValuesMatch(data, "x = 'banana'", {false, true});
  noneValuesMatch(data, "x = 'elephant'", {true, true});
  noneValuesMatch(data, "length(x) > 5", {false, true});
}

TEST_P(MapValuesMatchTest, noneMatchDoubleValues) {
  auto data = makeMapVectorFromJson<int32_t, double>({
      "{1: 1.5, 2: 2.5, 3: 3.5}",
      "{1: 0.1, 2: 0.2}",
  });

  noneValuesMatch(data, "x > 10.0", {true, true});
  noneValuesMatch(data, "x < 1.0", {true, false});
  noneValuesMatch(data, "x > 0.0", {false, false});
}

TEST_P(MapValuesMatchTest, noneMatchVarcharKeys) {
  auto data = makeMapVectorFromJson<std::string, int64_t>({
      R"({"a": 10, "b": 20, "c": 30})",
      R"({"x": 5, "y": 15})",
  });

  noneValuesMatch(data, "x > 25", {false, true});
  noneValuesMatch(data, "x < 0", {true, true});
}

TEST_P(MapValuesMatchTest, noneMatchNullMapValues) {
  auto data = makeNullableMapVector<int32_t, int64_t>({
      {{{1, 10}, {2, std::nullopt}, {3, 30}}},
      {{{1, std::nullopt}, {2, std::nullopt}}},
  });

  // Row 0: 10>25=false, null->null, 30>25=true => false (early return)
  // Row 1: null->null, null->null => NULL
  noneValuesMatch(data, "x > 25", {false, std::nullopt});
}

TEST_P(MapValuesMatchTest, noneMatchConsistencyWithNoValuesMatch) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
      "{}",
      "{1: 5}",
  });

  auto lambdas = {"x > 15", "x = 10", "x < 0", "x IN (20, 11)", "x % 2 = 0"};
  for (const auto& lambda : lambdas) {
    SCOPED_TRACE(lambda);
    auto newFn = evaluateParameterized(
        fmt::format("map_values_none_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    auto oldFn = evaluateParameterized(
        fmt::format("no_values_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    velox::test::assertEqualVectors(oldFn, newFn);
  }
}

TEST_P(MapValuesMatchTest, noneMatchEquivalenceWithMapValues) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{-1: 11, -2: 22}",
      "{}",
      "{1: 5}",
  });

  auto lambdas = {"x > 0", "x > 15", "x <= 30", "x = 10", "x % 2 = 0"};
  for (const auto& lambda : lambdas) {
    SCOPED_TRACE(lambda);
    auto direct = evaluateParameterized(
        fmt::format("map_values_none_match(c0, x -> ({}))", lambda),
        makeRowVector({data}));
    auto viaMapValues = evaluateParameterized(
        fmt::format("none_match(map_values(c0), x -> ({}))", lambda),
        makeRowVector({data}));
    velox::test::assertEqualVectors(viaMapValues, direct);
  }
}

TEST_P(MapValuesMatchTest, noneMatchErrorPropagation) {
  auto data = makeMapVectorFromJson<int32_t, int64_t>({
      "{1: 10, 2: 0, 3: 20}",
      "{1: 0, 2: 0}",
  });

  // Row 0: x=10 -> true -> earlyReturn -> false.
  // Row 1: x=0 -> error, x=0 -> error -> try wraps to NULL.
  auto result = evaluateParameterized(
      "try(map_values_none_match(c0, x -> (if(x = 0, 1/0 > 0, x > 5))))",
      makeRowVector({data}));
  auto expected = makeNullableFlatVector<bool>({false, std::nullopt});
  velox::test::assertEqualVectors(expected, result);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    MapValuesMatchTest,
    MapValuesMatchTest,
    testing::ValuesIn(MapValuesMatchTest::getTestParams()));

} // namespace
} // namespace facebook::velox::functions
