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

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapTopNValuesTest : public test::FunctionBaseTest {};

TEST_F(MapTopNValuesTest, emptyMap) {
  RowVectorPtr input = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{}",
      }),
  });

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 3)", input),
      makeArrayVectorFromJson<int64_t>({
          "[]",
      }));

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 3, (x,y) -> x)", input),
      makeArrayVectorFromJson<int64_t>({
          "[]",
      }));
}

TEST_F(MapTopNValuesTest, basic) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:3, 2:5, 3:1, 4:4, 5:2}",
          "{1:3, 2:5, 3:null, 4:4, 5:2}",
          "{1:null, 2:null, 3:1, 4:4, 5:null}",
          "{1:10, 2:7, 3:11, 5:4}",
          "{1:10, 2:10, 3:10, 4:10, 5:10}",
          "{1:10, 2:7, 3:0}",
          "{1:null, 2:10}",
          "{}",
          "{1:null, 2:null, 3:null}",
      }),
  });

  auto result = evaluate("map_top_n_values(c0, 3)", data);

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[5, 4, 3]",
      "[5, 4, 3]",
      "[4, 1, null]",
      "[11, 10, 7]",
      "[10, 10, 10]",
      "[10, 7, 0]",
      "[10, null]",
      "[]",
      "[null, null, null]",
  });

  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 3, (x,y) -> y)", data);
  assertEqualVectors(expected, result);

  // n = 0. Expect empty maps.
  result = evaluate("map_top_n_values(c0, 0)", data);

  expected = makeArrayVectorFromJson<int64_t>({
      "[]",
      "[]",
      "[]",
      "[]",
      "[]",
      "[]",
      "[]",
      "[]",
      "[]",
  });

  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 0, (x,y) -> y)", data);
  assertEqualVectors(expected, result);

  // n is negative. Expect an error.
  VELOX_ASSERT_THROW(
      evaluate("map_top_n_values(c0, -1)", data),
      "n must be greater than or equal to 0");

  VELOX_ASSERT_THROW(
      evaluate("map_top_n_values(c0, -1, (x,y) -> y)", data),
      "n must be greater than or equal to 0");
}

TEST_F(MapTopNValuesTest, tieBreak) {
  // Two logically identical maps stored in a different entry order. Ties on the
  // transformed value are broken on the keys, so both rows return the same top
  // values no matter how the entries are laid out.
  auto map = makeMapVector(
      {0, 3},
      makeFlatVector<int32_t>({1, 2, 3, 3, 2, 1}),
      makeFlatVector<int64_t>({10, 20, 30, 30, 20, 10}));

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 2, (k,v) -> 0)", makeRowVector({map})),
      makeArrayVectorFromJson<int64_t>({
          "[30, 20]",
          "[30, 20]",
      }));
}

TEST_F(MapTopNValuesTest, lambdaReturnsNull) {
  RowVectorPtr input = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:10, 2:20, 3:30, 4:40}",
      }),
  });

  // The entry whose transformed value is null ranks last.
  assertEqualVectors(
      evaluate(
          "map_top_n_values(c0, 4, (k,v) -> if(k = 2, cast(null as integer), k))",
          input),
      makeArrayVectorFromJson<int64_t>({"[40, 30, 10, 20]"}));
}

TEST_F(MapTopNValuesTest, nIsZeroWithFailingLambda) {
  RowVectorPtr input = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{0:5, 1:10}",
      }),
  });

  // n = 0 discards the whole result, so the lambda is never evaluated and its
  // division by zero is not raised.
  assertEqualVectors(
      evaluate("map_top_n_values(c0, 0, (k,v) -> v / k)", input),
      makeArrayVectorFromJson<int64_t>({"[]"}));
}

TEST_F(MapTopNValuesTest, complexKeys) {
  RowVectorPtr input =
      makeRowVector({makeMapVectorFromJson<std::string, std::string>(
          {R"( {"x":"ab", "y":"y"} )",
           R"( {"x":"dw", "x2":"-2"} )",
           R"( {"ac":"ac", "cc":"cc", "dd": "dd"} )"})});

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 1)", input),
      makeArrayVectorFromJson<std::string>({
          "[\"y\"]",
          "[\"dw\"]",
          "[\"dd\"]",
      }));

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 1, (x,y) -> y)", input),
      makeArrayVectorFromJson<std::string>({
          "[\"y\"]",
          "[\"dw\"]",
          "[\"dd\"]",
      }));
}

TEST_F(MapTopNValuesTest, tryFuncWithInputHasNullInArray) {
  auto arrayVector = makeNullableFlatVector<std::int32_t>({1, 2, 3});
  auto flatVector = makeNullableArrayVector<bool>(
      {{true, std::nullopt}, {true, std::nullopt}, {false, std::nullopt}});
  auto mapvector = makeMapVector(
      /*offsets=*/{0},
      /*keyVector=*/arrayVector,
      /*valueVector=*/flatVector);
  auto rst = evaluate(
      "try(map_top_n_values(c0, 6455219767830808341))",
      makeRowVector({mapvector}));
  assert(rst->size() == 1);
  assert(rst->isNullAt(0));
}

TEST_F(MapTopNValuesTest, floatingPointValues) {
  // Test with floating point values
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, double>({
          "{1:3.5, 2:5.1, 3:1.2, 4:4.9, 5:2.0}",
          "{1:3.5, 2:5.1, 3:null, 4:4.9, 5:2.0}",
      }),
  });

  auto result = evaluate("map_top_n_values(c0, 3)", data);

  auto expected = makeArrayVectorFromJson<double>({
      "[5.1, 4.9, 3.5]",
      "[5.1, 4.9, 3.5]",
  });

  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 3, (x,y) -> y)", data);
  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 3, (x,y) -> x)", data);
  expected = makeArrayVectorFromJson<double>({
      "[2.0, 4.9, 1.2]",
      "[2.0, 4.9, null]",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapTopNValuesTest, edgeCaseValues) {
  // Test with minimum/maximum values
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:" + std::to_string(INT64_MIN) +
              ", 2:" + std::to_string(INT64_MAX) + "}",
          "{1:0, 2:" + std::to_string(INT64_MAX) + "}",
      }),
  });

  auto result = evaluate("map_top_n_values(c0, 2)", data);

  auto expected = makeArrayVectorFromJson<int64_t>({
      "[" + std::to_string(INT64_MAX) + ", " + std::to_string(INT64_MIN) + "]",
      "[" + std::to_string(INT64_MAX) + ", 0]",
  });

  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 2, (x,y) -> y)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapTopNValuesTest, stringValuesWithSpecialChars) {
  // Test string values with special characters
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, std::string>({
          R"({1:"a,b", 2:"a\nb", 3:"a\tb"})",
          R"({1:"", 2:" ", 3:"  "})",
      }),
  });

  auto result = evaluate("map_top_n_values(c0, 2)", data);

  auto expected = makeArrayVectorFromJson<std::string>({
      R"(["a,b", "a\nb"])",
      R"(["  ", " "])",
  });

  assertEqualVectors(expected, result);

  result = evaluate("map_top_n_values(c0, 2, (x,y) -> y)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapTopNValuesTest, nullLambda) {
  // Test that passing NULL as the lambda falls back to sorting by values
  // (same as the 2-argument version).
  RowVectorPtr input = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:30, 2:10, 3:50, 4:20, 5:40}",
          "{1:30, 2:10, 3:50}",
          "{1:30, 2:10}",
      }),
  });

  assertEqualVectors(
      evaluate("map_top_n_values(c0, 3, null)", input),
      makeArrayVectorFromJson<int64_t>({
          "[50, 40, 30]",
          "[50, 30, 10]",
          "[30, 10]",
      }));

  // Verify it matches the 2-argument version.
  assertEqualVectors(
      evaluate("map_top_n_values(c0, 3)", input),
      evaluate("map_top_n_values(c0, 3, null)", input));
}

TEST_F(MapTopNValuesTest, lambdaErrorHandling) {
  RowVectorPtr input = makeRowVector({
      makeMapVectorFromJson<int32_t, int64_t>({
          "{1:5, 2:10, 0:20}",
      }),
  });

  // Lambda that divides by key - should error on key=0.
  VELOX_ASSERT_THROW(
      evaluate("map_top_n_values(c0, 2, (k,v) -> v / k)", input),
      "division by zero");

  // With TRY, the error row should return null.
  assertEqualVectors(
      evaluate("try(map_top_n_values(c0, 2, (k,v) -> v / k))", input),
      makeNullableArrayVector<int64_t>({std::nullopt}));
}

} // namespace
} // namespace facebook::velox::functions
