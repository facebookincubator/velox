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
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapValuesInRangeTest : public test::FunctionBaseTest {
 protected:
  void testMapValuesInRange(
      const std::string& expression,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    auto result = evaluate(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapValuesInRangeTest, basicInteger) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
      "{1:5, 2:15, 3:25, 4:35, 5:100}",
      "{}",
      "{1:1, 2:2, 3:3}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:20, 3:30, 4:40}",
      "{2:15, 3:25, 4:35}",
      "{}",
      "{}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(15 as bigint), cast(40 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, basicDouble) {
  auto inputMap = makeMapVectorFromJson<int64_t, double>({
      "{1:1.5, 2:2.5, 3:3.5, 4:4.5, 5:5.5}",
      "{1:0.5, 2:1.5, 3:10.5}",
  });

  auto expected = makeMapVectorFromJson<int64_t, double>({
      "{2:2.5, 3:3.5, 4:4.5}",
      "{}",
  });

  auto result =
      evaluate("map_values_in_range(c0, 2.0, 5.0)", makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, varcharKey) {
  auto inputMap = makeMapVectorFromJson<std::string, int64_t>({
      R"({"apple":10, "banana":20, "cherry":30, "date":40})",
      R"({"x":5, "y":50, "z":25})",
  });

  auto expected = makeMapVectorFromJson<std::string, int64_t>({
      R"({"banana":20, "cherry":30})",
      R"({"z":25})",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(15 as bigint), cast(35 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, emptyMap) {
  auto inputMap =
      makeMapVectorFromJson<int64_t, int64_t>({"{}", "{}", "{1:10}"});

  auto expected =
      makeMapVectorFromJson<int64_t, int64_t>({"{}", "{}", "{1:10}"});

  auto result = evaluate(
      "map_values_in_range(c0, cast(0 as bigint), cast(100 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, allValuesOutOfRange) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:1, 2:2, 3:3}",
      "{1:100, 2:200, 3:300}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{}",
      "{}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(50 as bigint), cast(60 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, allValuesInRange) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30}",
      "{1:15, 2:25}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30}",
      "{1:15, 2:25}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(0 as bigint), cast(100 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, boundaryValues) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(10 as bigint), cast(50 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, nullValuesPreserved) {
  auto inputMap = makeNullableMapVector<int64_t, int64_t>({
      {{{1, 10}, {2, std::nullopt}, {3, 30}, {4, std::nullopt}, {5, 50}}},
      {{{1, std::nullopt}, {2, 20}}},
  });

  auto expected = makeNullableMapVector<int64_t, int64_t>({
      {{{2, std::nullopt}, {3, 30}, {4, std::nullopt}}},
      {{{1, std::nullopt}, {2, 20}}},
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(15 as bigint), cast(45 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, floatValues) {
  auto inputMap = makeMapVectorFromJson<int64_t, double>({
      "{1:1.1, 2:2.2, 3:3.3, 4:4.4, 5:5.5}",
      "{1:0.1, 2:10.1}",
  });

  auto expected = makeMapVectorFromJson<int64_t, double>({
      "{2:2.2, 3:3.3, 4:4.4}",
      "{}",
  });

  auto result =
      evaluate("map_values_in_range(c0, 2.0, 5.0)", makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, negativeBounds) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:-50, 2:-20, 3:0, 4:20, 5:50}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:-20, 3:0, 4:20}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(-30 as bigint), cast(30 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, singleElementMap) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:25}",
      "{1:5}",
      "{1:100}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:25}",
      "{}",
      "{}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(10 as bigint), cast(50 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, int32KeyAndValue) {
  auto inputMap = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40}",
      "{1:5, 2:25, 3:45}",
  });

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{2:20, 3:30}",
      "{2:25}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(15 as integer), cast(35 as integer))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, varcharKeyDoubleValue) {
  auto inputMap = makeMapVectorFromJson<std::string, double>({
      R"({"a":1.5, "b":2.5, "c":3.5, "d":4.5})",
      R"({"x":0.5, "y":5.5})",
  });

  auto expected = makeMapVectorFromJson<std::string, double>({
      R"({"b":2.5, "c":3.5})",
      "{}",
  });

  auto result =
      evaluate("map_values_in_range(c0, 2.0, 4.0)", makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, sameLowerAndUpperBound) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:25, 3:25, 4:30}",
      "{1:25, 2:25}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:25, 3:25}",
      "{1:25, 2:25}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(25 as bigint), cast(25 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, largeMap) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80, 9:90, 10:100}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(20 as bigint), cast(80 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, varcharKeyFloatValue) {
  auto inputMap = makeMapVectorFromJson<std::string, float>({
      R"({"temp1":20.5, "temp2":25.0, "temp3":30.5, "temp4":35.0})",
      R"({"sensor1":10.0, "sensor2":50.0})",
  });

  auto expected = makeMapVectorFromJson<std::string, float>({
      R"({"temp2":25.0, "temp3":30.5})",
      "{}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(22.0 as real), cast(32.0 as real))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

TEST_F(MapValuesInRangeTest, extremeValues) {
  auto inputMap = makeMapVectorFromJson<int64_t, int64_t>({
      "{1:-9223372036854775808, 2:0, 3:9223372036854775807}",
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{2:0}",
  });

  auto result = evaluate(
      "map_values_in_range(c0, cast(-100 as bigint), cast(100 as bigint))",
      makeRowVector({inputMap}));
  assertEqualVectors(expected, result);
}

} // namespace

// Custom fuzzer tests to compare map_values_in_range with equivalent expression
// using existing UDFs. The equivalent expression is:
// map_filter(map, (k, v) -> IF(v IS NULL, true, v >= lower_bound AND v <=
// upper_bound))
class MapValuesInRangeFuzzerTest : public test::FunctionBaseTest {
 protected:
  // Get a SelectivityVector that excludes rows where input map is null.
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto inputMap = data->childAt(0);
    SelectivityVector nonNullRows(data->size());
    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (inputMap->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  // Verify that map_values_in_range correctly filters values outside the range.
  // Properties verified:
  // 1. Result map size <= input map size
  // 2. All non-null values in result are within [lowerBound, upperBound]
  // 3. Null values are preserved
  void testMapValuesInRangeProperties(
      const RowVectorPtr& data,
      double lowerBound,
      double upperBound) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    try {
      result = evaluate(
          fmt::format(
              "map_values_in_range(c0, cast({} as double), cast({} as double))",
              lowerBound,
              upperBound),
          data);
    } catch (...) {
      return;
    }

    if (!result) {
      return;
    }

    auto inputMap = data->childAt(0);
    auto inputMapVector = inputMap->as<MapVector>();
    if (!inputMapVector) {
      return;
    }

    auto resultMap = result->as<MapVector>();
    if (!resultMap) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i) || result->isNullAt(i)) {
        continue;
      }

      if (inputMap->isNullAt(i)) {
        continue;
      }

      auto origSize = inputMapVector->sizeAt(i);
      auto resultSize = resultMap->sizeAt(i);
      ASSERT_LE(resultSize, origSize)
          << "Result map should be at most as large as input map at row " << i;
    }
  }

  // Test equivalence with map_filter expression for double values.
  void testEquivalenceDouble(
      const RowVectorPtr& data,
      double lowerBound,
      double upperBound) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    auto mapValuesInRangeExpr = fmt::format(
        "map_values_in_range(c0, cast({} as double), cast({} as double))",
        lowerBound,
        upperBound);

    // Equivalent expression: keep entries where value is null OR within bounds
    auto equivalentExpr = fmt::format(
        "map_filter(c0, (k, v) -> v IS NULL OR (v >= cast({} as double) AND v <= cast({} as double)))",
        lowerBound,
        upperBound);

    VectorPtr result;
    VectorPtr expected;
    try {
      result = evaluate(mapValuesInRangeExpr, data);
      expected = evaluate(equivalentExpr, data);
    } catch (...) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (nonNullRows.isValid(i)) {
        ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
            << "Mismatch at row " << i << ": expected " << expected->toString(i)
            << ", got " << result->toString(i);
      }
    }
  }

  // Test equivalence with map_filter expression for bigint values.
  void testEquivalenceBigint(
      const RowVectorPtr& data,
      int64_t lowerBound,
      int64_t upperBound) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    auto mapValuesInRangeExpr = fmt::format(
        "map_values_in_range(c0, cast({} as bigint), cast({} as bigint))",
        lowerBound,
        upperBound);

    auto equivalentExpr = fmt::format(
        "map_filter(c0, (k, v) -> v IS NULL OR (v >= cast({} as bigint) AND v <= cast({} as bigint)))",
        lowerBound,
        upperBound);

    VectorPtr result;
    VectorPtr expected;
    try {
      result = evaluate(mapValuesInRangeExpr, data);
      expected = evaluate(equivalentExpr, data);
    } catch (...) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (nonNullRows.isValid(i)) {
        ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
            << "Mismatch at row " << i << ": expected " << expected->toString(i)
            << ", got " << result->toString(i);
      }
    }
  }
};

TEST_F(MapValuesInRangeFuzzerTest, fuzzIntegerKeyDoubleValue) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, -50.0, 50.0);
    testMapValuesInRangeProperties(data, -50.0, 50.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzBigintKeyBigintValue) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(BIGINT(), BIGINT()));
    auto data = makeRowVector({inputMap});
    testEquivalenceBigint(data, -1000, 1000);
    testMapValuesInRangeProperties(data, -1000.0, 1000.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzVarcharKeyDoubleValue) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.stringLength = 20;
  opts.stringVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(VARCHAR(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, 0.0, 100.0);
    testMapValuesInRangeProperties(data, 0.0, 100.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzHighNullRatio) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.5;
  opts.containerHasNulls = true;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, -100.0, 100.0);
    testMapValuesInRangeProperties(data, -100.0, 100.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzSmallContainers) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 2;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, -10.0, 10.0);
    testMapValuesInRangeProperties(data, -10.0, 10.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzLargeVectors) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 500;
  opts.nullRatio = 0.1;
  opts.containerLength = 20;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 5; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, -1000.0, 1000.0);
    testMapValuesInRangeProperties(data, -1000.0, 1000.0);
  }
}

TEST_F(MapValuesInRangeFuzzerTest, fuzzNarrowBounds) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), DOUBLE()));
    auto data = makeRowVector({inputMap});
    testEquivalenceDouble(data, -0.5, 0.5);
    testMapValuesInRangeProperties(data, -0.5, 0.5);
  }
}

} // namespace facebook::velox::functions
