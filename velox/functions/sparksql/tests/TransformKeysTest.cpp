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

#include <bit>
#include <cstdint>
#include <limits>

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

class TransformKeysTest : public SparkFunctionBaseTest {
 protected:
  // The function builds its result differently under each duplicate-key
  // policy. These cases give the same result under both, so each one runs once
  // under each policy.

  void testWithoutDuplicates() {
    auto data = makeRowVector({
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1: 10, 2: 20, 3: 30}",
            "{}",
            "null",
            "{4: 40}",
        }),
    });
    auto expected = makeMapVectorFromJson<int64_t, int64_t>({
        "{11: 10, 12: 20, 13: 30}",
        "{}",
        "null",
        "{14: 40}",
    });
    assertEqualVectors(
        expected, evaluate("transform_keys(c0, (k, v) -> add(k, 10))", data));
  }

  void testLambdaReadsValue() {
    auto data = makeRowVector({
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1: 10, 2: 20, 3: 30}",
            "{}",
            "null",
            "{4: 40}",
        }),
    });
    auto expected = makeMapVectorFromJson<int64_t, int64_t>({
        "{11: 10, 22: 20, 33: 30}",
        "{}",
        "null",
        "{44: 40}",
    });
    assertEqualVectors(
        expected, evaluate("transform_keys(c0, (k, v) -> add(k, v))", data));
  }

  void testNullKey() {
    auto data = makeRowVector({
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1: 10, 2: 20}",
            "{3: 30}",
        }),
        makeNullableFlatVector<int64_t>({std::nullopt, 7}),
    });
    VELOX_ASSERT_USER_THROW(
        evaluate("transform_keys(c0, (k, v) -> c1)", data),
        "Cannot use null as map key!");
  }

  void testTryNullKey() {
    auto data = makeRowVector({
        makeMapVectorFromJson<int64_t, int64_t>({
            "{1: 10, 2: 20}",
            "{3: 30}",
        }),
        makeNullableFlatVector<int64_t>({std::nullopt, 7}),
    });
    auto expected = makeMapVectorFromJson<int64_t, int64_t>({
        "null",
        "{7: 30}",
    });
    assertEqualVectors(
        expected, evaluate("try(transform_keys(c0, (k, v) -> c1))", data));
  }
};

TEST_F(TransformKeysTest, lastWinWithoutDuplicates) {
  setThrowExceptionOnDuplicateMapKeys(false);
  testWithoutDuplicates();
}

TEST_F(TransformKeysTest, throwPolicyWithoutDuplicates) {
  setThrowExceptionOnDuplicateMapKeys(true);
  testWithoutDuplicates();
}

TEST_F(TransformKeysTest, lastWinLambdaReadsValue) {
  setThrowExceptionOnDuplicateMapKeys(false);
  testLambdaReadsValue();
}

TEST_F(TransformKeysTest, throwPolicyLambdaReadsValue) {
  setThrowExceptionOnDuplicateMapKeys(true);
  testLambdaReadsValue();
}

TEST_F(TransformKeysTest, lastWinDuplicateKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
          {{5, 50}},
          {{2, 20}, {4, 40}, {6, 60}},
      }),
  });

  // A repeated key keeps the position of its first occurrence and takes the
  // value of its last one. The assertion reads map_entries because map equality
  // ignores entry order.
  auto expected = makeArrayVector(
      {0, 2, 4, 5},
      makeRowVector({
          makeFlatVector<int64_t>({1, 0, 1, 0, 1, 0}),
          makeFlatVector<int64_t>({30, 20, 30, 40, 50, 60}),
      }));
  assertEqualVectors(
      expected,
      evaluate(
          "map_entries(transform_keys(c0, (k, v) -> remainder(k, 2)))", data));
}

TEST_F(TransformKeysTest, throwOnDuplicateKeys) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{5, 50}},
      }),
  });

  VELOX_ASSERT_USER_THROW(
      evaluate("transform_keys(c0, (k, v) -> remainder(k, 2))", data),
      "Duplicate map key was found: 1");
}

TEST_F(TransformKeysTest, throwPolicyTryDuplicateKeys) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{5, 50}},
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "null",
      "{1: 50}",
  });
  assertEqualVectors(
      expected,
      evaluate("try(transform_keys(c0, (k, v) -> remainder(k, 2)))", data));
}

TEST_F(TransformKeysTest, lastWinFloatingPointKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // The lambda turns keys 1.0 and -1.0 into 0.0 and -0.0 in the first row and
  // into NaN in the second. Spark keeps 0.0 and -0.0 as two keys and treats
  // every NaN as one key.
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector({
      makeMapVector<double, int64_t>({
          {{1.0, 10}, {-1.0, 20}},
          {{1.0, 30}, {-1.0, 40}},
      }),
      makeFlatVector<double>({0.0, kNaN}),
  });

  auto expected = makeArrayVector(
      {0, 2},
      makeRowVector({
          makeFlatVector<double>({0.0, -0.0, kNaN}),
          makeFlatVector<int64_t>({10, 20, 40}),
      }));
  assertEqualVectors(
      expected,
      evaluate(
          "map_entries(transform_keys(c0, (k, v) -> multiply(k, c1)))", data));
  EXPECT_THAT(
      elementBitPatterns(evaluate(
          "map_keys(transform_keys(c0, (k, v) -> multiply(k, c1)))", data)),
      testing::ElementsAre(
          std::bit_cast<uint64_t>(0.0),
          std::bit_cast<uint64_t>(-0.0),
          std::bit_cast<uint64_t>(kNaN)));
}

TEST_F(TransformKeysTest, throwPolicyFloatingPointKeys) {
  setThrowExceptionOnDuplicateMapKeys(true);

  {
    auto data = makeRowVector({
        makeMapVector<double, int64_t>({{{1.0, 10}, {-1.0, 20}}}),
        makeFlatVector<double>({0.0}),
    });
    auto expected = makeArrayVector(
        {0},
        makeRowVector({
            makeFlatVector<double>({0.0, -0.0}),
            makeFlatVector<int64_t>({10, 20}),
        }));
    assertEqualVectors(
        expected,
        evaluate(
            "map_entries(transform_keys(c0, (k, v) -> multiply(k, c1)))",
            data));
    EXPECT_THAT(
        elementBitPatterns(evaluate(
            "map_keys(transform_keys(c0, (k, v) -> multiply(k, c1)))", data)),
        testing::ElementsAre(
            std::bit_cast<uint64_t>(0.0), std::bit_cast<uint64_t>(-0.0)));
  }

  {
    auto data = makeRowVector({
        makeMapVector<double, int64_t>({{{1.0, 10}, {-1.0, 20}}}),
        makeFlatVector<double>({std::numeric_limits<double>::quiet_NaN()}),
    });
    VELOX_ASSERT_USER_THROW(
        evaluate("transform_keys(c0, (k, v) -> multiply(k, c1))", data),
        "Duplicate map key was found: NaN");
  }
}

TEST_F(TransformKeysTest, lastWinNestedFloatingPointKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Spark compares array and row keys with SQL ordering, where 0.0 equals
  // -0.0, so [0.0] and [-0.0] are one key.
  auto data = makeRowVector({
      makeMapVector(
          {0},
          makeFlatVector<int64_t>({1, 2}),
          makeArrayVector<double>({{0.0}, {-0.0}})),
  });

  auto expected = makeArrayVector(
      {0},
      makeRowVector({
          makeArrayVector<double>({{0.0}}),
          makeArrayVector<double>({{-0.0}}),
      }));
  assertEqualVectors(
      expected, evaluate("map_entries(transform_keys(c0, (k, v) -> v))", data));
  EXPECT_THAT(
      elementBitPatterns(
          evaluate("flatten(map_keys(transform_keys(c0, (k, v) -> v)))", data)),
      testing::ElementsAre(std::bit_cast<uint64_t>(0.0)));
  EXPECT_THAT(
      elementBitPatterns(evaluate(
          "flatten(map_values(transform_keys(c0, (k, v) -> v)))", data)),
      testing::ElementsAre(std::bit_cast<uint64_t>(-0.0)));
}

TEST_F(TransformKeysTest, lastWinNullKey) {
  setThrowExceptionOnDuplicateMapKeys(false);
  testNullKey();
}

TEST_F(TransformKeysTest, throwPolicyNullKey) {
  setThrowExceptionOnDuplicateMapKeys(true);
  testNullKey();
}

TEST_F(TransformKeysTest, lastWinTryNullKey) {
  setThrowExceptionOnDuplicateMapKeys(false);
  testTryNullKey();
}

TEST_F(TransformKeysTest, throwPolicyTryNullKey) {
  setThrowExceptionOnDuplicateMapKeys(true);
  testTryNullKey();
}

TEST_F(TransformKeysTest, throwPolicyNullAndRepeatedKeysFailInKeyOrder) {
  setThrowExceptionOnDuplicateMapKeys(true);
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "transform_keys(c0, (k, v) -> v)",
          makeRowVector({makeMapVector<int64_t, int64_t>(
              {{{1, 1}, {2, 1}, {3, std::nullopt}}})})),
      "Duplicate map key was found: 1");
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "transform_keys(c0, (k, v) -> v)",
          makeRowVector({makeMapVector<int64_t, int64_t>(
              {{{1, 1}, {2, std::nullopt}, {3, 1}}})})),
      "Cannot use null as map key!");
}

TEST_F(TransformKeysTest, lastWinNullAndRepeatedKeysFailInKeyOrder) {
  setThrowExceptionOnDuplicateMapKeys(false);
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "transform_keys(c0, (k, v) -> v)",
          makeRowVector({makeMapVector<int64_t, int64_t>(
              {{{1, 1}, {2, 1}, {3, std::nullopt}}})})),
      "Cannot use null as map key!");
  VELOX_ASSERT_USER_THROW(
      evaluate(
          "transform_keys(c0, (k, v) -> v)",
          makeRowVector({makeMapVector<int64_t, int64_t>(
              {{{1, 1}, {2, std::nullopt}, {3, 1}}})})),
      "Cannot use null as map key!");
}

TEST_F(TransformKeysTest, lastWinArrayKeysWithNestedNulls) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
          {{3, 30}},
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, null]",
          "[2, null]",
      }),
  });

  auto expected = makeMapVector(
      {0, 1},
      makeArrayVectorFromJson<int64_t>({
          "[1, null]",
          "[2, null]",
      }),
      makeFlatVector<int64_t>({20, 30}));
  assertEqualVectors(
      expected, evaluate("transform_keys(c0, (k, v) -> c1)", data));
}

TEST_F(TransformKeysTest, throwOnDuplicateArrayKeysWithNestedNulls) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
          {{3, 30}},
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, null]",
          "[2, null]",
      }),
  });

  VELOX_ASSERT_USER_THROW(
      evaluate("transform_keys(c0, (k, v) -> c1)", data),
      "Duplicate map key was found: {1, null}");
}

TEST_F(TransformKeysTest, throwPolicyTryArrayKeysWithNestedNulls) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}},
          {{3, 30}},
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, null]",
          "[2, null]",
      }),
  });

  auto expected = makeMapVector(
      {0, 0},
      makeArrayVectorFromJson<int64_t>({"[2, null]"}),
      makeFlatVector<int64_t>({30}));
  expected->setNull(0, true);
  assertEqualVectors(
      expected, evaluate("try(transform_keys(c0, (k, v) -> c1))", data));
}

TEST_F(TransformKeysTest, keyTypeContainingMap) {
  // Spark rejects a key type that is or contains a map.
  auto maps = makeMapVectorFromJson<int64_t, int64_t>({"{1: 10}"});
  auto data = makeRowVector({
      maps,
      maps,
      makeArrayVector({0}, maps),
      makeRowVector({"m"}, {maps}),
  });

  VELOX_ASSERT_USER_THROW(
      evaluate("transform_keys(c0, (k, v) -> c1)", data),
      "Scalar function signature is not supported: transform_keys(MAP<BIGINT,BIGINT>, FUNCTION<BIGINT, BIGINT, MAP<BIGINT,BIGINT>>)");
  VELOX_ASSERT_USER_THROW(
      evaluate("transform_keys(c0, (k, v) -> c2)", data),
      "Scalar function signature is not supported: transform_keys(MAP<BIGINT,BIGINT>, FUNCTION<BIGINT, BIGINT, ARRAY<MAP<BIGINT,BIGINT>>>)");
  VELOX_ASSERT_USER_THROW(
      evaluate("transform_keys(c0, (k, v) -> c3)", data),
      "Scalar function signature is not supported: transform_keys(MAP<BIGINT,BIGINT>, FUNCTION<BIGINT, BIGINT, ROW<m:MAP<BIGINT,BIGINT>>>)");
}

TEST_F(TransformKeysTest, dictionaryEncodedInput) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto baseMaps = makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {3, 30}},
      {{4, 40}, {5, 50}},
  });
  // Rows are read out of order and the second base row is read twice, so the
  // result offsets must follow the dictionary rather than the base vector. The
  // flat capture keeps the dictionary from being peeled off before the function
  // runs.
  auto data = makeRowVector({
      wrapInDictionary(makeIndices({1, 0, 1}), 3, baseMaps),
      makeFlatVector<int64_t>({10, 20, 30}),
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{0: 40, 1: 50}",
      "{1: 30, 0: 20}",
      "{0: 40, 1: 50}",
  });
  assertEqualVectors(
      expected,
      evaluate("transform_keys(c0, (k, v) -> remainder(add(k, c1), 2))", data));
}

TEST_F(TransformKeysTest, conditional) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Each branch writes a disjoint subset of rows into the same result vector.
  auto data = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {{1, 10}, {2, 20}, {3, 30}},
          {{1, 10}, {2, 20}, {3, 30}},
          {{4, 40}},
      }),
      makeFlatVector<int64_t>({1, 2, 3}),
  });

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{11: 10, 12: 20, 13: 30}",
      "{1: 30, 0: 20}",
      "{0: 40}",
  });
  assertEqualVectors(
      expected,
      evaluate(
          "if(greaterthan(c1, 1), transform_keys(c0, (k, v) -> remainder(k, 2)), "
          "transform_keys(c0, (k, v) -> add(k, 10)))",
          data));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
