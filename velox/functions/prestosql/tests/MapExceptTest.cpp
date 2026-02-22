/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/functions/prestosql/tests/utils/FuzzerTestUtils.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapExceptTest : public test::FunctionBaseTest {
 public:
  template <typename T>
  void testFloatNaNs() {
    static const auto kNaN = std::numeric_limits<T>::quiet_NaN();
    static const auto kSNaN = std::numeric_limits<T>::signaling_NaN();

    // Case 1: Non-constant search keys.
    auto data = makeRowVector(
        {makeMapVectorFromJson<T, int32_t>({
             "{1:10, NaN:20, 3:null, 4:40, 5:50, 6:60}",
             "{NaN:20}",
         }),
         makeArrayVector<T>({{1, kNaN, 5}, {kSNaN, 3}})});

    auto expected = makeMapVectorFromJson<T, int32_t>({
        "{3:null, 4:40, 6:60}",
        "{}",
    });
    auto result = evaluate("map_except(c0, c1)", data);
    assertEqualVectors(expected, result);

    // Case 2: Constant search keys.
    data = makeRowVector(
        {makeMapVectorFromJson<T, int32_t>({
             "{1:10, NaN:20, 3:null, 4:40, 5:50, 6:60}",
             "{NaN:20}",
         }),
         BaseVector::wrapInConstant(2, 0, makeArrayVector<T>({{1, kNaN, 5}}))});
    expected = makeMapVectorFromJson<T, int32_t>({
        "{3:null, 4:40, 6:60}",
        "{}",
    });
    result = evaluate("map_except(c0, c1)", data);
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapExceptTest, bigintKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:null, 4:40, 5:50, 6:60}",
          "{1:10, 2:20, 4:40, 5:50}",
          "{}",
          "{2:20, 4:40, 6:60}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 3, 5]",
          "[1, 3, 5, 7]",
          "[3, 5]",
          "[1, 3]",
      }),
  });

  // Constant keys.
  auto result = evaluate("map_except(c0, array_constructor(1, 3, 5))", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{2:20, 4:40, 6:60}",
      "{2:20, 4:40}",
      "{}",
      "{2:20, 4:40, 6:60}",
  });

  assertEqualVectors(expected, result);

  // Non-constant keys.
  result = evaluate("map_except(c0, c1)", data);
  assertEqualVectors(expected, result);

  // Empty list of keys. Expect all map entries returned.
  result = evaluate("map_except(c0, array_constructor()::bigint[])", data);

  expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:null, 4:40, 5:50, 6:60}",
      "{1:10, 2:20, 4:40, 5:50}",
      "{}",
      "{2:20, 4:40, 6:60}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapExceptTest, varcharKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<std::string, int32_t>({
          R"({"apple": 1, "banana": 2, "Cucurbitaceae": null, "date": 4, "eggplant": 5, "fig": 6})",
          R"({"banana": 2, "orange": 4})",
          R"({"banana": 2, "fig": 4, "date": 5})",
      }),
      makeArrayVectorFromJson<std::string>({
          R"(["apple", "Cucurbitaceae", "fig"])",
          R"(["apple", "Cucurbitaceae", "date", "eggplant"])",
          R"(["fig"])",
      }),
  });

  // Constant keys.
  auto result = evaluate(
      "map_except(c0, array_constructor('apple', 'some very looooong name', 'fig', 'Cucurbitaceae'))",
      data);

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"banana": 2, "date": 4, "eggplant": 5})",
      R"({"banana": 2, "orange": 4})",
      R"({"banana": 2, "date": 5})",
  });

  assertEqualVectors(expected, result);

  // Non-constant keys.
  result = evaluate("map_except(c0, c1)", data);
  assertEqualVectors(expected, result);

  // Empty list of keys. Expect all map entries returned.
  result = evaluate("map_except(c0, array_constructor()::varchar[])", data);

  expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple": 1, "banana": 2, "Cucurbitaceae": null, "date": 4, "eggplant": 5, "fig": 6})",
      R"({"banana": 2, "orange": 4})",
      R"({"banana": 2, "fig": 4, "date": 5})",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapExceptTest, floatNaNs) {
  testFloatNaNs<float>();
  testFloatNaNs<double>();
}

// Custom fuzzer tests

// Custom fuzzer tests to compare map_except with equivalent expression
// using existing UDFs. The equivalent expression is:
// map_filter(map, (k, v) -> NOT contains(keys_array, k))
//
// Note: We already have explicit tests for NULL handling in MapExceptTest.
// These fuzzer tests focus on comparing behavior with non-null inputs.
class MapExceptFuzzerTest : public test::FunctionBaseTest {
 protected:
  // Helper to flatten vectors for consistent comparison across encodings
  template <typename T>
  static VectorPtr flatten(const std::shared_ptr<T>& vector) {
    SelectivityVector allRows(vector->size());
    auto result =
        BaseVector::create(vector->type(), vector->size(), vector->pool());
    result->copy(vector.get(), allRows, nullptr);
    return result;
  }

  // The equivalent SQL expression for map_except using existing UDFs.
  // map_except(map, keys_array) is equivalent to:
  // map_filter(map, (k, v) -> NOT coalesce(contains(keys_array, k), false))
  //
  // Note: We use coalesce to handle the case where contains returns null
  // (when comparing with null elements). map_except ignores null keys in
  // the exclusion array, so we treat null as "not found" (false).
  static constexpr const char* kEquivalentExpression =
      "map_filter(c0, (k, v) -> NOT coalesce(contains(c1, k), false))";

  // Get a SelectivityVector that excludes rows where either input is null.
  // This is because map_except returns NULL when either input is NULL.
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto inputMap = data->childAt(0);
    auto keysArray = data->childAt(1);
    SelectivityVector nonNullRows(data->size());

    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (inputMap->isNullAt(i) || keysArray->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testEquivalence(const RowVectorPtr& data) {
    auto result = evaluate("map_except(c0, c1)", data);
    auto expected = evaluate(kEquivalentExpression, data);

    // Get rows where neither input is null
    auto nonNullRows = getNonNullRows(data);

    // Compare only non-null rows (null propagation is tested separately)
    for (auto i = 0; i < data->size(); ++i) {
      if (nonNullRows.isValid(i)) {
        ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
            << "Mismatch at row " << i << ": expected " << expected->toString(i)
            << ", got " << result->toString(i);
      }
    }
  }

  void runFuzzTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      const test::FuzzerTestOptions& opts) {
    test::FuzzerTestHelper helper(pool());
    helper.runMapArrayTest(
        keyType,
        valueType,
        [this](const VectorPtr& inputMap, const VectorPtr& keysArray) {
          auto data = makeRowVector({inputMap, keysArray});
          testEquivalence(data);
        },
        opts);
  }
};

// Fuzz test with flat vectors, no nulls, fixed-size maps
TEST_F(MapExceptFuzzerTest, fuzzFlatNoNulls) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .nullRatio = 0.0,
       .containerLength = 10,
       .iterations = 10});
}

// Fuzz test with nulls in maps and keys array
TEST_F(MapExceptFuzzerTest, fuzzWithNulls) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with dictionary-encoded vectors
TEST_F(MapExceptFuzzerTest, fuzzDictionaryEncoded) {
  test::FuzzerTestHelper helper(pool());
  test::FuzzerTestOptions opts{
      .vectorSize = 100,
      .containerLength = 10,
      .containerHasNulls = true,
      .containerVariableLength = true};
  auto fuzzer = helper.createFuzzer(opts);

  for (auto i = 0; i < 10; ++i) {
    auto baseInputMap = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));
    auto baseKeysArray = fuzzer.fuzz(ARRAY(INTEGER()));

    auto inputMap = fuzzer.fuzzDictionary(baseInputMap, opts.vectorSize);
    auto keysArray = fuzzer.fuzzDictionary(baseKeysArray, opts.vectorSize);

    auto data = makeRowVector({inputMap, keysArray});
    auto flatData = makeRowVector({flatten(inputMap), flatten(keysArray)});

    auto result = evaluate("map_except(c0, c1)", data);
    auto expectedResult = evaluate("map_except(c0, c1)", flatData);
    assertEqualVectors(expectedResult, result);

    testEquivalence(flatData);
  }
}

// Fuzz test with variable-length maps and high null ratio
TEST_F(MapExceptFuzzerTest, fuzzVariableLengthWithHighNullRatio) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .nullRatio = 0.3,
       .containerLength = 20,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with string keys
TEST_F(MapExceptFuzzerTest, fuzzStringKeys) {
  runFuzzTest(
      VARCHAR(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with bigint keys
TEST_F(MapExceptFuzzerTest, fuzzBigintKeys) {
  runFuzzTest(
      BIGINT(),
      VARCHAR(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with empty maps and arrays
TEST_F(MapExceptFuzzerTest, fuzzWithEmptyContainers) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 2,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with various value types
TEST_F(MapExceptFuzzerTest, fuzzVariousValueTypes) {
  test::FuzzerTestOptions opts{
      .vectorSize = 100,
      .containerLength = 10,
      .containerHasNulls = true,
      .containerVariableLength = true,
      .iterations = 5};

  // Test with DOUBLE values
  runFuzzTest(INTEGER(), DOUBLE(), opts);

  // Test with BOOLEAN values
  runFuzzTest(INTEGER(), BOOLEAN(), opts);

  // Test with VARCHAR values
  runFuzzTest(INTEGER(), VARCHAR(), opts);
}

// Fuzz test with constant vectors
TEST_F(MapExceptFuzzerTest, fuzzConstantVectors) {
  test::FuzzerTestHelper helper(pool());
  test::FuzzerTestOptions opts{
      .vectorSize = 100, .nullRatio = 0.0, .containerLength = 5};
  auto fuzzer = helper.createFuzzer(opts);

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));
    auto keysArray = fuzzer.fuzzConstant(ARRAY(INTEGER()), opts.vectorSize);

    auto data = makeRowVector({inputMap, keysArray});
    testEquivalence(data);
  }
}

// Stress test with large vectors
TEST_F(MapExceptFuzzerTest, fuzzLargeVectors) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 1000,
       .containerLength = 50,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 5});
}

} // namespace
} // namespace facebook::velox::functions
