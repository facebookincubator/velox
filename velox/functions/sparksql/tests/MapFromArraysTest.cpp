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
#include <cstdint>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class MapFromArraysTest : public SparkFunctionBaseTest {
 protected:
  void testMapFromArrays(
      const VectorPtr& keys,
      const VectorPtr& values,
      const VectorPtr& expected) {
    auto result =
        evaluate("map_from_arrays(c0, c1)", makeRowVector({keys, values}));
    ::facebook::velox::test::assertEqualVectors(expected, result);
  }

  void testMapFromArraysFails(
      const VectorPtr& keys,
      const VectorPtr& values,
      const std::string& errorMessage) {
    VELOX_ASSERT_USER_THROW(
        evaluate("map_from_arrays(c0, c1)", makeRowVector({keys, values})),
        errorMessage);
  }

  // Sets the Spark 'spark.sql.mapKeyDedupPolicy' equivalent query config.
  void setThrowExceptionOnDuplicateMapKeys(bool value) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kThrowExceptionOnDuplicateMapKeys,
         value ? "true" : "false"},
    });
  }
};

TEST_F(MapFromArraysTest, basic) {
  auto keys = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3]",
      "[4, 5]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[40, 50]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{4: 40, 5: 50}",
  });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, emptyArrays) {
  auto keys = makeArrayVectorFromJson<int64_t>({"[]"});
  auto values = makeArrayVectorFromJson<int64_t>({"[]"});
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({"{}"});
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, nullInputArray) {
  auto keys = makeArrayVectorFromJson<int64_t>({
      "null",
      "[1, 2]",
      "[3]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "null",
      "[30]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "null",
      "null",
      "{3: 30}",
  });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, lastWinDuplicateKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto keys = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 1]",
      "[3, 4]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[40, 50]",
  });
  // Key 1 appears twice; the last value (30) wins and the map holds a single
  // entry per distinct key.
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1: 30, 2: 20}",
      "{3: 40, 4: 50}",
  });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, lastWinPreservesFirstKeyPosition) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto keys = makeArrayVectorFromJson<int64_t>({"[2, 1, 2]"});
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20, 30]"});
  auto data = makeRowVector({keys, values});

  // Spark's ArrayBasedMapBuilder overwrites the value of the first occurrence
  // in place, so the repeated key stays at its first position and the entries
  // keep insertion order. Pinned via map_keys/map_values because map equality
  // ignores entry order.
  ::facebook::velox::test::assertEqualVectors(
      makeArrayVectorFromJson<int64_t>({"[2, 1]"}),
      evaluate("map_keys(map_from_arrays(c0, c1))", data));
  ::facebook::velox::test::assertEqualVectors(
      makeArrayVectorFromJson<int64_t>({"[30, 20]"}),
      evaluate("map_values(map_from_arrays(c0, c1))", data));
}

TEST_F(MapFromArraysTest, lastWinKeyRepeatedThreeOrMoreTimes) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto keys = makeArrayVectorFromJson<int64_t>({
      "[7, 7, 7]",
      "[8, 8, 8, 8]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3]",
      "[4, 5, 6, 7]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{7: 3}",
      "{8: 7}",
  });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, throwOnDuplicateKeys) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto keys = makeArrayVectorFromJson<int64_t>({"[1, 2, 1]"});
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20, 30]"});
  testMapFromArraysFails(keys, values, "Duplicate map key (1) was found.");
}

// Under EXCEPTION no row can shrink, so the result references the input arrays
// instead of building an entry index for each side. The tests below cover that
// path; the ones above only reach its error cases.
TEST_F(MapFromArraysTest, throwPolicyWithoutDuplicates) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto keys = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3]",
      "[]",
      "[4, 5]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[]",
      "[40, 50]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1: 10, 2: 20, 3: 30}",
      "{}",
      "{4: 40, 5: 50}",
  });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, throwPolicyComplexTypeKeys) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto keys = makeNestedArrayVectorFromJson<int64_t>({
      "[[1, 2], [3, 4]]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20]"});
  auto expected = makeMapVector(
      {0},
      makeArrayVectorFromJson<int64_t>({"[1, 2]", "[3, 4]"}),
      makeFlatVector<int64_t>({10, 20}));
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, throwPolicyTryDeselectsFailedRows) {
  setThrowExceptionOnDuplicateMapKeys(true);

  auto keys = makeArrayVectorFromJson<int64_t>({
      "[1, 1]",
      "[2, null]",
      "[3, 4]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[30, 40]",
      "[50, 60]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "null",
      "null",
      "{3: 50, 4: 60}",
  });
  ::facebook::velox::test::assertEqualVectors(
      expected,
      evaluate("try(map_from_arrays(c0, c1))", makeRowVector({keys, values})));
}

TEST_F(MapFromArraysTest, nullKey) {
  auto keys = makeArrayVectorFromJson<int64_t>({"[1, null]"});
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20]"});
  testMapFromArraysFails(keys, values, "Cannot use null as map key!");

  setThrowExceptionOnDuplicateMapKeys(true);
  testMapFromArraysFails(keys, values, "Cannot use null as map key!");
}

TEST_F(MapFromArraysTest, mismatchedArrayLengths) {
  auto keys = makeArrayVectorFromJson<int64_t>({"[1, 2, 3]"});
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20]"});
  testMapFromArraysFails(
      keys,
      values,
      "The key array and value array of MapData must have the same length.");
}

TEST_F(MapFromArraysTest, complexTypeKeys) {
  auto keys = makeNestedArrayVectorFromJson<int64_t>({
      "[[1, 2], [3, 4]]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20]"});
  auto expected = makeMapVector(
      {0},
      makeArrayVectorFromJson<int64_t>({"[1, 2]", "[3, 4]"}),
      makeFlatVector<int64_t>({10, 20}));
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, complexTypeKeysDuplicate) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto keys = makeNestedArrayVectorFromJson<int64_t>({
      "[[1, 2], [1, 2]]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({"[10, 20]"});
  auto expected = makeMapVector(
      {0},
      makeArrayVectorFromJson<int64_t>({"[1, 2]"}),
      makeFlatVector<int64_t>({20}));
  testMapFromArrays(keys, values, expected);

  setThrowExceptionOnDuplicateMapKeys(true);
  testMapFromArraysFails(keys, values, "Duplicate map key ({1, 2}) was found.");
}

TEST_F(MapFromArraysTest, allRowsNullInput) {
  auto keys = makeAllNullArrayVector(2, BIGINT());
  auto values = makeArrayVectorFromJson<int64_t>({"[1]", "[2]"});
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({"null", "null"});
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, dictionaryEncodedInputs) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto baseKeys = makeArrayVectorFromJson<int64_t>({
      "[1, 2]",
      "[3, 3, 4]",
  });
  auto baseValues = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[30, 40, 50]",
  });
  // Rows are read out of order and the second base row is read twice, so the
  // per-row sizes and offsets must come from the decoded index, not the row.
  auto indices = makeIndices({1, 0, 1});
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{3: 40, 4: 50}",
      "{1: 10, 2: 20}",
      "{3: 40, 4: 50}",
  });
  testMapFromArrays(
      wrapInDictionary(indices, 3, baseKeys),
      wrapInDictionary(indices, 3, baseValues),
      expected);
}

TEST_F(MapFromArraysTest, dictionaryEncodedInputsThrowPolicy) {
  setThrowExceptionOnDuplicateMapKeys(true);

  // Under EXCEPTION a row cannot shrink, so the result is otherwise expressible
  // as a reference to the input arrays. Only the encoding rules that out here,
  // and the result must follow the dictionary order rather than the order the
  // rows sit in the base vector.
  //
  // Wrapping only the keys defeats the shared-wrapping peel that would
  // otherwise hand the function two identity-mapped vectors. Every row is the
  // same length, so the base and the values still agree on offset and size and
  // the alignment check alone would let this through.
  auto baseKeys = makeArrayVectorFromJson<int64_t>({
      "[1, 2]",
      "[3, 4]",
      "[5, 6]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[30, 40]",
      "[50, 60]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{5: 10, 6: 20}",
      "{3: 30, 4: 40}",
      "{1: 50, 2: 60}",
  });
  testMapFromArrays(
      wrapInDictionary(makeIndices({2, 1, 0}), 3, baseKeys), values, expected);
}

TEST_F(MapFromArraysTest, constantEncodedKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto baseKeys = makeArrayVectorFromJson<int64_t>({"[1, 2, 1]"});
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[40, 50, 60]",
      "[70, 80, 90]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1: 30, 2: 20}",
      "{1: 60, 2: 50}",
      "{1: 90, 2: 80}",
  });
  testMapFromArrays(
      BaseVector::wrapInConstant(3, 0, baseKeys), values, expected);
}

TEST_F(MapFromArraysTest, constantEncodedKeysThrowOnDuplicate) {
  setThrowExceptionOnDuplicateMapKeys(true);

  // The key array is shared by every row, so a duplicate in it fails all of
  // them rather than just the row that happened to be inspected first.
  auto baseKeys = makeArrayVectorFromJson<int64_t>({"[1, 2, 1]"});
  auto keys = BaseVector::wrapInConstant(2, 0, baseKeys);
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20, 30]",
      "[40, 50, 60]",
  });
  testMapFromArraysFails(keys, values, "Duplicate map key (1) was found.");

  auto expected = makeMapVectorFromJson<int64_t, int64_t>({"null", "null"});
  ::facebook::velox::test::assertEqualVectors(
      expected,
      evaluate("try(map_from_arrays(c0, c1))", makeRowVector({keys, values})));
}

TEST_F(MapFromArraysTest, constantEncodedKeysMismatchedLengths) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // The length check stays per-row even though the keys are shared, and a row
  // that fails it must not shift the offsets of the rows after it.
  auto baseKeys = makeArrayVectorFromJson<int64_t>({"[1, 2]"});
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[30]",
      "[40, 50]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{1: 10, 2: 20}",
      "null",
      "{1: 40, 2: 50}",
  });
  ::facebook::velox::test::assertEqualVectors(
      expected,
      evaluate(
          "try(map_from_arrays(c0, c1))",
          makeRowVector({BaseVector::wrapInConstant(3, 0, baseKeys), values})));
}

TEST_F(MapFromArraysTest, tryDeselectsFailedRows) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Rows 0 and 1 fail on a length mismatch and a null key respectively. Both
  // sit before rows that succeed, so a failed row must not shift the offsets
  // of the rows that follow it.
  auto keys = makeArrayVectorFromJson<int64_t>({
      "[1, 2, 3]",
      "[4, null]",
      "[5, 5]",
      "[6]",
  });
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[40, 50]",
      "[60, 70]",
      "[80]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "null",
      "null",
      "{5: 70}",
      "{6: 80}",
  });
  ::facebook::velox::test::assertEqualVectors(
      expected,
      evaluate("try(map_from_arrays(c0, c1))", makeRowVector({keys, values})));
}

TEST_F(MapFromArraysTest, resultSize) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Each branch writes a disjoint subset of rows into the same result vector,
  // so the second call has to extend a partially populated result.
  auto condition = makeFlatVector<int64_t>({1, 2, 3});
  auto keys = makeArrayVectorFromJson<int64_t>({"[1, 1]", "[2]", "[3, 3]"});
  auto values = makeArrayVectorFromJson<int64_t>({
      "[10, 20]",
      "[30]",
      "[40, 41]",
  });
  auto expected = makeMapVectorFromJson<int64_t, int64_t>({
      "{10: 1, 20: 1}",
      "{30: 2}",
      "{3: 41}",
  });
  ::facebook::velox::test::assertEqualVectors(
      expected,
      evaluate(
          "if(greaterthan(c2, 2), map_from_arrays(c0, c1), map_from_arrays(c1, c0))",
          makeRowVector({keys, values, condition})));
}

TEST_F(MapFromArraysTest, wideKeyArray) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Deduplication hashes the key value, so a wide key array stays linear. This
  // checks correctness at a width the pairwise scan it replaced could not serve
  // in reasonable time; it is NOT a performance guard. A slower strategy still
  // produces the right answer here, so a regression would pass rather than
  // fail.
  constexpr vector_size_t kNumKeys = 50'000;
  constexpr vector_size_t kNumDistinctKeys = kNumKeys / 2;

  auto keys = makeArrayVector<int64_t>(
      1,
      [=](auto /*row*/) { return kNumKeys; },
      [=](auto index) { return index % kNumDistinctKeys; });
  auto values = makeArrayVector<int64_t>(
      1,
      [=](auto /*row*/) { return kNumKeys; },
      [](auto index) { return index; });

  // Every key occurs exactly twice, so LAST_WIN keeps the value from the
  // second half of the array.
  auto expected = makeMapVector<int64_t, int64_t>(
      1,
      [=](auto /*row*/) { return kNumDistinctKeys; },
      [](auto index) { return index; },
      [=](auto index) { return index + kNumDistinctKeys; });
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, wideComplexTypeKeyArray) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Complex keys cannot be hashed on a primitive value, so they order by
  // BaseVector::compare instead. Same intent as wideKeyArray: correctness at a
  // width the pairwise scan could not serve, not a performance guard.
  constexpr vector_size_t kNumKeys = 2'000;
  constexpr vector_size_t kNumDistinctKeys = kNumKeys / 2;

  auto keys = makeArrayVector(
      {0},
      makeArrayVector<int64_t>(
          kNumKeys,
          [](auto /*row*/) { return 1; },
          [=](auto index) { return index % kNumDistinctKeys; }));
  auto values = makeArrayVector<int64_t>(
      1,
      [=](auto /*row*/) { return kNumKeys; },
      [](auto index) { return index; });

  // Every key occurs exactly twice, so LAST_WIN keeps the value from the second
  // half while the key holds the position of its first occurrence.
  auto expected = makeMapVector(
      {0},
      makeArrayVector<int64_t>(
          kNumDistinctKeys,
          [](auto /*row*/) { return 1; },
          [](auto index) { return index; }),
      makeFlatVector<int64_t>(
          kNumDistinctKeys, [](auto row) { return row + kNumDistinctKeys; }));
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, stringKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  auto keys = makeArrayVector<std::string>({{"a", "b", "a", ""}});
  auto values = makeArrayVectorFromJson<int64_t>({"[1, 2, 3, 4]"});
  auto expected =
      makeMapVector<std::string, int64_t>({{{"a", 3}, {"b", 2}, {"", 4}}});
  testMapFromArrays(keys, values, expected);
}

TEST_F(MapFromArraysTest, floatingPointKeys) {
  setThrowExceptionOnDuplicateMapKeys(false);

  // Pins the equality the deduplication inherits from BaseVector: NaNs are
  // equal to each other, and negative zero is equal to positive zero. Spark
  // instead keys on Double.equals and treats -0.0 as distinct from 0.0; that
  // divergence is Velox-wide and tracked separately.
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  auto keys = makeArrayVector<double>({{0.0, -0.0, kNaN, kNaN, 1.5}});
  auto values = makeArrayVectorFromJson<int64_t>({"[1, 2, 3, 4, 5]"});
  auto expected = makeMapVector(
      {0},
      makeFlatVector<double>({0.0, kNaN, 1.5}),
      makeFlatVector<int64_t>({2, 4, 5}));
  testMapFromArrays(keys, values, expected);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
