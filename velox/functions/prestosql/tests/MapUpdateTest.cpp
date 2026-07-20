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
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapUpdateTest : public test::FunctionBaseTest {
 public:
  template <typename T>
  void testFloatNaNs() {
    static const auto kNaN = std::numeric_limits<T>::quiet_NaN();

    auto data = makeRowVector({
        makeMapVectorFromJson<T, int32_t>({
            "{1:10, NaN:20, 3:30, 4:40, 5:50}",
            "{NaN:20}",
        }),
        makeArrayVector<T>({{1, kNaN, 5}, {kNaN}}),
        makeArrayVector<int32_t>({{100, 200, 500}, {999}}),
    });

    auto expected = makeMapVectorFromJson<T, int32_t>({
        "{1:100, NaN:200, 3:30, 4:40, 5:500}",
        "{NaN:999}",
    });
    auto result = evaluate("map_update(c0, c1, c2)", data);
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapUpdateTest, basicUpdate) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
          "{1:10, 2:20}",
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 3]",
          "[1]",
          "[1, 2]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[200, 300]",
          "[100]",
          "[10, 20]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:200, 3:300}",
      "{1:100, 2:20}",
      "{1:10, 2:20}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, addNewKeys) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[3, 4]",
          "[1, 2, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[30, 40]",
          "[10, 20, 30]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40}",
      "{1:10, 2:20, 3:30}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, mixedUpdateAndAdd) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
          "{1:10, 2:20, 3:30, 4:40}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 4, 5]",
          "[1, 5]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[200, 400, 500]",
          "[100, 500]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:200, 3:30, 4:400, 5:500}",
      "{1:100, 2:20, 3:30, 4:40, 5:500}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, emptyKeysAndValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
          "{}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[]",
          "[]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[]",
          "[]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, nullValuesInInput) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:null, 3:30}",
          "{1:null, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 4]",
          "[1]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[200, 400]",
          "[100]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:200, 3:30, 4:400}",
      "{1:100, 2:20}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, nullValuesInUpdate) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 4]",
          "[1, 3]",
      }),
      makeNullableArrayVector<int32_t>({
          {std::nullopt, 400},
          {std::nullopt, 300},
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:null, 3:30, 4:400}",
      "{1:null, 2:20, 3:300}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, varcharKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<std::string, int32_t>({
          R"({"apple": 1, "banana": 2, "cherry": 3})",
          R"({"banana": 2, "orange": 4})",
      }),
      makeArrayVectorFromJson<std::string>({
          R"(["apple", "date"])",
          R"(["banana", "grape"])",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 400]",
          "[200, 500]",
      }),
  });

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple": 100, "banana": 2, "cherry": 3, "date": 400})",
      R"({"banana": 200, "orange": 4, "grape": 500})",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, duplicateKeysInUpdateArrayThrows) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 1, 2]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 200, 300]",
      }),
  });

  VELOX_ASSERT_THROW(
      evaluate("map_update(c0, c1, c2)", data), "Duplicate key in keys array");
}

TEST_F(MapUpdateTest, mismatchedKeysValuesLengthThrows) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 2, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 200]",
      }),
  });

  VELOX_ASSERT_THROW(
      evaluate("map_update(c0, c1, c2)", data),
      "Keys and values arrays must have the same length");
}

TEST_F(MapUpdateTest, nullKeysInUpdateArrayAreIgnored) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
      }),
      makeNullableArrayVector<int64_t>({
          {std::nullopt, 2, std::nullopt, 4},
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 200, 300, 400]",
      }),
  });

  auto expected =
      makeMapVectorFromJson<int64_t, int32_t>({"{1:10, 2:200, 3:30, 4:400}"});

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, floatNaNs) {
  testFloatNaNs<float>();
  testFloatNaNs<double>();
}

TEST_F(MapUpdateTest, booleanKey) {
  auto data = makeRowVector({
      makeMapVector<bool, int32_t>({{{true, 10}, {false, 20}}}),
      makeArrayVector<bool>({{true}}),
      makeArrayVector<int32_t>({{100}}),
  });

  auto expected = makeMapVector<bool, int32_t>({{{true, 100}, {false, 20}}});

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, allKeysUpdated) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 2, 3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 200, 300]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:100, 2:200, 3:300}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, largeMap) {
  // Create a map with 10 entries and update/add 3
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80, 9:90, 10:100}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[5, 10, 11]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[500, 1000, 1100]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:500, 6:60, 7:70, 8:80, 9:90, 10:1000, 11:1100}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, timestampKeys) {
  auto data = makeRowVector({
      makeMapVector<Timestamp, int32_t>(
          {{{Timestamp(1, 0), 10}, {Timestamp(2, 0), 20}}}),
      makeArrayVector<Timestamp>({{Timestamp(2, 0), Timestamp(3, 0)}}),
      makeArrayVector<int32_t>({{200, 30}}),
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);

  auto expected = makeMapVector<Timestamp, int32_t>(
      {{{Timestamp(1, 0), 10}, {Timestamp(2, 0), 200}, {Timestamp(3, 0), 30}}});

  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, preserveUnchangedKeys) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{5:50, 3:30, 1:10}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 4]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[20, 40]",
      }),
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{5:50, 3:30, 1:10, 2:20, 4:40}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, updateToNull) {
  // Using explicit vector type to avoid ambiguity
  const std::vector<std::vector<std::optional<int32_t>>> valuesData = {
      {std::nullopt},
  };

  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1]",
      }),
      makeNullableArrayVector<int32_t>(valuesData),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:null, 2:20}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

// Tests specifically for order preservation behavior.
TEST_F(MapUpdateTest, orderPreservationUpdateFirstKey) {
  // Update the first key - it should remain in position 1.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:100, 2:20, 3:30, 4:40, 5:50}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationUpdateMiddleKey) {
  // Update the middle key - it should remain in its original position.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[3]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[300]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:300, 4:40, 5:50}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationUpdateLastKey) {
  // Update the last key - it should remain in its original position.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[5]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[500]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:500}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationUpdateMultipleNonConsecutive) {
  // Update multiple non-consecutive keys - they should all remain in place.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 3, 5]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[100, 300, 500]",
      }),
  });

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:100, 2:20, 3:300, 4:40, 5:500}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationNewKeysAppendedInOrder) {
  // New keys should be appended at the end in the order they appear
  // in the update arrays.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[5, 3, 4]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[50, 30, 40]",
      }),
  });

  // New keys 5, 3, 4 should be appended in that order.
  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 5:50, 3:30, 4:40}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationMixedUpdateAndNewKeys) {
  // Mix of updates and new keys - updates stay in place, new keys append.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[2, 5, 4]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[200, 50, 40]",
      }),
  });

  // Key 2 stays in position 2 with new value, keys 5 and 4 are appended.
  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:200, 3:30, 5:50, 4:40}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationVarcharKeys) {
  // Test order preservation with varchar keys.
  auto data = makeRowVector({
      makeMapVectorFromJson<std::string, int32_t>({
          R"({"alpha": 1, "beta": 2, "gamma": 3, "delta": 4})",
      }),
      makeArrayVectorFromJson<std::string>({
          R"(["beta", "delta", "epsilon"])",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[200, 400, 500]",
      }),
  });

  // beta and delta should stay in place, epsilon appended at end.
  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"alpha": 1, "beta": 200, "gamma": 3, "delta": 400, "epsilon": 500})",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

TEST_F(MapUpdateTest, orderPreservationUpdateKeysInReverseOrder) {
  // Update keys are provided in reverse order of their position in input map.
  // The output should preserve the original input map order, not the update
  // array order.
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[4, 2, 1]",
      }),
      makeArrayVectorFromJson<int32_t>({
          "[400, 200, 100]",
      }),
  });

  // Keys should remain in original order: 1, 2, 3, 4 (not 4, 2, 1).
  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:100, 2:200, 3:30, 4:400}",
  });

  auto result = evaluate("map_update(c0, c1, c2)", data);
  assertEqualVectors(expected, result);
}

} // namespace

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests use VectorFuzzer to generate random inputs and verify
// properties of the map_update function. Since map_update has complex error
// conditions (duplicate keys, mismatched lengths), the tests use
// try(map_update(...)) to gracefully handle errors and verify that:
// - When the function succeeds, the result map size is correct
// - Null handling is correct
// ============================================================================

class MapUpdateFuzzerTest : public test::FunctionBaseTest {
 protected:
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    SelectivityVector nonNullRows(data->size());
    for (vector_size_t i = 0; i < data->size(); ++i) {
      bool hasNull = false;
      for (vector_size_t j = 0; j < data->childrenSize(); ++j) {
        if (data->childAt(j)->isNullAt(i)) {
          hasNull = true;
          break;
        }
      }
      if (hasNull) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testMapUpdateProperties(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    try {
      result = evaluate("try(map_update(c0, c1, c2))", data);
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

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i) || result->isNullAt(i)) {
        continue;
      }

      if (inputMap->isNullAt(i)) {
        continue;
      }

      auto resultMap = result->as<MapVector>();
      if (!resultMap) {
        continue;
      }

      auto origSize = inputMapVector->sizeAt(i);
      auto resultSize = resultMap->sizeAt(i);
      ASSERT_GE(resultSize, origSize)
          << "Result map should be at least as large as input map at row " << i;
    }
  }

  template <typename KeyType, typename ValueType>
  void runFuzzerTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      vector_size_t vectorSize = 50,
      double nullRatio = 0.1,
      vector_size_t containerLength = 5) {
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.nullRatio = nullRatio;
    opts.containerLength = containerLength;
    VectorFuzzer fuzzer(opts, pool());

    auto inputMap = fuzzer.fuzz(MAP(keyType, valueType));
    auto keys = fuzzer.fuzz(ARRAY(keyType));
    auto values = fuzzer.fuzz(ARRAY(valueType));
    auto data = makeRowVector({inputMap, keys, values});
    testMapUpdateProperties(data);
  }
};

TEST_F(MapUpdateFuzzerTest, fuzzInteger) {
  runFuzzerTest<int32_t, int32_t>(INTEGER(), INTEGER());
}

TEST_F(MapUpdateFuzzerTest, fuzzBigint) {
  runFuzzerTest<int64_t, int64_t>(BIGINT(), BIGINT());
}

TEST_F(MapUpdateFuzzerTest, fuzzVarchar) {
  runFuzzerTest<StringView, StringView>(VARCHAR(), VARCHAR());
}

TEST_F(MapUpdateFuzzerTest, fuzzDouble) {
  runFuzzerTest<double, double>(DOUBLE(), DOUBLE());
}

TEST_F(MapUpdateFuzzerTest, fuzzHighNullRatio) {
  runFuzzerTest<int32_t, int32_t>(INTEGER(), INTEGER(), 50, 0.5, 5);
}

TEST_F(MapUpdateFuzzerTest, fuzzSmallint) {
  runFuzzerTest<int16_t, int16_t>(SMALLINT(), SMALLINT());
}

TEST_F(MapUpdateFuzzerTest, fuzzLargeVectors) {
  runFuzzerTest<int32_t, int32_t>(INTEGER(), INTEGER(), 200, 0.1, 3);
}

} // namespace facebook::velox::functions
