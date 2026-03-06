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

#include <limits>

#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class MarkSortedTest : public OperatorTestBase {
 protected:
  void assertMarkSortedResults(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::string& markerName,
      const std::vector<bool>& expectedMarkers) {
    auto plan = PlanBuilder()
                    .values(input)
                    .markSorted(markerName, sortingKeys, sortingOrders)
                    .planNode();

    auto results = AssertQueryBuilder(plan).copyResults(pool());

    // Verify marker column exists and has correct values
    auto markerVector = results->childAt(results->childrenSize() - 1);
    ASSERT_EQ(markerVector->size(), expectedMarkers.size());

    auto flatMarker = markerVector->as<FlatVector<bool>>();
    for (vector_size_t i = 0; i < expectedMarkers.size(); ++i) {
      ASSERT_EQ(flatMarker->valueAt(i), expectedMarkers[i])
          << "Mismatch at row " << i;
    }
  }
};

TEST_F(MarkSortedTest, singleKeyAscSorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, singleKeyDescSorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({5, 4, 3, 2, 1}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kDescNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, singleKeyAscUnsorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 3, 2, 4, 5}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, false, true, true});
}

TEST_F(MarkSortedTest, singleKeyDescUnsorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({5, 4, 6, 2, 1}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kDescNullsLast},
      "is_sorted",
      {true, true, false, true, true});
}

TEST_F(MarkSortedTest, multipleKeysSorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
      makeFlatVector<int32_t>({1, 2, 1, 2, 1}),
  });

  assertMarkSortedResults(
      {data},
      {"c0", "c1"},
      {core::kAscNullsLast, core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, multipleKeysUnsorted) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
      makeFlatVector<int32_t>({1, 3, 2, 1, 1}),
  });

  assertMarkSortedResults(
      {data},
      {"c0", "c1"},
      {core::kAscNullsLast, core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true});
}

TEST_F(MarkSortedTest, nullsFirstSorted) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt, 1, 2, 3}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsFirst},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, nullsLastSorted) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, 3, std::nullopt, std::nullopt}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, nullsFirstUnsorted) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 2, 3, 4}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsFirst},
      "is_sorted",
      {true, false, true, true, true});
}

TEST_F(MarkSortedTest, crossBatchSorted) {
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  assertMarkSortedResults(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, crossBatchUnsorted) {
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 5}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({3, 4, 6}),
  });

  assertMarkSortedResults(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true, true});
}

TEST_F(MarkSortedTest, emptyBatch) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({}),
  });

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", {});
}

TEST_F(MarkSortedTest, allNullValues) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  });

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", {true, true, true});
}

TEST_F(MarkSortedTest, firstRowAlwaysTrue) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({100, 1, 2, 3}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, false, true, true});
}

TEST_F(MarkSortedTest, singleRow) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>({42}),
  });

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", {true});
}

TEST_F(MarkSortedTest, stringKey) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"apple", "banana", "cherry", "date"}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true});
}

TEST_F(MarkSortedTest, stringKeyUnsorted) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"apple", "cherry", "banana", "date"}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, false, true});
}

// Tests for D3 optimizations: zero-copy, copy mode, and SIMD path.

TEST_F(MarkSortedTest, zeroCopySmallBatch) {
  // Default threshold is 1000 rows. Small batches should use zero-copy mode.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  assertMarkSortedResults(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, copyLargeBatch) {
  // Create batches larger than default threshold (1000 rows) to trigger
  // copy mode. Use a config override to set threshold to 10.
  std::vector<int32_t> values1(20);
  std::vector<int32_t> values2(20);
  for (int i = 0; i < 20; ++i) {
    values1[i] = i;
    values2[i] = 20 + i;
  }
  auto batch1 = makeRowVector({makeFlatVector<int32_t>(values1)});
  auto batch2 = makeRowVector({makeFlatVector<int32_t>(values2)});

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .markSorted("is_sorted", {"c0"}, {core::kAscNullsLast})
                  .planNode();

  // Use config to set threshold to 10, so 20-row batches trigger copy mode.
  auto results =
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kMarkSortedZeroCopyThreshold, "10")
          .copyResults(pool());

  auto markerVector = results->childAt(results->childrenSize() - 1);
  ASSERT_EQ(markerVector->size(), 40);

  auto flatMarker = markerVector->as<FlatVector<bool>>();
  for (vector_size_t i = 0; i < 40; ++i) {
    ASSERT_TRUE(flatMarker->valueAt(i)) << "Mismatch at row " << i;
  }
}

TEST_F(MarkSortedTest, zeroCopyThresholdConfig) {
  // Test that the threshold config works correctly.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({4, 5, 6, 7, 8}), // 4 < 5, so first row unsorted
  });

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .markSorted("is_sorted", {"c0"}, {core::kAscNullsLast})
                  .planNode();

  // Threshold of 3 means batch1 (5 rows) uses copy mode.
  auto results =
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kMarkSortedZeroCopyThreshold, "3")
          .copyResults(pool());

  auto markerVector = results->childAt(results->childrenSize() - 1);
  ASSERT_EQ(markerVector->size(), 10);

  auto flatMarker = markerVector->as<FlatVector<bool>>();
  // First batch: all sorted
  for (vector_size_t i = 0; i < 5; ++i) {
    ASSERT_TRUE(flatMarker->valueAt(i)) << "Mismatch at row " << i;
  }
  // Second batch: first row (index 5) compares 4 vs 5, which is unsorted
  ASSERT_FALSE(flatMarker->valueAt(5)) << "Row 5 should be unsorted";
  for (vector_size_t i = 6; i < 10; ++i) {
    ASSERT_TRUE(flatMarker->valueAt(i)) << "Mismatch at row " << i;
  }
}

TEST_F(MarkSortedTest, fastPathInteger) {
  // Create a batch large enough for fast path (>= 16 rows).
  std::vector<int32_t> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = i;
  }
  auto data = makeRowVector({makeFlatVector<int32_t>(values)});

  std::vector<bool> expected(32, true);
  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, fastPathBigint) {
  std::vector<int64_t> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = static_cast<int64_t>(i) * 1000000000LL;
  }
  auto data = makeRowVector({makeFlatVector<int64_t>(values)});

  std::vector<bool> expected(32, true);
  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, genericPathDouble) {
  std::vector<double> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = i * 1.5;
  }
  auto data = makeRowVector({makeFlatVector<double>(values)});

  std::vector<bool> expected(32, true);
  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, genericPathMultiKey) {
  // Multi-key should fall back to generic path.
  std::vector<int32_t> values1(32);
  std::vector<int32_t> values2(32);
  for (int i = 0; i < 32; ++i) {
    values1[i] = i / 4; // Groups of 4
    values2[i] = i % 4;
  }
  auto data = makeRowVector(
      {makeFlatVector<int32_t>(values1), makeFlatVector<int32_t>(values2)});

  std::vector<bool> expected(32, true);
  assertMarkSortedResults(
      {data},
      {"c0", "c1"},
      {core::kAscNullsLast, core::kAscNullsLast},
      "is_sorted",
      expected);
}

TEST_F(MarkSortedTest, genericPathVarchar) {
  // VARCHAR should fall back to generic path.
  std::vector<std::string> values(20);
  for (int i = 0; i < 20; ++i) {
    values[i] = fmt::format("value_{:03d}", i);
  }
  auto data = makeRowVector({makeFlatVector<std::string>(values)});

  std::vector<bool> expected(20, true);
  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, genericPathWithNulls) {
  // Nulls should cause fallback to generic path.
  // With kAscNullsLast, nulls are considered larger than any value.
  std::vector<std::optional<int32_t>> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = i;
  }
  // Put null at position 16 - next row (17) will have value 17,
  // but since NULL is "larger" in NullsLast, row 17 is NOT sorted.
  values[16] = std::nullopt;

  auto data = makeRowVector({makeNullableFlatVector<int32_t>(values)});

  std::vector<bool> expected(32, true);
  expected[17] = false; // Row 17 is unsorted because 17 < NULL (NullsLast)

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, genericPathSmallBatch) {
  // Small batch (<16 rows) should fall back to generic path.
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, fastPathDescending) {
  // Test fast path with descending order.
  std::vector<int32_t> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = 100 - i;
  }
  auto data = makeRowVector({makeFlatVector<int32_t>(values)});

  std::vector<bool> expected(32, true);
  assertMarkSortedResults(
      {data}, {"c0"}, {core::kDescNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, allEqualValues) {
  // All equal keys should be considered sorted (BUG-3 regression test).
  auto data = makeRowVector({
      makeFlatVector<int32_t>({5, 5, 5, 5, 5}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, crossBatchNonFirstColumnKey) {
  // Sort key on c1 (not c0) to verify cross-batch comparison uses correct
  // column indices (BUG-1 regression test).
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int32_t>({1, 2, 3}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({40, 50, 60}),
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  assertMarkSortedResults(
      {batch1, batch2},
      {"c1"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, crossBatchNonFirstColumnKeyUnsorted) {
  // Sort key on c1 with unsorted cross-batch boundary.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int32_t>({1, 2, 5}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({40, 50, 60}),
      makeFlatVector<int32_t>({3, 4, 6}),
  });

  assertMarkSortedResults(
      {batch1, batch2},
      {"c1"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true, true});
}

TEST_F(MarkSortedTest, copyModeCrossBatchNonFirstColumnKey) {
  // Copy mode with sort key on c1 (not c0) to verify copy mode handles
  // column index remapping correctly (BUG-1 regression test for copy mode).
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30}),
      makeFlatVector<int32_t>({1, 2, 3}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({40, 50, 60}),
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .markSorted("is_sorted", {"c1"}, {core::kAscNullsLast})
                  .planNode();

  // Threshold of 1 forces copy mode for all batches.
  auto results =
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kMarkSortedZeroCopyThreshold, "1")
          .copyResults(pool());

  auto markerVector = results->childAt(results->childrenSize() - 1);
  ASSERT_EQ(markerVector->size(), 6);

  auto flatMarker = markerVector->as<FlatVector<bool>>();
  for (vector_size_t i = 0; i < 6; ++i) {
    ASSERT_TRUE(flatMarker->valueAt(i)) << "Mismatch at row " << i;
  }
}

TEST_F(MarkSortedTest, fastPathUnsortedLargeBatch) {
  // Test fast path detects violations in large batch.
  std::vector<int32_t> values(32);
  for (int i = 0; i < 32; ++i) {
    values[i] = i;
  }
  // Introduce violations at positions 10 and 25.
  values[10] = 5;
  values[25] = 20;

  auto data = makeRowVector({makeFlatVector<int32_t>(values)});

  std::vector<bool> expected(32, true);
  expected[10] = false;
  expected[25] = false;

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", expected);
}

TEST_F(MarkSortedTest, nanValuesDouble) {
  // NaN is treated as greater than all non-NaN values by Velox's compare().
  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 2.0, nan, 3.0, nan, nan}),
  });

  // ASC: 1.0 <= 2.0 (sorted), 2.0 <= NaN (sorted, NaN is largest),
  //      NaN > 3.0 (unsorted), 3.0 <= NaN (sorted), NaN == NaN (sorted).
  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true, true});
}

TEST_F(MarkSortedTest, nanValuesFloat) {
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto data = makeRowVector({
      makeFlatVector<float>({1.0f, nan, 2.0f, 3.0f}),
  });

  // ASC: 1.0 <= NaN (sorted), NaN > 2.0 (unsorted), 2.0 <= 3.0 (sorted).
  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, false, true});
}
