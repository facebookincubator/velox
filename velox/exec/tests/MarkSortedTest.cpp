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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/ConstantVector.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

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

  void assertMarkSortedResultsWithConfig(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      const std::string& markerName,
      const std::vector<bool>& expectedMarkers,
      const std::unordered_map<std::string, std::string>& configs) {
    auto plan = PlanBuilder()
                    .values(input)
                    .markSorted(markerName, sortingKeys, sortingOrders)
                    .planNode();

    auto builder = AssertQueryBuilder(plan);
    for (const auto& [key, value] : configs) {
      builder.config(key, value);
    }
    auto results = builder.copyResults(pool());

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

// --- Optimization tests ---

TEST_F(MarkSortedTest, constantVectorSorted) {
  // All key columns are constant non-null — trivially sorted.
  auto data = makeRowVector({
      BaseVector::createConstant(INTEGER(), 42, 5, pool()),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, constantVectorWithNull) {
  // Constant null key — falls back to generic path.
  auto data = makeRowVector({
      BaseVector::createNullConstant(INTEGER(), 5, pool()),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, simdPathInteger) {
  // Single flat non-null INTEGER key — uses SIMD path (ascending).
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, simdPathBigint) {
  // SIMD path with BIGINT key (descending), with unsorted violations.
  auto data = makeRowVector({
      makeFlatVector<int64_t>({100, 90, 80, 85, 70, 60, 50, 40}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kDescNullsLast},
      "is_sorted",
      {true, true, true, false, true, true, true, true});
}

TEST_F(MarkSortedTest, genericPathDouble) {
  // DOUBLE key — SIMD not applicable for floating point, uses generic path.
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 2.0, 3.0, 2.5, 4.0}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true});
}

TEST_F(MarkSortedTest, simdFallbackMultiKey) {
  // Multi-key: SIMD not applicable, falls back to generic.
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

TEST_F(MarkSortedTest, simdFallbackVarchar) {
  // VARCHAR key: SIMD not applicable, falls back to generic.
  auto data = makeRowVector({
      makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true});
}

TEST_F(MarkSortedTest, simdFallbackWithNulls) {
  // Flat vector with nulls — SIMD not applicable, falls back to generic.
  // With kAscNullsLast, null > any value, so row 3 (4 after null) is unsorted.
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, std::nullopt, 4, 5}),
  });

  assertMarkSortedResults(
      {data},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true});
}

TEST_F(MarkSortedTest, zeroCopySmallBatch) {
  // Small batches (< threshold) use zero-copy cross-batch. Verify correctness.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  // Default threshold is 1000, so 3-row batches use zero-copy.
  assertMarkSortedResults(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, true, true, true});
}

TEST_F(MarkSortedTest, copyLargeBatch) {
  // Set threshold very low so even small batches use copy mode.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 5}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({3, 4, 6}),
  });

  assertMarkSortedResultsWithConfig(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, true, true, false, true, true},
      {{QueryConfig::kMarkSortedZeroCopyThreshold, "1"}});
}

TEST_F(MarkSortedTest, zeroCopyThresholdConfig) {
  // Verify threshold is configurable: set to 2 so batch of 3 uses copy mode
  // and batch of 1 uses zero-copy.
  auto batch1 = makeRowVector({
      makeFlatVector<int32_t>({10}),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int32_t>({5, 6, 7}),
  });

  // batch1 (1 row) < threshold(2) → zero-copy
  // batch2 first row (5) < prev (10) → unsorted for ascending
  assertMarkSortedResultsWithConfig(
      {batch1, batch2},
      {"c0"},
      {core::kAscNullsLast},
      "is_sorted",
      {true, false, true, true},
      {{QueryConfig::kMarkSortedZeroCopyThreshold, "2"}});
}

TEST_F(MarkSortedTest, simdSmallBatch) {
  // SIMD works correctly with very small batches (< 16 rows).
  auto data = makeRowVector({
      makeFlatVector<int32_t>({3, 1, 2}),
  });

  assertMarkSortedResults(
      {data}, {"c0"}, {core::kAscNullsLast}, "is_sorted", {true, false, true});
}
