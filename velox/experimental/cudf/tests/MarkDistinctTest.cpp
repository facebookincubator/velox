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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CudfMarkDistinctTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }
};

// Test 1: Single batch, all distinct keys -> all markers true
TEST_F(CudfMarkDistinctTest, allDistinct) {
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });

  auto expectedMarkers = makeFlatVector<bool>({true, true, true, true, true});
  auto expected = makeRowVector({input->childAt(0), expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 2: Single batch, duplicate keys -> first occurrence true, rest false
TEST_F(CudfMarkDistinctTest, duplicateKeys) {
  // key: 1, 2, 1, 2, 3 -> first occurrences at indices 0, 1, 4
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 1, 2, 3}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
  });

  auto expectedMarkers = makeFlatVector<bool>({true, true, false, false, true});
  auto expected =
      makeRowVector({input->childAt(0), input->childAt(1), expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 3: Two batches - key seen in batch 1 reappears in batch 2 -> false in
// batch 2
TEST_F(CudfMarkDistinctTest, multiBatch) {
  auto batch1 = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});
  auto batch2 = makeRowVector({makeFlatVector<int32_t>({3, 4, 1})});

  // batch1: all first occurrences -> [T, T, T]
  // batch2: 3 seen -> F, 4 new -> T, 1 seen -> F

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .markDistinct("m", {"c0"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // Result should have 6 rows total; check markers
  ASSERT_EQ(6, result->size());
  auto markers = result->childAt(1)->asFlatVector<bool>();
  ASSERT_NE(nullptr, markers);
  EXPECT_TRUE(markers->valueAt(0)); // 1: first
  EXPECT_TRUE(markers->valueAt(1)); // 2: first
  EXPECT_TRUE(markers->valueAt(2)); // 3: first
  EXPECT_FALSE(markers->valueAt(3)); // 3: duplicate
  EXPECT_TRUE(markers->valueAt(4)); // 4: first
  EXPECT_FALSE(markers->valueAt(5)); // 1: duplicate
}

// Test 4: BIGINT key column (matches Q16/Q28 usage)
TEST_F(CudfMarkDistinctTest, bigintKey) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({100L, 200L, 100L, 300L, 200L}),
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });

  auto expectedMarkers = makeFlatVector<bool>({true, true, false, true, false});
  auto expected =
      makeRowVector({input->childAt(0), input->childAt(1), expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 5: Composite key (two columns)
TEST_F(CudfMarkDistinctTest, compositeKey) {
  // (c0, c1) composite key: (1,10), (1,20), (2,10), (2,20), (1,10)
  // First occurrences: rows 0,1,2,3 -> T; row 4 -> F (duplicate of row 0)
  auto input = makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 1}),
      makeFlatVector<int32_t>({10, 20, 10, 20, 10}),
      makeFlatVector<int32_t>({100, 200, 300, 400, 500}),
  });

  auto expectedMarkers = makeFlatVector<bool>({true, true, true, true, false});
  auto expected = makeRowVector(
      {input->childAt(0),
       input->childAt(1),
       input->childAt(2),
       expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0", "c1"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 6: Compare GPU output with CPU MarkDistinct via assertQuery (DuckDB)
TEST_F(CudfMarkDistinctTest, allDistinctMatchesDuckDb) {
  // All distinct - DuckDB SQL can express this simply
  auto input = makeRowVector({
      makeFlatVector<int32_t>({10, 20, 30, 40}),
      makeFlatVector<double>({1.1, 2.2, 3.3, 4.4}),
  });

  createDuckDbTable({input});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  // All keys are distinct, so all markers are true
  assertQuery(plan, "SELECT c0, c1, true AS m FROM tmp");
}

// Test 7: Null handling in keys - nulls are treated as equal
TEST_F(CudfMarkDistinctTest, nullKeys) {
  auto input = makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 2, std::nullopt, 1}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
  });

  // First occurrences: 1 at row 0, null at row 1, 2 at row 2
  // Duplicates: null at row 3, 1 at row 4
  auto expectedMarkers = makeFlatVector<bool>({true, true, true, false, false});
  auto expected =
      makeRowVector({input->childAt(0), input->childAt(1), expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 8: VARCHAR/string keys
TEST_F(CudfMarkDistinctTest, stringKeys) {
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          {"apple", "banana", "apple", "cherry", "banana"}),
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
  });

  auto expectedMarkers = makeFlatVector<bool>({true, true, false, true, false});
  auto expected =
      makeRowVector({input->childAt(0), input->childAt(1), expectedMarkers});

  auto plan =
      PlanBuilder().values({input}).markDistinct("m", {"c0"}).planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  assertEqualVectors(expected, result);
}

// Test 9: Three or more batches to verify state accumulation
TEST_F(CudfMarkDistinctTest, threeBatches) {
  auto batch1 = makeRowVector({makeFlatVector<int32_t>({1, 2})});
  auto batch2 = makeRowVector({makeFlatVector<int32_t>({3, 1})});
  auto batch3 = makeRowVector({makeFlatVector<int32_t>({2, 4, 3})});

  // batch1: 1 -> T, 2 -> T
  // batch2: 3 -> T, 1 -> F (seen in batch1)
  // batch3: 2 -> F (seen in batch1), 4 -> T, 3 -> F (seen in batch2)

  auto plan = PlanBuilder()
                  .values({batch1, batch2, batch3})
                  .markDistinct("m", {"c0"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(7, result->size());
  auto markers = result->childAt(1)->asFlatVector<bool>();
  EXPECT_TRUE(markers->valueAt(0)); // 1: first
  EXPECT_TRUE(markers->valueAt(1)); // 2: first
  EXPECT_TRUE(markers->valueAt(2)); // 3: first
  EXPECT_FALSE(markers->valueAt(3)); // 1: duplicate
  EXPECT_FALSE(markers->valueAt(4)); // 2: duplicate
  EXPECT_TRUE(markers->valueAt(5)); // 4: first
  EXPECT_FALSE(markers->valueAt(6)); // 3: duplicate
}

// Test 10: All duplicates - every row in batch 2 is a duplicate
TEST_F(CudfMarkDistinctTest, allDuplicates) {
  auto batch1 = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});
  auto batch2 = makeRowVector({makeFlatVector<int32_t>({2, 1, 3, 1})});

  // batch1: all first -> [T, T, T]
  // batch2: all duplicates -> [F, F, F, F]

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .markDistinct("m", {"c0"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(7, result->size());
  auto markers = result->childAt(1)->asFlatVector<bool>();
  EXPECT_TRUE(markers->valueAt(0)); // 1: first
  EXPECT_TRUE(markers->valueAt(1)); // 2: first
  EXPECT_TRUE(markers->valueAt(2)); // 3: first
  EXPECT_FALSE(markers->valueAt(3)); // 2: duplicate
  EXPECT_FALSE(markers->valueAt(4)); // 1: duplicate
  EXPECT_FALSE(markers->valueAt(5)); // 3: duplicate
  EXPECT_FALSE(markers->valueAt(6)); // 1: duplicate
}

// Test: Empty batch handling
TEST_F(CudfMarkDistinctTest, emptyBatch) {
  // First batch has data, second batch is empty, third has data
  auto batch1 = makeRowVector({makeFlatVector<int32_t>({1, 2})});
  auto emptyBatch = makeRowVector({makeFlatVector<int32_t>({})});
  auto batch3 = makeRowVector({makeFlatVector<int32_t>({2, 3})});

  auto plan = PlanBuilder()
                  .values({batch1, emptyBatch, batch3})
                  .markDistinct("m", {"c0"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // Result should have 4 rows (2 + 0 + 2)
  ASSERT_EQ(4, result->size());
  auto markers = result->childAt(1)->asFlatVector<bool>();
  EXPECT_TRUE(markers->valueAt(0)); // 1: first
  EXPECT_TRUE(markers->valueAt(1)); // 2: first
  // Empty batch contributes nothing
  EXPECT_FALSE(markers->valueAt(2)); // 2: duplicate
  EXPECT_TRUE(markers->valueAt(3)); // 3: first
}
