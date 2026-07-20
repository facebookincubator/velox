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
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CudfEnforceSingleRowTest : public HiveConnectorTestBase {
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

TEST_F(CudfEnforceSingleRowTest, singleRow) {
  // Single row of input - should pass
  auto singleRow =
      makeRowVector({makeFlatVector<int32_t>(1, [](auto row) { return row; })});

  createDuckDbTable({singleRow});

  auto plan = PlanBuilder().values({singleRow}).enforceSingleRow().planNode();
  assertQuery(plan, "SELECT * FROM tmp");
}

TEST_F(CudfEnforceSingleRowTest, twoSingleRowBatches) {
  // Two rows of input in two separate single-row batches - should fail
  // Note: The test framework may concatenate batches, so error may show total
  // rows
  auto singleRow =
      makeRowVector({makeFlatVector<int32_t>(1, [](auto row) { return row; })});

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(
          PlanBuilder()
              .values({singleRow, singleRow})
              .enforceSingleRow()
              .planNode())
          .copyResults(pool()),
      "Expected single row of input");
}

TEST_F(CudfEnforceSingleRowTest, multipleRows) {
  // Multiple rows of input in a single batch - should fail
  auto multipleRows = makeRowVector(
      {makeFlatVector<int32_t>(27, [](auto row) { return row; })});

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(
          PlanBuilder().values({multipleRows}).enforceSingleRow().planNode())
          .copyResults(pool()),
      "Expected single row of input. Received 27 rows.");
}

TEST_F(CudfEnforceSingleRowTest, emptyInput) {
  // Empty input - should pass and return a single row of nulls
  auto singleRow =
      makeRowVector({makeFlatVector<int32_t>(1, [](auto row) { return row; })});

  auto plan = PlanBuilder()
                  .values({singleRow})
                  .filter("c0 = 12345")
                  .enforceSingleRow()
                  .planNode();
  assertQuery(plan, "SELECT null");
}

TEST_F(CudfEnforceSingleRowTest, multipleColumns) {
  // Test with multiple columns of different types
  auto singleRow = makeRowVector({
      makeFlatVector<int32_t>(1, [](auto /*row*/) { return 42; }),
      makeFlatVector<int64_t>(1, [](auto /*row*/) { return 100L; }),
      makeFlatVector<double>(1, [](auto /*row*/) { return 3.14; }),
      makeFlatVector<std::string>(1, [](auto /*row*/) { return "test"; }),
  });

  createDuckDbTable({singleRow});

  auto plan = PlanBuilder().values({singleRow}).enforceSingleRow().planNode();
  assertQuery(plan, "SELECT * FROM tmp");
}

TEST_F(CudfEnforceSingleRowTest, withNulls) {
  // Test with null values
  auto singleRow = makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt}),
      makeNullableFlatVector<std::string>({std::nullopt}),
  });

  createDuckDbTable({singleRow});

  auto plan = PlanBuilder().values({singleRow}).enforceSingleRow().planNode();
  assertQuery(plan, "SELECT * FROM tmp");
}

TEST_F(CudfEnforceSingleRowTest, largeBatch) {
  // Large batch (more than 1 row) - should fail
  auto largeRows = makeRowVector(
      {makeFlatVector<int32_t>(1000, [](auto row) { return row; })});

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(
          PlanBuilder().values({largeRows}).enforceSingleRow().planNode())
          .copyResults(pool()),
      "Expected single row of input. Received 1000 rows.");
}
