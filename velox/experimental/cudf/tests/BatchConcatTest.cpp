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
#include "velox/experimental/cudf/CudfQueryConfig.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

class CudfBatchConcatTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // Enable cuDF debug logs to help trace operator adaptation and runtime
    // behavior during tests.
    facebook::velox::cudf_velox::CudfConfig::getInstance().debugEnabled = true;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void updateCudfConfig(int32_t min, std::optional<int32_t> max) {
    auto& config = CudfConfig::getInstance();
    config.batchSizeMinThreshold = min;
    config.batchSizeMaxThreshold = max;
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  // Forces fragmented input that Values won't coalesce
  core::PlanNodePtr createFragmentedPlan(
      const std::vector<RowVectorPtr>& vectors,
      std::shared_ptr<core::PlanNodeIdGenerator> generator) {
    std::vector<core::PlanNodePtr> sources;
    for (const auto& vec : vectors) {
      sources.push_back(PlanBuilder(generator).values({vec}).planNode());
    }
    return PlanBuilder(generator).localPartition({}, sources).planNode();
  }
};

TEST_F(CudfBatchConcatTest, fragmentedInputGetsConcatenated) {
  updateCudfConfig(/*min=*/30, /*max=*/20);

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodePtr> sources;
  for (const auto& vec : vectors) {
    sources.push_back(PlanBuilder(generator).values({vec}).planNode());
  }

  auto plan = PlanBuilder(generator)
                  .localPartitionRoundRobin(sources)
                  .limit(0, 100, false)
                  .filter("c0 >= 0") // forces FilterProject
                  .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  // Enable concat optimization for this query so CudfBatchConcat can be
  // inserted by the adapter.
  params.queryConfigs
      [cudf_velox::CudfQueryConfig::kCudfConcatOptimizationEnabled] = "true";

  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  // correctness
  ASSERT_EQ(batches.size(), 3);

  ASSERT_EQ(batches[0]->size(), 20);
  ASSERT_EQ(batches[1]->size(), 20);
  ASSERT_EQ(batches[2]->size(), 20);
}

// To test if the optimization is not performed after certain operators.
TEST_F(CudfBatchConcatTest, fragmentedInputDoesNotGetConcatenated) {
  updateCudfConfig(/*min=*/1e5, 1e7);

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodePtr> sources;
  for (const auto& vec : vectors) {
    sources.push_back(PlanBuilder(generator).values({vec}).planNode());
  }

  auto plan = PlanBuilder(generator)
                  .localPartitionRoundRobin(sources)
                  .filter("c0 >= 0") // forces FilterProject
                  .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.queryConfigs
      [cudf_velox::CudfQueryConfig::kCudfConcatOptimizationEnabled] = "true";

  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  // correctness
  ASSERT_EQ(batches.size(), 6);
}

TEST_F(CudfBatchConcatTest, concatDisabledWithQueryConfig) {
  updateCudfConfig(/*min=*/30, /*max=*/20);

  std::vector<RowVectorPtr> vectors;
  for (int i = 0; i < 6; ++i) {
    vectors.push_back(makeRowVector({makeFlatSequence<int64_t>(i * 10, 10)}));
  }

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodePtr> sources;
  for (const auto& vec : vectors) {
    sources.push_back(PlanBuilder(generator).values({vec}).planNode());
  }

  auto plan = PlanBuilder(generator)
                  .localPartitionRoundRobin(sources)
                  .limit(0, 100, false)
                  .filter("c0 >= 0") // forces FilterProject
                  .planNode();

  exec::CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.queryConfigs
      [cudf_velox::CudfQueryConfig::kCudfConcatOptimizationEnabled] = "false";

  auto [cursor, batches] = exec::test::readCursor(
      params, [](auto* c) { c->setNoMoreSplits(); }, 3000);

  // correctness
  ASSERT_EQ(batches.size(), 6);
}
