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

#include "velox/exec/tests/utils/DistributedPlanBuilder.h"
#include "velox/exec/tests/utils/LocalRunnerTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class LocalRunnerTest : public LocalRunnerTestBase {};

void makeAscending(const RowVectorPtr& rows, int32_t& counter) {
  auto ints = rows->childAt(0)->as<FlatVector<int64_t>>();
  for (auto i = 0; i < ints->size(); ++i) {
    ints->set(i, counter + i);
  }
  counter += ints->size();
}

TEST_F(LocalRunnerTest, count) {
  constexpr int32_t kNumFiles = 5;
  constexpr int32_t kNumVectors = 5;
  constexpr int32_t kRowsPerVector = 10000;
  constexpr int32_t kNumRows = kNumFiles * kNumVectors * kRowsPerVector;

  auto rowType = ROW({"c0"}, {BIGINT()});
  int32_t counter1 = 0;
  auto patch1 = [&](const RowVectorPtr& rows) {
    makeAscending(rows, counter1);
  };

  int32_t counter2 = 0;
  auto patch2 = [&](const RowVectorPtr& rows) {
    makeAscending(rows, counter2);
  };

  std::vector<TableSpec> specs = {
      TableSpec{
          .name = "T",
          .columns = rowType,
          .rowsPerVector = kRowsPerVector,
          .numVectorsPerFile = kNumVectors,
          .numFiles = kNumFiles,
          .patch = patch1},
      TableSpec{
          .name = "U",
          .columns = rowType,
          .rowsPerVector = kRowsPerVector,
          .numVectorsPerFile = kNumVectors,
          .numFiles = kNumFiles,
          .patch = patch2}};

  std::shared_ptr<TempDirectoryPath> files;
  auto schema = makeTables(specs, files);

  ExecutablePlanOptions options = {
      .queryId = "test.", .numWorkers = 4, .numDrivers = 2};
  const int32_t width = 3;
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  DistributedPlanBuilder rootBuilder(options, ids, pool_.get());
  rootBuilder.tableScan("T", rowType)
      .shuffle({"c0"}, 3, false)
      .hashJoin(
          {"c0"},
          {"b0"},
          DistributedPlanBuilder(rootBuilder)
              .tableScan("U", rowType)
              .project({"c0 as b0"})
              .shuffleResult({"b0"}, width, false),
          "",
          {"c0", "b0"})
      .shuffle({}, 1, false)
      .finalAggregation({}, {"count(1)"}, {{BIGINT()}});
  auto stages = rootBuilder.fragments();

  auto sourceFactory = std::make_shared<LocalSplitSourceFactory>(schema, 2);
  auto localRunner = std::make_shared<LocalRunner>(
      std::move(stages), makeQueryCtx("q1"), sourceFactory, options);
  auto results = readCursor(localRunner);
  auto stats = localRunner->stats();
  EXPECT_EQ(1, results.size());
  EXPECT_EQ(1, results[0]->size());
  EXPECT_EQ(
      kNumRows, results[0]->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0));
  results.clear();
  LocalRunner::waitForAllDeleted(std::move(localRunner), 5000);
}
