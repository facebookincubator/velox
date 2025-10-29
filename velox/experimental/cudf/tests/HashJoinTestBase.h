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

#pragma once

#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HashJoinTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/VectorTestUtil.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <fmt/format.h>
#include <re2/re2.h>

namespace facebook::velox::cudf_velox::exec::test {

using facebook::velox::exec::Split;
using facebook::velox::exec::test::HashJoinBuilder;
using facebook::velox::exec::test::HashJoinTestBase;
using SplitInput = std::unordered_map<core::PlanNodeId, std::vector<Split>>;
using facebook::velox::exec::test::makeBatches;
using facebook::velox::exec::test::PlanBuilder;
using facebook::velox::exec::test::TempFilePath;
using facebook::velox::exec::test::TestParam;

class CudfHashJoinTestBase : public CudfHiveConnectorTestBase,
                             public HashJoinTestBase {
 protected:
  CudfHashJoinTestBase() : HashJoinTestBase(TestParam(1)) {}

  CudfHashJoinTestBase(const TestParam& param) : HashJoinTestBase(param) {}

  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    // Initialize the probeType_, buildType_, and fuzzerOpts_ that would
    // normally be initialized by HashJoinTestBase::SetUp(). We can't call
    // HashJoinTestBase::SetUp() directly because it would cause
    // OperatorTestBase::SetUp() to be invoked twice (once through
    // CudfHiveConnectorTestBase and once through HiveConnectorTestBase).
    HashJoinTestBase::MemberSetUp();
  }

  std::vector<Split> makeSplit(
      const std::vector<std::shared_ptr<TempFilePath>>& files) override {
    std::vector<Split> splits;
    splits.reserve(files.size());
    for (const auto& file : files) {
      splits.push_back(Split(makeCudfHiveConnectorSplit(file->getPath())));
    }
    return splits;
  }

  void testLazyVectorsWithFilter(
      const core::JoinType joinType,
      const std::string& filter,
      const std::vector<std::string>& outputLayout,
      const std::string& referenceQuery) override {
    const vector_size_t vectorSize = 1'000;
    auto probeVectors = makeBatches(1, [&](int32_t /*unused*/) {
      return makeRowVector(
          {makeFlatVector<int32_t>(vectorSize, folly::identity),
           makeFlatVector<int64_t>(
               vectorSize, [](auto row) { return row % 23; }),
           makeFlatVector<int32_t>(
               vectorSize, [](auto row) { return row % 31; })});
    });

    std::vector<RowVectorPtr> buildVectors =
        makeBatches(1, [&](int32_t /*unused*/) {
          return makeRowVector({makeFlatVector<int32_t>(
              vectorSize, [](auto row) { return row * 3; })});
        });

    std::shared_ptr<TempFilePath> probeFile = TempFilePath::create();
    CudfHiveConnectorTestBase::writeToFile(probeFile->getPath(), probeVectors);

    std::shared_ptr<TempFilePath> buildFile = TempFilePath::create();
    CudfHiveConnectorTestBase::writeToFile(buildFile->getPath(), buildVectors);

    createDuckDbTable("t", probeVectors);
    createDuckDbTable("u", buildVectors);

    // Lazy vector is part of the filter but never gets loaded.
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    auto op =
        PlanBuilder(planNodeIdGenerator)
            .startTableScan()
            .outputType(asRowType(probeVectors[0]->type()))
            .tableHandle(CudfHiveConnectorTestBase::makeTableHandle())
            .endTableScan()
            .capturePlanNodeId(probeScanId)
            .hashJoin(
                {"c0"},
                {"c0"},
                PlanBuilder(planNodeIdGenerator)
                    .startTableScan()
                    .outputType(asRowType(buildVectors[0]->type()))
                    .tableHandle(CudfHiveConnectorTestBase::makeTableHandle())
                    .endTableScan()
                    .capturePlanNodeId(buildScanId)
                    .planNode(),
                filter,
                outputLayout,
                joinType)
            .planNode();
    SplitInput splitInput = {
        {probeScanId,
         {Split(makeCudfHiveConnectorSplit(probeFile->getPath()))}},
        {buildScanId,
         {Split(makeCudfHiveConnectorSplit(buildFile->getPath()))}},
    };
    HashJoinBuilder(*pool_, duckDbQueryRunner_, driverExecutor_.get())
        .planNode(std::move(op))
        .inputSplits(splitInput)
        .checkSpillStats(false)
        .referenceQuery(referenceQuery)
        .run();
  }

  void TearDown() override {
    CudfHiveConnectorTestBase::TearDown();
  }
};

} // namespace facebook::velox::cudf_velox::exec::test
