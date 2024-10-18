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

#include <boost/random/uniform_int_distribution.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>

#include "folly/experimental/EventCount.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/QueryDataReader.h"
#include "velox/exec/QueryTraceUtil.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/tool/trace/HashJoinReplayer.h"

#include <common/file/Utils.h>

#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::common;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::common::hll;

namespace facebook::velox::tool::trace::test {
class HashJoinReplayerTest : public HiveConnectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    HiveConnectorTestBase::SetUpTestCase();
    filesystems::registerLocalFileSystem();
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    Type::registerSerDe();
    common::Filter::registerSerDe();
    connector::hive::HiveTableHandle::registerSerDe();
    connector::hive::LocationHandle::registerSerDe();
    connector::hive::HiveColumnHandle::registerSerDe();
    connector::hive::HiveInsertTableHandle::registerSerDe();
    connector::hive::HiveConnectorSplit::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    registerPartitionFunctionSerDe();
  }

  struct PlanWithSplits {
    core::PlanNodePtr plan;
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    std::unordered_map<core::PlanNodeId, std::vector<exec::Split>> splits;

    explicit PlanWithSplits(
        const core::PlanNodePtr& _plan,
        const core::PlanNodeId& _probeScanId = "",
        const core::PlanNodeId& _buildScanId = "",
        const std::unordered_map<
            core::PlanNodeId,
            std::vector<velox::exec::Split>>& _splits = {})
        : plan(_plan),
          probeScanId(_probeScanId),
          buildScanId(_buildScanId),
          splits(_splits) {}
  };

  int32_t randInt(int32_t min, int32_t max) {
    return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
  }

  RowTypePtr concat(const RowTypePtr& a, const RowTypePtr& b) {
    std::vector<std::string> names = a->names();
    std::vector<TypePtr> types = a->children();

    for (auto i = 0; i < b->size(); ++i) {
      names.push_back(b->nameOf(i));
      types.push_back(b->childAt(i));
    }

    return ROW(std::move(names), std::move(types));
  }

  std::vector<RowVectorPtr>
  makeVectors(int32_t count, int32_t rowsPerVector, const RowTypePtr& rowType) {
    return HiveConnectorTestBase::makeVectors(rowType, count, rowsPerVector);
  }

  std::vector<Split> makeSplits(
      const std::vector<RowVectorPtr>& inputs,
      const std::string& path,
      memory::MemoryPool* writerPool) {
    std::vector<Split> splits;
    for (auto i = 0; i < 4; ++i) {
      const std::string filePath = fmt::format("{}/{}", path, i);
      writeToFile(filePath, inputs);
      splits.emplace_back(makeHiveConnectorSplit(filePath));
    }

    return splits;
  }

  PlanWithSplits createPlan(
      const std::string& tableDir,
      core::JoinType joinType,
      const std::vector<std::string>& probeKeys,
      const std::vector<std::string>& buildKeys,
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<RowVectorPtr>& buildInput,
      const std::vector<std::string>& outputColumns) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const std::vector<Split> probeSplits =
        makeSplits(probeInput, fmt::format("{}/probe", tableDir), pool());
    const std::vector<Split> buildSplits =
        makeSplits(buildInput, fmt::format("{}/build", tableDir), pool());
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .tableScan(probeType_)
                    .capturePlanNodeId(probeScanId)
                    .hashJoin(
                        probeKeys,
                        buildKeys,
                        PlanBuilder(planNodeIdGenerator)
                            .tableScan(buildType_)
                            .capturePlanNodeId(buildScanId)
                            .planNode(),
                        /*filter=*/"",
                        outputColumns,
                        joinType,
                        false)
                    .capturePlanNodeId(traceNodeId_)
                    .planNode();
    return PlanWithSplits{
        plan,
        probeScanId,
        buildScanId,
        {{probeScanId, probeSplits}, {buildScanId, buildSplits}}};
  }

  core::PlanNodeId traceNodeId_;
  std::mt19937 rng_;
  const RowTypePtr probeType_{
      ROW({"t0", "t1", "t2", "t3"}, {BIGINT(), INTEGER(), SMALLINT(), REAL()})};

  const RowTypePtr buildType_{
      ROW({"u0", "u1", "u2", "u3"}, {BIGINT(), INTEGER(), SMALLINT(), REAL()})};
};

TEST_F(HashJoinReplayerTest, test) {
  static const std::vector<core::JoinType> kJoinTypes = {
      core::JoinType::kInner,
      core::JoinType::kLeft,
      core::JoinType::kRight,
      core::JoinType::kFull,
      core::JoinType::kLeftSemiFilter,
      core::JoinType::kLeftSemiProject,
      core::JoinType::kAnti};
  const auto testDir = TempDirectoryPath::create();
  const auto tableDir = fmt::format("{}/{}", testDir->getPath(), "table");
  std::vector<std::string> probeKeys{"t0", "t1"};
  std::vector<std::string> buildKeys{"u0", "u1"};
  const auto probeInput = makeVectors(randInt(2, 5), 100, probeType_);
  const auto buildInput = makeVectors(randInt(2, 5), 100, buildType_);

  for (const auto joinType : kJoinTypes) {
    auto outputColumns =
        (core::isLeftSemiProjectJoin(joinType) ||
         core::isLeftSemiFilterJoin(joinType) || core::isAntiJoin(joinType))
        ? asRowType(probeInput[0]->type())->names()
        : concat(
              asRowType(probeInput[0]->type()),
              asRowType(buildInput[0]->type()))
              ->names();

    if (core::isLeftSemiProjectJoin(joinType) ||
        core::isRightSemiProjectJoin(joinType)) {
      outputColumns.emplace_back("match");
    }

    const auto planWithSplits = createPlan(
        tableDir,
        joinType,
        probeKeys,
        buildKeys,
        probeInput,
        buildInput,
        outputColumns);
    AssertQueryBuilder builder(planWithSplits.plan);
    for (const auto& [planNodeId, nodeSplits] : planWithSplits.splits) {
      builder.splits(planNodeId, nodeSplits);
    }
    const auto result = builder.copyResults(pool());

    const auto traceRoot =
        fmt::format("{}/{}/traceRoot/", testDir->getPath(), joinType);
    std::shared_ptr<Task> task;
    auto tracePlanWithSplits = createPlan(
        tableDir,
        joinType,
        probeKeys,
        buildKeys,
        probeInput,
        buildInput,
        outputColumns);
    AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
    traceBuilder.maxDrivers(4)
        .config(core::QueryConfig::kQueryTraceEnabled, true)
        .config(core::QueryConfig::kQueryTraceDir, traceRoot)
        .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
        .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
        .config(core::QueryConfig::kQueryTraceNodeIds, traceNodeId_);
    for (const auto& [planNodeId, nodeSplits] : tracePlanWithSplits.splits) {
      traceBuilder.splits(planNodeId, nodeSplits);
    }
    auto traceResult = traceBuilder.copyResults(pool(), task);

    assertEqualResults({result}, {traceResult});

    const auto taskId = task->taskId();
    const auto replayingResult =
        HashJoinReplayer(traceRoot, task->taskId(), traceNodeId_, 0, "HashJoin")
            .run();
    assertEqualResults({result}, {replayingResult});
  }
}
} // namespace facebook::velox::tool::trace::test
