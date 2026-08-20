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

#include "velox/experimental/cudf/exec/GpuMemoryPlanPath.h"

#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox::test {

class GpuMemoryPlanPathTest : public OperatorTestBase {
 protected:
  /// Returns the plan node identifiers along a resolved chain, which is the
  /// order the counter hierarchy nests by.
  static std::vector<std::string> ids(const GpuMemoryPlanLocation& location) {
    std::vector<std::string> result;
    result.reserve(location.path.size());
    for (const auto& entry : location.path) {
      result.push_back(entry.planNodeId);
    }
    return result;
  }

  RowVectorPtr makeInput() {
    return makeRowVector(
        {makeFlatVector<int64_t>(10, [](auto row) { return row; })});
  }
};

TEST_F(GpuMemoryPlanPathTest, resolvesEachNodeToItsChainAndDisplayType) {
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId valuesId;
  core::PlanNodeId filterId;
  const auto plan = PlanBuilder(idGenerator)
                        .values({makeInput()})
                        .capturePlanNodeId(valuesId)
                        .filter("c0 > 0")
                        .capturePlanNodeId(filterId)
                        .planNode();

  EXPECT_EQ(
      ids(gpuMemoryPlanLocationFromRoot(plan.get(), filterId)),
      (std::vector<std::string>{filterId}));

  const auto values = gpuMemoryPlanLocationFromRoot(plan.get(), valuesId);
  EXPECT_EQ(ids(values), (std::vector<std::string>{filterId, valuesId}));
  // The display type is what the counter label carries.
  EXPECT_EQ(values.node()->planNodeType, "ValuesNode");
}

/// A node with several sources is where the operator-instance graph stops being
/// a tree, so both branches must resolve to chains that share only the common
/// prefix. This is what lets fan-in become sibling leaves in the counter tree.
TEST_F(GpuMemoryPlanPathTest, multipleSourcesResolveThroughTheirOwnBranch) {
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId leftId;
  core::PlanNodeId rightId;
  core::PlanNodeId partitionId;

  auto left = PlanBuilder(idGenerator)
                  .values({makeInput()})
                  .capturePlanNodeId(leftId)
                  .planNode();
  auto right = PlanBuilder(idGenerator)
                   .values({makeInput()})
                   .capturePlanNodeId(rightId)
                   .planNode();
  const auto plan = PlanBuilder(idGenerator)
                        .localPartition({}, {left, right})
                        .capturePlanNodeId(partitionId)
                        .planNode();

  EXPECT_EQ(
      ids(gpuMemoryPlanLocationFromRoot(plan.get(), leftId)),
      (std::vector<std::string>{partitionId, leftId}));
  EXPECT_EQ(
      ids(gpuMemoryPlanLocationFromRoot(plan.get(), rightId)),
      (std::vector<std::string>{partitionId, rightId}));

  // Sources are walked left to right, and both sit one level below the parent.
  EXPECT_EQ(gpuMemoryPlanLocationFromRoot(plan.get(), partitionId).order, 0);
  EXPECT_EQ(gpuMemoryPlanLocationFromRoot(plan.get(), leftId).order, 1);
  EXPECT_EQ(gpuMemoryPlanLocationFromRoot(plan.get(), rightId).order, 2);
  EXPECT_EQ(gpuMemoryPlanLocationFromRoot(plan.get(), leftId).depth(), 1);
  EXPECT_EQ(gpuMemoryPlanLocationFromRoot(plan.get(), rightId).depth(), 1);
}

TEST_F(GpuMemoryPlanPathTest, conversionSuffixesResolveThroughSourceNode) {
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId valuesId;
  const auto plan = PlanBuilder(idGenerator)
                        .values({makeInput()})
                        .capturePlanNodeId(valuesId)
                        .planNode();

  for (const std::string suffix : {"-from-velox", "-to-velox"}) {
    const auto location =
        gpuMemoryPlanLocationFromRoot(plan.get(), valuesId + suffix);
    ASSERT_EQ(location.path.size(), 1) << "suffix " << suffix;
    EXPECT_EQ(location.node()->planNodeId, valuesId) << "suffix " << suffix;
  }
}

TEST_F(GpuMemoryPlanPathTest, unresolvableIdentifiersReportUnmapped) {
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  const auto plan = PlanBuilder(idGenerator).values({makeInput()}).planNode();

  // An unknown identifier, an unknown identifier that still carries a
  // conversion suffix, an empty identifier, and an absent plan fragment all
  // report an empty chain rather than guessing a place in the hierarchy.
  for (const auto* id : {"no-such-node", "no-such-node-to-velox", ""}) {
    const auto location = gpuMemoryPlanLocationFromRoot(plan.get(), id);
    EXPECT_TRUE(location.path.empty()) << "id " << id;
    EXPECT_EQ(location.node(), nullptr) << "id " << id;
    // Not given a plausible-looking position either.
    EXPECT_EQ(location.order, -1) << "id " << id;
  }
  EXPECT_TRUE(gpuMemoryPlanLocationFromRoot(nullptr, "0").path.empty());
}

/// The plan node label is sorted lexically by Nsight, so the pre-order index
/// and depth are what reproduce EXPLAIN order and nesting in a flat row list.
TEST_F(GpuMemoryPlanPathTest, reportsPreOrderIndexAndDepth) {
  auto idGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId valuesId;
  core::PlanNodeId filterId;
  core::PlanNodeId projectId;
  const auto plan = PlanBuilder(idGenerator)
                        .values({makeInput()})
                        .capturePlanNodeId(valuesId)
                        .filter("c0 > 0")
                        .capturePlanNodeId(filterId)
                        .project({"c0"})
                        .capturePlanNodeId(projectId)
                        .planNode();

  // Pre-order from the fragment root: project, filter, values.
  const auto project = gpuMemoryPlanLocationFromRoot(plan.get(), projectId);
  EXPECT_EQ(project.order, 0);
  EXPECT_EQ(project.depth(), 0);

  const auto filter = gpuMemoryPlanLocationFromRoot(plan.get(), filterId);
  EXPECT_EQ(filter.order, 1);
  EXPECT_EQ(filter.depth(), 1);

  const auto values = gpuMemoryPlanLocationFromRoot(plan.get(), valuesId);
  EXPECT_EQ(values.order, 2);
  EXPECT_EQ(values.depth(), 2);
}

} // namespace facebook::velox::cudf_velox::test
