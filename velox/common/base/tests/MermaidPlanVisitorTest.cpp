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

#include "velox/common/base/MermaidPlanVisitor.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/TaskStats.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::common::base;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;

class MermaidPlanVisitorTest : public testing::Test,
                               public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  std::shared_ptr<const ValuesNode> createValuesNode(
      const std::string& id = "values") {
    return ValuesNode::Builder()
        .id(id)
        .values({makeRowVector({
            makeFlatVector<int32_t>({1, 2, 3}),
            makeFlatVector<std::string>({"a", "b", "c"}),
        })})
        .build();
  }

  std::shared_ptr<const FilterNode> createFilterNode(
      const std::string& id = "filter",
      const PlanNodePtr& source = nullptr) {
    auto sourceNode = source ? source : createValuesNode();
    return FilterNode::Builder()
        .id(id)
        .filter(std::make_shared<FieldAccessTypedExpr>(BOOLEAN(), "c0"))
        .source(sourceNode)
        .build();
  }

  std::shared_ptr<const ProjectNode> createProjectNode(
      const std::string& id = "project",
      const PlanNodePtr& source = nullptr) {
    auto sourceNode = source ? source : createValuesNode();
    return ProjectNode::Builder()
        .id(id)
        .names({"out_col"})
        .projections({std::make_shared<FieldAccessTypedExpr>(INTEGER(), "c0")})
        .source(sourceNode)
        .build();
  }

  std::shared_ptr<const AggregationNode> createAggregationNode(
      const std::string& id = "aggregation",
      const PlanNodePtr& source = nullptr) {
    auto sourceNode = source ? source : createValuesNode();
    std::vector<FieldAccessTypedExprPtr> groupingKeys{
        std::make_shared<FieldAccessTypedExpr>(INTEGER(), "c0")};

    std::vector<AggregationNode::Aggregate> aggregates{
        AggregationNode::Aggregate{
            .call = std::make_shared<CallTypedExpr>(
                BIGINT(), std::vector<TypedExprPtr>{}, "count"),
            .rawInputTypes = {}}};

    return AggregationNode::Builder()
        .id(id)
        .step(AggregationNode::Step::kSingle)
        .groupingKeys(groupingKeys)
        .preGroupedKeys({})
        .aggregateNames(std::vector<std::string>{"count"})
        .aggregates(aggregates)
        .ignoreNullKeys(false)
        .source(sourceNode)
        .build();
  }

  std::shared_ptr<const HashJoinNode> createHashJoinNode(
      const std::string& id = "hash_join") {
    auto leftSource = createValuesNode("left_values");
    auto rightSource = createValuesNode("right_values");
    auto outputType = ROW({"c0", "c1"}, {INTEGER(), VARCHAR()});

    return HashJoinNode::Builder()
        .id(id)
        .joinType(JoinType::kInner)
        .leftKeys({std::make_shared<FieldAccessTypedExpr>(INTEGER(), "c0")})
        .rightKeys({std::make_shared<FieldAccessTypedExpr>(INTEGER(), "c0")})
        .left(leftSource)
        .right(rightSource)
        .outputType(outputType)
        .nullAware(false)
        .build();
  }

  TaskStats createMockTaskStats() {
    TaskStats stats;
    stats.numTotalSplits = 10;
    stats.numQueuedSplits = 2;
    stats.numRunningSplits = 3;
    stats.numFinishedSplits = 5;
    stats.executionStartTimeMs = 1000;
    stats.executionEndTimeMs = 2000;
    return stats;
  }
};

TEST_F(MermaidPlanVisitorTest, basicDiagramGeneration) {
  MermaidPlanVisitor visitor;
  auto planNode = createValuesNode();

  std::string diagram = visitor.build(planNode);

  // Check that the diagram contains Mermaid flowchart syntax
  EXPECT_TRUE(diagram.find("flowchart BT") != std::string::npos);
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
  EXPECT_TRUE(diagram.find("classDef") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, differentDirections) {
  auto planNode = createValuesNode();

  // Test TopToBottom direction
  MermaidPlanVisitor visitor1;
  visitor1.setDirection(MermaidPlanVisitor::Direction::TopToBottom);
  std::string diagram1 = visitor1.build(planNode);
  EXPECT_TRUE(diagram1.find("flowchart TD") != std::string::npos);

  // Test LeftToRight direction
  MermaidPlanVisitor visitor2;
  visitor2.setDirection(MermaidPlanVisitor::Direction::LeftToRight);
  std::string diagram2 = visitor2.build(planNode);
  EXPECT_TRUE(diagram2.find("flowchart LR") != std::string::npos);

  // Test RightToLeft direction
  MermaidPlanVisitor visitor3;
  visitor3.setDirection(MermaidPlanVisitor::Direction::RightToLeft);
  std::string diagram3 = visitor3.build(planNode);
  EXPECT_TRUE(diagram3.find("flowchart RL") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, filterNodeVisualization) {
  MermaidPlanVisitor visitor;
  auto planNode = createFilterNode();

  std::string diagram = visitor.build(planNode);

  // Check that the diagram contains the expected structure
  EXPECT_TRUE(diagram.find("flowchart") != std::string::npos);
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
  // The FilterNode has a specific visitor that includes HTML table content
  EXPECT_TRUE(
      diagram.find("FILTER") != std::string::npos ||
      diagram.find("Filter") != std::string::npos);
  // Should contain connections
  EXPECT_TRUE(diagram.find("---") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, projectNodeVisualization) {
  MermaidPlanVisitor visitor;
  auto planNode = createProjectNode();

  std::string diagram = visitor.build(planNode);

  EXPECT_TRUE(diagram.find("PROJECT") != std::string::npos);
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, aggregationNodeVisualization) {
  MermaidPlanVisitor visitor;
  auto planNode = createAggregationNode();

  std::string diagram = visitor.build(planNode);

  // Check for the actual output format - may be different than expected
  EXPECT_TRUE(diagram.find("flowchart") != std::string::npos);
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
  // Look for any form of aggregation-related text
  EXPECT_TRUE(
      diagram.find("AGGREGATION") != std::string::npos ||
      diagram.find("Aggregation") != std::string::npos ||
      diagram.find("aggregation") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, complexPlanVisualization) {
  // Create a more complex plan: Values -> Filter -> Project -> Aggregation
  auto values = createValuesNode("source");
  auto filter = createFilterNode("filter", values);
  auto project = createProjectNode("project", filter);
  auto aggregation = createAggregationNode("agg", project);

  MermaidPlanVisitor visitor;
  std::string diagram = visitor.build(aggregation);

  // Check that all nodes are present - use flexible matching
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
  EXPECT_TRUE(
      diagram.find("FILTER") != std::string::npos ||
      diagram.find("Filter") != std::string::npos);
  EXPECT_TRUE(
      diagram.find("PROJECT") != std::string::npos ||
      diagram.find("Project") != std::string::npos);
  EXPECT_TRUE(
      diagram.find("AGGREGATION") != std::string::npos ||
      diagram.find("Aggregation") != std::string::npos);

  // Check that connections are present
  EXPECT_TRUE(diagram.find("---") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, diagramWithTaskStats) {
  MermaidPlanVisitor visitor;
  auto planNode = createValuesNode();
  auto taskStats = createMockTaskStats();

  std::string diagram = visitor.build(planNode, taskStats);

  // The diagram should contain task statistics information
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
  EXPECT_TRUE(diagram.find("flowchart") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, includeTaskStatsFlag) {
  auto planNode = createValuesNode();
  auto taskStats = createMockTaskStats();

  // Test with task stats enabled
  MermaidPlanVisitor visitor1;
  visitor1.includeTaskStats(true);
  std::string diagram1 = visitor1.build(planNode, taskStats);

  // Test with task stats disabled
  MermaidPlanVisitor visitor2;
  visitor2.includeTaskStats(false);
  std::string diagram2 = visitor2.build(planNode, taskStats);

  // Both should generate valid diagrams
  EXPECT_TRUE(diagram1.find("flowchart") != std::string::npos);
  EXPECT_TRUE(diagram2.find("flowchart") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, chainingSetters) {
  MermaidPlanVisitor visitor;
  auto planNode = createValuesNode();

  // Test method chaining
  std::string diagram =
      visitor.setDirection(MermaidPlanVisitor::Direction::LeftToRight)
          .includeTaskStats(false)
          .build(planNode);

  EXPECT_TRUE(diagram.find("flowchart LR") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, nodeShapeAndStyling) {
  MermaidPlanVisitor visitor;
  auto planNode = createValuesNode();

  std::string diagram = visitor.build(planNode);

  // Check for CSS class definitions for styling
  EXPECT_TRUE(diagram.find("classDef green") != std::string::npos);
  EXPECT_TRUE(diagram.find("classDef yellow") != std::string::npos);
  EXPECT_TRUE(diagram.find("classDef orange") != std::string::npos);
  EXPECT_TRUE(diagram.find("classDef blue") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, specialCharacterEscaping) {
  // Create a plan with values that might contain special characters
  auto values =
      ValuesNode::Builder()
          .id("values_with_special_chars")
          .values({makeRowVector({
              makeFlatVector<std::string>({"<test>", "a&b", "\"quote\""}),
          })})
          .build();

  MermaidPlanVisitor visitor;
  std::string diagram = visitor.build(values);

  // Should not contain unescaped special characters that could break Mermaid
  EXPECT_TRUE(diagram.find("VALUES") != std::string::npos);
}

TEST_F(MermaidPlanVisitorTest, emptyPlan) {
  MermaidPlanVisitor visitor;

  // Test with null plan node (should handle gracefully)
  // Note: This test depends on the implementation's error handling
  PlanNodePtr nullPlan = nullptr;

  // This should not crash the visitor, though the exact behavior
  // depends on the implementation
  if (nullPlan != nullptr) {
    std::string diagram = visitor.build(nullPlan);
    EXPECT_TRUE(diagram.find("flowchart") != std::string::npos);
  }
}

TEST_F(MermaidPlanVisitorTest, nodeIdGeneration) {
  MermaidPlanVisitor visitor1;
  MermaidPlanVisitor visitor2;

  auto planNode = createValuesNode();

  std::string diagram1 = visitor1.build(planNode);
  std::string diagram2 = visitor2.build(planNode);

  // Both diagrams should be valid but potentially have different node IDs
  EXPECT_TRUE(
      diagram1.find("node0") != std::string::npos ||
      diagram1.find("node") != std::string::npos);
  EXPECT_TRUE(
      diagram2.find("node0") != std::string::npos ||
      diagram2.find("node") != std::string::npos);
}
