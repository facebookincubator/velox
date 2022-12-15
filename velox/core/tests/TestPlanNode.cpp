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
#include <gtest/gtest.h>

#include "velox/core/PlanNode.h"

using namespace ::facebook::velox;
using namespace ::facebook::velox::core;

TEST(TestPlanNode, findFirstNode) {
  auto rowType = ROW({"name1"}, {BIGINT()});

  std::shared_ptr<connector::ConnectorTableHandle> tableHandle;
  std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
      assignments;

  std::shared_ptr<PlanNode> tableScan3 =
      std::make_shared<TableScanNode>("3", rowType, tableHandle, assignments);
  std::shared_ptr<PlanNode> tableScan2 =
      std::make_shared<TableScanNode>("2", rowType, tableHandle, assignments);

  std::vector<FieldAccessTypedExprPtr> sortingKeys;
  std::vector<SortOrder> sortingOrders;
  std::shared_ptr<PlanNode> localMerge1 = std::make_shared<LocalMergeNode>(
      "1",
      sortingKeys,
      sortingOrders,
      std::vector<PlanNodePtr>{tableScan2, tableScan3});

  std::vector<std::string> names;
  std::vector<TypedExprPtr> projections;
  std::shared_ptr<PlanNode> project0 =
      std::make_shared<ProjectNode>("0", names, projections, localMerge1);

  EXPECT_EQ(
      tableScan3.get(),
      PlanNode::findFirstNode(project0.get(), [](const PlanNode* node) {
        return node->id() == "3";
      }));

  EXPECT_EQ(
      project0.get(),
      PlanNode::findFirstNode(project0.get(), [](const PlanNode* node) {
        return node->name() == "Project";
      }));

  EXPECT_EQ(
      nullptr,
      PlanNode::findFirstNode(project0.get(), [](const PlanNode* node) {
        return node->name() == "Unknown";
      }));
}
