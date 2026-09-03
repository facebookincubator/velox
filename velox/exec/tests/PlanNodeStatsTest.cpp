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

#include "velox/exec/PlanNodeStats.h"
#include <gtest/gtest.h>

namespace facebook::velox::exec::test {

TEST(PlanNodeStatsTest, exprStatsTotal) {
  PlanNodeStats stats;
  stats.expressionStats["foo"] = ExprStats{
      .timing = {.wallNanos = 1, .cpuNanos = 2},
      .numProcessedRows = 3,
      .numProcessedVectors = 4};

  PlanNodeStats total;
  total += stats;
  EXPECT_EQ(total.expressionStats["foo"], stats.expressionStats["foo"]);
}

TEST(PlanNodeStatsTest, planNodeBoundary) {
  auto makeOperatorStats = [](int32_t operatorId,
                              std::string operatorType,
                              core::PlanNode::Boundary boundary) {
    OperatorStats stats(operatorId, 0, "1", std::move(operatorType));
    stats.planNodeBoundary = boundary;
    stats.inputPositions = 100;
    stats.outputPositions = 100;
    stats.blockedWallNanos = 5;
    return stats;
  };

  PlanNodeStats stats;
  stats.add(makeOperatorStats(0, "Producer", core::PlanNode::Boundary::kInput));
  stats.add(
      makeOperatorStats(1, "Consumer", core::PlanNode::Boundary::kOutput));

  EXPECT_EQ(stats.inputRows, 100);
  EXPECT_EQ(stats.outputRows, 100);

  // Resource metrics sum over all operators regardless of boundary.
  EXPECT_EQ(stats.blockedWallNanos, 10);

  // The per-operator-type breakdown reports each operator in full.
  for (const auto& operatorType : {"Producer", "Consumer"}) {
    SCOPED_TRACE(operatorType);
    EXPECT_EQ(stats.operatorStatsFor(operatorType).inputRows, 100);
    EXPECT_EQ(stats.operatorStatsFor(operatorType).outputRows, 100);
  }
}

} // namespace facebook::velox::exec::test
