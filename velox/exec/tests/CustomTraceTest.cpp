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
#include <unordered_map>
#include <vector>

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/trace/TraceCtx.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::trace::test {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;
using velox::test::assertEqualVectors;

using TCapturedVectors = std::unordered_map<core::PlanNodeId, VectorPtr>;

// A custom test trace implementation that only captures pointers to internal
// vectors. It only traces operators from `traceIds`, and assumes
// single-threaded execution. Vectors are captured in TCapturedVectors.
class TestTraceCtx : public TraceCtx {
 public:
  TestTraceCtx(
      const std::vector<core::PlanNodeId>& tracedIds,
      TCapturedVectors& tracedVectors)
      : TraceCtx(false),
        tracedIds_(tracedIds.begin(), tracedIds.end()),
        tracedVectors_(tracedVectors) {}

  bool shouldTrace(const Operator& op) const override {
    return tracedIds_.contains(op.planNodeId());
  }

  class TestTraceInputWriter : public TraceInputWriter {
   public:
    TestTraceInputWriter(
        const core::PlanNodeId& planId,
        TCapturedVectors& tracedVectors)
        : planId_(planId), tracedVectors_(tracedVectors) {}

    void write(const RowVectorPtr& rows) override {
      tracedVectors_[planId_] = rows;
    }

    void finish() override {}

   private:
    const core::PlanNodeId planId_;
    TCapturedVectors& tracedVectors_;
  };

  std::unique_ptr<TraceInputWriter> createInputTracer(
      Operator& op) const override {
    return std::make_unique<TestTraceInputWriter>(
        op.planNodeId(), tracedVectors_);
  }

 private:
  std::unordered_set<core::PlanNodeId> tracedIds_;

  TCapturedVectors& tracedVectors_;
};

class CustomTraceTest : public exec::test::HiveConnectorTestBase {};

TEST_F(CustomTraceTest, customTrace) {
  auto vector = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(10, [](auto row) { return row; })});

  core::PlanNodeId traceNodeId1;
  core::PlanNodeId traceNodeId2;

  // Trace the inputs from two operators.
  auto plan = PlanBuilder()
                  .values({vector})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(traceNodeId1)
                  .project({"a * 10 as a"})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(traceNodeId2)
                  .planNode();

  TCapturedVectors tracedVectors;
  auto queryCtx =
      core::QueryCtx::Builder()
          .executor(executor_.get())
          .traceCtxProvider([&](core::QueryCtx&, const core::PlanFragment&) {
            return std::make_unique<TestTraceCtx>(
                std::vector<core::PlanNodeId>{traceNodeId1, traceNodeId2},
                tracedVectors);
          })
          .build();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .queryCtx(queryCtx)
      .countResults(task);

  auto it1 = tracedVectors.find(traceNodeId1);
  auto it2 = tracedVectors.find(traceNodeId2);

  ASSERT_TRUE(it1 != tracedVectors.end());
  ASSERT_TRUE(it2 != tracedVectors.end());

  auto expected1 = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(10, [](auto row) { return row; })});
  auto expected2 = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(10, [](auto row) { return row * 10 * 10; })});

  assertEqualVectors(it1->second, expected1);
  assertEqualVectors(it2->second, expected2);

  // Vectors need to be destructed before the pool in the task dies.
  tracedVectors.clear();
}

} // namespace
} // namespace facebook::velox::exec::trace::test
