/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <utility>
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

    bool write(const RowVectorPtr& vector, ContinueFuture*) override {
      tracedVectors_[planId_] = vector;
      return false;
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

// Helper to convert a vector of plan node IDs to a breakpoints map with null
// callbacks.
CursorParameters::TBreakpointMap toBreakpointsMap(
    const std::vector<core::PlanNodeId>& ids) {
  CursorParameters::TBreakpointMap result;
  for (const auto& id : ids) {
    result[id] = nullptr;
  }
  return result;
}

void assertCursorOutput(
    const core::PlanNodePtr& plan,
    const std::vector<core::PlanNodeId>& breakpoints,
    const std::vector<RowVectorPtr>& expectation) {
  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = toBreakpointsMap(breakpoints),
  });
  size_t i = 0;

  while (cursor->moveStep()) {
    if (i < expectation.size()) {
      assertEqualVectors(cursor->current(), expectation[i++]);
    } else {
      ADD_FAILURE() << "Cursor output is longer than expectation: " << i;
    }
  }
  EXPECT_EQ(i, expectation.size());
}

TEST_F(CustomTraceTest, taskDebuggerCursor) {
  const size_t size = 10;
  auto makeData = [&](std::function<int64_t(vector_size_t)> values) {
    return makeRowVector(
        {"a"}, {makeFlatVector<int64_t>(size, std::move(values))});
  };

  // Two input vectors.
  auto input1 = makeData([](auto row) { return row; });
  auto input2 = makeData([](auto row) { return row + 10; });

  // Now, the expected input for a series of operators.
  auto input1Project1 = makeData([](auto row) { return row; });
  auto input1Project2 = makeData([](auto row) { return row * 10; });
  auto input1Project3 = makeData([](auto row) { return row * 100; });
  auto input1Project4 = makeData([](auto row) { return row * 1'000; });
  auto output1 = makeData([](auto row) { return row * 10'000; });

  auto input2Project1 = makeData([](auto row) { return (row + 10); });
  auto input2Project2 = makeData([](auto row) { return (row + 10) * 10; });
  auto input2Project3 = makeData([](auto row) { return (row + 10) * 100; });
  auto input2Project4 = makeData([](auto row) { return (row + 10) * 1'000; });
  auto output2 = makeData([](auto row) { return (row + 10) * 10'000; });

  core::PlanNodeId project1, project2, project3, project4;
  auto plan = PlanBuilder()
                  .values({input1, input2})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project2)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project3)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project4)
                  .planNode();

  // Test a series of combinations.
  assertCursorOutput(plan, {}, {output1, output2});
  assertCursorOutput(
      plan, {project1}, {input1Project1, output1, input2Project1, output2});
  assertCursorOutput(
      plan,
      {project1, project2},
      {
          input1Project1,
          input1Project2,
          output1,
          input2Project1,
          input2Project2,
          output2,
      });
  assertCursorOutput(
      plan,
      {project2, project4},
      {
          input1Project2,
          input1Project4,
          output1,
          input2Project2,
          input2Project4,
          output2,
      });
  assertCursorOutput(
      plan,
      {project1, project2, project3, project4},
      {
          input1Project1,
          input1Project2,
          input1Project3,
          input1Project4,
          output1,
          input2Project1,
          input2Project2,
          input2Project3,
          input2Project4,
          output2,
      });
}

TEST_F(CustomTraceTest, cursorAt) {
  const size_t size = 10;
  auto input1 = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(size, [](auto row) { return row; })});
  auto input2 = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(size, [](auto row) { return row + 10; })});

  core::PlanNodeId project1, project2, project3, project4;
  auto plan = PlanBuilder()
                  .values({input1, input2})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project2)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project3)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project4)
                  .planNode();

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = toBreakpointsMap({project1, project3}),
  });

  // Before any step, at() should return empty string.
  EXPECT_EQ(cursor->at(), "");

  // First step stops at project1 (first breakpoint).
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project1);

  // Second step stops at project3 (second breakpoint).
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project3);

  // Third step produces final output (no breakpoint, empty at()).
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");

  // Fourth step stops at project1 for second input batch.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project1);

  // Fifth step stops at project3 for second input batch.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project3);

  // Sixth step produces final output for second batch.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");

  // No more data.
  EXPECT_FALSE(cursor->moveStep());

  // Test that moveNext() skips breakpoints and at() returns empty.
  auto cursor2 = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = toBreakpointsMap({project1, project3}),
  });

  EXPECT_EQ(cursor2->at(), "");

  // moveNext() should skip to final output.
  EXPECT_TRUE(cursor2->moveNext());
  EXPECT_EQ(cursor2->at(), "");

  EXPECT_TRUE(cursor2->moveNext());
  EXPECT_EQ(cursor2->at(), "");

  EXPECT_FALSE(cursor2->moveNext());
}

TEST_F(CustomTraceTest, breakpointCallbackAlwaysStop) {
  const size_t size = 10;
  auto input = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(size, [](auto row) { return row; })});

  core::PlanNodeId project1;
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .planNode();

  // Callback that always returns true (always stop).
  int callbackCount = 0;
  CursorParameters::TBreakpointMap breakpoints;
  breakpoints[project1] = [&](const RowVectorPtr& vector) {
    ++callbackCount;
    EXPECT_EQ(vector->size(), size);
    return true;
  };

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = std::move(breakpoints),
  });

  // First step should stop at breakpoint.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project1);
  EXPECT_EQ(callbackCount, 1);

  // Second step should produce final output.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");

  EXPECT_FALSE(cursor->moveStep());
}

TEST_F(CustomTraceTest, breakpointCallbackNeverStop) {
  const size_t size = 10;
  auto input = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(size, [](auto row) { return row; })});

  core::PlanNodeId project1;
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .planNode();

  // Callback that always returns false (never stop).
  int callbackCount = 0;
  CursorParameters::TBreakpointMap breakpoints;
  breakpoints[project1] = [&](const RowVectorPtr& vector) {
    ++callbackCount;
    EXPECT_EQ(vector->size(), size);
    return false;
  };

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = std::move(breakpoints),
  });

  // Step should skip the breakpoint and go directly to final output.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");
  EXPECT_EQ(callbackCount, 1); // Callback was still invoked.

  EXPECT_FALSE(cursor->moveStep());
}

TEST_F(CustomTraceTest, breakpointCallbackConditional) {
  const size_t size = 10;
  auto input1 = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(size, [](auto row) { return row; })});
  auto input2 = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(size, [](auto row) { return row + 100; })});

  core::PlanNodeId project1;
  auto plan = PlanBuilder()
                  .values({input1, input2})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .planNode();

  // Callback that stops only when first element is >= 100.
  int callbackCount = 0;
  CursorParameters::TBreakpointMap breakpoints;
  breakpoints[project1] = [&](const RowVectorPtr& vector) {
    ++callbackCount;
    auto values = vector->childAt(0)->asFlatVector<int64_t>();
    return values->valueAt(0) >= 100;
  };

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = std::move(breakpoints),
  });

  // First batch: callback returns false (first element is 0), skips breakpoint.
  // Goes to final output for first batch.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");
  EXPECT_EQ(callbackCount, 1);

  // Second batch: callback returns true (first element is 100), stops at
  // breakpoint.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project1);
  EXPECT_EQ(callbackCount, 2);

  // Final output for second batch.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");

  EXPECT_FALSE(cursor->moveStep());
}

TEST_F(CustomTraceTest, breakpointMixedCallbacks) {
  const size_t size = 10;
  auto input = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>(size, [](auto row) { return row; })});

  core::PlanNodeId project1, project2;
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project1)
                  .project({"a * 10 as a"})
                  .capturePlanNodeId(project2)
                  .planNode();

  // project1 has callback returning false (don't stop).
  // project2 has null callback (always stop).
  int callbackCount = 0;
  CursorParameters::TBreakpointMap breakpoints;
  breakpoints[project1] = [&](const RowVectorPtr&) {
    ++callbackCount;
    return false;
  };
  breakpoints[project2] = nullptr; // null callback = always stop.

  auto cursor = TaskCursor::create({
      .planNode = plan,
      .serialExecution = true,
      .breakpoints = std::move(breakpoints),
  });

  // project1 callback returns false, so it's skipped.
  // project2 has null callback, so it stops.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), project2);
  EXPECT_EQ(callbackCount, 1);

  // Final output.
  EXPECT_TRUE(cursor->moveStep());
  EXPECT_EQ(cursor->at(), "");

  EXPECT_FALSE(cursor->moveStep());
}

} // namespace
} // namespace facebook::velox::exec::trace::test
