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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class AdapterOperatorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedCpuFallback_ = cudf_velox::CudfConfig::getInstance().allowCpuFallback;
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = savedCpuFallback_;
    OperatorTestBase::TearDown();
  }

  // Puts 'adapter' ahead of every built-in adapter, because
  // OperatorAdapterRegistry::findAdapter() returns the first match and
  // registerAdapter() appends. The built-ins are restored behind it so every
  // other operator in the plan still resolves; clearing them instead would make
  // those operators pure-CPU and trip the allowCpuFallback check for a reason
  // that has nothing to do with the case under test.
  void registerAdapterFirst(
      std::unique_ptr<cudf_velox::OperatorAdapter> adapter) {
    // registerCudf() in SetUp() already installed the built-ins. Put the test
    // adapter ahead of them; calling registerAllOperatorAdapters() here would
    // clear the registry and discard it.
    cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapterFront(
        std::move(adapter));
  }

  bool savedCpuFallback_{true};
};

namespace {
// Claims FilterProject and reports it as GPU-capable, but contributes no
// operators. Whether that is a failure depends on keepOperator(): a replacing
// adapter has produced nothing to replace 'op' with, while a keeping adapter
// has simply nothing to append.
class EmptyReplacementAdapter : public cudf_velox::OperatorAdapter {
 public:
  EmptyReplacementAdapter(bool keepOperator, bool producesGpuOutput = false)
      : cudf_velox::OperatorAdapter("EmptyReplacement"),
        keepOperator_{keepOperator},
        producesGpuOutput_{producesGpuOutput} {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return producesGpuOutput_;
  }

  bool keepOperator() const override {
    return keepOperator_;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

 private:
  const bool keepOperator_;
  const bool producesGpuOutput_;
};

// Keeps its operator and describes two operators that must run after it. The
// pair converts to GPU and back so the appended span is type-balanced whatever
// the neighbours are, which keeps the case about insertion and renumbering
// rather than about conversion placement.
//
// The plan node ids carry the "-from-velox" and "-to-velox" suffixes because
// CudfFromVelox and CudfToVelox recover the plan node they belong to by
// stripping exactly those strings, so any other suffix files their stats under
// a plan node of its own instead of the one being expanded.
class AppendingAdapter : public cudf_velox::OperatorAdapter {
 public:
  AppendingAdapter() : cudf_velox::OperatorAdapter("Appending") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  bool keepOperator() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    std::vector<std::unique_ptr<exec::Operator>> appended;
    appended.push_back(
        std::make_unique<cudf_velox::CudfFromVelox>(
            operatorId,
            planNode->outputType(),
            ctx,
            planNode->id() + "-from-velox"));
    appended.push_back(
        std::make_unique<cudf_velox::CudfToVelox>(
            operatorId,
            planNode->outputType(),
            ctx,
            planNode->id() + "-to-velox"));
    return appended;
  }
};
} // namespace

TEST_F(AdapterOperatorTest, adapterStatsMergedIntoPlanNode) {
  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);

  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_TRUE(projStats.operatorStats.count("CudfToVelox"));
}

// A replacing adapter that contributes nothing has not replaced the operator,
// so with fallback disabled the pipeline must be rejected rather than run with
// the CPU operator still in place. The decision has to come from what
// createReplacements() returned, not from having called it.
TEST_F(AdapterOperatorTest, emptyReplacementIsRejectedWithoutFallback) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/false));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Replacement with cuDF operator failed");
}

// The mirror case, so that rejecting an empty replacement cannot be implemented
// by rejecting every empty result: a keeping adapter that appends nothing
// leaves the operator running on its own, which is what lets one plan node
// expand into several operators.
TEST_F(AdapterOperatorTest, keptOperatorNeedsNoAppendedOperators) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);
  EXPECT_EQ(results->size(), 5);
}

// The same failure, but with the adapter claiming GPU output so that
// CompileState appends a CudfToVelox conversion behind it. Deciding the failure
// from the accumulated replacement list rather than from what the adapter
// returned would see that conversion operator and conclude the replacement
// succeeded, so this case is what pins the decision to the adapter's own
// result. Left undetected the conversion replaces the operator outright, which
// drops the plan node from the pipeline.
TEST_F(
    AdapterOperatorTest,
    emptyReplacementIsRejectedDespiteConversionOperator) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(
          /*keepOperator=*/false, /*producesGpuOutput=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Replacement with cuDF operator failed");
}

// The capability this contract change exists for: a kept operator describing
// operators that run after it, which is how one plan node expands into several.
// No built-in adapter both keeps its operator and returns any, so nothing else
// exercises the insert-behind path or the operator-id renumbering it relies on.
TEST_F(AdapterOperatorTest, keptOperatorGetsAppendedOperators) {
  registerAdapterFirst(std::make_unique<AppendingAdapter>());

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);

  facebook::velox::test::assertEqualVectors(
      makeRowVector({"x"}, {makeFlatVector<int64_t>({2, 4, 6, 8, 10})}),
      results);

  // The kept operator and both appended operators report under the one plan
  // node, which is the expansion of a single plan node into several operators
  // that keepOperator() together with a non-empty createReplacements() exists
  // to express. FilterProject still being there is what distinguishes the
  // append from a replacement.
  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);
  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_EQ(projStats.operatorStats.count("FilterProject"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfFromVelox"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfToVelox"), 1);
}
