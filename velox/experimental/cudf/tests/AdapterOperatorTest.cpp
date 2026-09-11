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

  // Gives the test adapter priority while preserving built-ins for other
  // operators.
  void registerAdapterFirst(
      std::unique_ptr<cudf_velox::OperatorAdapter> adapter) {
    cudf_velox::OperatorAdapterRegistry::getInstance().registerAdapterFront(
        std::move(adapter));
  }

  // Re-registers the driver adapter to capture the updated CPU fallback
  // setting.
  void enableCpuFallback() {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = true;
    cudf_velox::registerCudf();
  }

  bool savedCpuFallback_{true};
};

namespace {
// Returns no replacements to exercise both keep and replace contracts.
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

// Keeps FilterProject and appends a GPU round trip. The standard conversion
// suffixes preserve plan-node stat attribution.
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
// Declines FilterProject; createReplacements() must not be called.
class DecliningAdapter : public cudf_velox::OperatorAdapter {
 public:
  DecliningAdapter() : cudf_velox::OperatorAdapter("Declining") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  bool keepOperator() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    VELOX_FAIL(
        "createReplacements() must not be called for a declined operator");
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

// An empty replacement is an adapter error, not CPU fallback.
TEST_F(AdapterOperatorTest, emptyReplacementIsRejectedWithoutFallback) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/false));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Adapter replaced an operator with nothing");
}

// Empty additions are valid when the original operator is kept.
TEST_F(AdapterOperatorTest, keptOperatorNeedsNoAppendedOperators) {
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(/*keepOperator=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  auto results = AssertQueryBuilder(plan).copyResults(pool(), task);
  EXPECT_EQ(results->size(), 5);
}

// Reject before CudfToVelox can make an empty replacement appear non-empty.
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
      "Adapter replaced an operator with nothing");
}

// Adapter errors are rejected even when CPU fallback is enabled.
TEST_F(AdapterOperatorTest, emptyReplacementIsRejectedWithCpuFallbackEnabled) {
  enableCpuFallback();
  registerAdapterFirst(
      std::make_unique<EmptyReplacementAdapter>(
          /*keepOperator=*/false, /*producesGpuOutput=*/true));

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Adapter replaced an operator with nothing");
}

// A declined GPU path requires CPU fallback.
TEST_F(AdapterOperatorTest, declinedOperatorIsRejectedWithoutFallback) {
  registerAdapterFirst(std::make_unique<DecliningAdapter>());

  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});
  auto plan = PlanBuilder().values({data}).project({"c0 * 2 as x"}).planNode();

  std::shared_ptr<exec::Task> task;
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool(), task),
      "Replacement with cuDF operator failed");
}

// Verify that fallback keeps FilterProject on CPU.
TEST_F(AdapterOperatorTest, declinedOperatorRunsOnCpuWithFallback) {
  enableCpuFallback();
  registerAdapterFirst(std::make_unique<DecliningAdapter>());

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

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);
  EXPECT_EQ(projStats.operatorStats.count("FilterProject"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfFilterProject"), 0);
}

// Exercises appending after a kept operator and operator-ID renumbering.
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

  // All three operators must report under the original plan node.
  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);
  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_EQ(projStats.operatorStats.count("FilterProject"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfFromVelox"), 1);
  EXPECT_EQ(projStats.operatorStats.count("CudfToVelox"), 1);
}
