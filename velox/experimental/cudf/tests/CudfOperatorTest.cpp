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

#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"

namespace facebook::velox::cudf_velox {
namespace {

class TestCudfOperator final : public CudfOperatorBase {
 public:
  TestCudfOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId)
      : CudfOperatorBase(
            operatorId,
            driverCtx,
            std::move(outputType),
            planNodeId,
            "TestCudfOperator") {}

  bool needsInput() const override {
    return false;
  }

  bool isFinished() override {
    return true;
  }

  memory::MemoryPool* gpuPool() const {
    return customPool(kCudfMemoryResourceTag);
  }

 protected:
  void doAddInput(RowVectorPtr /*input*/) override {}

  RowVectorPtr doGetOutput() override {
    return nullptr;
  }
};

class CudfOperatorTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

TEST_F(CudfOperatorTest, installsGpuLeafReclaimerDuringInitialize) {
  auto resource = cudfCustomMemoryResource();
  ASSERT_NE(resource, nullptr);
  auto queryCtx = core::QueryCtx::Builder()
                      .executor(driverExecutor_.get())
                      .queryId("cudf-operator-reclaimer")
                      .customMemoryResource(resource)
                      .build();

  const core::PlanNodeId planNodeId{"values"};
  auto input = makeRowVector({makeFlatVector<int64_t>({1})});
  core::PlanFragment planFragment;
  planFragment.planNode =
      std::make_shared<core::ValuesNode>(planNodeId, std::vector{input});
  auto task = exec::Task::create(
      "CudfOperatorTest",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  auto driver = exec::Driver::testingCreate(
      std::make_unique<exec::DriverCtx>(task, 0, 0, 0, 0));
  TestCudfOperator testOperator{
      0,
      driver->driverCtx(),
      std::dynamic_pointer_cast<const RowType>(input->type()),
      planNodeId};

  auto* gpuPool = testOperator.gpuPool();
  ASSERT_NE(gpuPool, nullptr);
  EXPECT_EQ(gpuPool->reclaimer(), nullptr);

  testOperator.initialize();

  ASSERT_NE(gpuPool->reclaimer(), nullptr);
  uint64_t reclaimableBytes{1};
  EXPECT_FALSE(
      gpuPool->reclaimer()->reclaimableBytes(*gpuPool, reclaimableBytes));
  EXPECT_EQ(reclaimableBytes, 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox
