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
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

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
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Uploads a host batch into a device-resident CudfVector, matching what an
  // upstream GPU operator emits when its output crosses an execution boundary
  // without a CudfToVelox conversion.
  RowVectorPtr uploadToDevice(const RowVectorPtr& input) {
    auto stream = cudf::get_default_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        input, pool(), stream, cudf::get_current_device_resource_ref());
    const auto numRows = table->num_rows();
    return std::make_shared<cudf_velox::CudfVector>(
        pool(), asRowType(input->type()), numRows, std::move(table), stream);
  }
};

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

TEST_F(AdapterOperatorTest, fromVeloxPassesThroughDeviceInput) {
  auto batch = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<StringView>({"a", "bb", "ccc", "dd", "e"})});

  // Two device-resident batches force CudfFromVelox to merge its queued
  // inputs, which reads host children a CudfVector does not carry.
  auto plan = PlanBuilder()
                  .values({uploadToDevice(batch), uploadToDevice(batch)})
                  .project({"c0 * 2 as x", "c1"})
                  .planNode();

  auto expected = makeRowVector(
      {"x", "c1"},
      {makeFlatVector<int64_t>({2, 4, 6, 8, 10}),
       makeFlatVector<StringView>({"a", "bb", "ccc", "dd", "e"})});
  AssertQueryBuilder(plan).assertResults(
      std::vector<RowVectorPtr>{expected, expected});
}

TEST_F(AdapterOperatorTest, fromVeloxMergesHostInputsAheadOfDeviceInput) {
  auto hostBatch = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto deviceBatch = uploadToDevice(
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({4, 5, 6})}));

  // A host batch queued ahead of a device-resident batch must be converted
  // on its own; the device batch passes through unconverted.
  auto plan = PlanBuilder()
                  .values({hostBatch, deviceBatch})
                  .project({"c0 + 10 as x"})
                  .planNode();

  auto expected =
      makeRowVector({"x"}, {makeFlatVector<int64_t>({11, 12, 13, 14, 15, 16})});
  AssertQueryBuilder(plan).assertResults(expected);
}
