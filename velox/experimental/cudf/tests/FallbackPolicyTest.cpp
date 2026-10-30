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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox::test {
namespace {

// Exercises what the cuDF driver adapter does with an operator that has no
// GPU implementation. Unnest is used as that operator: it has no adapter, and
// Values (which does) feeds it, so the plan is the smallest GPU -> CPU edge.
class FallbackPolicyTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    saved_ = CudfConfig::getInstance();
  }

  void TearDown() override {
    if (cudfIsRegistered()) {
      unregisterCudf();
    }
    CudfConfig::getInstance() = saved_;
    OperatorTestBase::TearDown();
  }

  // The adapter reads allowCpuFallback at registration time, so the policy
  // under test must be set before registerCudf().
  void registerCudfWith(
      bool allowCpuFallback,
      std::unordered_set<std::string> allowedCpuOperators) {
    auto& config = CudfConfig::getInstance();
    config.allowCpuFallback = allowCpuFallback;
    config.allowedCpuOperators = std::move(allowedCpuOperators);
    if (cudfIsRegistered()) {
      unregisterCudf();
    }
    registerCudf();
  }

  core::PlanNodePtr unnestPlan() {
    auto input = makeRowVector(
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeArrayVector<int64_t>({{10, 11}, {20}, {}})});
    return PlanBuilder().values({input}).unnest({"c0"}, {"c1"}).planNode();
  }

  RowVectorPtr unnestExpected() {
    return makeRowVector(
        {makeFlatVector<int64_t>({1, 1, 2}),
         makeFlatVector<int64_t>({10, 11, 20})});
  }

  // Number of operators in the finished task carrying 'stat', and the type of
  // each such operator.
  std::vector<std::string> operatorsWithStat(
      const std::shared_ptr<Task>& task,
      std::string_view stat) {
    std::vector<std::string> types;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.runtimeStats.count(std::string(stat))) {
          types.push_back(op.operatorType);
        }
      }
    }
    return types;
  }

 private:
  CudfConfig saved_;
};

TEST_F(FallbackPolicyTest, strictModeFailsAndNamesTheOperator) {
  registerCudfWith(/*allowCpuFallback=*/false, {});
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(unnestPlan()).copyResults(pool()),
      "Replacement with cuDF operator failed: strict mode rejects operator 'Unnest'");
}

TEST_F(FallbackPolicyTest, strictModeHonoursAllowlist) {
  registerCudfWith(
      /*allowCpuFallback=*/false, {std::string(OperatorType::kUnnest)});
  auto task = AssertQueryBuilder(unnestPlan()).assertResults(unnestExpected());

  EXPECT_EQ(
      operatorsWithStat(task, kCudfCpuAllowlistedStat),
      std::vector<std::string>{std::string(OperatorType::kUnnest)});
  EXPECT_TRUE(operatorsWithStat(task, kCudfCpuFallbackStat).empty());
}

TEST_F(FallbackPolicyTest, fallbackIsRecordedOnTheCpuOperator) {
  registerCudfWith(/*allowCpuFallback=*/true, {});
  auto task = AssertQueryBuilder(unnestPlan()).assertResults(unnestExpected());

  EXPECT_EQ(
      operatorsWithStat(task, kCudfCpuFallbackStat),
      std::vector<std::string>{std::string(OperatorType::kUnnest)});
  EXPECT_TRUE(operatorsWithStat(task, kCudfCpuAllowlistedStat).empty());
}

TEST_F(FallbackPolicyTest, allowlistedFallbackIsRecordedAsAllowlisted) {
  registerCudfWith(
      /*allowCpuFallback=*/true, {std::string(OperatorType::kUnnest)});
  auto task = AssertQueryBuilder(unnestPlan()).assertResults(unnestExpected());

  EXPECT_EQ(
      operatorsWithStat(task, kCudfCpuAllowlistedStat),
      std::vector<std::string>{std::string(OperatorType::kUnnest)});
  EXPECT_TRUE(operatorsWithStat(task, kCudfCpuFallbackStat).empty());
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
