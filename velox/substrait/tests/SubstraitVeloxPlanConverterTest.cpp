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

#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/VeloxToSubstraitPlan.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::substrait;

using facebook::velox::test::BatchMaker;

class SubstraitVeloxPlanConvertorTest : public OperatorTestBase {
 protected:
  std::shared_ptr<const RowType> rowType_{
      ROW({"c0", "c1", "c2", "c3"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()})};

  /*!
   *
   * @param size  the number of RowVectorPtr.
   * @param childSize the size of rowType_, the number of columns
   * @param batchSize batch size.
   * @return std::vector<RowVectorPtr>
   */
  std::vector<RowVectorPtr>
  makeVector(int64_t size, int64_t childSize, int64_t batchSize) {
    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < size; i++) {
      std::vector<VectorPtr> children;
      for (int j = 0; j < childSize; j++) {
        VectorPtr child = VELOX_DYNAMIC_TYPE_DISPATCH(
            BatchMaker::createVector,
            rowType_->childAt(j)->kind(),
            rowType_->childAt(j),
            batchSize,
            *pool_);
        children.emplace_back(child);
      }

      auto rowVector = std::make_shared<RowVector>(
          pool_.get(), rowType_, BufferPtr(), batchSize, children);
      vectors.emplace_back(rowVector);
    }
    return vectors;
  };

  void assertVeloxSubstraitRoundTripProject(
      std::vector<RowVectorPtr>&& vectors) {
    auto vPlan = PlanBuilder()
                     .values(vectors)
                     .project(std::vector<std::string>{"c0", "c1"})
                     .planNode();

    assertQuery(vPlan, "SELECT c0, c1  FROM tmp");

    // Convert Velox Plan to Substrait Plan
    v2SPlanConvertor->veloxToSubstraitIR(vPlan, *sPlan);

    // convert back to Velox PlanNode.
    std::shared_ptr<const PlanNode> vPlan2 =
        s2VPlanConvertor->toVeloxPlan(*sPlan);
    assertQuery(vPlan2, "SELECT c0, c1 FROM tmp");
  }

  void assertVeloxToSubstraitProject(std::vector<RowVectorPtr>&& vectors) {
    auto vPlan = PlanBuilder()
                     .values(vectors)
                     .project(std::vector<std::string>{"c0 + c1", "c1-c2"})
                     .planNode();

    assertQuery(vPlan, "SELECT  c0 + c1, c1-c2 FROM tmp");

    // Convert Velox Plan to Substrait Plan
    v2SPlanConvertor->veloxToSubstraitIR(vPlan, *sPlan);
  }

  void SetUp() override {
    v2SPlanConvertor = new VeloxToSubstraitPlanConvertor();
    s2VPlanConvertor = new SubstraitToVeloxPlanConvertor();
    sPlan = new ::substrait::Plan();
  }

  void TearDown() override {
    delete v2SPlanConvertor;
    delete s2VPlanConvertor;
    delete sPlan;
  }

  VeloxToSubstraitPlanConvertor* v2SPlanConvertor;
  SubstraitToVeloxPlanConvertor* s2VPlanConvertor;
  ::substrait::Plan* sPlan;
};

TEST_F(SubstraitVeloxPlanConvertorTest, veloxToSubstraitProjectNode) {
  std::vector<RowVectorPtr> vectors;
  vectors = makeVector(3, 4, 2);
  createDuckDbTable(vectors);

  assertVeloxToSubstraitProject(std::move(vectors));
}
