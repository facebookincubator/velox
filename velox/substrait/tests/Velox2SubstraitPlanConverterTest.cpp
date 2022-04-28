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

#include <folly/Random.h>

#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/VectorMaker.h"

#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/VeloxToSubstraitPlan.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::substrait;

class SubstraitVeloxPlanConvertorTest : public OperatorTestBase {
 protected:
  /// The function to make RowVectorPtr data which use the specific rowType_ and
  /// batchSize. \param size the number of RowVectorPtr. \param batchSize The
  /// batch Size of the data.
  std::vector<RowVectorPtr> makeVector(int64_t size, int64_t batchSize) {
    std::vector<RowVectorPtr> vectors;
    std::mt19937 gen(std::mt19937::default_seed);
    int64_t childSize = rowType_->size();
    for (int i = 0; i < size; i++) {
      std::vector<VectorPtr> children;
      for (int j = 0; j < childSize; j++) {
        children.emplace_back(makeFlatVector<int32_t>(
            batchSize,
            [&](auto row) {
              return folly::Random::rand32(INT32_MAX / 4, INT32_MAX / 2, gen);
            },
            nullEvery(2)));
      }

      vectors.push_back(makeRowVector({children}));
    }
    return vectors;
  };

  void assertPlanConversion(
      const std::shared_ptr<const core::PlanNode>& plan,
      const std::string& duckDbSql) {
    assertQuery(plan, duckDbSql);
    // Convert Velox Plan to Substrait Plan.
    google::protobuf::Arena arena;
    convertor_.toSubstrait(arena, plan);
  }

  VeloxToSubstraitPlanConvertor convertor_;
  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()})};
};

TEST_F(SubstraitVeloxPlanConvertorTest, project) {
  auto vectors = makeVector(3, 2);
  createDuckDbTable(vectors);
  auto plan =
      PlanBuilder().values(vectors).project({"c0 + c1", "c1 - c2"}).planNode();
  assertPlanConversion(plan, "SELECT  c0 + c1, c1 - c2 FROM tmp");
}

TEST_F(SubstraitVeloxPlanConvertorTest, filter) {
  auto vectors = makeVector(3, 2);
  createDuckDbTable(vectors);

  const std::string& filter =
      "(c2 < 1000) and (c1 between 0.6 and 1.6) and (c0 >= 100)";
  auto plan = PlanBuilder().values(vectors).filter(filter).planNode();

  assertPlanConversion(plan, "SELECT * FROM tmp WHERE " + filter);
}

TEST_F(SubstraitVeloxPlanConvertorTest, values) {
  auto vectors = makeVector(3, 2);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder().values(vectors).planNode();

  assertPlanConversion(plan, "SELECT * FROM tmp");
}
