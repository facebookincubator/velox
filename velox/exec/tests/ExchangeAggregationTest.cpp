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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"

namespace facebook::velox::exec::test {

class ExchangeAggregationTest : public OperatorTestBase {
 protected:
  const int64_t kNumInputVectors = 10;
  const int32_t kNumDrivers = 8;

  ExchangeAggregationTest() {
    velox::aggregate::prestosql::registerAllAggregateFunctions("", false);
  }
};

TEST_F(ExchangeAggregationTest, groupBy) {
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kNumInputVectors);
  for (int i = 0; i < kNumInputVectors; ++i) {
    inputs.emplace_back(makeRowVector(
        {makeFlatVector<int32_t>(1000, [](auto row) { return row; }),
         makeFlatVector<int32_t>(
             1000, [&](auto row) { return (i * 1000 + row) % 83; }),
         makeFlatVector<int64_t>(1000, [](auto /*row*/) { return 1; })}));
  }

  auto plan = PlanBuilder()
                  .values(inputs)
                  .localPartition({"c1"})
                  .partialAggregation({"c0"}, {"sum(c2)"})
                  .localPartition({"c0"})
                  .finalAggregation()
                  .localPartition({})
                  .planNode();

  auto results = AssertQueryBuilder(plan)
                     .config(core::QueryConfig::kUseExchangeAggregation, "true")
                     .maxDrivers(kNumDrivers)
                     .copyResults(pool());

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>(1000, [](auto row) { return row; }),
       makeFlatVector<int64_t>(
           1000, [this](auto /*row*/) { return kNumInputVectors; })});

  assertEqualResults({expected}, {results});
}

} // namespace facebook::velox::exec::test
