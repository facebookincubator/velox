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
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

class CorrelationAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
    registerAggregateFunctions("spark_");
  }
};

TEST_F(CorrelationAggregationTest, precision) {
  // Construct data to make covar = 0.5, both stddev = 0.5.
  auto data = makeRowVector({
      makeFlatVector<double>({0, 1}),
      makeFlatVector<double>({0, 1}),
  });
  auto expected =
      makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>({1}))});

  // Add an extra cast to test the result is not 0.999...
  testAggregations(
      {data},
      {},
      {"spark_corr(c0, c1)"},
      {"cast(a0 as BIGINT)"},
      {expected},
      std::unordered_map<std::string, std::string>(
          {{facebook::velox::core::QueryConfig::kCastToIntByTruncate,
            "true"}}));
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
