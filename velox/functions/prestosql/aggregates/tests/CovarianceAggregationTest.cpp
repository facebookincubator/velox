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

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

class CovarianceAggregationTest
    : public virtual AggregationTestBase,
      public testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }

  void testGroupBy(const std::string& aggName, const RowVectorPtr& data) {
    auto partialAgg = fmt::format("{}(c1, c2)", aggName);
    auto sql =
        fmt::format("SELECT c0, {}(c1, c2) FROM tmp GROUP BY 1", aggName);

    testAggregations({data}, {"c0"}, {partialAgg}, sql);
  }

  void testGlobalAgg(const std::string& aggName, const RowVectorPtr& data) {
    auto partialAgg = fmt::format("{}(c1, c2)", aggName);
    auto sql = fmt::format("SELECT {}(c1, c2) FROM tmp", aggName);

    testAggregations({data}, {}, {partialAgg}, sql);
  }
};

//[(1-2)(1-3)+(2-2)(4-3)+(3-2)(4-3)]/3=1
//[(2-2)(1-1.5)+(2-2)(2-1.5)]/2=0
TEST_F(CovarianceAggregationTest, distinctWithMultipleColumns) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<double>({1, 2}),
       makeFlatVector<double>({1, 4})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<double>({2, 3}),
       makeFlatVector<double>({4, 4})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<double>({2, 2}),
       makeFlatVector<double>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c0"}, {"covar_pop(c1, c2)"}, {}, {true})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<double>({1, 0})});
  facebook::velox::test::assertEqualVectors(expected, actual[0]);
}

TEST_P(CovarianceAggregationTest, doubleNoNulls) {
  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<double>(size, [](auto row) { return row * 0.1; }),
      makeFlatVector<double>(size, [](auto row) { return row * 0.2; }),
  });

  createDuckDbTable({data});

  auto aggName = GetParam();
  testGlobalAgg(aggName, data);

  testGroupBy(aggName, data);
}

TEST_P(CovarianceAggregationTest, doubleSomeNulls) {
  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<double>(
          size, [](auto row) { return row * 0.1; }, nullEvery(11)),
      makeFlatVector<double>(
          size, [](auto row) { return row * 0.2; }, nullEvery(17)),
  });

  createDuckDbTable({data});

  auto aggName = GetParam();
  testGlobalAgg(aggName, data);
  testGroupBy(aggName, data);
}

TEST_P(CovarianceAggregationTest, floatNoNulls) {
  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<float>(size, [](auto row) { return row * 0.1; }),
      makeFlatVector<float>(size, [](auto row) { return row * 0.2; }),
  });

  createDuckDbTable({data});

  auto aggName = GetParam();
  testGlobalAgg(aggName, data);
  testGroupBy(aggName, data);
}

TEST_P(CovarianceAggregationTest, floatSomeNulls) {
  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
      makeFlatVector<float>(
          size, [](auto row) { return row * 0.1; }, nullEvery(11)),
      makeFlatVector<float>(
          size, [](auto row) { return row * 0.2; }, nullEvery(17)),
  });

  createDuckDbTable({data});

  auto aggName = GetParam();
  testGlobalAgg(aggName, data);
  testGroupBy(aggName, data);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    CovarianceAggregationTest,
    CovarianceAggregationTest,
    testing::Values(
        "covar_samp",
        "covar_pop",
        "corr",
        "regr_intercept",
        "regr_slope"));
} // namespace facebook::velox::aggregate::test
