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

#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {

namespace {

class ReservoirSampleTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    disableTestStreaming();
    disableTestIncremental();
  }
};

TEST_F(ReservoirSampleTest, basicSampling) {
  auto data = makeRowVector(
      {makeAllNullArrayVector(5, INTEGER()),
       makeFlatVector<int64_t>(std::vector<int64_t>(5, 0)),
       makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int32_t>(std::vector<int32_t>(5, 3))});

  auto expected = makeRowVector({makeRowVector({
      makeFlatVector<int64_t>({5}), // processed_count
      makeArrayVectorFromJson<int32_t>(
          {"[1, 2, 4]"}) // sample (first 3 elements)
  })});
  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, differentTypes) {
  // INTEGER
  {
    auto data = makeRowVector(
        {makeAllNullArrayVector(5, INTEGER()),
         makeFlatVector<int64_t>(std::vector<int64_t>(5, 0)),
         makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
         makeFlatVector<int32_t>(std::vector<int32_t>(5, 3))});
    auto expected = makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({5}), // processed_count
        makeArrayVectorFromJson<int32_t>(
            {"[1, 2, 4]"}) // sample (first 3 elements)
    })});
    testAggregations(
        {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
  }

  // BIGINT
  {
    auto data = makeRowVector(
        {makeAllNullArrayVector(3, BIGINT()),
         makeFlatVector<int64_t>(std::vector<int64_t>(3, 0)),
         makeFlatVector<int64_t>({100L, 200L, 300L}),
         makeFlatVector<int32_t>(std::vector<int32_t>(3, 5))});
    auto expected = makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({3}), // processed_count
        makeArrayVectorFromJson<int64_t>(
            {"[100, 200, 300]"}) // sample (all elements fit)
    })});
    testAggregations(
        {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
  }

  // VARCHAR
  {
    auto data = makeRowVector(
        {makeAllNullArrayVector(4, VARCHAR()),
         makeFlatVector<int64_t>(std::vector<int64_t>(4, 0)),
         makeFlatVector<std::string>({"a", "b", "c", "d"}),
         makeFlatVector<int32_t>(std::vector<int32_t>(4, 2))});
    auto expected = makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({4}), // processed_count
        makeArrayVectorFromJson<std::string>(
            {"[\"a\", \"d\"]"}) // sample (first 2 elements)
    })});
    testAggregations(
        {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
  }
}

TEST_F(ReservoirSampleTest, edgeCases) {
  // Sample size larger than input
  {
    auto data = makeRowVector(
        {makeAllNullArrayVector(3, INTEGER()),
         makeFlatVector<int64_t>(std::vector<int64_t>(3, 0)),
         makeFlatVector<int32_t>({1, 2, 3}),
         makeFlatVector<int32_t>(std::vector<int32_t>(3, 5))});
    auto expected = makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({3}), // processed_count
        makeArrayVectorFromJson<int32_t>(
            {"[1, 2, 3]"}) // all input values fit in sample
    })});
    testAggregations(
        {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
  }

  // Sample size of 1
  {
    auto data = makeRowVector(
        {makeAllNullArrayVector(5, INTEGER()),
         makeFlatVector<int64_t>(std::vector<int64_t>(5, 0)),
         makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
         makeFlatVector<int32_t>(std::vector<int32_t>(5, 1))});
    auto expected = makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({5}), // processed_count
        makeArrayVectorFromJson<int32_t>(
            {"[20]"}) // only first element due to sample size 1
    })});
    testAggregations(
        {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
  }
}

TEST_F(ReservoirSampleTest, groupBy) {
  auto data = makeRowVector(
      {makeAllNullArrayVector(6, INTEGER()),
       makeFlatVector<int64_t>(std::vector<int64_t>(6, 0)),
       makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}), // group_id
       makeFlatVector<int32_t>({10, 20, 30, 40, 50, 60}), // values
       makeFlatVector<int32_t>(std::vector<int32_t>(6, 2))});

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeRowVector(
           {makeFlatVector<int64_t>({2, 2, 2}),
            makeArrayVectorFromJson<int32_t>(
                {"[10, 20]", "[30,40]", "[50, 60]"})})});
  testAggregations(
      {data}, {"c2"}, {"reservoir_sample(c0, c1, c3, c4)"}, {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql
