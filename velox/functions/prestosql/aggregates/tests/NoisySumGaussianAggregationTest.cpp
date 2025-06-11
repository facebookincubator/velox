// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

class NoisySumGaussianAggregationTest
    : public functions::aggregate::test::AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  RowTypePtr doubleRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), DOUBLE()})};
};

TEST_F(NoisySumGaussianAggregationTest, simpleTestNoNoise) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 5.0})})};

  // Expect the result to be 15.0
  auto expectedResult = makeRowVector({makeConstant(15.0, 1)});
  testAggregations(
      {vectors}, {}, {"noisy_sum_gaussian(c2, 0.0)"}, {expectedResult});

  // test nosie scale of bigint type.
  testAggregations(
      {vectors}, {}, {"noisy_sum_gaussian(c2, 0)"}, {expectedResult});
}
} // namespace facebook::velox::aggregate::test
