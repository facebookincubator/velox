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
  RowTypePtr bigintRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), BIGINT()})};
  RowTypePtr decimalRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), DECIMAL(20, 5)})};
  RowTypePtr realRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), REAL()})};
  RowTypePtr integerRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), INTEGER()})};
  RowTypePtr smallintRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), SMALLINT()})};
  RowTypePtr tinyintRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), TINYINT()})};
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

TEST_F(NoisySumGaussianAggregationTest, inputTypeTestNoNoise) {
  // Test that the function supports various input types.
  auto doubleVector = makeVectors(doubleRowType_, 5, 3);
  auto bigintVector = makeVectors(bigintRowType_, 5, 3);
  auto decimalVector = makeVectors(decimalRowType_, 5, 3);
  auto realVector = makeVectors(realRowType_, 5, 3);
  auto integerVector = makeVectors(integerRowType_, 5, 3);
  auto smallintVector = makeVectors(smallintRowType_, 5, 3);
  auto tinyintVector = makeVectors(tinyintRowType_, 5, 3);

  createDuckDbTable(doubleVector);
  testAggregations(
      doubleVector,
      {},
      {"noisy_sum_gaussian(c2, 0.0)"}, // double noise_scale
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(bigintVector);
  testAggregations(
      bigintVector,
      {},
      {"noisy_sum_gaussian(c2, 0.0)"},
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(decimalVector);
  testAggregations(
      decimalVector,
      {},
      {"noisy_sum_gaussian(c2, 0.0)"},
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(realVector);
  testAggregations(
      realVector,
      {},
      {"noisy_sum_gaussian(c2, 0)"}, // bigint noise_scale
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(integerVector);
  testAggregations(
      integerVector,
      {},
      {"noisy_sum_gaussian(c2, 0)"},
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(smallintVector);
  testAggregations(
      smallintVector,
      {},
      {"noisy_sum_gaussian(c2, 0)"},
      "SELECT sum(c2) FROM tmp");

  createDuckDbTable(tinyintVector);
  testAggregations(
      tinyintVector,
      {},
      {"noisy_sum_gaussian(c2, 0)"},
      "SELECT sum(c2) FROM tmp");
}
} // namespace facebook::velox::aggregate::test
