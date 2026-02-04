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

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {

int64_t computeAvgRaw(const std::vector<int64_t>& values) {
  int128_t sum = 0;
  for (auto value : values) {
    sum += value;
  }
  int128_t avg = 0;
  facebook::velox::DecimalUtil::computeAverage(
      avg, sum, values.size(), 0);
  return static_cast<int64_t>(avg);
}

class CudfDecimalTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    CudfConfig::getInstance().allowCpuFallback = false;
    // Ensure a CUDA device is selected and initialized (RMM asserts otherwise).
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(0)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }
};

TEST_F(CudfDecimalTest, decimal64And128ArithmeticAndComparison) {
  // Short decimal (64-bit) uses scale 2, long decimal (128-bit) uses scale 10.
  auto rowType = ROW({
      {"d64_a", DECIMAL(12, 2)},
      {"d64_b", DECIMAL(12, 2)},
      {"d128_a", DECIMAL(38, 10)},
      {"d128_b", DECIMAL(38, 10)},
  });

  // Raw values are already scaled.
  auto input = makeRowVector(
      {"d64_a", "d64_b", "d128_a", "d128_b"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 999999}, DECIMAL(12, 2)), // 123.45, -25.00, 9999.99
          makeFlatVector<int64_t>(
              {6789, 1500, -50000}, DECIMAL(12, 2)), // 67.89, 15.00, -500.00
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(123'456'789'012),   // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(555'000'000'000),  // 55.5000000000
              },
              DECIMAL(38, 10)),
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(222'222'222'222),  // 22.2222222222
                  static_cast<int128_t>(333'333'333'333),  // 33.3333333333
                  static_cast<int128_t>(-111'111'111'111), // -11.1111111111
              },
              DECIMAL(38, 10)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "d64_a + d64_b AS sum64",
                      "d64_a - d64_b AS diff64",
                      "d64_a > d64_b AS gt64",
                      "d128_a + d128_b AS sum128",
                      "d128_a - d128_b AS diff128",
                      "d128_a < d128_b AS lt128",
                  })
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
      "SELECT d64_a + d64_b AS sum64, "
      "d64_a - d64_b AS diff64, "
      "d64_a > d64_b AS gt64, "
      "d128_a + d128_b AS sum128, "
      "d128_a - d128_b AS diff128, "
      "d128_a < d128_b AS lt128 "
      "FROM tmp");
}

TEST_F(CudfDecimalTest, decimalIdentityProjection64And128) {
  auto rowType = ROW({
      {"d64", DECIMAL(12, 2)},
      {"d128", DECIMAL(38, 10)},
  });

  // Max absolute raw value for DECIMAL(38,10) is 10^28 - 1 (28 integer digits).
  const int128_t max38p10 = facebook::velox::DecimalUtil::kPowersOfTen[28] - 1;

  auto input = makeRowVector(
      {"d64", "d128"},
      {
          makeFlatVector<int64_t>(
              {
                  // Near max/min for DECIMAL(12,2): +/- 99,999,999,999.99
                  9'999'999'999'999,   // 99,999,999,999.99
                  -9'999'999'999'999,  // -99,999,999,999.99
                  // Mid-range values
                  123'45,    // 1,23.45
                  -2'500,    // -25.00
                  999'999,   // 9,999.99
                  -1'000,    // -10.00
                  0,
                  1,         // 0.01
                  -1,        // -0.01
              },
              DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {
                  // Near max/min for DECIMAL(38,10): +/- (10^28 - 1) with scale 10
                  max38p10,
                  -max38p10,
                  // Mid-range values
                  static_cast<int128_t>(123'456'789'012),   // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(555'000'000'000),  // 55.5000000000
                  static_cast<int128_t>(44'388'888'889),   // 4.4388888889
                  static_cast<int128_t>(1),                // 0.0000000001
                  static_cast<int128_t>(-1),               // -0.0000000001
                  static_cast<int128_t>(0),
              },
              DECIMAL(38, 10)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"d64", "d128"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT d64, d128 FROM tmp");
}

TEST_F(CudfDecimalTest, decimalAddition64And128) {
  auto rowType = ROW({
      {"d64_a", DECIMAL(12, 2)},
      {"d64_b", DECIMAL(12, 2)},
      {"d128_a", DECIMAL(38, 10)},
      {"d128_b", DECIMAL(38, 10)},
  });

  const int128_t max38p10 = facebook::velox::DecimalUtil::kPowersOfTen[28] - 1;
  const int128_t min38p10 = -max38p10;

  auto input = makeRowVector(
      {"d64_a", "d64_b", "d128_a", "d128_b"},
      {
          makeFlatVector<int64_t>(
              {
                  9'999'999'999'99,   // 9,999,999,999.99 (near max for 12,2)
                  -9'999'999'999'99,  // -9,999,999,999.99
                  123'45,             // 1,23.45
                  -2'500,             // -25.00
                  0,
              },
              DECIMAL(12, 2)),
          makeFlatVector<int64_t>(
              {
                  1,     // 0.01
                  -1,    // -0.01
                  9'999, // 99.99
                  -100,  // -1.00
                  50,    // 0.50
              },
              DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {
                  max38p10,
                  min38p10,
                  static_cast<int128_t>(123'456'789'012),   // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(0),
              },
              DECIMAL(38, 10)),
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(1),                // 0.0000000001
                  static_cast<int128_t>(-1),               // -0.0000000001
                  static_cast<int128_t>(44'388'888'889),   // 4.4388888889
                  static_cast<int128_t>(555'000'000'000),  // 55.5000000000
                  max38p10,
              },
              DECIMAL(38, 10)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "d64_a + d64_b AS sum64",
                      "d128_a + d128_b AS sum128",
                  })
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT d64_a + d64_b AS sum64, d128_a + d128_b AS sum128 FROM tmp");
}

TEST_F(CudfDecimalTest, decimalMultiplyPromotesToLong) {
  // Two short decimals whose product requires long decimal precision.
  auto rowType = ROW({
      {"a", DECIMAL(10, 0)},
      {"b", DECIMAL(10, 0)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(
              {9'999'999'999, 1'234'567'890, -2'000'000'000},
              DECIMAL(10, 0)),
          makeFlatVector<int64_t>(
              {9'999'999'999, -2, 4},
              DECIMAL(10, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a * b AS prod"})
                  .planNode();

  const int128_t expected0 =
      static_cast<int128_t>(9'999'999'999LL) * static_cast<int128_t>(9'999'999'999LL);
  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<int128_t>(
          {expected0,
           static_cast<int128_t>(-2'469'135'780LL),
           static_cast<int128_t>(-8'000'000'000LL)},
          DECIMAL(20, 0))});

  // CPU (no cuDF adapter registered).
  unregisterCudf();
  auto cpuResult = facebook::velox::exec::test::AssertQueryBuilder(plan)
                       .copyResults(pool());
  registerCudf();

  // GPU (enable cuDF, no fallback).
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  // Verify promotion to long decimal and exact results on CPU/GPU.
  ASSERT_TRUE(cpuResult->childAt(0)->type()->isLongDecimal());
  ASSERT_TRUE(gpuResult->childAt(0)->type()->isLongDecimal());
  facebook::velox::test::assertEqualVectors(expected, cpuResult);
  facebook::velox::test::assertEqualVectors(expected, gpuResult);
}

TEST_F(CudfDecimalTest, decimalAddPromotesToLong) {
  // Two short decimals whose sum requires long decimal precision.
  auto rowType = ROW({
      {"a", DECIMAL(18, 0)},
      {"b", DECIMAL(18, 0)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(
              {999'999'999'999'999'999LL,
               -900'000'000'000'000'000LL,
               123'456'789'012'345'678LL},
              DECIMAL(18, 0)),
          makeFlatVector<int64_t>(
              {999'999'999'999'999'999LL,
               -900'000'000'000'000'000LL,
               -123'456'789'012'345'678LL},
              DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a + b AS sum"})
                  .planNode();

  auto expected = makeRowVector(
      {"sum"},
      {makeFlatVector<int128_t>(
          {static_cast<int128_t>(1'999'999'999'999'999'998LL),
           static_cast<int128_t>(-1'800'000'000'000'000'000LL),
           static_cast<int128_t>(0)},
          DECIMAL(19, 0))});

  // CPU (no cuDF adapter registered).
  unregisterCudf();
  auto cpuResult = facebook::velox::exec::test::AssertQueryBuilder(plan)
                       .copyResults(pool());
  registerCudf();

  // GPU (cuDF enabled).
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_TRUE(cpuResult->childAt(0)->type()->isLongDecimal());
  ASSERT_TRUE(gpuResult->childAt(0)->type()->isLongDecimal());
  facebook::velox::test::assertEqualVectors(expected, cpuResult);
  facebook::velox::test::assertEqualVectors(expected, gpuResult);
}

TEST_F(CudfDecimalTest, decimalAvgAndSumTimesDouble) {
  auto rowType = ROW({
      {"l_quantity", DECIMAL(15, 2)},
  });

  // Values chosen to keep the AVG and SUM exact in double.
  auto input = makeRowVector(
      {"l_quantity"},
      {makeFlatVector<int64_t>(
          {125, 250, 375, 400}, // 1.25, 2.50, 3.75, 4.00
          DECIMAL(15, 2))});

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  // Force CPU-only path to validate this fails without cuDF involvement.
  unregisterCudf();

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"l_quantity * 2.0 AS qty2"})
                  .singleAggregation(
                      {},
                      {"avg(qty2) AS avg_qty", "sum(qty2) AS sum_qty"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT avg(l_quantity * 2.0) AS avg_qty, "
          "sum(l_quantity * 2.0) AS sum_qty "
          "FROM tmp");
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInput) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {100, 200, 300, 400}, // 1.00, 2.00, 3.00, 4.00
          DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"},
      {makeFlatVector<int64_t>({250}, DECIMAL(12, 2))}); // 2.50

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInputRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  // Sum = 1.60, count = 7 => 0.22857..., rounds to 0.23 at scale 2.
  std::vector<int64_t> rawValues = {100, 10, 10, 10, 10, 10, 10};
  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>(rawValues, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"},
      {makeFlatVector<int64_t>(
          {computeAvgRaw(rawValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  std::vector<int32_t> keys = {1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3};
  std::vector<int64_t> values = {100, 10, 10, 10, 10, 10, 10, 100, 1, -100, -1};

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 3}),
          makeFlatVector<int64_t>({100, 10, 100, -100}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1, 2, 3}),
          makeFlatVector<int64_t>(
              {10, 10, 10, 10, 10, 1, -1}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(
          {computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(
          {computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {100, 10, 10, 10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>(
          {computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000},
              DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int64_t>(
              {200, -300},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalSingle) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {12345, -2500, 10000, 200, -300},
          DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

} // namespace
} // namespace facebook::velox::cudf_velox
