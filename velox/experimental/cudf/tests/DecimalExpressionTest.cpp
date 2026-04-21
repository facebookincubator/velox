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
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <cuda_runtime_api.h>

namespace facebook::velox::cudf_velox {
namespace {

class CudfDecimalTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
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
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
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
              {12345, -2500, 999999},
              DECIMAL(12, 2)), // 123.45, -25.00, 9999.99
          makeFlatVector<int64_t>(
              {6789, 1500, -50000}, DECIMAL(12, 2)), // 67.89, 15.00, -500.00
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(123'456'789'012), // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(555'000'000'000), // 55.5000000000
              },
              DECIMAL(38, 10)),
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(222'222'222'222), // 22.2222222222
                  static_cast<int128_t>(333'333'333'333), // 33.3333333333
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
                  9'999'999'999'999, // 99,999,999,999.99
                  -9'999'999'999'999, // -99,999,999,999.99
                                      // Mid-range values
                  123'45, // 1,23.45
                  -2'500, // -25.00
                  999'999, // 9,999.99
                  -1'000, // -10.00
                  0,
                  1, // 0.01
                  -1, // -0.01
              },
              DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {
                  // Near max/min for DECIMAL(38,10): +/- (10^28 - 1) with scale
                  // 10
                  max38p10,
                  -max38p10,
                  // Mid-range values
                  static_cast<int128_t>(123'456'789'012), // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(555'000'000'000), // 55.5000000000
                  static_cast<int128_t>(44'388'888'889), // 4.4388888889
                  static_cast<int128_t>(1), // 0.0000000001
                  static_cast<int128_t>(-1), // -0.0000000001
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
                  9'999'999'999'99, // 9,999,999,999.99 (near max for 12,2)
                  -9'999'999'999'99, // -9,999,999,999.99
                  123'45, // 1,23.45
                  -2'500, // -25.00
                  0,
              },
              DECIMAL(12, 2)),
          makeFlatVector<int64_t>(
              {
                  1, // 0.01
                  -1, // -0.01
                  9'999, // 99.99
                  -100, // -1.00
                  50, // 0.50
              },
              DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {
                  max38p10,
                  min38p10,
                  static_cast<int128_t>(123'456'789'012), // 12.3456789012
                  static_cast<int128_t>(-987'654'321'098), // -98.7654321098
                  static_cast<int128_t>(0),
              },
              DECIMAL(38, 10)),
          makeFlatVector<int128_t>(
              {
                  static_cast<int128_t>(1), // 0.0000000001
                  static_cast<int128_t>(-1), // -0.0000000001
                  static_cast<int128_t>(44'388'888'889), // 4.4388888889
                  static_cast<int128_t>(555'000'000'000), // 55.5000000000
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
              {9'999'999'999, 1'234'567'890, -2'000'000'000}, DECIMAL(10, 0)),
          makeFlatVector<int64_t>({9'999'999'999, -2, 4}, DECIMAL(10, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a * b AS prod"})
                  .planNode();

  const int128_t expected0 = static_cast<int128_t>(9'999'999'999LL) *
      static_cast<int128_t>(9'999'999'999LL);
  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<int128_t>(
          {expected0,
           static_cast<int128_t>(-2'469'135'780LL),
           static_cast<int128_t>(-8'000'000'000LL)},
          DECIMAL(20, 0))});

  // CPU (no cuDF adapter registered).
  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
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
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  // GPU (cuDF enabled).
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_TRUE(cpuResult->childAt(0)->type()->isLongDecimal());
  ASSERT_TRUE(gpuResult->childAt(0)->type()->isLongDecimal());
  facebook::velox::test::assertEqualVectors(expected, cpuResult);
  facebook::velox::test::assertEqualVectors(expected, gpuResult);
}

TEST_F(CudfDecimalTest, decimalAddDifferentScales) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 1)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a + b AS sum"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT a + b AS sum FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSubtractDifferentScales) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 1)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a - b AS diff"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT a - b AS diff FROM tmp");
}

TEST_F(CudfDecimalTest, decimalMultiplyDifferentScales) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 1)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a * b AS prod"})
                  .planNode();

  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<int128_t>(
          {
              static_cast<int128_t>(12345) * static_cast<int128_t>(10),
              static_cast<int128_t>(-2500) * static_cast<int128_t>(-25),
              static_cast<int128_t>(100) * static_cast<int128_t>(3),
          },
          DECIMAL(20, 3))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalCompareDecimalDecimal) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({120, -250, 10}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({110, -250, 30}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a = b AS eq",
                      "a != b AS neq",
                      "a < b AS lt",
                      "a <= b AS lte",
                      "a > b AS gt",
                      "a >= b AS gte",
                  })
                  .planNode();

  auto expected = makeRowVector(
      {"eq", "neq", "lt", "lte", "gt", "gte"},
      {
          makeNullableFlatVector<bool>({false, true, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, true, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, false, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, true, false}, BOOLEAN()),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalCompareWithLiteral) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>(
          {120, 110, 130, std::nullopt}, DECIMAL(10, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a = CAST('1.20' AS DECIMAL(10, 2)) AS eq_r",
                      "a != CAST('1.20' AS DECIMAL(10, 2)) AS neq_r",
                      "a < CAST('1.20' AS DECIMAL(10, 2)) AS lt_r",
                      "a <= CAST('1.20' AS DECIMAL(10, 2)) AS lte_r",
                      "a > CAST('1.20' AS DECIMAL(10, 2)) AS gt_r",
                      "a >= CAST('1.20' AS DECIMAL(10, 2)) AS gte_r",
                      "CAST('1.20' AS DECIMAL(10, 2)) = a AS eq_l",
                      "CAST('1.20' AS DECIMAL(10, 2)) != a AS neq_l",
                      "CAST('1.20' AS DECIMAL(10, 2)) < a AS lt_l",
                      "CAST('1.20' AS DECIMAL(10, 2)) <= a AS lte_l",
                      "CAST('1.20' AS DECIMAL(10, 2)) > a AS gt_l",
                      "CAST('1.20' AS DECIMAL(10, 2)) >= a AS gte_l",
                  })
                  .planNode();

  auto expected = makeRowVector(
      {"eq_r",
       "neq_r",
       "lt_r",
       "lte_r",
       "gt_r",
       "gte_r",
       "eq_l",
       "neq_l",
       "lt_l",
       "lte_l",
       "gt_l",
       "gte_l"},
      {
          makeNullableFlatVector<bool>(
              {true, false, false, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, true, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, true, false, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, true, false, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, false, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, false, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, false, false, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, true, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, false, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, false, true, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, true, false, std::nullopt}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, true, false, std::nullopt}, BOOLEAN()),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, DISABLED_decimalLogicalAndOrProject) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({100, 200, 300, 400, 500}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({250, 150, 350, 100, 500}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "(a > CAST('1.00' AS DECIMAL(10, 2)) "
                      "AND b < CAST('2.00' AS DECIMAL(10, 2))) AS and2",
                      "(a > CAST('1.00' AS DECIMAL(10, 2)) "
                      "AND b < CAST('3.00' AS DECIMAL(10, 2)) "
                      "AND a < CAST('4.00' AS DECIMAL(10, 2))) AS and3",
                      "(a < CAST('1.00' AS DECIMAL(10, 2)) "
                      "OR b > CAST('3.00' AS DECIMAL(10, 2)) "
                      "OR a = CAST('2.00' AS DECIMAL(10, 2))) AS or3",
                  })
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT "
          "(a > CAST('1.00' AS DECIMAL(10, 2)) "
          " AND b < CAST('2.00' AS DECIMAL(10, 2))) AS and2, "
          "(a > CAST('1.00' AS DECIMAL(10, 2)) "
          " AND b < CAST('3.00' AS DECIMAL(10, 2)) "
          " AND a < CAST('4.00' AS DECIMAL(10, 2))) AS and3, "
          "(a < CAST('1.00' AS DECIMAL(10, 2)) "
          " OR b > CAST('3.00' AS DECIMAL(10, 2)) "
          " OR a = CAST('2.00' AS DECIMAL(10, 2))) AS or3 "
          "FROM tmp");
}

TEST_F(CudfDecimalTest, DISABLED_decimalLogicalAndOrFilter) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({100, 200, 300, 400, 500}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({250, 150, 350, 100, 500}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  const std::string filter =
      "((a between CAST('1.50' AS DECIMAL(10, 2)) AND "
      "CAST('3.50' AS DECIMAL(10, 2)) "
      "AND b < CAST('3.00' AS DECIMAL(10, 2)) "
      "AND a > CAST('1.00' AS DECIMAL(10, 2))) "
      "OR a = CAST('4.00' AS DECIMAL(10, 2)) "
      "OR a = CAST('5.00' AS DECIMAL(10, 2)))";

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .filter(filter)
                  .project({"a", "b"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT a, b FROM tmp WHERE "
          "((a between CAST('1.50' AS DECIMAL(10, 2)) AND "
          "CAST('3.50' AS DECIMAL(10, 2)) "
          "AND b < CAST('3.00' AS DECIMAL(10, 2)) "
          "AND a > CAST('1.00' AS DECIMAL(10, 2))) "
          "OR a = CAST('4.00' AS DECIMAL(10, 2)) "
          "OR a = CAST('5.00' AS DECIMAL(10, 2)))");
}

TEST_F(CudfDecimalTest, decimalBinaryNullPropagation) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 300, std::nullopt}, DECIMAL(10, 2)),
          makeNullableFlatVector<int64_t>(
              {200, 200, std::nullopt, std::nullopt}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a + b AS sum",
                      "a / b AS div",
                      "a = b AS eq",
                  })
                  .planNode();

  auto expected = makeRowVector(
      {"sum", "div", "eq"},
      {
          makeNullableFlatVector<int64_t>(
              {300, std::nullopt, std::nullopt, std::nullopt}, DECIMAL(11, 2)),
          makeNullableFlatVector<int64_t>(
              {50, std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
          makeNullableFlatVector<bool>(
              {false, std::nullopt, std::nullopt, std::nullopt}, BOOLEAN()),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalMultiplyDoubleCast) {
  auto rowType = ROW({
      {"d", DECIMAL(10, 2)},
      {"x", DOUBLE()},
  });

  auto input = makeRowVector(
      {"d", "x"},
      {
          makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2)),
          makeFlatVector<double>({2.0, -4.0, 0.0}),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto expected =
      makeRowVector({"prod"}, {makeFlatVector<double>({2.5, 10.0, 0.0})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as double) * x AS prod"})
                    .planNode();
    auto result =
        facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(
            pool());
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  runAndAssert(true);
  runAndAssert(false);
}

TEST_F(CudfDecimalTest, decimalMultiplyDoubleCastRight) {
  auto rowType = ROW({
      {"d", DECIMAL(10, 2)},
      {"x", DOUBLE()},
  });

  auto input = makeRowVector(
      {"d", "x"},
      {
          makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2)),
          makeFlatVector<double>({2.0, -4.0, 0.0}),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto expected =
      makeRowVector({"prod"}, {makeFlatVector<double>({2.5, 10.0, 0.0})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"x * cast(d as double) AS prod"})
                    .planNode();
    auto result =
        facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(
            pool());
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  runAndAssert(true);
  runAndAssert(false);
}

TEST_F(CudfDecimalTest, decimalAstRecursiveMixedScaleAdd) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 1)},
      {"x", DOUBLE()},
  });

  auto input = makeRowVector(
      {"a", "b", "x"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
          makeFlatVector<double>({2.0, -4.0, 0.0}),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto expected =
      makeRowVector({"prod"}, {makeFlatVector<double>({126.45, -31.5, 1.3})});

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"cast(a + b as double) + x AS prod"})
                  .planNode();

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalCastToDoubleProjection) {
  auto rowType = ROW({
      {"d", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto expected =
      makeRowVector({"d_double"}, {makeFlatVector<double>({1.25, -2.5, 0.5})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as double) AS d_double"})
                    .planNode();
    auto result =
        facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(
            pool());
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  runAndAssert(true);
  runAndAssert(false);
}

TEST_F(CudfDecimalTest, decimalCastToRealProjection) {
  auto rowType = ROW({
      {"d", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto expected =
      makeRowVector({"d_real"}, {makeFlatVector<float>({1.25f, -2.5f, 0.5f})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as real) AS d_real"})
                    .planNode();
    auto result =
        facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(
            pool());
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  runAndAssert(true);
  runAndAssert(false);
}

// Parameterized test that verifies cudf decimal binary ops match CPU results
// across different precision/scale combinations
struct DecimalBinaryParam {
  std::string name;
  std::string op;
  TypePtr aType;
  TypePtr bType;
};

class CudfDecimalBinaryTest
    : public CudfDecimalTest,
      public testing::WithParamInterface<DecimalBinaryParam> {};

TEST_P(CudfDecimalBinaryTest, cpuGpuMatch) {
  const auto& param = GetParam();

  VectorPtr aVec;
  VectorPtr bVec;
  if (param.aType->equivalent(*DECIMAL(20, 4)) ||
      param.aType->equivalent(*DECIMAL(20, 3))) {
    aVec = makeFlatVector<int128_t>({12345, -2500, 100}, param.aType);
    bVec = makeFlatVector<int128_t>({300, -25, 400}, param.bType);
  } else {
    aVec = makeFlatVector<int64_t>({12345, -2500, 100}, param.aType);
    bVec = makeFlatVector<int64_t>({300, -25, 400}, param.bType);
  }
  auto input = makeRowVector({"a", "b"}, {aVec, bVec});
  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({param.op + " AS result"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalBinaryOps,
    CudfDecimalBinaryTest,
    testing::Values(
        // Same scale, DECIMAL64.
        DecimalBinaryParam{
            "add_same_scale",
            "a + b",
            DECIMAL(10, 2),
            DECIMAL(10, 2)},
        DecimalBinaryParam{
            "sub_same_scale",
            "a - b",
            DECIMAL(10, 2),
            DECIMAL(10, 2)},
        DecimalBinaryParam{
            "mul_same_scale",
            "a * b",
            DECIMAL(10, 2),
            DECIMAL(10, 2)},
        DecimalBinaryParam{
            "div_same_scale",
            "a / b",
            DECIMAL(10, 2),
            DECIMAL(10, 2)},
        DecimalBinaryParam{
            "mod_same_scale",
            "a % b",
            DECIMAL(10, 2),
            DECIMAL(10, 2)},
        // Different scales, DECIMAL64.
        DecimalBinaryParam{
            "add_diff_scale",
            "a + b",
            DECIMAL(10, 2),
            DECIMAL(10, 1)},
        DecimalBinaryParam{
            "sub_diff_scale",
            "a - b",
            DECIMAL(10, 2),
            DECIMAL(10, 1)},
        DecimalBinaryParam{
            "mul_diff_scale",
            "a * b",
            DECIMAL(10, 2),
            DECIMAL(10, 1)},
        DecimalBinaryParam{
            "div_diff_scale",
            "a / b",
            DECIMAL(10, 2),
            DECIMAL(10, 1)},
        DecimalBinaryParam{
            "mod_diff_scale",
            "a % b",
            DECIMAL(10, 2),
            DECIMAL(10, 1)},
        // DECIMAL64 inputs, result overflows to DECIMAL128.
        DecimalBinaryParam{
            "add_64to128",
            "a + b",
            DECIMAL(10, 2),
            DECIMAL(10, 3)},
        DecimalBinaryParam{
            "sub_64to128",
            "a - b",
            DECIMAL(10, 2),
            DECIMAL(10, 3)},
        DecimalBinaryParam{
            "mul_64to128",
            "a * b",
            DECIMAL(10, 2),
            DECIMAL(10, 3)},
        DecimalBinaryParam{
            "div_64to128",
            "a / b",
            DECIMAL(10, 2),
            DECIMAL(10, 3)},
        DecimalBinaryParam{
            "mod_64to128",
            "a % b",
            DECIMAL(10, 2),
            DECIMAL(10, 3)},
        // Native DECIMAL128 inputs.
        DecimalBinaryParam{"add_128", "a + b", DECIMAL(20, 4), DECIMAL(20, 4)},
        DecimalBinaryParam{"sub_128", "a - b", DECIMAL(20, 4), DECIMAL(20, 4)},
        DecimalBinaryParam{"mul_128", "a * b", DECIMAL(20, 4), DECIMAL(20, 4)},
        DecimalBinaryParam{"div_128", "a / b", DECIMAL(20, 4), DECIMAL(20, 4)},
        DecimalBinaryParam{"mod_128", "a % b", DECIMAL(20, 4), DECIMAL(20, 4)},
        // DECIMAL128 with different scales.
        DecimalBinaryParam{
            "add_128_diff",
            "a + b",
            DECIMAL(20, 4),
            DECIMAL(20, 3)},
        DecimalBinaryParam{
            "sub_128_diff",
            "a - b",
            DECIMAL(20, 4),
            DECIMAL(20, 3)},
        DecimalBinaryParam{
            "mul_128_diff",
            "a * b",
            DECIMAL(20, 4),
            DECIMAL(20, 3)},
        DecimalBinaryParam{
            "div_128_diff",
            "a / b",
            DECIMAL(20, 4),
            DECIMAL(20, 3)},
        DecimalBinaryParam{
            "mod_128_diff",
            "a % b",
            DECIMAL(20, 4),
            DECIMAL(20, 3)}),
    [](const testing::TestParamInfo<DecimalBinaryParam>& info) {
      return info.param.name;
    });

TEST_F(CudfDecimalTest, decimalDivideRounds) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({200, 100, -200}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({300, 300, 300}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a / b AS div"})
                  .planNode();

  auto computeDiv = [](int64_t a, int64_t b) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, __int128_t>(
            out,
            static_cast<__int128_t>(a),
            static_cast<__int128_t>(b),
            false,
            2,
            0);
    return static_cast<int64_t>(out);
  };

  auto expected = makeRowVector(
      {"div"},
      {makeFlatVector<int64_t>(
          {computeDiv(200, 300), computeDiv(100, 300), computeDiv(-200, 300)},
          DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalDivideByZero) {
  auto rowType = ROW({
      {"a", DECIMAL(10, 2)},
      {"b", DECIMAL(10, 2)},
  });

  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({0}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a / b AS div"})
                  .planNode();

  VELOX_ASSERT_USER_THROW(
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool()),
      "Division by zero");
}

TEST_F(CudfDecimalTest, decimalModulo) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({700, 100, -700, 0, 500}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({300, 300, 300, 300, 200}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a % b AS mod"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalModuloDifferentScales) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a % b AS mod"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalSubtractPromotesToLong) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>(
              {999'999'999'999'999'999LL,
               -900'000'000'000'000'000LL,
               123'456'789'012'345'678LL},
              DECIMAL(18, 0)),
          makeFlatVector<int64_t>(
              {-999'999'999'999'999'999LL,
               900'000'000'000'000'000LL,
               123'456'789'012'345'678LL},
              DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a - b AS diff"})
                  .planNode();

  auto expected = makeRowVector(
      {"diff"},
      {makeFlatVector<int128_t>(
          {static_cast<int128_t>(1'999'999'999'999'999'998LL),
           static_cast<int128_t>(-1'800'000'000'000'000'000LL),
           static_cast<int128_t>(0)},
          DECIMAL(19, 0))});

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_TRUE(cpuResult->childAt(0)->type()->isLongDecimal());
  ASSERT_TRUE(gpuResult->childAt(0)->type()->isLongDecimal());
  facebook::velox::test::assertEqualVectors(expected, cpuResult);
  facebook::velox::test::assertEqualVectors(expected, gpuResult);
}

TEST_F(CudfDecimalTest, decimalDivideDifferentScales) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({12345, -2500, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({10, -25, 3}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a / b AS div"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalModuloNullPropagation) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 300, std::nullopt}, DECIMAL(10, 2)),
          makeNullableFlatVector<int64_t>(
              {200, 200, std::nullopt, std::nullopt}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a % b AS mod",
                      "a - b AS diff",
                      "a * b AS prod",
                  })
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalArithmeticWithScalarRight) {
  auto input = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({500, -250, 100}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a + CAST('1.00' AS DECIMAL(10, 2)) AS add_r",
                      "a - CAST('1.00' AS DECIMAL(10, 2)) AS sub_r",
                      "a * CAST('2.00' AS DECIMAL(10, 2)) AS mul_r",
                      "a / CAST('2.00' AS DECIMAL(10, 2)) AS div_r",
                      "a % CAST('3.00' AS DECIMAL(10, 2)) AS mod_r",
                  })
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalArithmeticWithScalarLeft) {
  auto input = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({500, -250, 100}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "CAST('10.00' AS DECIMAL(10, 2)) + a AS add_l",
                      "CAST('10.00' AS DECIMAL(10, 2)) - a AS sub_l",
                      "CAST('2.00' AS DECIMAL(10, 2)) * a AS mul_l",
                      "CAST('10.00' AS DECIMAL(10, 2)) / a AS div_l",
                      "CAST('10.00' AS DECIMAL(10, 2)) % a AS mod_l",
                  })
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalDivideNullScalar) {
  auto input = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({500, -250, 100}, DECIMAL(10, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a / CAST(NULL AS DECIMAL(10, 2)) AS div_null_r",
                      "CAST(NULL AS DECIMAL(10, 2)) / a AS div_null_l",
                  })
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();

  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());

  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

// Velox comparison functions require both arguments to have the same type
// (Generic<T1>, Generic<T1>), so raw different-scale comparisons fail at the
// type-resolution stage before cuDF evaluation.
TEST_F(CudfDecimalTest, DISABLED_decimalCompareDifferentScales) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({120, 150, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({12, 10, 20}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a = b AS eq",
                      "a != b AS neq",
                      "a < b AS lt",
                      "a <= b AS lte",
                      "a > b AS gt",
                      "a >= b AS gte",
                  })
                  .planNode();

  // 1.20 == 1.2, 1.50 > 1.0, 1.00 < 2.0
  auto expected = makeRowVector(
      {"eq", "neq", "lt", "lte", "gt", "gte"},
      {
          makeNullableFlatVector<bool>({true, false, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, true, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, true, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, true, false}, BOOLEAN()),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalCompareDifferentScalesWithCast) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({120, 150, 100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({12, 10, 20}, DECIMAL(10, 1)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({
                      "a = CAST(b AS DECIMAL(10, 2)) AS eq",
                      "a != CAST(b AS DECIMAL(10, 2)) AS neq",
                      "a < CAST(b AS DECIMAL(10, 2)) AS lt",
                      "a <= CAST(b AS DECIMAL(10, 2)) AS lte",
                      "a > CAST(b AS DECIMAL(10, 2)) AS gt",
                      "a >= CAST(b AS DECIMAL(10, 2)) AS gte",
                  })
                  .planNode();

  // 1.20 == 1.2, 1.50 > 1.0, 1.00 < 2.0
  auto expected = makeRowVector(
      {"eq", "neq", "lt", "lte", "gt", "gte"},
      {
          makeNullableFlatVector<bool>({true, false, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, true, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>({false, true, false}, BOOLEAN()),
          makeNullableFlatVector<bool>({true, true, false}, BOOLEAN()),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalGreatestLeastAllColumns) {
  auto input = makeRowVector(
      {"a", "b", "c"},
      {
          makeFlatVector<int64_t>({100, 500, -300}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({400, 200, -100}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({300, 300, -200}, DECIMAL(10, 2)),
      });
  std::vector<RowVectorPtr> vectors = {input};
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"greatest(a, b, c) AS g", "least(a, b, c) AS l"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalGreatestLeastMixed) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeFlatVector<int64_t>({100, 500, -300}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({400, 200, -100}, DECIMAL(10, 2)),
      });
  std::vector<RowVectorPtr> vectors = {input};
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project(
                      {"greatest(a, CAST('5.00' AS DECIMAL(10, 2)), b) AS g",
                       "least(a, CAST('0.00' AS DECIMAL(10, 2)), b) AS l"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalGreatestLeastWithNulls) {
  // The cuDF implementation uses NULL_MAX / NULL_MIN which skip nulls
  // (returning null only when all inputs are null). This differs from
  // Presto (which propagates null if any input is null), so we verify
  // against manually constructed expected results.
  auto input = makeRowVector(
      {"a", "b", "c"},
      {
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, std::nullopt}, DECIMAL(10, 2)),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, 200, std::nullopt}, DECIMAL(10, 2)),
          makeNullableFlatVector<int64_t>(
              {300, std::nullopt, std::nullopt}, DECIMAL(10, 2)),
      });
  std::vector<RowVectorPtr> vectors = {input};
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"greatest(a, b, c) AS g", "least(a, b, c) AS l"})
                  .planNode();

  auto expected = makeRowVector(
      {"g", "l"},
      {
          makeNullableFlatVector<int64_t>(
              {300, 200, std::nullopt}, DECIMAL(10, 2)),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt}, DECIMAL(10, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalBetween) {
  auto input = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>({-100, 50, 150, 300}, DECIMAL(10, 2))});
  std::vector<RowVectorPtr> vectors = {input};
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"a BETWEEN CAST('0.00' AS DECIMAL(10, 2)) "
                            "AND CAST('2.00' AS DECIMAL(10, 2)) AS result"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalCoalesceColumnWithLiteral) {
  auto input = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>(
          {100, std::nullopt, 300, std::nullopt, 500}, DECIMAL(10, 2))});
  std::vector<RowVectorPtr> vectors = {input};
  auto plan =
      exec::test::PlanBuilder()
          .values(vectors)
          .project({"coalesce(a, CAST('0.00' AS DECIMAL(10, 2))) AS result"})
          .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

// Disabled: a non-null literal in the first position makes the entire coalesce
// trivially constant. For simple types Velox folds this away before it reaches
// cuDF, but the CAST wrapper needed for decimal literals prevents that folding,
// causing the cuDF compiler to reject the expression.
TEST_F(CudfDecimalTest, DISABLED_decimalCoalesceLiteralFirst) {
  auto input = makeRowVector(
      {"a"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(10, 2))});
  std::vector<RowVectorPtr> vectors = {input};
  auto plan =
      exec::test::PlanBuilder()
          .values(vectors)
          .project({"coalesce(CAST('5.00' AS DECIMAL(10, 2)), a) AS result"})
          .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

TEST_F(CudfDecimalTest, decimalCoalesceStopsAtFirstLiteral) {
  auto input = makeRowVector(
      {"a", "b"},
      {
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 300, std::nullopt}, DECIMAL(10, 2)),
          makeFlatVector<int64_t>({900, 800, 700, 600}, DECIMAL(10, 2)),
      });
  std::vector<RowVectorPtr> vectors = {input};
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .project({"coalesce(a, CAST('1.00' AS DECIMAL(10, 2)), b) "
                            "AS result"})
                  .planNode();

  unregisterCudf();
  auto cpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  registerCudf();
  auto gpuResult =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(cpuResult, gpuResult);
}

} // namespace
} // namespace facebook::velox::cudf_velox
