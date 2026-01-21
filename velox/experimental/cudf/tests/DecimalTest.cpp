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

} // namespace
} // namespace facebook::velox::cudf_velox
