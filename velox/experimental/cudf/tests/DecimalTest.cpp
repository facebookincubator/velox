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
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>
#include <optional>
#include <type_traits>

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

constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

std::pair<rmm::device_buffer, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    rmm::cuda_stream_view stream) {
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {rmm::device_buffer{}, 0};
  }
  auto maskBytes = cudf::bitmask_allocation_size_bytes(numBits);
  auto numWords = maskBytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      auto word = i / kBitsPerWord;
      auto bit = i % kBitsPerWord;
      host[word] |= (cudf::bitmask_type{1} << bit);
    } else {
      ++nullCount;
    }
  }
  rmm::device_buffer mask(maskBytes, stream);
  if (!host.empty()) {
    auto status = cudaMemcpyAsync(
        mask.data(),
        host.data(),
        host.size() * sizeof(cudf::bitmask_type),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.synchronize();
  }
  return {std::move(mask), nullCount};
}

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    rmm::cuda_stream_view stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    auto status = cudaMemcpyAsync(
        col->mutable_view().data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.value());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.synchronize();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

template <typename T>
std::unique_ptr<cudf::column> makeDecimalColumn(
    const std::vector<T>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    rmm::cuda_stream_view stream) {
  cudf::type_id typeId =
      std::is_same_v<T, int64_t> ? cudf::type_id::DECIMAL64
                                 : cudf::type_id::DECIMAL128;
  cudf::data_type type{typeId, -scale};
  return makeFixedWidthColumn(type, values, valid, stream);
}

std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    const std::vector<bool>* valid,
    rmm::cuda_stream_view stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, valid, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    rmm::cuda_stream_view stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.data<T>(),
      view.size() * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.value());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.synchronize();
  return host;
}

std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    rmm::cuda_stream_view stream) {
  auto numWords = cudf::num_bitmask_words(view.size());
  std::vector<cudf::bitmask_type> host(numWords, 0);
  if (!view.nullable() || numWords == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.null_mask(),
      host.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.value());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.synchronize();
  return host;
}

bool isValidAt(const std::vector<cudf::bitmask_type>& mask, size_t idx) {
  if (mask.empty()) {
    return true;
  }
  auto word = idx / kBitsPerWord;
  auto bit = idx % kBitsPerWord;
  return (mask[word] >> bit) & 1;
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
          makeNullableFlatVector<bool>(
              {false, true, false}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, false, true}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {false, true, true}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, false, false}, BOOLEAN()),
          makeNullableFlatVector<bool>(
              {true, true, false}, BOOLEAN()),
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

  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<double>({2.5, 10.0, 0.0})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
  auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as double) * x AS prod"})
                    .planNode();
    auto result = facebook::velox::exec::test::AssertQueryBuilder(plan)
                      .copyResults(pool());
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

  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<double>({2.5, 10.0, 0.0})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
  auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"x * cast(d as double) AS prod"})
                    .planNode();
    auto result = facebook::velox::exec::test::AssertQueryBuilder(plan)
                      .copyResults(pool());
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

  auto expected = makeRowVector(
      {"prod"},
      {makeFlatVector<double>({126.45, -31.5, 1.3})});

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
      {"d"},
      {makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto expected = makeRowVector(
      {"d_double"},
      {makeFlatVector<double>({1.25, -2.5, 0.5})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as double) AS d_double"})
                    .planNode();
    auto result = facebook::velox::exec::test::AssertQueryBuilder(plan)
                      .copyResults(pool());
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
      {"d"},
      {makeFlatVector<int64_t>({125, -250, 50}, DECIMAL(10, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto expected = makeRowVector(
      {"d_real"},
      {makeFlatVector<float>({1.25f, -2.5f, 0.5f})});

  auto runAndAssert = [&](bool useCudf) {
    if (!useCudf) {
      unregisterCudf();
    }
    auto plan = exec::test::PlanBuilder()
                    .values(vectors)
                    .project({"cast(d as real) AS d_real"})
                    .planNode();
    auto result = facebook::velox::exec::test::AssertQueryBuilder(plan)
                      .copyResults(pool());
    facebook::velox::test::assertEqualVectors(expected, result);
  };

  runAndAssert(true);
  runAndAssert(false);
}

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
    facebook::velox::DecimalUtil::divideWithRoundUp<
        __int128_t,
        __int128_t,
        __int128_t>(out, static_cast<__int128_t>(a), static_cast<__int128_t>(b), false, 2, 0);
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

TEST_F(CudfDecimalTest, DISABLED_decimalAvgAndSumTimesDouble) {
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

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"},
      {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

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

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal64) {
  auto stream = cudf::get_default_stream();
  std::vector<int64_t> sums = {100, -200, 300};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream);
  auto sumOnly =
      deserializeDecimalSumState(stateCol->view(), 2, stream);

  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumOnly->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumOnly->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal128) {
  auto stream = cudf::get_default_stream();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, false, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream);
  auto sumOnly =
      deserializeDecimalSumState(stateCol->view(), 3, stream);

  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumOnly->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumOnly->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64) {
  auto stream = cudf::get_default_stream();
  std::vector<int64_t> sums = {100, 105, 250, -125};
  std::vector<int64_t> counts = {4, 2, 0, 2};
  std::vector<bool> sumValid = {true, true, true, true};
  std::vector<bool> countValid = {true, false, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::divideWithRoundUp<
        __int128_t,
        __int128_t,
        int64_t>(out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128) {
  auto stream = cudf::get_default_stream();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {3, 2, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol =
      computeDecimalAverage(sumCol->view(), countCol->view(), stream);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::divideWithRoundUp<
        __int128_t,
        __int128_t,
        int64_t>(out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal64) {
  auto stream = cudf::get_default_stream();
  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto decoded =
      deserializeDecimalSumStateWithCount(stateCol->view(), 2, stream);
  auto outSumView = decoded.sum->view();
  auto outCountView = decoded.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal128) {
  auto stream = cudf::get_default_stream();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto decoded =
      deserializeDecimalSumStateWithCount(stateCol->view(), 3, stream);
  auto outSumView = decoded.sum->view();
  auto outCountView = decoded.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTrip) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto stream = cudf::get_default_stream();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARCHAR);

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARCHAR())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

} // namespace
} // namespace facebook::velox::cudf_velox
