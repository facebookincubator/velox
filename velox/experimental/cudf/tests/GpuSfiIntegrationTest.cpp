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
#include "velox/experimental/cudf/exec/GpuSfiExpression.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;

class GpuSfiIntegrationTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    functions::prestosql::registerAllScalarFunctions();
  }

  void SetUp() override {
    cudaSetDevice(0);
    pool_ = memory::memoryManager()->addLeafPool("", false);
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());

    CudfConfig::getInstance().gpuSfiExpressionEnabled = true;
    CudfConfig::getInstance().gpuSfiExpressionPriority = 200;
    CudfConfig::getInstance().memoryResource = "cuda";
    registerCudf();

    rowType_ = ROW({{"a", DOUBLE()}, {"b", DOUBLE()}, {"c", BIGINT()}});
    parse::registerTypeResolver();
  }

  void TearDown() override {
    unregisterFunctions();
    unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  std::unique_ptr<cudf::column> makeDoubleColumn(
      const std::vector<double>& data) {
    auto col = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        data.size(),
        cudf::mask_state::UNALLOCATED,
        stream_,
        mr());
    cudaMemcpy(
        col->mutable_view().data<double>(),
        data.data(),
        data.size() * sizeof(double),
        cudaMemcpyHostToDevice);
    return col;
  }

  std::unique_ptr<cudf::column> makeInt64Column(
      const std::vector<int64_t>& data) {
    auto col = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        data.size(),
        cudf::mask_state::UNALLOCATED,
        stream_,
        mr());
    cudaMemcpy(
        col->mutable_view().data<int64_t>(),
        data.data(),
        data.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice);
    return col;
  }

  std::vector<double> readDouble(const cudf::column_view& col) {
    std::vector<double> result(col.size());
    cudaMemcpy(
        result.data(),
        col.data<double>(),
        col.size() * sizeof(double),
        cudaMemcpyDeviceToHost);
    return result;
  }

  rmm::device_async_resource_ref mr() {
    return cudf::get_current_device_resource_ref();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
  rmm::cuda_stream_view stream_ = rmm::cuda_stream_default;
};

TEST_F(GpuSfiIntegrationTest, gpuSfiIsHighestPriority) {
  auto expr = compileExecExpr("a + b", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);
  auto* sfiExpr = dynamic_cast<GpuSfiExpression*>(cudfExpr.get());
  ASSERT_NE(sfiExpr, nullptr)
      << "GPU SFI should be selected as highest priority evaluator for a + b";
}

TEST_F(GpuSfiIntegrationTest, gpuSfiClaimsAllExpressions) {
  EXPECT_TRUE(GpuSfiExpression::canEvaluate(nullptr));

  auto expr = compileExecExpr("lower(cast(c as varchar))", rowType_, execCtx_.get());
  EXPECT_TRUE(GpuSfiExpression::canEvaluate(expr));
}

TEST_F(GpuSfiIntegrationTest, endToEndArithmetic) {
  auto expr = compileExecExpr("a + b", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);

  auto colA = makeDoubleColumn({1.0, 2.0, 3.0});
  auto colB = makeDoubleColumn({10.0, 20.0, 30.0});
  auto colC = makeInt64Column({100, 200, 300});

  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  auto result = cudfExpr->eval(inputViews, stream_, mr());
  auto view = asView(result);
  auto hostResult = readDouble(view);

  EXPECT_DOUBLE_EQ(11.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(22.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(33.0, hostResult[2]);
}

TEST_F(GpuSfiIntegrationTest, endToEndCompound) {
  // a * (1.0 - b)  -- TPC-H revenue pattern
  auto expr = compileExecExpr(
      "a * (1.0e0 - b)", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);

  auto colA = makeDoubleColumn({100.0, 200.0, 300.0});
  auto colB = makeDoubleColumn({0.05, 0.10, 0.0});
  auto colC = makeInt64Column({0, 0, 0});

  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  auto result = cudfExpr->eval(inputViews, stream_, mr());
  auto view = asView(result);
  auto hostResult = readDouble(view);

  EXPECT_DOUBLE_EQ(95.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(180.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(300.0, hostResult[2]);
}

TEST_F(GpuSfiIntegrationTest, cpuFallbackForUnsupportedExpr) {
  // "coalesce" has no GPU SFI implementation, so it should fall back to
  // CPU Velox's expression engine via D->H->eval->H->D.
  auto expr = compileExecExpr(
      "coalesce(a, b)", rowType_, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, rowType_);

  auto colA = makeDoubleColumn({1.0, 2.0, 3.0});
  auto colB = makeDoubleColumn({10.0, 20.0, 30.0});
  auto colC = makeInt64Column({100, 200, 300});

  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  auto result = cudfExpr->eval(inputViews, stream_, mr());
  auto view = asView(result);
  auto hostResult = readDouble(view);

  // coalesce(a, b) returns a when a is non-null.
  EXPECT_DOUBLE_EQ(1.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(2.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(3.0, hostResult[2]);
}

TEST_F(GpuSfiIntegrationTest, cpuFallbackMixedGpuAndCpu) {
  // "a + b" runs on GPU, but the overall expression includes a cast
  // that may need CPU fallback. Tests mixed GPU/CPU expression trees.
  auto mixedType = ROW({{"a", DOUBLE()}, {"b", DOUBLE()}, {"c", BIGINT()}});
  // "if(c > 150, a + b, 0.0)" -- 'if' likely falls back, but 'a + b' can
  // run on GPU. The entire expression should produce correct results
  // regardless of where each sub-tree runs.
  auto expr = compileExecExpr(
      "if(c > 150, a + b, 0.0e0)", mixedType, execCtx_.get());
  auto cudfExpr = createCudfExpression(expr, mixedType);

  auto colA = makeDoubleColumn({1.0, 2.0, 3.0});
  auto colB = makeDoubleColumn({10.0, 20.0, 30.0});
  auto colC = makeInt64Column({100, 200, 300});

  std::vector<cudf::column_view> inputViews = {
      colA->view(), colB->view(), colC->view()};

  auto result = cudfExpr->eval(inputViews, stream_, mr());
  auto view = asView(result);
  auto hostResult = readDouble(view);

  // c=100 (<=150): 0.0, c=200 (>150): 2+20=22, c=300 (>150): 3+30=33
  EXPECT_DOUBLE_EQ(0.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(22.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(33.0, hostResult[2]);
}

TEST_F(GpuSfiIntegrationTest, configKeys) {
  EXPECT_TRUE(CudfConfig::getInstance().gpuSfiExpressionEnabled);
  EXPECT_EQ(200, CudfConfig::getInstance().gpuSfiExpressionPriority);

  EXPECT_STREQ(
      "cudf.gpu_sfi_expression_enabled",
      CudfConfig::kCudfGpuSfiExpressionEnabled);
  EXPECT_STREQ(
      "cudf.gpu_sfi_expression_priority",
      CudfConfig::kCudfGpuSfiExpressionPriority);
}
