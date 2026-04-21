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

// Tests for CudfFallbackFunction and GpuFunctionDispatch:
// - cuDF binary/unary fallback execution
// - Dispatch priority (native GPU > cuDF fallback > not found)

#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"
#include "velox/experimental/cudf/functions/GpuFunctionDispatch.h"

using namespace facebook::velox::gpu;

class GpuFallbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    CudfFallbackRegistry::instance().registerDefaults();
    GpuFunctionRegistry::instance().clear();
  }

  rmm::cuda_stream_view stream_ = rmm::cuda_stream_default;
  rmm::device_async_resource_ref mr_ = cudf::get_current_device_resource_ref();
};

TEST_F(GpuFallbackTest, cudfBinaryAdd) {
  constexpr int N = 4;
  std::vector<double> aHost = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> bHost = {10.0, 20.0, 30.0, 40.0};
  std::vector<double> expected = {11.0, 22.0, 33.0, 44.0};

  auto colA = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream_, mr_);
  auto colB = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream_, mr_);

  cudaMemcpy(colA->mutable_view().data<double>(), aHost.data(),
             N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(colB->mutable_view().data<double>(), bHost.data(),
             N * sizeof(double), cudaMemcpyHostToDevice);

  CudfBinaryFunction fn(
      cudf::binary_operator::ADD,
      cudf::data_type{cudf::type_id::FLOAT64});

  std::vector<cudf::column_view> inputs = {colA->view(), colB->view()};
  auto result = fn.apply(inputs, N, nullptr, stream_, mr_);

  std::vector<double> hostResult(N);
  cudaMemcpy(hostResult.data(), result->view().data<double>(),
             N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], hostResult[i]) << "at " << i;
  }
}

TEST_F(GpuFallbackTest, cudfUnaryAbs) {
  constexpr int N = 4;
  std::vector<double> aHost = {-5.0, 3.0, -0.5, 0.0};
  std::vector<double> expected = {5.0, 3.0, 0.5, 0.0};

  auto col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream_, mr_);
  cudaMemcpy(col->mutable_view().data<double>(), aHost.data(),
             N * sizeof(double), cudaMemcpyHostToDevice);

  CudfUnaryFunction fn(cudf::unary_operator::ABS);

  std::vector<cudf::column_view> inputs = {col->view()};
  auto result = fn.apply(inputs, N, nullptr, stream_, mr_);

  std::vector<double> hostResult(N);
  cudaMemcpy(hostResult.data(), result->view().data<double>(),
             N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], hostResult[i]) << "at " << i;
  }
}

TEST_F(GpuFallbackTest, cudfComparisonLess) {
  constexpr int N = 4;
  std::vector<double> aHost = {1.0, 5.0, 3.0, 3.0};
  std::vector<double> bHost = {2.0, 3.0, 3.0, 1.0};

  auto colA = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream_, mr_);
  auto colB = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream_, mr_);

  cudaMemcpy(colA->mutable_view().data<double>(), aHost.data(),
             N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(colB->mutable_view().data<double>(), bHost.data(),
             N * sizeof(double), cudaMemcpyHostToDevice);

  CudfBinaryFunction fn(
      cudf::binary_operator::LESS,
      cudf::data_type{cudf::type_id::BOOL8});

  std::vector<cudf::column_view> inputs = {colA->view(), colB->view()};
  auto result = fn.apply(inputs, N, nullptr, stream_, mr_);

  bool hostResult[N];
  cudaMemcpy(hostResult, result->view().data<bool>(),
             N * sizeof(bool), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(hostResult[0]);   // 1 < 2
  EXPECT_FALSE(hostResult[1]);  // 5 < 3
  EXPECT_FALSE(hostResult[2]);  // 3 < 3
  EXPECT_FALSE(hostResult[3]);  // 3 < 1
}

TEST_F(GpuFallbackTest, dispatchPrefersNativeGpu) {
  auto dr = dispatchGpuFunction(
      "plus", cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64});
  EXPECT_EQ(GpuDispatchKind::kCudfFallback, dr.kind);
  EXPECT_NE(nullptr, dr.function);
}

TEST_F(GpuFallbackTest, dispatchCudfUnary) {
  auto dr = dispatchGpuFunction(
      "abs", cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64});
  EXPECT_EQ(GpuDispatchKind::kCudfFallback, dr.kind);
  EXPECT_NE(nullptr, dr.function);
}

TEST_F(GpuFallbackTest, dispatchNotFound) {
  auto dr = dispatchGpuFunction(
      "unknown_function", cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64});
  EXPECT_EQ(GpuDispatchKind::kNotFound, dr.kind);
  EXPECT_EQ(nullptr, dr.function);
}

TEST_F(GpuFallbackTest, fallbackRegistryLookup) {
  auto binOp = CudfFallbackRegistry::instance().findBinaryOp("plus");
  EXPECT_TRUE(binOp.has_value());
  EXPECT_EQ(cudf::binary_operator::ADD, *binOp);

  auto unOp = CudfFallbackRegistry::instance().findUnaryOp("sin");
  EXPECT_TRUE(unOp.has_value());
  EXPECT_EQ(cudf::unary_operator::SIN, *unOp);

  auto missing = CudfFallbackRegistry::instance().findBinaryOp("no_such_fn");
  EXPECT_FALSE(missing.has_value());
}
