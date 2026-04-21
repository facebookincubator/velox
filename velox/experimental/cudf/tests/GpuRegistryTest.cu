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

// Tests for GPU Function Registry: registration, lookup, and end-to-end
// execution through the GpuVectorFunction interface using cuDF columns.

#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include "velox/experimental/cudf/functions/GpuSimpleFunction.cuh"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Comparisons.h"

using namespace facebook::velox::gpu;

class GpuRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    GpuFunctionRegistry::instance().clear();
  }

  void TearDown() override {
    GpuFunctionRegistry::instance().clear();
  }
};

TEST_F(GpuRegistryTest, registerAndLookup) {
  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  registerGpuFunction<PlusFn, double, double, double>("plus");
  registerGpuFunction<PlusFn, int64_t, int64_t, int64_t>("plus");

  EXPECT_EQ(2, GpuFunctionRegistry::instance().size());

  GpuFunctionKey keyDouble{
      "plus",
      cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};
  EXPECT_TRUE(GpuFunctionRegistry::instance().hasFunction(keyDouble));

  GpuFunctionKey keyInt{
      "plus",
      cudf::type_id::INT64,
      {cudf::type_id::INT64, cudf::type_id::INT64}};
  EXPECT_TRUE(GpuFunctionRegistry::instance().hasFunction(keyInt));

  GpuFunctionKey keyMissing{
      "plus",
      cudf::type_id::INT32,
      {cudf::type_id::INT32, cudf::type_id::INT32}};
  EXPECT_FALSE(GpuFunctionRegistry::instance().hasFunction(keyMissing));
}

TEST_F(GpuRegistryTest, resolveAndExecute) {
  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  registerGpuFunction<PlusFn, double, double, double>("plus");

  GpuFunctionKey key{
      "plus",
      cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};

  auto* fn = GpuFunctionRegistry::instance().resolveFunction(key);
  ASSERT_NE(nullptr, fn);

  auto stream = rmm::cuda_stream_default;
  auto mr = cudf::get_current_device_resource_ref();

  constexpr int N = 4;
  std::vector<double> aHost = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> bHost = {10.0, 20.0, 30.0, 40.0};
  std::vector<double> expected = {11.0, 22.0, 33.0, 44.0};

  auto colA = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream, mr);
  auto colB = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream, mr);

  cudaMemcpy(
      colA->mutable_view().data<double>(), aHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
      colB->mutable_view().data<double>(), bHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);

  std::vector<cudf::column_view> inputs = {colA->view(), colB->view()};
  auto result = fn->apply(inputs, N, nullptr, stream, mr);

  std::vector<double> hostResult(N);
  cudaMemcpy(
      hostResult.data(), result->view().data<double>(),
      N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(expected[i], hostResult[i]) << "at " << i;
  }
}

TEST_F(GpuRegistryTest, executeWithNulls) {
  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  registerGpuFunction<PlusFn, double, double, double>("plus");

  GpuFunctionKey key{
      "plus",
      cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};
  auto* fn = GpuFunctionRegistry::instance().resolveFunction(key);
  ASSERT_NE(nullptr, fn);

  auto stream = rmm::cuda_stream_default;
  auto mr = cudf::get_current_device_resource_ref();
  constexpr int N = 4;

  std::vector<double> aHost = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> bHost = {10.0, 20.0, 30.0, 40.0};

  auto colA = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::ALL_VALID, stream, mr);
  auto colB = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::ALL_VALID, stream, mr);

  cudaMemcpy(
      colA->mutable_view().data<double>(), aHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
      colB->mutable_view().data<double>(), bHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);

  // Set row 1 of colA as null
  cudf::bitmask_type aMask = 0b00001101; // rows 0,2,3 valid; row 1 null
  cudaMemcpy(
      colA->mutable_view().null_mask(), &aMask,
      sizeof(cudf::bitmask_type), cudaMemcpyHostToDevice);
  colA->set_null_count(1);

  std::vector<cudf::column_view> inputs = {colA->view(), colB->view()};
  auto result = fn->apply(inputs, N, nullptr, stream, mr);

  std::vector<double> hostResult(N);
  cudaMemcpy(
      hostResult.data(), result->view().data<double>(),
      N * sizeof(double), cudaMemcpyDeviceToHost);

  EXPECT_DOUBLE_EQ(11.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(33.0, hostResult[2]);
  EXPECT_DOUBLE_EQ(44.0, hostResult[3]);

  auto resultMask = result->view().null_mask();
  cudf::bitmask_type hostMask;
  cudaMemcpy(&hostMask, resultMask, sizeof(cudf::bitmask_type),
             cudaMemcpyDeviceToHost);
  EXPECT_FALSE(cudf::bit_is_set(&hostMask, 1));
}

TEST_F(GpuRegistryTest, multipleFunctions) {
  using PlusFn = facebook::velox::functions::PlusFunction<GpuExec>;
  using MinusFn = facebook::velox::functions::MinusFunction<GpuExec>;
  using LtFn = facebook::velox::functions::LtFunction<GpuExec>;

  registerGpuFunction<PlusFn, double, double, double>("plus");
  registerGpuFunction<MinusFn, double, double, double>("minus");
  registerGpuFunction<LtFn, bool, double, double>("lt");

  EXPECT_EQ(3, GpuFunctionRegistry::instance().size());

  auto stream = rmm::cuda_stream_default;
  auto mr = cudf::get_current_device_resource_ref();
  constexpr int N = 2;

  std::vector<double> aHost = {10.0, 5.0};
  std::vector<double> bHost = {3.0, 8.0};

  auto colA = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream, mr);
  auto colB = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::FLOAT64}, N,
      cudf::mask_state::UNALLOCATED, stream, mr);

  cudaMemcpy(
      colA->mutable_view().data<double>(), aHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
      colB->mutable_view().data<double>(), bHost.data(),
      N * sizeof(double), cudaMemcpyHostToDevice);

  std::vector<cudf::column_view> inputs = {colA->view(), colB->view()};

  {
    GpuFunctionKey key{"minus", cudf::type_id::FLOAT64,
                       {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};
    auto* fn = GpuFunctionRegistry::instance().resolveFunction(key);
    ASSERT_NE(nullptr, fn);
    auto result = fn->apply(inputs, N, nullptr, stream, mr);
    std::vector<double> hr(N);
    cudaMemcpy(hr.data(), result->view().data<double>(),
               N * sizeof(double), cudaMemcpyDeviceToHost);
    EXPECT_DOUBLE_EQ(7.0, hr[0]);
    EXPECT_DOUBLE_EQ(-3.0, hr[1]);
  }
  {
    GpuFunctionKey key{"lt", cudf::type_id::BOOL8,
                       {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};
    auto* fn = GpuFunctionRegistry::instance().resolveFunction(key);
    ASSERT_NE(nullptr, fn);
    auto result = fn->apply(inputs, N, nullptr, stream, mr);
    bool hr[N];
    cudaMemcpy(hr, result->view().data<bool>(),
               N * sizeof(bool), cudaMemcpyDeviceToHost);
    EXPECT_FALSE(hr[0]); // 10 < 3 = false
    EXPECT_TRUE(hr[1]);  // 5 < 8 = true
  }
}
