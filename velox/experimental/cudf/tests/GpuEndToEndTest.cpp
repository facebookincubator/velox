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

// End-to-end test: register all Presto GPU functions and evaluate
// TPC-H-like expressions via GpuExprEvaluator over cuDF tables.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"
#include "velox/experimental/cudf/functions/GpuExprEvaluator.h"
#include "velox/experimental/cudf/functions/GpuPrestoFunctions.h"

using namespace facebook::velox::gpu;

class GpuEndToEndTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    GpuFunctionRegistry::instance().clear();
    CudfFallbackRegistry::instance().registerDefaults();
    registerAllPrestoGpuFunctions();
  }

  void TearDown() override {
    GpuFunctionRegistry::instance().clear();
  }

  rmm::cuda_stream_view stream_ = rmm::cuda_stream_default;
  rmm::device_async_resource_ref mr_ = cudf::get_current_device_resource_ref();

  std::unique_ptr<cudf::column> makeDoubleColumn(
      const std::vector<double>& data) {
    auto col = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        data.size(),
        cudf::mask_state::UNALLOCATED,
        stream_,
        mr_);
    cudaMemcpy(
        col->mutable_view().data<double>(),
        data.data(),
        data.size() * sizeof(double),
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

  std::vector<bool> readBool(const cudf::column_view& col) {
    std::vector<bool> result(col.size());
    std::vector<bool> buf(col.size());
    bool* hostBuf = new bool[col.size()];
    cudaMemcpy(
        hostBuf,
        col.data<bool>(),
        col.size() * sizeof(bool),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < col.size(); ++i) {
      result[i] = hostBuf[i];
    }
    delete[] hostBuf;
    return result;
  }
};

// TPC-H Q1 revenue: l_extendedprice * (1 - l_discount)
TEST_F(GpuEndToEndTest, tpchQ1Revenue) {
  // l_extendedprice (field 0), l_discount (field 1)
  auto price = makeDoubleColumn({100.0, 200.0, 300.0, 150.0});
  auto discount = makeDoubleColumn({0.05, 0.10, 0.0, 0.25});

  std::vector<cudf::column_view> cols = {price->view(), discount->view()};
  cudf::table_view table(cols);

  // expr: field(0) * (literal(1.0) - field(1))
  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto lit1 = makeLiteralDouble(1.0);
  auto f1 = makeFieldAccess(1, cudf::type_id::FLOAT64);

  std::vector<std::unique_ptr<GpuExprNode>> subArgs;
  subArgs.push_back(std::move(lit1));
  subArgs.push_back(std::move(f1));
  auto oneMinus = makeFunctionCall(
      "minus", cudf::type_id::FLOAT64, std::move(subArgs));

  std::vector<std::unique_ptr<GpuExprNode>> mulArgs;
  mulArgs.push_back(std::move(f0));
  mulArgs.push_back(std::move(oneMinus));
  auto expr = makeFunctionCall(
      "multiply", cudf::type_id::FLOAT64, std::move(mulArgs));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readDouble(result->view());

  // 100*(1-0.05)=95, 200*(1-0.10)=180, 300*(1-0)=300, 150*(1-0.25)=112.5
  EXPECT_DOUBLE_EQ(95.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(180.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(300.0, hostResult[2]);
  EXPECT_DOUBLE_EQ(112.5, hostResult[3]);
}

// TPC-H Q1 charge: l_extendedprice * (1 - l_discount) * (1 + l_tax)
TEST_F(GpuEndToEndTest, tpchQ1Charge) {
  auto price = makeDoubleColumn({100.0, 200.0});
  auto discount = makeDoubleColumn({0.05, 0.10});
  auto tax = makeDoubleColumn({0.08, 0.06});

  std::vector<cudf::column_view> cols = {
      price->view(), discount->view(), tax->view()};
  cudf::table_view table(cols);

  // revenue = f0 * (1.0 - f1)
  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto lit1a = makeLiteralDouble(1.0);
  auto f1 = makeFieldAccess(1, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> subArgs;
  subArgs.push_back(std::move(lit1a));
  subArgs.push_back(std::move(f1));
  auto oneMinus = makeFunctionCall(
      "minus", cudf::type_id::FLOAT64, std::move(subArgs));
  std::vector<std::unique_ptr<GpuExprNode>> mulArgs1;
  mulArgs1.push_back(std::move(f0));
  mulArgs1.push_back(std::move(oneMinus));
  auto revenue = makeFunctionCall(
      "multiply", cudf::type_id::FLOAT64, std::move(mulArgs1));

  // charge = revenue * (1.0 + f2)
  auto lit1b = makeLiteralDouble(1.0);
  auto f2 = makeFieldAccess(2, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> addArgs;
  addArgs.push_back(std::move(lit1b));
  addArgs.push_back(std::move(f2));
  auto onePlus = makeFunctionCall(
      "plus", cudf::type_id::FLOAT64, std::move(addArgs));
  std::vector<std::unique_ptr<GpuExprNode>> mulArgs2;
  mulArgs2.push_back(std::move(revenue));
  mulArgs2.push_back(std::move(onePlus));
  auto charge = makeFunctionCall(
      "multiply", cudf::type_id::FLOAT64, std::move(mulArgs2));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*charge, table, stream_, mr_);
  auto hostResult = readDouble(result->view());

  // 100*(1-0.05)*(1+0.08) = 95*1.08 = 102.6
  // 200*(1-0.10)*(1+0.06) = 180*1.06 = 190.8
  EXPECT_NEAR(102.6, hostResult[0], 1e-10);
  EXPECT_NEAR(190.8, hostResult[1], 1e-10);
}

// TPC-H Q6 filter: l_discount between 0.05 and 0.07
TEST_F(GpuEndToEndTest, tpchQ6DiscountFilter) {
  auto discount = makeDoubleColumn({0.04, 0.05, 0.06, 0.07, 0.08});

  std::vector<cudf::column_view> cols = {discount->view()};
  cudf::table_view table(cols);

  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto lo = makeLiteralDouble(0.05);
  auto hi = makeLiteralDouble(0.07);
  std::vector<std::unique_ptr<GpuExprNode>> betArgs;
  betArgs.push_back(std::move(f0));
  betArgs.push_back(std::move(lo));
  betArgs.push_back(std::move(hi));
  auto expr = makeFunctionCall(
      "between", cudf::type_id::BOOL8, std::move(betArgs));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readBool(result->view());

  EXPECT_FALSE(hostResult[0]); // 0.04 not in [0.05,0.07]
  EXPECT_TRUE(hostResult[1]);  // 0.05 in [0.05,0.07]
  EXPECT_TRUE(hostResult[2]);  // 0.06 in [0.05,0.07]
  EXPECT_TRUE(hostResult[3]);  // 0.07 in [0.05,0.07]
  EXPECT_FALSE(hostResult[4]); // 0.08 not in [0.05,0.07]
}

// Mixed: native GPU + cuDF fallback in same expression
// (field(0) + field(1)) > literal(10.0)
// Plus is native GPU, gt is native GPU
TEST_F(GpuEndToEndTest, nativeGpuExecution) {
  auto colA = makeDoubleColumn({3.0, 7.0, 5.0});
  auto colB = makeDoubleColumn({8.0, 2.0, 6.0});

  std::vector<cudf::column_view> cols = {colA->view(), colB->view()};
  cudf::table_view table(cols);

  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto f1 = makeFieldAccess(1, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> addArgs;
  addArgs.push_back(std::move(f0));
  addArgs.push_back(std::move(f1));
  auto sum = makeFunctionCall(
      "plus", cudf::type_id::FLOAT64, std::move(addArgs));

  auto lit10 = makeLiteralDouble(10.0);
  std::vector<std::unique_ptr<GpuExprNode>> gtArgs;
  gtArgs.push_back(std::move(sum));
  gtArgs.push_back(std::move(lit10));
  auto expr = makeFunctionCall(
      "gt", cudf::type_id::BOOL8, std::move(gtArgs));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readBool(result->view());

  EXPECT_TRUE(hostResult[0]);   // 3+8=11 > 10
  EXPECT_FALSE(hostResult[1]);  // 7+2=9 > 10
  EXPECT_TRUE(hostResult[2]);   // 5+6=11 > 10
}

TEST_F(GpuEndToEndTest, registryHasExpectedFunctions) {
  auto& reg = GpuFunctionRegistry::instance();
  EXPECT_GT(reg.size(), 30u);

  GpuFunctionKey plusDouble{
      "plus",
      cudf::type_id::FLOAT64,
      {cudf::type_id::FLOAT64, cudf::type_id::FLOAT64}};
  EXPECT_TRUE(reg.hasFunction(plusDouble));

  GpuFunctionKey ltInt{
      "lt", cudf::type_id::BOOL8, {cudf::type_id::INT64, cudf::type_id::INT64}};
  EXPECT_TRUE(reg.hasFunction(ltInt));

  GpuFunctionKey sinDouble{
      "sin", cudf::type_id::FLOAT64, {cudf::type_id::FLOAT64}};
  EXPECT_TRUE(reg.hasFunction(sinDouble));
}
