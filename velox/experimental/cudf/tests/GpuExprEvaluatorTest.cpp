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

// Tests for GpuExprEvaluator: evaluates expression trees over cuDF tables.

#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"
#include "velox/experimental/cudf/functions/GpuExprEvaluator.h"

using namespace facebook::velox::gpu;

class GpuExprEvaluatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    CudfFallbackRegistry::instance().registerDefaults();
    GpuFunctionRegistry::instance().clear();
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

  std::vector<double> readDoubleColumn(const cudf::column_view& col) {
    std::vector<double> result(col.size());
    cudaMemcpy(
        result.data(),
        col.data<double>(),
        col.size() * sizeof(double),
        cudaMemcpyDeviceToHost);
    return result;
  }
};

// Simple: field(0) + field(1)
TEST_F(GpuExprEvaluatorTest, simpleAddition) {
  auto colA = makeDoubleColumn({1.0, 2.0, 3.0, 4.0});
  auto colB = makeDoubleColumn({10.0, 20.0, 30.0, 40.0});

  std::vector<cudf::column_view> cols = {colA->view(), colB->view()};
  cudf::table_view table(cols);

  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto f1 = makeFieldAccess(1, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> args;
  args.push_back(std::move(f0));
  args.push_back(std::move(f1));
  auto expr = makeFunctionCall("plus", cudf::type_id::FLOAT64, std::move(args));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readDoubleColumn(result->view());

  EXPECT_DOUBLE_EQ(11.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(22.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(33.0, hostResult[2]);
  EXPECT_DOUBLE_EQ(44.0, hostResult[3]);
}

// Nested: (field(0) + field(1)) * field(0)
TEST_F(GpuExprEvaluatorTest, nestedExpression) {
  auto colA = makeDoubleColumn({2.0, 3.0, 4.0});
  auto colB = makeDoubleColumn({10.0, 20.0, 30.0});

  std::vector<cudf::column_view> cols = {colA->view(), colB->view()};
  cudf::table_view table(cols);

  // Inner: field(0) + field(1)
  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto f1 = makeFieldAccess(1, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> addArgs;
  addArgs.push_back(std::move(f0));
  addArgs.push_back(std::move(f1));
  auto addExpr = makeFunctionCall(
      "plus", cudf::type_id::FLOAT64, std::move(addArgs));

  // Outer: (f0+f1) * f0
  auto f0again = makeFieldAccess(0, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> mulArgs;
  mulArgs.push_back(std::move(addExpr));
  mulArgs.push_back(std::move(f0again));
  auto mulExpr = makeFunctionCall(
      "multiply", cudf::type_id::FLOAT64, std::move(mulArgs));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*mulExpr, table, stream_, mr_);
  auto hostResult = readDoubleColumn(result->view());

  // (2+10)*2=24, (3+20)*3=69, (4+30)*4=136
  EXPECT_DOUBLE_EQ(24.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(69.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(136.0, hostResult[2]);
}

// Literal: field(0) + 100.0
TEST_F(GpuExprEvaluatorTest, literalConstant) {
  auto colA = makeDoubleColumn({1.0, 2.0, 3.0});

  std::vector<cudf::column_view> cols = {colA->view()};
  cudf::table_view table(cols);

  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  auto lit = makeLiteralDouble(100.0);
  std::vector<std::unique_ptr<GpuExprNode>> args;
  args.push_back(std::move(f0));
  args.push_back(std::move(lit));
  auto expr = makeFunctionCall("plus", cudf::type_id::FLOAT64, std::move(args));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readDoubleColumn(result->view());

  EXPECT_DOUBLE_EQ(101.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(102.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(103.0, hostResult[2]);
}

// Unary: abs(field(0))
TEST_F(GpuExprEvaluatorTest, unaryAbs) {
  auto colA = makeDoubleColumn({-5.0, 3.0, -0.5, 0.0});

  std::vector<cudf::column_view> cols = {colA->view()};
  cudf::table_view table(cols);

  auto f0 = makeFieldAccess(0, cudf::type_id::FLOAT64);
  std::vector<std::unique_ptr<GpuExprNode>> args;
  args.push_back(std::move(f0));
  auto expr = makeFunctionCall("abs", cudf::type_id::FLOAT64, std::move(args));

  GpuExprEvaluator evaluator;
  auto result = evaluator.evaluate(*expr, table, stream_, mr_);
  auto hostResult = readDoubleColumn(result->view());

  EXPECT_DOUBLE_EQ(5.0, hostResult[0]);
  EXPECT_DOUBLE_EQ(3.0, hostResult[1]);
  EXPECT_DOUBLE_EQ(0.5, hostResult[2]);
  EXPECT_DOUBLE_EQ(0.0, hostResult[3]);
}

// Deep nesting: sqrt(field(0) * field(0) + field(1) * field(1))
// i.e., hypot without the name
TEST_F(GpuExprEvaluatorTest, deepNesting) {
  auto colA = makeDoubleColumn({3.0, 5.0});
  auto colB = makeDoubleColumn({4.0, 12.0});

  std::vector<cudf::column_view> cols = {colA->view(), colB->view()};
  cudf::table_view table(cols);

  // a*a
  {
    auto f0a = makeFieldAccess(0, cudf::type_id::FLOAT64);
    auto f0b = makeFieldAccess(0, cudf::type_id::FLOAT64);
    std::vector<std::unique_ptr<GpuExprNode>> mulArgs;
    mulArgs.push_back(std::move(f0a));
    mulArgs.push_back(std::move(f0b));
    auto aa = makeFunctionCall(
        "multiply", cudf::type_id::FLOAT64, std::move(mulArgs));

    // b*b
    auto f1a = makeFieldAccess(1, cudf::type_id::FLOAT64);
    auto f1b = makeFieldAccess(1, cudf::type_id::FLOAT64);
    std::vector<std::unique_ptr<GpuExprNode>> mulArgs2;
    mulArgs2.push_back(std::move(f1a));
    mulArgs2.push_back(std::move(f1b));
    auto bb = makeFunctionCall(
        "multiply", cudf::type_id::FLOAT64, std::move(mulArgs2));

    // a*a + b*b
    std::vector<std::unique_ptr<GpuExprNode>> addArgs;
    addArgs.push_back(std::move(aa));
    addArgs.push_back(std::move(bb));
    auto sum = makeFunctionCall(
        "plus", cudf::type_id::FLOAT64, std::move(addArgs));

    // sqrt(...)
    std::vector<std::unique_ptr<GpuExprNode>> sqrtArgs;
    sqrtArgs.push_back(std::move(sum));
    auto expr = makeFunctionCall(
        "sqrt", cudf::type_id::FLOAT64, std::move(sqrtArgs));

    GpuExprEvaluator evaluator;
    auto result = evaluator.evaluate(*expr, table, stream_, mr_);
    auto hostResult = readDoubleColumn(result->view());

    EXPECT_NEAR(5.0, hostResult[0], 1e-10);   // sqrt(9+16) = 5
    EXPECT_NEAR(13.0, hostResult[1], 1e-10);  // sqrt(25+144) = 13
  }
}
