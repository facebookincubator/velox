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

// GPU Expression Evaluator: walks a tree of GpuExprNode and evaluates
// each node bottom-up, producing cuDF columns as intermediates.
#pragma once

#include "velox/experimental/cudf/functions/GpuFunctionDispatch.h"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace facebook::velox::gpu {

enum class GpuExprNodeKind {
  kFieldAccess,
  kFunctionCall,
  kLiteral,
  kCpuFallback,
  kAnd,
  kOr,
  kNot,
  kSwitch,
  kCoalesce,
  kCast,
};

struct GpuExprNode {
  GpuExprNodeKind kind;
  cudf::type_id resultType;

  int fieldIndex{-1};

  std::string functionName;
  std::vector<std::unique_ptr<GpuExprNode>> children;

  double literalDouble{0.0};
  int64_t literalInt64{0};
  bool literalBool{false};
  bool literalIsNull{false};
  std::string literalString;

  // For kCpuFallback: type-erased pointers to avoid Velox header deps here.
  // Holds shared_ptr<exec::Expr> and shared_ptr<RowType> respectively.
  std::shared_ptr<void> fallbackExpr;
  std::shared_ptr<void> fallbackSchema;
};

using CpuFallbackFn = std::function<std::unique_ptr<cudf::column>(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)>;

class GpuExprEvaluator {
 public:
  void setCpuFallbackHandler(CpuFallbackFn handler) {
    cpuFallbackHandler_ = std::move(handler);
  }

  std::unique_ptr<cudf::column> evaluate(
      const GpuExprNode& expr,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

 private:
  struct EvalResult {
    std::unique_ptr<cudf::column> owned;
    cudf::column_view view;
  };

  EvalResult evalNode(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalFieldAccess(
      const GpuExprNode& node,
      const cudf::table_view& input);

  EvalResult evalFunctionCall(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalLiteral(
      const GpuExprNode& node,
      cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalCpuFallback(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalAnd(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalOr(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalNot(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalSwitch(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalCoalesce(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  EvalResult evalCast(
      const GpuExprNode& node,
      const cudf::table_view& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  CpuFallbackFn cpuFallbackHandler_;
};

std::unique_ptr<GpuExprNode> makeFieldAccess(
    int index,
    cudf::type_id type);

std::unique_ptr<GpuExprNode> makeFunctionCall(
    const std::string& name,
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children);

std::unique_ptr<GpuExprNode> makeLiteralDouble(double value);
std::unique_ptr<GpuExprNode> makeLiteralInt64(int64_t value);
std::unique_ptr<GpuExprNode> makeLiteralBool(bool value);
std::unique_ptr<GpuExprNode> makeLiteralString(const std::string& value);
std::unique_ptr<GpuExprNode> makeLiteralInt32(int32_t value);
std::unique_ptr<GpuExprNode> makeLiteralNull(cudf::type_id type);

std::unique_ptr<GpuExprNode> makeCpuFallback(
    cudf::type_id resultType,
    std::shared_ptr<void> fallbackExpr,
    std::shared_ptr<void> fallbackSchema);

std::unique_ptr<GpuExprNode> makeAnd(
    std::vector<std::unique_ptr<GpuExprNode>> children);
std::unique_ptr<GpuExprNode> makeOr(
    std::vector<std::unique_ptr<GpuExprNode>> children);
std::unique_ptr<GpuExprNode> makeNot(
    std::unique_ptr<GpuExprNode> child);
std::unique_ptr<GpuExprNode> makeSwitch(
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children);
std::unique_ptr<GpuExprNode> makeCoalesce(
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children);
std::unique_ptr<GpuExprNode> makeCast(
    cudf::type_id targetType,
    std::unique_ptr<GpuExprNode> child);

} // namespace facebook::velox::gpu
