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
#include "velox/experimental/cudf/functions/GpuExprEvaluator.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <rmm/device_uvector.hpp>

#include <stdexcept>

namespace facebook::velox::gpu {

std::unique_ptr<cudf::column> GpuExprEvaluator::evaluate(
    const GpuExprNode& expr,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto result = evalNode(expr, input, stream, mr);
  if (result.owned) {
    return std::move(result.owned);
  }
  return std::make_unique<cudf::column>(result.view, stream, mr);
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalNode(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (node.kind) {
    case GpuExprNodeKind::kFieldAccess:
      return evalFieldAccess(node, input);
    case GpuExprNodeKind::kFunctionCall:
      return evalFunctionCall(node, input, stream, mr);
    case GpuExprNodeKind::kLiteral:
      return evalLiteral(node, input.num_rows(), stream, mr);
    case GpuExprNodeKind::kCpuFallback:
      return evalCpuFallback(node, input, stream, mr);
  }
  throw std::runtime_error("Unknown GpuExprNode kind");
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalFieldAccess(
    const GpuExprNode& node,
    const cudf::table_view& input) {
  return {nullptr, input.column(node.fieldIndex)};
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalFunctionCall(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<EvalResult> childResults;
  childResults.reserve(node.children.size());
  for (auto& child : node.children) {
    childResults.push_back(evalNode(*child, input, stream, mr));
  }

  std::vector<cudf::column_view> childViews;
  childViews.reserve(childResults.size());
  std::vector<cudf::type_id> argTypes;
  argTypes.reserve(childResults.size());
  for (auto& cr : childResults) {
    childViews.push_back(cr.owned ? cr.owned->view() : cr.view);
    argTypes.push_back(childViews.back().type().id());
  }

  auto dispatch = dispatchGpuFunction(
      node.functionName, node.resultType, argTypes);

  if (!dispatch.function) {
    throw std::runtime_error(
        "No GPU implementation found for function: " + node.functionName);
  }

  // Retain owned to keep the function object alive through apply().
  auto ownedFn = std::move(dispatch.owned);
  GpuVectorFunction* fn = ownedFn ? ownedFn.get() : dispatch.function;

  auto col = fn->apply(childViews, input.num_rows(), nullptr, stream, mr);
  auto view = col->view();
  return {std::move(col), view};
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalLiteral(
    const GpuExprNode& node,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::scalar> scalar;
  cudf::data_type dtype{node.resultType};

  switch (node.resultType) {
    case cudf::type_id::FLOAT64:
      scalar = cudf::make_fixed_width_scalar(node.literalDouble, stream, mr);
      break;
    case cudf::type_id::FLOAT32:
      scalar = cudf::make_fixed_width_scalar(
          static_cast<float>(node.literalDouble), stream, mr);
      break;
    case cudf::type_id::INT64:
      scalar = cudf::make_fixed_width_scalar(node.literalInt64, stream, mr);
      break;
    case cudf::type_id::INT32:
      scalar = cudf::make_fixed_width_scalar(
          static_cast<int32_t>(node.literalInt64), stream, mr);
      break;
    case cudf::type_id::BOOL8:
      scalar = cudf::make_fixed_width_scalar(node.literalBool, stream, mr);
      break;
    default:
      throw std::runtime_error("Unsupported literal type");
  }
  scalar->set_valid_async(true, stream);

  auto col = cudf::make_column_from_scalar(*scalar, numRows, stream, mr);
  auto view = col->view();
  return {std::move(col), view};
}

std::unique_ptr<GpuExprNode> makeFieldAccess(int index, cudf::type_id type) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kFieldAccess;
  node->resultType = type;
  node->fieldIndex = index;
  return node;
}

std::unique_ptr<GpuExprNode> makeFunctionCall(
    const std::string& name,
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kFunctionCall;
  node->resultType = resultType;
  node->functionName = name;
  node->children = std::move(children);
  return node;
}

std::unique_ptr<GpuExprNode> makeLiteralDouble(double value) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = cudf::type_id::FLOAT64;
  node->literalDouble = value;
  return node;
}

std::unique_ptr<GpuExprNode> makeLiteralInt64(int64_t value) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = cudf::type_id::INT64;
  node->literalInt64 = value;
  return node;
}

std::unique_ptr<GpuExprNode> makeCpuFallback(
    cudf::type_id resultType,
    std::shared_ptr<void> fallbackExpr,
    std::shared_ptr<void> fallbackSchema) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kCpuFallback;
  node->resultType = resultType;
  node->fallbackExpr = std::move(fallbackExpr);
  node->fallbackSchema = std::move(fallbackSchema);
  return node;
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalCpuFallback(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!cpuFallbackHandler_) {
    throw std::runtime_error(
        "kCpuFallback node encountered but no CPU fallback handler is set");
  }
  auto col = cpuFallbackHandler_(node, input, stream, mr);
  auto view = col->view();
  return {std::move(col), view};
}

} // namespace facebook::velox::gpu
