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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>

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
    case GpuExprNodeKind::kAnd:
      return evalAnd(node, input, stream, mr);
    case GpuExprNodeKind::kOr:
      return evalOr(node, input, stream, mr);
    case GpuExprNodeKind::kNot:
      return evalNot(node, input, stream, mr);
    case GpuExprNodeKind::kSwitch:
      return evalSwitch(node, input, stream, mr);
    case GpuExprNodeKind::kCoalesce:
      return evalCoalesce(node, input, stream, mr);
    case GpuExprNodeKind::kCast:
      return evalCast(node, input, stream, mr);
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

  if (node.literalIsNull) {
    scalar = cudf::make_default_constructed_scalar(
        cudf::data_type{node.resultType}, stream, mr);
    scalar->set_valid_async(false, stream);
  } else if (node.resultType == cudf::type_id::STRING) {
    scalar = std::make_unique<cudf::string_scalar>(
        node.literalString, true, stream, mr);
  } else {
    switch (node.resultType) {
      case cudf::type_id::FLOAT64:
        scalar =
            cudf::make_fixed_width_scalar(node.literalDouble, stream, mr);
        break;
      case cudf::type_id::FLOAT32:
        scalar = cudf::make_fixed_width_scalar(
            static_cast<float>(node.literalDouble), stream, mr);
        break;
      case cudf::type_id::INT64:
        scalar =
            cudf::make_fixed_width_scalar(node.literalInt64, stream, mr);
        break;
      case cudf::type_id::INT32:
        scalar = cudf::make_fixed_width_scalar(
            static_cast<int32_t>(node.literalInt64), stream, mr);
        break;
      case cudf::type_id::INT16:
        scalar = cudf::make_fixed_width_scalar(
            static_cast<int16_t>(node.literalInt64), stream, mr);
        break;
      case cudf::type_id::INT8:
        scalar = cudf::make_fixed_width_scalar(
            static_cast<int8_t>(node.literalInt64), stream, mr);
        break;
      case cudf::type_id::BOOL8:
        scalar =
            cudf::make_fixed_width_scalar(node.literalBool, stream, mr);
        break;
      default:
        throw std::runtime_error(
            "Unsupported literal type: " +
            std::to_string(static_cast<int>(node.resultType)));
    }
    scalar->set_valid_async(true, stream);
  }

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

std::unique_ptr<GpuExprNode> makeLiteralBool(bool value) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = cudf::type_id::BOOL8;
  node->literalBool = value;
  return node;
}

std::unique_ptr<GpuExprNode> makeLiteralString(const std::string& value) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = cudf::type_id::STRING;
  node->literalString = value;
  return node;
}

std::unique_ptr<GpuExprNode> makeLiteralInt32(int32_t value) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = cudf::type_id::INT32;
  node->literalInt64 = value;
  return node;
}

std::unique_ptr<GpuExprNode> makeLiteralNull(cudf::type_id type) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kLiteral;
  node->resultType = type;
  node->literalIsNull = true;
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

// SQL three-valued AND: false AND null = false, true AND null = null.
GpuExprEvaluator::EvalResult GpuExprEvaluator::evalAnd(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto result = evalNode(*node.children[0], input, stream, mr);
  auto accum = result.owned
      ? std::move(result.owned)
      : std::make_unique<cudf::column>(result.view, stream, mr);

  for (size_t i = 1; i < node.children.size(); ++i) {
    auto rhs = evalNode(*node.children[i], input, stream, mr);
    auto rhsView = rhs.owned ? rhs.owned->view() : rhs.view;
    accum = cudf::binary_operation(
        accum->view(),
        rhsView,
        cudf::binary_operator::NULL_LOGICAL_AND,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  auto view = accum->view();
  return {std::move(accum), view};
}

// SQL three-valued OR: true OR null = true, false OR null = null.
GpuExprEvaluator::EvalResult GpuExprEvaluator::evalOr(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto result = evalNode(*node.children[0], input, stream, mr);
  auto accum = result.owned
      ? std::move(result.owned)
      : std::make_unique<cudf::column>(result.view, stream, mr);

  for (size_t i = 1; i < node.children.size(); ++i) {
    auto rhs = evalNode(*node.children[i], input, stream, mr);
    auto rhsView = rhs.owned ? rhs.owned->view() : rhs.view;
    accum = cudf::binary_operation(
        accum->view(),
        rhsView,
        cudf::binary_operator::NULL_LOGICAL_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  auto view = accum->view();
  return {std::move(accum), view};
}

GpuExprEvaluator::EvalResult GpuExprEvaluator::evalNot(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto child = evalNode(*node.children[0], input, stream, mr);
  auto childView = child.owned ? child.owned->view() : child.view;
  auto col = cudf::unary_operation(childView, cudf::unary_operator::NOT,
                                   stream, mr);
  auto view = col->view();
  return {std::move(col), view};
}

// SWITCH/IF: children are [cond1, val1, cond2, val2, ..., default].
// Evaluates conditions left to right; picks the value of the first true
// condition, or the default if none match.
GpuExprEvaluator::EvalResult GpuExprEvaluator::evalSwitch(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  size_t n = node.children.size();
  bool hasDefault = (n % 2 == 1);
  size_t numPairs = n / 2;

  // Start with the default (last child) or a null column.
  std::unique_ptr<cudf::column> result;
  if (hasDefault) {
    auto def = evalNode(*node.children[n - 1], input, stream, mr);
    result = def.owned
        ? std::move(def.owned)
        : std::make_unique<cudf::column>(def.view, stream, mr);
  } else {
    auto scalar = cudf::make_default_constructed_scalar(
        cudf::data_type{node.resultType}, stream, mr);
    scalar->set_valid_async(false, stream);
    result = cudf::make_column_from_scalar(*scalar, input.num_rows(),
                                           stream, mr);
  }

  // Process condition/value pairs in reverse so the first matching condition
  // wins (later copy_if_else overwrites earlier results).
  for (int i = static_cast<int>(numPairs) - 1; i >= 0; --i) {
    auto cond = evalNode(*node.children[2 * i], input, stream, mr);
    auto val = evalNode(*node.children[2 * i + 1], input, stream, mr);
    auto condView = cond.owned ? cond.owned->view() : cond.view;
    auto valView = val.owned ? val.owned->view() : val.view;
    result = cudf::copy_if_else(
        valView, result->view(), condView, stream, mr);
  }

  auto view = result->view();
  return {std::move(result), view};
}

// COALESCE: returns first non-null value across children.
GpuExprEvaluator::EvalResult GpuExprEvaluator::evalCoalesce(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto first = evalNode(*node.children[0], input, stream, mr);
  auto result = first.owned
      ? std::move(first.owned)
      : std::make_unique<cudf::column>(first.view, stream, mr);

  for (size_t i = 1; i < node.children.size(); ++i) {
    if (!result->has_nulls()) {
      break;
    }
    auto next = evalNode(*node.children[i], input, stream, mr);
    auto nextView = next.owned ? next.owned->view() : next.view;
    result = cudf::replace_nulls(result->view(), nextView, stream, mr);
  }

  auto view = result->view();
  return {std::move(result), view};
}

// CAST: type conversion using cudf::cast.
GpuExprEvaluator::EvalResult GpuExprEvaluator::evalCast(
    const GpuExprNode& node,
    const cudf::table_view& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto child = evalNode(*node.children[0], input, stream, mr);
  auto childView = child.owned ? child.owned->view() : child.view;
  auto col = cudf::cast(childView, cudf::data_type{node.resultType},
                         stream, mr);
  auto view = col->view();
  return {std::move(col), view};
}

// -- Factory functions for new node kinds --

std::unique_ptr<GpuExprNode> makeAnd(
    std::vector<std::unique_ptr<GpuExprNode>> children) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kAnd;
  node->resultType = cudf::type_id::BOOL8;
  node->children = std::move(children);
  return node;
}

std::unique_ptr<GpuExprNode> makeOr(
    std::vector<std::unique_ptr<GpuExprNode>> children) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kOr;
  node->resultType = cudf::type_id::BOOL8;
  node->children = std::move(children);
  return node;
}

std::unique_ptr<GpuExprNode> makeNot(
    std::unique_ptr<GpuExprNode> child) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kNot;
  node->resultType = cudf::type_id::BOOL8;
  node->children.push_back(std::move(child));
  return node;
}

std::unique_ptr<GpuExprNode> makeSwitch(
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kSwitch;
  node->resultType = resultType;
  node->children = std::move(children);
  return node;
}

std::unique_ptr<GpuExprNode> makeCoalesce(
    cudf::type_id resultType,
    std::vector<std::unique_ptr<GpuExprNode>> children) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kCoalesce;
  node->resultType = resultType;
  node->children = std::move(children);
  return node;
}

std::unique_ptr<GpuExprNode> makeCast(
    cudf::type_id targetType,
    std::unique_ptr<GpuExprNode> child) {
  auto node = std::make_unique<GpuExprNode>();
  node->kind = GpuExprNodeKind::kCast;
  node->resultType = targetType;
  node->children.push_back(std::move(child));
  return node;
}

} // namespace facebook::velox::gpu
