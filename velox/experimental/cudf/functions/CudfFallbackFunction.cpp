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
#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"

namespace facebook::velox::gpu {

std::unique_ptr<cudf::column> CudfBinaryFunction::apply(
    const std::vector<cudf::column_view>& inputs,
    cudf::size_type /*numRows*/,
    const cudf::bitmask_type* /*activeRows*/,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      inputs.at(0), inputs.at(1), op_, outputType_, stream, mr);
}

std::unique_ptr<cudf::column> CudfUnaryFunction::apply(
    const std::vector<cudf::column_view>& inputs,
    cudf::size_type /*numRows*/,
    const cudf::bitmask_type* /*activeRows*/,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::unary_operation(inputs.at(0), op_, stream, mr);
}

CudfFallbackRegistry& CudfFallbackRegistry::instance() {
  static CudfFallbackRegistry reg;
  return reg;
}

std::optional<cudf::binary_operator> CudfFallbackRegistry::findBinaryOp(
    const std::string& name) const {
  auto it = binaryOps_.find(name);
  if (it != binaryOps_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<cudf::unary_operator> CudfFallbackRegistry::findUnaryOp(
    const std::string& name) const {
  auto it = unaryOps_.find(name);
  if (it != unaryOps_.end()) {
    return it->second;
  }
  return std::nullopt;
}

void CudfFallbackRegistry::registerBinaryOp(
    const std::string& name,
    cudf::binary_operator op) {
  binaryOps_[name] = op;
}

void CudfFallbackRegistry::registerUnaryOp(
    const std::string& name,
    cudf::unary_operator op) {
  unaryOps_[name] = op;
}

void CudfFallbackRegistry::registerDefaults() {
  registerBinaryOp("plus", cudf::binary_operator::ADD);
  registerBinaryOp("minus", cudf::binary_operator::SUB);
  registerBinaryOp("multiply", cudf::binary_operator::MUL);
  registerBinaryOp("divide", cudf::binary_operator::DIV);
  registerBinaryOp("mod", cudf::binary_operator::MOD);
  registerBinaryOp("power", cudf::binary_operator::POW);

  registerBinaryOp("eq", cudf::binary_operator::EQUAL);
  registerBinaryOp("neq", cudf::binary_operator::NOT_EQUAL);
  registerBinaryOp("lt", cudf::binary_operator::LESS);
  registerBinaryOp("gt", cudf::binary_operator::GREATER);
  registerBinaryOp("lte", cudf::binary_operator::LESS_EQUAL);
  registerBinaryOp("gte", cudf::binary_operator::GREATER_EQUAL);

  registerUnaryOp("abs", cudf::unary_operator::ABS);
  registerUnaryOp("negate", cudf::unary_operator::NEGATE);
  registerUnaryOp("ceil", cudf::unary_operator::CEIL);
  registerUnaryOp("floor", cudf::unary_operator::FLOOR);
  registerUnaryOp("sqrt", cudf::unary_operator::SQRT);
  registerUnaryOp("cbrt", cudf::unary_operator::CBRT);
  registerUnaryOp("exp", cudf::unary_operator::EXP);
  registerUnaryOp("ln", cudf::unary_operator::LOG);
  registerUnaryOp("sin", cudf::unary_operator::SIN);
  registerUnaryOp("cos", cudf::unary_operator::COS);
  registerUnaryOp("tan", cudf::unary_operator::TAN);
  registerUnaryOp("asin", cudf::unary_operator::ARCSIN);
  registerUnaryOp("acos", cudf::unary_operator::ARCCOS);
  registerUnaryOp("atan", cudf::unary_operator::ARCTAN);
  registerUnaryOp("sinh", cudf::unary_operator::SINH);
  registerUnaryOp("cosh", cudf::unary_operator::COSH);
  registerUnaryOp("tanh", cudf::unary_operator::TANH);
}

} // namespace facebook::velox::gpu
