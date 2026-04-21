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

// Fallback GPU execution using cuDF's built-in binary_operation() and
// unary_operation() for functions not yet compiled from Velox source.
#pragma once

#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/unary.hpp>

#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::gpu {

class CudfBinaryFunction : public GpuVectorFunction {
 public:
  CudfBinaryFunction(
      cudf::binary_operator op,
      cudf::data_type outputType)
      : op_(op), outputType_(outputType) {}

  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type numRows,
      const cudf::bitmask_type* activeRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override;

 private:
  cudf::binary_operator op_;
  cudf::data_type outputType_;
};

class CudfUnaryFunction : public GpuVectorFunction {
 public:
  explicit CudfUnaryFunction(cudf::unary_operator op) : op_(op) {}

  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type numRows,
      const cudf::bitmask_type* activeRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override;

 private:
  cudf::unary_operator op_;
};

struct CudfFallbackRegistry {
  static CudfFallbackRegistry& instance();

  std::optional<cudf::binary_operator> findBinaryOp(
      const std::string& name) const;
  std::optional<cudf::unary_operator> findUnaryOp(
      const std::string& name) const;

  void registerBinaryOp(
      const std::string& name,
      cudf::binary_operator op);
  void registerUnaryOp(
      const std::string& name,
      cudf::unary_operator op);

  void registerDefaults();

 private:
  CudfFallbackRegistry() = default;
  std::unordered_map<std::string, cudf::binary_operator> binaryOps_;
  std::unordered_map<std::string, cudf::unary_operator> unaryOps_;
};

} // namespace facebook::velox::gpu
