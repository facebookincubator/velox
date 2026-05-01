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
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"

#include "velox/experimental/cudf/gpu_portable/DecimalDivide.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstdint>
#include <span>

namespace facebook::velox::cudf_velox {
namespace {

// Pick the right gpu_portable getter for the requested input/output combo.
// All three lifts emit kernels with the same arithmetic; only the cuDF
// boundary types differ (numeric::decimal64 vs numeric::decimal128).
std::string decimalDivideJitSource(
    cudf::data_type lhsType,
    cudf::data_type outputType,
    __int128_t pow10) {
  using cudf::type_id;
  if (lhsType.id() == type_id::DECIMAL64) {
    if (outputType.id() == type_id::DECIMAL64) {
      return gpu_portable::velox_decimal_divide_int64_int64_source(false, pow10);
    }
    CUDF_EXPECTS(
        outputType.id() == type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
    return gpu_portable::velox_decimal_divide_int64_int128_source(false, pow10);
  }
  CUDF_EXPECTS(
      lhsType.id() == type_id::DECIMAL128,
      "Unsupported input type for decimal divide");
  CUDF_EXPECTS(
      outputType.id() == type_id::DECIMAL128,
      "Unexpected output type for decimal divide");
  return gpu_portable::velox_decimal_divide_int128_int128_source(false, pow10);
}

// Run transform_extended over the given inputs. The lifted kernel always
// writes a scale-0 column; we relabel the result to outputType (metadata
// only, no data movement) on return. Mirrors the CPU contract where the
// output vector carries the scale, not the function.
std::unique_ptr<cudf::column> decimalDivideViaJit(
    cudf::data_type lhsType,
    std::span<const cudf::transform_input> inputs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto src = decimalDivideJitSource(
      lhsType, outputType, DecimalUtil::kPowersOfTen[aRescale]);
  const cudf::data_type intermediate{
      outputType.id() == cudf::type_id::DECIMAL64
          ? cudf::type_id::DECIMAL64
          : cudf::type_id::DECIMAL128,
      0};
  auto raw = cudf::transform_extended(
      inputs,
      src,
      intermediate,
      cudf::udf_source_type::CUDA,
      std::nullopt,
      cudf::null_aware::NO,
      std::nullopt,
      cudf::output_nullability::PRESERVE,
      stream,
      mr);
  const auto size = raw->size();
  const auto nullCount = raw->null_count();
  auto contents = raw->release();
  return std::make_unique<cudf::column>(
      outputType,
      size,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount,
      std::move(contents.children));
}

// Materialize a scalar as a 1-element column so it can be wrapped in
// scalar_column_view and handed to transform_extended as a broadcast input.
std::unique_ptr<cudf::column> scalarToColumn(
    const cudf::scalar& s,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_column_from_scalar(s, 1, stream, mr);
}

} // namespace

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Decimal divide requires equal sizes");
  CUDF_EXPECTS(
      lhs.type().id() == rhs.type().id(),
      "Decimal divide requires matching input types");
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");
  const std::vector<cudf::transform_input> inputs = {lhs, rhs};
  return decimalDivideViaJit(
      lhs.type(), inputs, outputType, aRescale, stream, mr);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");
  auto rhsCol = scalarToColumn(rhs, stream, mr);
  const std::vector<cudf::transform_input> inputs = {
      lhs, cudf::scalar_column_view(rhsCol->view())};
  return decimalDivideViaJit(
      lhs.type(), inputs, outputType, aRescale, stream, mr);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");
  auto lhsCol = scalarToColumn(lhs, stream, mr);
  const std::vector<cudf::transform_input> inputs = {
      cudf::scalar_column_view(lhsCol->view()), rhs};
  return decimalDivideViaJit(
      rhs.type(), inputs, outputType, aRescale, stream, mr);
}

} // namespace facebook::velox::cudf_velox
