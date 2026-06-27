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
#include "velox/experimental/cudf/expression/AstPrinter.h"
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace facebook::velox::cudf_velox {
namespace {

const char* decimalOverflowMessage(cudf::binary_operator op) {
  switch (op) {
    case cudf::binary_operator::ADD:
      return "Decimal overflow in add";
    case cudf::binary_operator::SUB:
      return "Decimal overflow in subtract";
    case cudf::binary_operator::MUL:
      return "Decimal overflow in multiply";
    case cudf::binary_operator::DIV:
      return "Decimal overflow in divide";
    case cudf::binary_operator::MOD:
      return "Decimal overflow in modulo";
    default:
      return "Decimal overflow";
  }
}

// Fail-fast on overflow, matching Presto / Velox CPU semantics: the kernel
// reports a single batch-wide flag and we fail the whole expression if set.
void throwIfDecimalOverflow(bool didOverflow, cudf::binary_operator op) {
  if (didOverflow) {
    VELOX_USER_FAIL("{}", decimalOverflowMessage(op));
  }
}

} // namespace

std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalBinaryOperationWithOverflow(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, op);
  return std::move(result);
}

std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalBinaryOperationWithOverflow(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, op);
  return std::move(result);
}

std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalBinaryOperationWithOverflow(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, op);
  return std::move(result);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalDivideWithOverflow(
      lhs, rhs, outputType, aRescale, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, cudf::binary_operator::DIV);
  return scatterNullsAtZeroDivisor(std::move(result), rhs, stream, mr);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalDivideWithOverflow(
      lhs, rhs, outputType, aRescale, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, cudf::binary_operator::DIV);
  return std::move(result);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto [result, didOverflow] = decimalDivideWithOverflow(
      lhs, rhs, outputType, aRescale, outputPrecision, stream, mr);
  throwIfDecimalOverflow(didOverflow, cudf::binary_operator::DIV);
  return scatterNullsAtZeroDivisor(std::move(result), rhs, stream, mr);
}

// Scatters null values to positions where the divisor is zero.
// Returns a new column with nulls at zero-divisor positions.
std::unique_ptr<cudf::column> scatterNullsAtZeroDivisor(
    std::unique_ptr<cudf::column> result,
    const cudf::column_view& divisor,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create zero scalar for comparison and null scalar for scattering.
  std::unique_ptr<cudf::scalar> zeroScalar;
  std::unique_ptr<cudf::scalar> nullScalar;
  auto divisorScale = numeric::scale_type{divisor.type().scale()};
  auto outputScale = numeric::scale_type{result->type().scale()};

  if (divisor.type().id() == cudf::type_id::DECIMAL64) {
    zeroScalar = cudf::make_fixed_point_scalar<numeric::decimal64>(
        0, divisorScale, stream, mr);
    nullScalar = cudf::make_fixed_point_scalar<numeric::decimal64>(
        0, outputScale, stream, mr);
  } else if (divisor.type().id() == cudf::type_id::DECIMAL128) {
    zeroScalar = cudf::make_fixed_point_scalar<numeric::decimal128>(
        0, divisorScale, stream, mr);
    nullScalar = cudf::make_fixed_point_scalar<numeric::decimal128>(
        0, outputScale, stream, mr);
  } else {
    VELOX_FAIL(
        "Unsupported decimal type {} for division",
        cudf::ast::type_id_to_string(divisor.type().id()));
  }
  nullScalar->set_valid_async(false, stream);

  // Create boolean column: TRUE where divisor == 0, FALSE otherwise.
  auto divisorIsZero = cudf::binary_operation(
      divisor,
      *zeroScalar,
      cudf::binary_operator::EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);

  // Scatter nulls where divisor is zero.
  return cudf::copy_if_else(
      *nullScalar, *result, divisorIsZero->view(), stream, mr);
}

} // namespace facebook::velox::cudf_velox
