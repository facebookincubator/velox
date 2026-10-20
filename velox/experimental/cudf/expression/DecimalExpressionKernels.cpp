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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"
#include "velox/experimental/cudf/expression/DecimalExpressionKernelsGpu.h"

#include "velox/common/base/Exceptions.h"
#include "velox/type/Type.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <tuple>

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
      VELOX_UNREACHABLE("BAD OPERATOR: {}", static_cast<int>(op));
  }
}

// Kept distinct from overflow so the GPU path reports the same fail-fast error
// as Velox CPU for a zero divisor on DIV or MOD.
const char* decimalDivisionByZeroMessage(cudf::binary_operator op) {
  switch (op) {
    case cudf::binary_operator::MOD:
      return "Modulus by zero";
    default:
      return "Division by zero";
  }
}

void checkDecimalBinaryOpStatus(
    DecimalBinaryOpStatus status,
    cudf::binary_operator op) {
  if (status == DecimalBinaryOpStatus::kDivisionByZero) {
    VELOX_USER_FAIL("{}", decimalDivisionByZeroMessage(op));
  }
  if (status == DecimalBinaryOpStatus::kOverflow) {
    VELOX_USER_FAIL("{}", decimalOverflowMessage(op));
  }
}

// Fails before any row is divided if the dividend must be rescaled by more than
// the maximum precision, as Velox CPU does when it initializes divide. Velox
// scales count fractional digits, so they are the negated cuDF scales.
void checkDecimalDivideRescale(
    cudf::data_type lhsType,
    cudf::data_type rhsType,
    cudf::data_type outputType) {
  auto const aRescale = static_cast<int64_t>(lhsType.scale()) -
      rhsType.scale() - outputType.scale();
  VELOX_USER_CHECK_LE(
      aRescale,
      LongDecimalType::kMaxPrecision,
      "{}",
      decimalOverflowMessage(cudf::binary_operator::DIV));
}

std::unique_ptr<cudf::scalar> makeZeroDecimalScalar(
    cudf::data_type type,
    cuda::stream_ref stream) {
  auto const scale = numeric::scale_type{type.scale()};
  switch (type.id()) {
    case cudf::type_id::DECIMAL32:
      return cudf::make_fixed_point_scalar<numeric::decimal32>(
          0, scale, stream, get_temp_mr());
    case cudf::type_id::DECIMAL64:
      return cudf::make_fixed_point_scalar<numeric::decimal64>(
          0, scale, stream, get_temp_mr());
    case cudf::type_id::DECIMAL128:
      return cudf::make_fixed_point_scalar<numeric::decimal128>(
          0, scale, stream, get_temp_mr());
    default:
      VELOX_UNREACHABLE("Decimal divide requires a decimal divisor");
  }
}

// Returns true if any row of `divisor` that is valid in `activeMask` is zero.
// A null `activeMask` means every row is active.
bool hasActiveZeroDivisor(
    const cudf::column_view& divisor,
    cudf::bitmask_type const* activeMask,
    cudf::size_type activeNullCount,
    cuda::stream_ref stream) {
  auto const zero = makeZeroDecimalScalar(divisor.type(), stream);
  auto const isZero = cudf::binary_operation(
      divisor,
      *zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      get_temp_mr());
  auto const active = cudf::column_view{
      isZero->type(),
      isZero->size(),
      isZero->view().head(),
      activeMask,
      activeNullCount};
  auto const any = cudf::reduce(
      active,
      *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      get_temp_mr());
  return any->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool> const&>(*any).value(stream);
}

// cudf::divide_decimal returns null for a row whose inputs are both valid but
// whose division fails, from a zero divisor or from overflow. Velox fails the
// batch instead, so any nulls beyond the inputs' own are reported as errors.
template <typename HasZeroDivisor>
void checkDecimalDivideResult(
    const cudf::column& result,
    cudf::size_type inputNullCount,
    HasZeroDivisor&& hasZeroDivisor) {
  if (result.null_count() == inputNullCount) {
    return;
  }
  if (hasZeroDivisor()) {
    VELOX_USER_FAIL(
        "{}", decimalDivisionByZeroMessage(cudf::binary_operator::DIV));
  }
  VELOX_USER_FAIL("{}", decimalOverflowMessage(cudf::binary_operator::DIV));
}

} // namespace

template <typename Lhs, typename Rhs>
std::unique_ptr<cudf::column> decimalBinaryOperation(
    const Lhs& lhs,
    const Rhs& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto [result, status] = decimalBinaryOperationWithOverflow(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
  checkDecimalBinaryOpStatus(status, op);
  return std::move(result);
}

template std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::column_view&,
    const cudf::column_view&,
    cudf::binary_operator,
    cudf::data_type,
    int32_t,
    cuda::stream_ref,
    rmm::device_async_resource_ref);

template std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::column_view&,
    const cudf::scalar&,
    cudf::binary_operator,
    cudf::data_type,
    int32_t,
    cuda::stream_ref,
    rmm::device_async_resource_ref);

template std::unique_ptr<cudf::column> decimalBinaryOperation(
    const cudf::scalar&,
    const cudf::column_view&,
    cudf::binary_operator,
    cudf::data_type,
    int32_t,
    cuda::stream_ref,
    rmm::device_async_resource_ref);

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(lhs.size(), rhs.size(), "Decimal divide requires equal sizes");
  checkDecimalDivideRescale(lhs.type(), rhs.type(), outputType);

  auto result = cudf::divide_decimal(
      lhs,
      rhs,
      outputType,
      cudf::rounding_method::HALF_UP,
      stream,
      cudf::memory_resources{mr, get_temp_mr()});

  // Rows where either input is null are null in the result without failing.
  auto activeMask = cudf::create_null_mask(
      0,
      cudf::mask_state::UNALLOCATED,
      stream,
      cudf::memory_resources{get_temp_mr(), get_temp_mr()});
  cudf::size_type inputNullCount = 0;
  if (lhs.has_nulls() || rhs.has_nulls()) {
    std::tie(activeMask, inputNullCount) = cudf::bitmask_and(
        cudf::table_view({lhs, rhs}),
        stream,
        cudf::memory_resources{get_temp_mr(), get_temp_mr()});
  }
  checkDecimalDivideResult(*result, inputNullCount, [&] {
    return hasActiveZeroDivisor(rhs, activeMask.data(), inputNullCount, stream);
  });
  return result;
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  checkDecimalDivideRescale(lhs.type(), rhs.type(), outputType);

  auto result = cudf::divide_decimal(
      lhs,
      rhs,
      outputType,
      cudf::rounding_method::HALF_UP,
      stream,
      cudf::memory_resources{mr, get_temp_mr()});

  auto const inputNullCount =
      rhs.is_valid(stream) ? lhs.null_count() : lhs.size();
  checkDecimalDivideResult(*result, inputNullCount, [&] {
    return detail::getDecimalScalarValue(rhs, stream) == 0;
  });
  return result;
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  checkDecimalDivideRescale(lhs.type(), rhs.type(), outputType);

  auto result = cudf::divide_decimal(
      lhs,
      rhs,
      outputType,
      cudf::rounding_method::HALF_UP,
      stream,
      cudf::memory_resources{mr, get_temp_mr()});

  auto const inputNullCount =
      lhs.is_valid(stream) ? rhs.null_count() : rhs.size();
  checkDecimalDivideResult(*result, inputNullCount, [&] {
    return hasActiveZeroDivisor(
        rhs, rhs.null_mask(), rhs.null_count(), stream);
  });
  return result;
}

} // namespace facebook::velox::cudf_velox
