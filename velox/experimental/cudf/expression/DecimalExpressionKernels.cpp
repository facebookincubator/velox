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
#include "velox/experimental/cudf/expression/DecimalExpressionKernelsGpu.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>

namespace facebook::velox::cudf_velox {
namespace {

__int128_t getDecimalScalarValue(
    const cudf::scalar& s,
    rmm::cuda_stream_view stream) {
  if (s.type().id() == cudf::type_id::DECIMAL64) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(s);
    return static_cast<__int128_t>(static_cast<int64_t>(dec.value(stream)));
  }
  auto const& dec =
      static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(s);
  return static_cast<__int128_t>(dec.value(stream));
}

/// Column of \p outputType with \p size rows, all null (e.g. NULL scalar
/// operand).
std::unique_ptr<cudf::column> makeAllNullDecimalColumn(
    cudf::data_type outputType,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (size == 0) {
    return cudf::make_empty_column(outputType);
  }
  return cudf::make_fixed_width_column(
      outputType, size, cudf::mask_state::ALL_NULL, stream, mr);
}

} // namespace

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

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(lhs.size() == rhs.size(), "Decimal divide requires equal sizes");
  // Use VELOX_CHECK (not _EQ) so failed checks do not pass cudf::type_id into
  // fmt, which has no formatter for that enum.
  VELOX_CHECK(
      lhs.type().id() == rhs.type().id(),
      "Decimal divide requires matching input types");
  VELOX_CHECK_GE(
      aRescale, 0, "Decimal divide requires non-negative rescale factor");

  const auto inType = lhs.type().id();
  const auto outType = outputType.id();
  VELOX_CHECK(
      inType == cudf::type_id::DECIMAL64 || inType == cudf::type_id::DECIMAL128,
      "Unsupported input type for decimal divide");
  if (inType == cudf::type_id::DECIMAL64) {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL64 ||
            outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  } else {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  }

  // Combine input null masks (lhs and rhs nulls).
  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream, mr);

  // Create output column with input null mask and perform division.
  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), std::move(nullMask), nullCount, stream, mr);

  detail::decimalDivideColumnColumn(
      inType, outType, lhs, rhs, out->mutable_view(), aRescale, stream);

  // Scatter nulls where divisor is zero.
  return scatterNullsAtZeroDivisor(std::move(out), rhs, stream, mr);
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GE(
      aRescale, 0, "Decimal divide requires non-negative rescale factor");

  if (!rhs.is_valid(stream)) {
    return makeAllNullDecimalColumn(outputType, lhs.size(), stream, mr);
  }

  auto nullMask = cudf::copy_bitmask(lhs, stream, mr);
  auto nullCount = lhs.null_count();
  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), std::move(nullMask), nullCount, stream, mr);

  auto rhsValue = getDecimalScalarValue(rhs, stream);

  const auto inType = lhs.type().id();
  const auto outType = outputType.id();
  VELOX_CHECK(
      inType == cudf::type_id::DECIMAL64 || inType == cudf::type_id::DECIMAL128,
      "Unsupported input type for decimal divide");
  if (inType == cudf::type_id::DECIMAL64) {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL64 ||
            outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  } else {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  }

  detail::decimalDivideColumnScalar(
      inType, outType, lhs, rhsValue, out->mutable_view(), aRescale, stream);

  return out;
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GE(
      aRescale, 0, "Decimal divide requires non-negative rescale factor");

  if (!lhs.is_valid(stream)) {
    return makeAllNullDecimalColumn(outputType, rhs.size(), stream, mr);
  }

  // Copy rhs null mask.
  auto nullMask = cudf::copy_bitmask(rhs, stream, mr);
  auto nullCount = rhs.null_count();

  // Create output column and perform division.
  auto out = cudf::make_fixed_width_column(
      outputType, rhs.size(), std::move(nullMask), nullCount, stream, mr);

  auto lhsValue = getDecimalScalarValue(lhs, stream);

  const auto inType = rhs.type().id();
  const auto outType = outputType.id();
  VELOX_CHECK(
      inType == cudf::type_id::DECIMAL64 || inType == cudf::type_id::DECIMAL128,
      "Unsupported input type for decimal divide");
  if (inType == cudf::type_id::DECIMAL64) {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL64 ||
            outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  } else {
    VELOX_CHECK(
        outType == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
  }

  detail::decimalDivideScalarColumn(
      inType, outType, lhsValue, rhs, out->mutable_view(), aRescale, stream);

  // Scatter nulls where divisor is zero.
  return scatterNullsAtZeroDivisor(std::move(out), rhs, stream, mr);
}

} // namespace facebook::velox::cudf_velox
