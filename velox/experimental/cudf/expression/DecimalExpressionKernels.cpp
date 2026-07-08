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
#include "velox/type/DecimalUtil.h"
#include "velox/type/Type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>

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
  return cudf::make_fixed_width_column(
      outputType, size, cudf::mask_state::ALL_NULL, stream, mr);
}

void checkDecimalDivideTypes(cudf::type_id inType, cudf::type_id outType) {
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
}

void finalizeDivideOutputNullCount(
    cudf::column& out,
    rmm::cuda_stream_view stream) {
  if (out.size() == 0) {
    return;
  }
  auto const nullCount =
      cudf::null_count(out.view().null_mask(), 0, out.size(), stream);
  out.set_null_count(nullCount);
}

} // namespace

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(lhs.size(), rhs.size(), "Decimal divide requires equal sizes");
  // Use VELOX_CHECK (not _EQ) so failed checks do not pass cudf::type_id into
  // fmt, which has no formatter for that enum.
  VELOX_CHECK(
      lhs.type().id() == rhs.type().id(),
      "Decimal divide requires matching input types");
  VELOX_CHECK_GE(
      aRescale, 0, "Decimal divide requires non-negative rescale factor");
  // Rescale indexes DecimalUtil::kPowersOfTen; same bound as Presto divide
  // init.
  VELOX_USER_CHECK_LE(
      aRescale, LongDecimalType::kMaxPrecision, "Decimal overflow");

  const auto inType = lhs.type().id();
  const auto outType = outputType.id();
  checkDecimalDivideTypes(inType, outType);

  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), cudf::mask_state::ALL_VALID, stream, mr);

  const __int128_t rescaleFactor = DecimalUtil::kPowersOfTen[aRescale];
  VELOX_USER_CHECK(
      detail::decimalDivideColumnColumn(
          inType,
          outType,
          lhs,
          rhs,
          out->mutable_view(),
          rescaleFactor,
          stream),
      "Decimal overflow");

  finalizeDivideOutputNullCount(*out, stream);
  return out;
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
  // Rescale indexes DecimalUtil::kPowersOfTen; same bound as Presto divide
  // init.
  VELOX_USER_CHECK_LE(
      aRescale, LongDecimalType::kMaxPrecision, "Decimal overflow");

  if (!rhs.is_valid(stream)) {
    return makeAllNullDecimalColumn(outputType, lhs.size(), stream, mr);
  }

  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), cudf::mask_state::ALL_VALID, stream, mr);

  auto rhsValue = getDecimalScalarValue(rhs, stream);

  const auto inType = lhs.type().id();
  const auto outType = outputType.id();
  checkDecimalDivideTypes(inType, outType);

  VELOX_USER_CHECK(
      detail::decimalDivideColumnScalar(
          inType,
          outType,
          lhs,
          rhsValue,
          out->mutable_view(),
          DecimalUtil::kPowersOfTen[aRescale],
          stream),
      "Decimal overflow");

  finalizeDivideOutputNullCount(*out, stream);
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
  // Rescale indexes DecimalUtil::kPowersOfTen; same bound as Presto divide
  // init.
  VELOX_USER_CHECK_LE(
      aRescale, LongDecimalType::kMaxPrecision, "Decimal overflow");

  if (!lhs.is_valid(stream)) {
    return makeAllNullDecimalColumn(outputType, rhs.size(), stream, mr);
  }

  auto out = cudf::make_fixed_width_column(
      outputType, rhs.size(), cudf::mask_state::ALL_VALID, stream, mr);

  auto lhsValue = getDecimalScalarValue(lhs, stream);

  const auto inType = rhs.type().id();
  const auto outType = outputType.id();
  checkDecimalDivideTypes(inType, outType);

  const __int128_t rescaleFactor = DecimalUtil::kPowersOfTen[aRescale];
  VELOX_USER_CHECK(
      detail::decimalDivideScalarColumn(
          inType,
          outType,
          lhsValue,
          rhs,
          out->mutable_view(),
          rescaleFactor,
          stream),
      "Decimal overflow");

  finalizeDivideOutputNullCount(*out, stream);
  return out;
}

} // namespace facebook::velox::cudf_velox
