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

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

template <typename OutT>
__device__ OutT
decimalDivideImpl(__int128_t numerator, __int128_t denom, __int128_t scale) {
  if (denom == 0) {
    return OutT{0};
  }
  int sign = 1;
  if (numerator < 0) {
    numerator = -numerator;
    sign = -sign;
  }
  if (denom < 0) {
    denom = -denom;
    sign = -sign;
  }
  __int128_t scaled = numerator * scale;
  __int128_t quotient = scaled / denom;
  __int128_t remainder = scaled % denom;
  if (remainder * 2 >= denom) {
    ++quotient;
  }
  if (sign < 0) {
    quotient = -quotient;
  }
  return static_cast<OutT>(quotient);
}

template <typename InT, typename OutT>
__global__ void decimalDivideKernel(
    const InT* lhs,
    const InT* rhs,
    OutT* out,
    int32_t numRows,
    __int128_t scale) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  out[idx] = decimalDivideImpl<OutT>(lhs[idx], rhs[idx], scale);
}

template <typename InColT, typename OutT>
__global__ void decimalDivideKernelLhsScalar(
    __int128_t lhsValue,
    const InColT* rhs,
    OutT* out,
    int32_t numRows,
    __int128_t scale) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  out[idx] = decimalDivideImpl<OutT>(lhsValue, rhs[idx], scale);
}

template <typename InColT, typename OutT>
__global__ void decimalDivideKernelRhsScalar(
    const InColT* lhs,
    __int128_t rhsValue,
    OutT* out,
    int32_t numRows,
    __int128_t scale) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  out[idx] = decimalDivideImpl<OutT>(lhs[idx], rhsValue, scale);
}

inline __int128_t pow10Int128(int32_t exp) {
  __int128_t value = 1;
  for (int32_t i = 0; i < exp; ++i) {
    value *= 10;
  }
  return value;
}

template <typename InT, typename OutT>
void launchDivideKernel(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  int32_t blockSize = 256;
  int32_t gridSize = (lhs.size() + blockSize - 1) / blockSize;
  auto scale = pow10Int128(aRescale);
  decimalDivideKernel<<<gridSize, blockSize, 0, stream.value()>>>(
      lhs.data<InT>(), rhs.data<InT>(), out.data<OutT>(), lhs.size(), scale);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename InColT, typename OutT>
void launchDivideKernelLhsScalar(
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (rhs.size() == 0) {
    return;
  }
  int32_t blockSize = 256;
  int32_t gridSize = (rhs.size() + blockSize - 1) / blockSize;
  auto scale = pow10Int128(aRescale);
  decimalDivideKernelLhsScalar<<<gridSize, blockSize, 0, stream.value()>>>(
      lhsValue, rhs.data<InColT>(), out.data<OutT>(), rhs.size(), scale);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename InColT, typename OutT>
void launchDivideKernelRhsScalar(
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  int32_t blockSize = 256;
  int32_t gridSize = (lhs.size() + blockSize - 1) / blockSize;
  auto scale = pow10Int128(aRescale);
  decimalDivideKernelRhsScalar<<<gridSize, blockSize, 0, stream.value()>>>(
      lhs.data<InColT>(), rhsValue, out.data<OutT>(), lhs.size(), scale);
  CUDF_CUDA_TRY(cudaGetLastError());
}

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

} // namespace

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Decimal divide requires equal sizes");
  CUDF_EXPECTS(
      lhs.type().id() == rhs.type().id(),
      "Decimal divide requires matching input types");
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream);
  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), std::move(nullMask), nullCount, stream);

  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernel<int64_t, int64_t>(
          lhs, rhs, out->mutable_view(), aRescale, stream);
    } else {
      CUDF_EXPECTS(
          outputType.id() == cudf::type_id::DECIMAL128,
          "Unexpected output type for decimal divide");
      launchDivideKernel<int64_t, __int128_t>(
          lhs, rhs, out->mutable_view(), aRescale, stream);
    }
  } else {
    CUDF_EXPECTS(
        lhs.type().id() == cudf::type_id::DECIMAL128,
        "Unsupported input type for decimal divide");
    CUDF_EXPECTS(
        outputType.id() == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
    launchDivideKernel<__int128_t, __int128_t>(
        lhs, rhs, out->mutable_view(), aRescale, stream);
  }

  return out;
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  auto nullMask = cudf::copy_bitmask(lhs, stream);
  auto nullCount = lhs.null_count();
  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), std::move(nullMask), nullCount, stream);

  auto rhsValue = getDecimalScalarValue(rhs, stream);

  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernelRhsScalar<int64_t, int64_t>(
          lhs, rhsValue, out->mutable_view(), aRescale, stream);
    } else {
      CUDF_EXPECTS(
          outputType.id() == cudf::type_id::DECIMAL128,
          "Unexpected output type for decimal divide");
      launchDivideKernelRhsScalar<int64_t, __int128_t>(
          lhs, rhsValue, out->mutable_view(), aRescale, stream);
    }
  } else {
    CUDF_EXPECTS(
        lhs.type().id() == cudf::type_id::DECIMAL128,
        "Unsupported input type for decimal divide");
    CUDF_EXPECTS(
        outputType.id() == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
    launchDivideKernelRhsScalar<__int128_t, __int128_t>(
        lhs, rhsValue, out->mutable_view(), aRescale, stream);
  }

  return out;
}

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  auto nullMask = cudf::copy_bitmask(rhs, stream);
  auto nullCount = rhs.null_count();
  auto out = cudf::make_fixed_width_column(
      outputType, rhs.size(), std::move(nullMask), nullCount, stream);

  auto lhsValue = getDecimalScalarValue(lhs, stream);

  if (rhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernelLhsScalar<int64_t, int64_t>(
          lhsValue, rhs, out->mutable_view(), aRescale, stream);
    } else {
      CUDF_EXPECTS(
          outputType.id() == cudf::type_id::DECIMAL128,
          "Unexpected output type for decimal divide");
      launchDivideKernelLhsScalar<int64_t, __int128_t>(
          lhsValue, rhs, out->mutable_view(), aRescale, stream);
    }
  } else {
    CUDF_EXPECTS(
        rhs.type().id() == cudf::type_id::DECIMAL128,
        "Unsupported input type for decimal divide");
    CUDF_EXPECTS(
        outputType.id() == cudf::type_id::DECIMAL128,
        "Unexpected output type for decimal divide");
    launchDivideKernelLhsScalar<__int128_t, __int128_t>(
        lhsValue, rhs, out->mutable_view(), aRescale, stream);
  }

  return out;
}

} // namespace facebook::velox::cudf_velox
