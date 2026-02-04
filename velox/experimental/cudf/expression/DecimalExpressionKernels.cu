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
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

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
  __int128_t numerator = static_cast<__int128_t>(lhs[idx]);
  __int128_t denom = static_cast<__int128_t>(rhs[idx]);
  if (denom == 0) {
    out[idx] = OutT{0};
    return;
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
  out[idx] = static_cast<OutT>(quotient);
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
      lhs.data<InT>(),
      rhs.data<InT>(),
      out.data<OutT>(),
      lhs.size(),
      scale);
  CUDF_CUDA_TRY(cudaGetLastError());
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
      aRescale >= 0,
      "Decimal divide requires non-negative rescale factor");

  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream);
  auto out = cudf::make_fixed_width_column(
      outputType,
      lhs.size(),
      std::move(nullMask),
      nullCount,
      stream);

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

} // namespace facebook::velox::cudf_velox
