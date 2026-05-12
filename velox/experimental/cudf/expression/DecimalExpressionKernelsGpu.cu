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

#include "velox/experimental/cudf/expression/DecimalExpressionKernelsGpu.h"

#include <cudf/utilities/error.hpp>

#include <cub/device/device_for.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

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

inline __int128_t pow10Int128(int32_t exp) {
  __int128_t value = 1;
  for (int32_t i = 0; i < exp; ++i) {
    value *= 10;
  }
  return value;
}

template <typename InT, typename OutT>
struct DivideFunctor {
  const InT* lhs;
  const InT* rhs;
  OutT* out;
  __int128_t scale;

  __device__ void operator()(int32_t idx) const {
    out[idx] = decimalDivideImpl<OutT>(lhs[idx], rhs[idx], scale);
  }
};

template <typename InColT, typename OutT>
struct DivideLhsScalarFunctor {
  __int128_t lhsValue;
  const InColT* rhs;
  OutT* out;
  __int128_t scale;

  __device__ void operator()(int32_t idx) const {
    out[idx] = decimalDivideImpl<OutT>(lhsValue, rhs[idx], scale);
  }
};

template <typename InColT, typename OutT>
struct DivideRhsScalarFunctor {
  const InColT* lhs;
  __int128_t rhsValue;
  OutT* out;
  __int128_t scale;

  __device__ void operator()(int32_t idx) const {
    out[idx] = decimalDivideImpl<OutT>(lhs[idx], rhsValue, scale);
  }
};

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
  DivideFunctor<InT, OutT> op{
      lhs.data<InT>(),
      rhs.data<InT>(),
      out.data<OutT>(),
      pow10Int128(aRescale)};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), lhs.size(), op, stream.value());
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
  DivideLhsScalarFunctor<InColT, OutT> op{
      lhsValue, rhs.data<InColT>(), out.data<OutT>(), pow10Int128(aRescale)};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), rhs.size(), op, stream.value());
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
  DivideRhsScalarFunctor<InColT, OutT> op{
      lhs.data<InColT>(), rhsValue, out.data<OutT>(), pow10Int128(aRescale)};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), lhs.size(), op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace

namespace detail {

void launchDecimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (inType == cudf::type_id::DECIMAL64) {
    if (outType == cudf::type_id::DECIMAL64) {
      launchDivideKernel<int64_t, int64_t>(lhs, rhs, out, aRescale, stream);
    } else {
      launchDivideKernel<int64_t, __int128_t>(lhs, rhs, out, aRescale, stream);
    }
  } else {
    launchDivideKernel<__int128_t, __int128_t>(lhs, rhs, out, aRescale, stream);
  }
}

void launchDecimalDivideColumnRhsScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (inType == cudf::type_id::DECIMAL64) {
    if (outType == cudf::type_id::DECIMAL64) {
      launchDivideKernelRhsScalar<int64_t, int64_t>(
          lhs, rhsValue, out, aRescale, stream);
    } else {
      launchDivideKernelRhsScalar<int64_t, __int128_t>(
          lhs, rhsValue, out, aRescale, stream);
    }
  } else {
    launchDivideKernelRhsScalar<__int128_t, __int128_t>(
        lhs, rhsValue, out, aRescale, stream);
  }
}

void launchDecimalDivideLhsScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (inType == cudf::type_id::DECIMAL64) {
    if (outType == cudf::type_id::DECIMAL64) {
      launchDivideKernelLhsScalar<int64_t, int64_t>(
          lhsValue, rhs, out, aRescale, stream);
    } else {
      launchDivideKernelLhsScalar<int64_t, __int128_t>(
          lhsValue, rhs, out, aRescale, stream);
    }
  } else {
    launchDivideKernelLhsScalar<__int128_t, __int128_t>(
        lhsValue, rhs, out, aRescale, stream);
  }
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
