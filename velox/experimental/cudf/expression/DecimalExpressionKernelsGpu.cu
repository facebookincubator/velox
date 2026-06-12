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
#include <cudf/utilities/type_dispatcher.hpp>

#include <cub/device/device_for.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <concepts>
#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

// Device-safe int128 bounds (std::numeric_limits is host-only in CUDA).
constexpr unsigned __int128 kUnsigned128Max =
    static_cast<unsigned __int128>(-1);
constexpr unsigned __int128 kInt128MinMagnitude =
    static_cast<unsigned __int128>(1) << 127;
constexpr unsigned __int128 kInt128MaxMagnitude = kInt128MinMagnitude - 1;
constexpr __int128_t kInt128Max =
    static_cast<__int128_t>(kInt128MaxMagnitude);
// Bit pattern 2^127 maps to INT128_MIN without negating INT128_MIN (UB).
constexpr __int128_t kInt128Min =
    static_cast<__int128_t>(kInt128MinMagnitude);

// Extract absolute value in unsigned space. Signed negation of INT128_MIN is
// undefined; negating the unsigned bit pattern is always defined.
__device__ inline unsigned __int128
absToUnsigned(__int128_t value, bool& negative) {
  if (value < 0) {
    negative = !negative;
    return -static_cast<unsigned __int128>(value);
  }
  return static_cast<unsigned __int128>(value);
}

// Reapply sign after unsigned divide/round. Magnitudes >= 2^127 cannot be
// represented as positive int128; magnitude == 2^127 is exactly INT128_MIN.
__device__ inline __int128_t
signedFromUnsigned(unsigned __int128 magnitude, bool negative) {
  if (!negative) {
    if (magnitude > kInt128MaxMagnitude) {
      return kInt128Max;
    }
    return static_cast<__int128_t>(magnitude);
  }
  if (magnitude >= kInt128MinMagnitude) {
    return kInt128Min;
  }
  return -static_cast<__int128_t>(magnitude);
}

// Decimal divide with rescale (numerator * scale / denom), half-up rounding.
// All intermediate math uses unsigned magnitudes so multiply, divide, mod, and
// abs never hit signed overflow or INT128_MIN negation UB.
template <typename OutT>
__device__ OutT
decimalDivideImpl(__int128_t numerator, __int128_t denom, __int128_t scale) {
  if (denom == 0) {
    return OutT{0};
  }

  bool negative = false;
  unsigned __int128 const uNum = absToUnsigned(numerator, negative);
  unsigned __int128 const uDenom = absToUnsigned(denom, negative);
  // scale comes from pow10Int128 and is always positive.
  unsigned __int128 const uScale = static_cast<unsigned __int128>(scale);

  unsigned __int128 scaled = uNum * uScale;
  // Detect unsigned multiply overflow; saturate to int128 min/max for sign.
  if (uScale != 0 && scaled / uScale != uNum) {
    return static_cast<OutT>(signedFromUnsigned(kUnsigned128Max, negative));
  }

  unsigned __int128 quotient = scaled / uDenom;
  unsigned __int128 const remainder = scaled % uDenom;

  // Half-up: round up when remainder >= denom/2. Equivalent to
  // 2 * remainder >= denom but avoids overflow when remainder is large.
  if (remainder > (uDenom - 1) / 2) {
    // Guard ++quotient when quotient is already UINT128_MAX.
    if (quotient < kUnsigned128Max) {
      ++quotient;
    }
  }

  return static_cast<OutT>(signedFromUnsigned(quotient, negative));
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

} // namespace

namespace detail {

template <typename InT, typename OutT>
concept ValidDecimalDivideStorageTypes =
    (std::same_as<InT, int64_t> &&
     (std::same_as<OutT, int64_t> || std::same_as<OutT, __int128_t>)) ||
    (std::same_as<InT, __int128_t> && std::same_as<OutT, __int128_t>);

struct divideColumnColumnKernel {
  const cudf::column_view& lhs;
  const cudf::column_view& rhs;
  cudf::mutable_column_view out;
  int32_t aRescale;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  void operator()() const {
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

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  void operator()() const {}
};

struct divideColumnScalarKernel {
  const cudf::column_view& lhs;
  __int128_t rhsValue;
  cudf::mutable_column_view out;
  int32_t aRescale;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  void operator()() const {
    if (lhs.size() == 0) {
      return;
    }
    DivideRhsScalarFunctor<InT, OutT> op{
        lhs.data<InT>(),
        rhsValue,
        out.data<OutT>(),
        pow10Int128(aRescale)};
    cub::DeviceFor::ForEachN(
        thrust::counting_iterator<int32_t>(0), lhs.size(), op, stream.value());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  void operator()() const {}
};

struct divideScalarColumnKernel {
  __int128_t lhsValue;
  const cudf::column_view& rhs;
  cudf::mutable_column_view out;
  int32_t aRescale;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  void operator()() const {
    if (rhs.size() == 0) {
      return;
    }
    DivideLhsScalarFunctor<InT, OutT> op{
        lhsValue,
        rhs.data<InT>(),
        out.data<OutT>(),
        pow10Int128(aRescale)};
    cub::DeviceFor::ForEachN(
        thrust::counting_iterator<int32_t>(0), rhs.size(), op, stream.value());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  void operator()() const {}
};

void decimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideColumnColumnKernel{lhs, rhs, out, aRescale, stream});
}

void decimalDivideColumnScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideColumnScalarKernel{lhs, rhsValue, out, aRescale, stream});
}

void decimalDivideScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideScalarColumnKernel{lhsValue, rhs, out, aRescale, stream});
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
