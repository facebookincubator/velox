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

#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/iterator>
#include <cuda_runtime.h>

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
// Bit pattern 2^127 maps to INT128_MIN without negating INT128_MIN (UB).
constexpr __int128_t kInt128Min = static_cast<__int128_t>(kInt128MinMagnitude);

// Match DecimalUtil::kLongDecimal{Min,Max} (10^38 bounds); duplicated here
// because Velox headers cannot be included in this translation unit (nvcc).
constexpr __int128_t kLongDecimalPowerOfTen38 = 1'000'000'000'000'000'000LL *
    static_cast<__int128_t>(1'000'000'000'000'000'000LL) * 100;
constexpr __int128_t kLongDecimalMax = kLongDecimalPowerOfTen38 - 1;
constexpr __int128_t kLongDecimalMin = -kLongDecimalPowerOfTen38 + 1;

// Device threads cannot throw; record overflow for launchDecimalDivide to
// report to the host caller, matching Velox CPU decimal divide errors.
__device__ inline void markDecimalOverflow(int32_t* overflowFlag) {
  atomicOr(overflowFlag, 1);
}

// Extract absolute value in unsigned space. Signed negation of INT128_MIN is
// undefined; negating the unsigned bit pattern is always defined.
__device__ inline unsigned __int128 absToUnsigned(
    __int128_t value,
    bool& negative) {
  if (value < 0) {
    negative = !negative;
    return -static_cast<unsigned __int128>(value);
  }
  return static_cast<unsigned __int128>(value);
}

// Reapply sign after unsigned divide/round. Caller must ensure magnitude fits.
__device__ inline __int128_t signedFromUnsigned(
    unsigned __int128 magnitude,
    bool negative) {
  if (!negative) {
    return static_cast<__int128_t>(magnitude);
  }
  if (magnitude >= kInt128MinMagnitude) {
    return kInt128Min;
  }
  return -static_cast<__int128_t>(magnitude);
}

// Quotient magnitude must fit in int128 before signedFromUnsigned; rounding can
// push a representable unsigned quotient past INT128_MAX / INT128_MIN.
__device__ inline bool fitsRepresentableInt128(
    unsigned __int128 magnitude,
    bool negative) {
  if (!negative) {
    return magnitude <= kInt128MaxMagnitude;
  }
  return magnitude <= kInt128MinMagnitude;
}

// Decimal divide with rescale (numerator * rescaleFactor / denom). Rounding
// matches Velox CPU DecimalUtil::divideWithRoundUp (increment unsigned
// quotient, then apply sign), not Java/Hive HALF_UP toward +infinity on ties.
// Overflow on rescale multiply, round-up, or out-of-range results sets
// overflowFlag (see launchDecimalDivide); intermediate math uses unsigned
// magnitudes so multiply, divide, mod, and abs never hit signed overflow UB.
template <typename OutT>
__device__ OutT decimalDivideImpl(
    __int128_t numerator,
    __int128_t denom,
    __int128_t rescaleFactor,
    int32_t* overflowFlag) {
  bool negative = false;
  unsigned __int128 const uNum = absToUnsigned(numerator, negative);
  unsigned __int128 const uDenom = absToUnsigned(denom, negative);
  // rescaleFactor is DecimalUtil::kPowersOfTen[aRescale] from the host caller.
  unsigned __int128 const uRescaleFactor =
      static_cast<unsigned __int128>(rescaleFactor);

  unsigned __int128 scaled = uNum * uRescaleFactor;
  // Match Velox CPU checkedMultiply on rescale.
  if (uRescaleFactor != 0 && scaled / uRescaleFactor != uNum) {
    markDecimalOverflow(overflowFlag);
    return OutT{0};
  }

  unsigned __int128 quotient = scaled / uDenom;
  unsigned __int128 const remainder = scaled % uDenom;

  // Round ties away from zero (e.g. -1.5 -> -2), same as Velox CPU divide.
  // Equivalent to 2 * remainder >= denom but avoids overflow when remainder is
  // large.
  if (remainder > (uDenom - 1) / 2) {
    // Round-up would wrap unsigned quotient; CPU path would overflow too.
    if (quotient >= kUnsigned128Max) {
      markDecimalOverflow(overflowFlag);
      return OutT{0};
    }
    ++quotient;
  }

  if (!fitsRepresentableInt128(quotient, negative)) {
    markDecimalOverflow(overflowFlag);
    return OutT{0};
  }

  __int128_t const result = signedFromUnsigned(quotient, negative);
  // Match Velox CPU DecimalUtil::valueInRange after divide.
  if (result < kLongDecimalMin || result > kLongDecimalMax) {
    markDecimalOverflow(overflowFlag);
    return OutT{0};
  }

  return static_cast<OutT>(result);
}

template <typename InT, typename OutT>
struct DivideFunctor {
  cudf::column_device_view lhs;
  cudf::column_device_view rhs;
  cudf::mutable_column_device_view out;
  __int128_t rescaleFactor;
  int32_t* overflowFlag;

  __device__ void operator()(cudf::size_type idx) const {
    if (lhs.is_null(idx) || rhs.is_null(idx) || rhs.element<InT>(idx) == 0) {
      out.set_null(idx);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhs.element<InT>(idx),
        rhs.element<InT>(idx),
        rescaleFactor,
        overflowFlag);
  }
};

template <typename InColT, typename OutT>
struct DivideLhsScalarFunctor {
  __int128_t lhsValue;
  cudf::column_device_view rhs;
  cudf::mutable_column_device_view out;
  __int128_t rescaleFactor;
  int32_t* overflowFlag;

  __device__ void operator()(cudf::size_type idx) const {
    if (rhs.is_null(idx) || rhs.element<InColT>(idx) == 0) {
      out.set_null(idx);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhsValue, rhs.element<InColT>(idx), rescaleFactor, overflowFlag);
  }
};

template <typename InColT, typename OutT>
struct DivideRhsScalarFunctor {
  cudf::column_device_view lhs;
  __int128_t rhsValue;
  cudf::mutable_column_device_view out;
  __int128_t rescaleFactor;
  int32_t* overflowFlag;

  __device__ void operator()(cudf::size_type idx) const {
    if (lhs.is_null(idx) || rhsValue == 0) {
      out.set_null(idx);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhs.element<InColT>(idx), rhsValue, rescaleFactor, overflowFlag);
  }
};

// Returns false if any row set overflowFlag during the kernel.
template <typename BuildOp>
bool launchDecimalDivide(
    cudf::size_type size,
    BuildOp buildOp,
    rmm::cuda_stream_view stream) {
  if (size == 0) {
    return true;
  }
  rmm::device_scalar<int32_t> overflowFlag{0, stream};
  auto op = buildOp(overflowFlag.data());
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, size, op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
  return overflowFlag.value(stream) == 0;
}

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
  __int128_t rescaleFactor;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  bool operator()() const {
    auto lhsDev = cudf::column_device_view::create(lhs, stream);
    auto rhsDev = cudf::column_device_view::create(rhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return launchDecimalDivide(
        lhs.size(),
        [&](int32_t* overflowFlag) {
          return DivideFunctor<InT, OutT>{
              *lhsDev, *rhsDev, *outDev, rescaleFactor, overflowFlag};
        },
        stream);
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  bool operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return false;
  }
};

struct divideColumnScalarKernel {
  const cudf::column_view& lhs;
  __int128_t rhsValue;
  cudf::mutable_column_view out;
  __int128_t rescaleFactor;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  bool operator()() const {
    auto lhsDev = cudf::column_device_view::create(lhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return launchDecimalDivide(
        lhs.size(),
        [&](int32_t* overflowFlag) {
          return DivideRhsScalarFunctor<InT, OutT>{
              *lhsDev, rhsValue, *outDev, rescaleFactor, overflowFlag};
        },
        stream);
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  bool operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return false;
  }
};

struct divideScalarColumnKernel {
  __int128_t lhsValue;
  const cudf::column_view& rhs;
  cudf::mutable_column_view out;
  __int128_t rescaleFactor;
  rmm::cuda_stream_view stream;

  template <typename InT, typename OutT>
    requires ValidDecimalDivideStorageTypes<InT, OutT>
  bool operator()() const {
    auto rhsDev = cudf::column_device_view::create(rhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return launchDecimalDivide(
        rhs.size(),
        [&](int32_t* overflowFlag) {
          return DivideLhsScalarFunctor<InT, OutT>{
              lhsValue, *rhsDev, *outDev, rescaleFactor, overflowFlag};
        },
        stream);
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  bool operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return false;
  }
};

bool decimalDivideColumnColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream) {
  return cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideColumnColumnKernel{lhs, rhs, out, rescaleFactor, stream});
}

bool decimalDivideColumnScalar(
    cudf::type_id inType,
    cudf::type_id outType,
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream) {
  return cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideColumnScalarKernel{lhs, rhsValue, out, rescaleFactor, stream});
}

bool decimalDivideScalarColumn(
    cudf::type_id inType,
    cudf::type_id outType,
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    __int128_t rescaleFactor,
    rmm::cuda_stream_view stream) {
  return cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{inType},
      cudf::data_type{outType},
      divideScalarColumnKernel{lhsValue, rhs, out, rescaleFactor, stream});
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
