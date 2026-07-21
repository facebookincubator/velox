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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/operators/checked_arithmetic.cuh>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/iterator>
#include <cuda_runtime.h>

#include <concepts>
#include <cstdint>
#include <type_traits>

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

// Device threads cannot throw, so overflow is recorded in a single per-launch
// device flag that launchOverflowChecked reports back to the host. Presto /
// Velox CPU decimal arithmetic is fail-fast (any overflow fails the whole
// expression), so every overflowing row simply ORs into one shared flag and no
// per-row (O(n)) overflow column is required. Shared by the divide and
// ADD/SUB/MUL/MOD kernels.
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
// overflowFlag (see launchOverflowChecked); intermediate math uses unsigned
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

// Runs buildOp(flag) across [0, size) rows behind a single per-launch overflow
// flag and reports whether any row overflowed. Shared by the divide and
// ADD/SUB/MUL/MOD kernels; buildOp receives the device flag pointer and returns
// the per-row functor. Returns true if any row set the overflow flag.
template <typename BuildOp>
bool launchOverflowChecked(
    cudf::size_type size,
    BuildOp buildOp,
    rmm::cuda_stream_view stream) {
  if (size == 0) {
    return false;
  }
  rmm::device_scalar<int32_t> overflowFlag{0, stream};
  auto op = buildOp(overflowFlag.data());
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, size, op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
  return overflowFlag.value(stream) != 0;
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
    return !launchOverflowChecked(
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
    return !launchOverflowChecked(
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
    return !launchOverflowChecked(
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

} // namespace detail

// ---------------------------------------------------------------------------
// Overflow-checked decimal binary-op kernels (ADD / SUB / MUL / MOD).
// Division overflow is handled by the detail:: kernels above; this path covers
// the remaining fixed-point arithmetic. Overflow is tracked with a single
// device-side flag (atomicOr) to match the fail-fast semantics of Presto /
// Velox CPU decimal arithmetic.
// ---------------------------------------------------------------------------
namespace {

using errc = cudf::errc;

__device__ inline bool
isRowActive(cudf::bitmask_type const* nullMask, bool hasNullMask, int32_t idx) {
  return !hasNullMask || cudf::bit_is_set(nullMask, idx);
}

template <typename Rep>
__device__ numeric::decimal<Rep> makeDecimal(
    Rep value,
    numeric::scale_type scale) {
  return numeric::decimal<Rep>{numeric::scaled_integer<Rep>{value, scale}};
}

template <typename Rep>
__device__ cuda::std::expected<numeric::decimal<Rep>, errc> checkedRescale(
    numeric::decimal<Rep> value,
    numeric::scale_type targetScale) {
  // Moving to a smaller scale widens the representation by a power of ten,
  // which can overflow Rep. cuDF's decimal::rescaled performs the shift but
  // does not detect that overflow, so guard the widening case explicitly and
  // delegate the actual rescale to cuDF.
  auto const growDigits =
      static_cast<int32_t>(value.scale()) - static_cast<int32_t>(targetScale);
  if (growDigits > 0) {
    auto const multiplier =
        numeric::detail::ipow<Rep, numeric::Radix::BASE_10>(growDigits);
    if (numeric::multiplication_overflow<Rep>(value.value(), multiplier)) {
      return cuda::std::unexpected{errc::OVERFLOW};
    }
  }
  return value.rescaled(targetScale);
}

template <typename Rep>
__device__ cuda::std::expected<numeric::decimal<Rep>, errc> applyCheckedBinOp(
    cudf::binary_operator op,
    numeric::decimal<Rep> lhs,
    numeric::decimal<Rep> rhs) {
  switch (op) {
    case cudf::binary_operator::ADD:
      return cudf::detail::ops::add_overflow(lhs, rhs);
    case cudf::binary_operator::SUB:
      return cudf::detail::ops::sub_overflow(lhs, rhs);
    case cudf::binary_operator::MUL:
      return cudf::detail::ops::mul_overflow(lhs, rhs);
    case cudf::binary_operator::MOD:
      return cudf::detail::ops::mod_overflow(lhs, rhs);
    default:
      return cuda::std::unexpected{errc::OVERFLOW};
  }
}

// Computes one row of a checked decimal binary op using cuDF's *_overflow
// operators. On overflow it ORs the shared flag and writes a well-defined 0
// (the value is irrelevant because the host fails the whole batch).
template <typename Rep, typename OutRep>
__device__ void evalDecimalBinaryRow(
    numeric::decimal<Rep> lhsDec,
    numeric::decimal<Rep> rhsDec,
    cudf::binary_operator op,
    numeric::scale_type outScale,
    int32_t outPrecision,
    OutRep* out,
    int32_t idx,
    int32_t* overflowFlag) {
  // ADD/SUB/MOD require both operands at a common scale (the output scale).
  // cuDF's *_overflow operators reach that scale via fixed_point::rescaled,
  // whose widening multiply is unchecked and can silently overflow Rep for
  // mixed-scale operands. Pre-rescale the operands here with overflow
  // detection so the conversion is covered by the same fail-fast path. MUL is
  // excluded: it adds operand scales and its product is rescaled to outScale
  // by the checkedRescale below.
  if (op == cudf::binary_operator::ADD ||
      op == cudf::binary_operator::SUB ||
      op == cudf::binary_operator::MOD) {
    auto lhsRescaled = checkedRescale<Rep>(lhsDec, outScale);
    auto rhsRescaled = checkedRescale<Rep>(rhsDec, outScale);
    if (!lhsRescaled.has_value() || !rhsRescaled.has_value()) {
      markDecimalOverflow(overflowFlag);
      out[idx] = OutRep{0};
      return;
    }
    lhsDec = lhsRescaled.value();
    rhsDec = rhsRescaled.value();
  }

  auto opResult = applyCheckedBinOp<Rep>(op, lhsDec, rhsDec);
  if (!opResult.has_value()) {
    markDecimalOverflow(overflowFlag);
    out[idx] = OutRep{0};
    return;
  }

  auto rescaled = checkedRescale<Rep>(opResult.value(), outScale);
  if (!rescaled.has_value()) {
    markDecimalOverflow(overflowFlag);
    out[idx] = OutRep{0};
    return;
  }

  auto precisionChecked =
      cudf::detail::ops::check_precision(rescaled.value(), outPrecision);
  if (!precisionChecked.has_value()) {
    markDecimalOverflow(overflowFlag);
    out[idx] = OutRep{0};
    return;
  }
  out[idx] = static_cast<OutRep>(precisionChecked.value().value());
}

template <typename Rep, typename OutRep>
struct DecimalBinaryColColFunctor {
  const Rep* lhs;
  const Rep* rhs;
  OutRep* out;
  int32_t* overflowFlag;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        makeDecimal<Rep>(lhs[idx], lhsScale),
        makeDecimal<Rep>(rhs[idx], rhsScale),
        op,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

template <typename Rep, typename OutRep>
struct DecimalBinaryLhsScalarFunctor {
  Rep lhsValue;
  const Rep* rhs;
  OutRep* out;
  int32_t* overflowFlag;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        makeDecimal<Rep>(lhsValue, lhsScale),
        makeDecimal<Rep>(rhs[idx], rhsScale),
        op,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

template <typename Rep, typename OutRep>
struct DecimalBinaryRhsScalarFunctor {
  const Rep* lhs;
  Rep rhsValue;
  OutRep* out;
  int32_t* overflowFlag;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        makeDecimal<Rep>(lhs[idx], lhsScale),
        makeDecimal<Rep>(rhsValue, rhsScale),
        op,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

template <typename InRep, typename OutRep>
bool launchDecimalBinaryColColKernel(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      lhs.size(),
      [&](int32_t* overflowFlag) {
        return DecimalBinaryColColFunctor<InRep, OutRep>{
            lhs.data<InRep>(),
            rhs.data<InRep>(),
            out.data<OutRep>(),
            overflowFlag,
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask,
            nullMask != nullptr};
      },
      stream);
}

template <typename InRep, typename OutRep>
bool launchDecimalBinaryRhsScalarKernel(
    cudf::column_view const& lhs,
    InRep rhsValue,
    numeric::scale_type rhsScale,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      lhs.size(),
      [&](int32_t* overflowFlag) {
        return DecimalBinaryRhsScalarFunctor<InRep, OutRep>{
            lhs.data<InRep>(),
            rhsValue,
            out.data<OutRep>(),
            overflowFlag,
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask,
            nullMask != nullptr};
      },
      stream);
}

template <typename InRep, typename OutRep>
bool launchDecimalBinaryLhsScalarKernel(
    InRep lhsValue,
    numeric::scale_type lhsScale,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      rhs.size(),
      [&](int32_t* overflowFlag) {
        return DecimalBinaryLhsScalarFunctor<InRep, OutRep>{
            lhsValue,
            rhs.data<InRep>(),
            out.data<OutRep>(),
            overflowFlag,
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask,
            nullMask != nullptr};
      },
      stream);
}

void validateDecimalBinaryOp(cudf::binary_operator op) {
  CUDF_EXPECTS(
      op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
          op == cudf::binary_operator::MUL || op == cudf::binary_operator::MOD,
      "Unsupported decimal binary operator for overflow-checked execution");
}

std::unique_ptr<cudf::column> makeResultColumn(
    cudf::size_type size,
    cudf::data_type outputType,
    rmm::device_buffer&& nullMask,
    cudf::size_type nullCount,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_fixed_width_column(
      outputType, size, std::move(nullMask), nullCount, stream, mr);
}

template <typename InRep, typename OutRep>
std::pair<std::unique_ptr<cudf::column>, bool> decimalBinaryOperationColColImpl(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Decimal binop requires equal sizes");
  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream, mr);
  auto result = makeResultColumn(
      lhs.size(), outputType, std::move(nullMask), nullCount, stream, mr);

  bool const didOverflow = launchDecimalBinaryColColKernel<InRep, OutRep>(
      lhs, rhs, result->mutable_view(), op, outputPrecision, stream);
  return {std::move(result), didOverflow};
}

std::pair<std::unique_ptr<cudf::column>, bool>
dispatchDecimalBinaryOperationColCol(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  bool const inputsAreDecimal64 = lhs.type().id() == cudf::type_id::DECIMAL64 &&
      rhs.type().id() == cudf::type_id::DECIMAL64;
  if (inputsAreDecimal64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      return decimalBinaryOperationColColImpl<int64_t, int64_t>(
          lhs, rhs, op, outputType, outputPrecision, stream, mr);
    }
    return decimalBinaryOperationColColImpl<int64_t, __int128_t>(
        lhs, rhs, op, outputType, outputPrecision, stream, mr);
  }
  return decimalBinaryOperationColColImpl<__int128_t, __int128_t>(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
}

template <typename InRep>
InRep getTypedDecimalScalarValue(
    const cudf::scalar& s,
    rmm::cuda_stream_view stream) {
  if constexpr (std::is_same_v<InRep, int64_t>) {
    return static_cast<int64_t>(detail::getDecimalScalarValue(s, stream));
  }
  return static_cast<InRep>(detail::getDecimalScalarValue(s, stream));
}

} // namespace

std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateDecimalBinaryOp(op);
  return dispatchDecimalBinaryOperationColCol(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
}

std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateDecimalBinaryOp(op);
  if (!rhs.is_valid(stream)) {
    auto result = cudf::make_fixed_width_column(
        outputType, lhs.size(), cudf::mask_state::ALL_NULL, stream, mr);
    return {std::move(result), false};
  }
  auto nullMask = cudf::copy_bitmask(lhs, stream, mr);
  auto result = makeResultColumn(
      lhs.size(),
      outputType,
      std::move(nullMask),
      lhs.null_count(),
      stream,
      mr);

  bool didOverflow = false;
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    auto const rhsValue = getTypedDecimalScalarValue<int64_t>(rhs, stream);
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      didOverflow = launchDecimalBinaryRhsScalarKernel<int64_t, int64_t>(
          lhs,
          rhsValue,
          rhsScale,
          result->mutable_view(),
          op,
          outputPrecision,
          stream);
    } else {
      didOverflow = launchDecimalBinaryRhsScalarKernel<int64_t, __int128_t>(
          lhs,
          rhsValue,
          rhsScale,
          result->mutable_view(),
          op,
          outputPrecision,
          stream);
    }
  } else {
    auto const rhsValue = getTypedDecimalScalarValue<__int128_t>(rhs, stream);
    didOverflow = launchDecimalBinaryRhsScalarKernel<__int128_t, __int128_t>(
        lhs,
        rhsValue,
        rhsScale,
        result->mutable_view(),
        op,
        outputPrecision,
        stream);
  }
  return {std::move(result), didOverflow};
}

std::pair<std::unique_ptr<cudf::column>, bool>
decimalBinaryOperationWithOverflow(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateDecimalBinaryOp(op);
  if (!lhs.is_valid(stream)) {
    auto result = cudf::make_fixed_width_column(
        outputType, rhs.size(), cudf::mask_state::ALL_NULL, stream, mr);
    return {std::move(result), false};
  }
  auto nullMask = cudf::copy_bitmask(rhs, stream, mr);
  auto result = makeResultColumn(
      rhs.size(),
      outputType,
      std::move(nullMask),
      rhs.null_count(),
      stream,
      mr);

  bool didOverflow = false;
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  if (rhs.type().id() == cudf::type_id::DECIMAL64) {
    auto const lhsValue = getTypedDecimalScalarValue<int64_t>(lhs, stream);
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      didOverflow = launchDecimalBinaryLhsScalarKernel<int64_t, int64_t>(
          lhsValue,
          lhsScale,
          rhs,
          result->mutable_view(),
          op,
          outputPrecision,
          stream);
    } else {
      didOverflow = launchDecimalBinaryLhsScalarKernel<int64_t, __int128_t>(
          lhsValue,
          lhsScale,
          rhs,
          result->mutable_view(),
          op,
          outputPrecision,
          stream);
    }
  } else {
    auto const lhsValue = getTypedDecimalScalarValue<__int128_t>(lhs, stream);
    didOverflow = launchDecimalBinaryLhsScalarKernel<__int128_t, __int128_t>(
        lhsValue,
        lhsScale,
        rhs,
        result->mutable_view(),
        op,
        outputPrecision,
        stream);
  }
  return {std::move(result), didOverflow};
}

} // namespace facebook::velox::cudf_velox
