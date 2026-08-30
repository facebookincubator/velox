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
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/operators/checked_arithmetic.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

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

// Device threads cannot throw, so failures are recorded in a single per-launch
// device flag that launchOverflowChecked reports back to the host. Presto /
// Velox CPU decimal arithmetic is fail-fast (any failing row fails the whole
// expression), so every failing row ORs its status bit into one shared flag and
// no per-row (O(n)) status column is required. Shared by the divide and
// ADD/SUB/MUL/MOD kernels.
//
// Distinct bits keep division-by-zero separate from overflow (matching
// cudf::errc OVERFLOW=1 / DIVISION_BY_ZERO=2), letting the host raise the
// matching error kind.
//
// Errors are accumulated in a thread-local register during the grid-stride
// loop; each thread performs at most one global atomicOr when it finishes.
constexpr int32_t kDecimalOverflowBit = 1;
constexpr int32_t kDecimalDivByZeroBit = 2;
constexpr int32_t kOverflowCheckedBlockSize = 256;

__device__ inline void markError(
    int32_t& localStatus,
    DecimalBinaryOpStatus status) {
  localStatus |= static_cast<int32_t>(status);
}

// Host precomputes the output null stencil (bitmask_and / copy_bitmask) before
// the kernel. A null mask pointer of nullptr means every row is active.
__device__ inline bool isRowActive(
    cudf::bitmask_type const* nullMask,
    int32_t idx) {
  return nullMask == nullptr || cudf::bit_is_set(nullMask, idx);
}

// Maps the raw device overflowFlag bits to a DecimalBinaryOpStatus. Division by
// zero takes precedence over overflow to match Velox CPU, which validates the
// divisor before the arithmetic.
DecimalBinaryOpStatus toDecimalBinaryOpStatus(int32_t overflowFlag) {
  if (overflowFlag & kDecimalDivByZeroBit) {
    return DecimalBinaryOpStatus::kDivisionByZero;
  }
  if (overflowFlag & kDecimalOverflowBit) {
    return DecimalBinaryOpStatus::kOverflow;
  }
  return DecimalBinaryOpStatus::kOk;
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
// Overflow on rescale multiply, round-up, or out-of-range results ORs into
// localStatus (flushed once per thread by launchOverflowChecked); intermediate
// math uses unsigned magnitudes so multiply, divide, mod, and abs never hit
// signed overflow UB.
template <typename OutT>
__device__ OutT decimalDivideImpl(
    __int128_t numerator,
    __int128_t denom,
    __int128_t rescaleFactor,
    int32_t& localStatus) {
  bool negative = false;
  unsigned __int128 const uNum = absToUnsigned(numerator, negative);
  unsigned __int128 const uDenom = absToUnsigned(denom, negative);
  // rescaleFactor is DecimalUtil::kPowersOfTen[aRescale] from the host caller.
  unsigned __int128 const uRescaleFactor =
      static_cast<unsigned __int128>(rescaleFactor);

  unsigned __int128 scaled = uNum * uRescaleFactor;
  // Match Velox CPU checkedMultiply on rescale.
  if (uRescaleFactor != 0 && scaled / uRescaleFactor != uNum) {
    markError(localStatus, DecimalBinaryOpStatus::kOverflow);
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
      markError(localStatus, DecimalBinaryOpStatus::kOverflow);
      return OutT{0};
    }
    ++quotient;
  }

  if (!fitsRepresentableInt128(quotient, negative)) {
    markError(localStatus, DecimalBinaryOpStatus::kOverflow);
    return OutT{0};
  }

  __int128_t const result = signedFromUnsigned(quotient, negative);
  // Match Velox CPU DecimalUtil::valueInRange after divide.
  if (result < kLongDecimalMin || result > kLongDecimalMax) {
    markError(localStatus, DecimalBinaryOpStatus::kOverflow);
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
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(cudf::size_type idx, int32_t& localStatus) const {
    // Null stencil is applied on the host (bitmask_and); skip inactive rows
    // instead of calling set_null (a device-wide atomic per row).
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    if (rhs.element<InT>(idx) == 0) {
      markError(localStatus, DecimalBinaryOpStatus::kDivisionByZero);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhs.element<InT>(idx),
        rhs.element<InT>(idx),
        rescaleFactor,
        localStatus);
  }
};

template <typename InColT, typename OutT>
struct DivideLhsScalarFunctor {
  __int128_t lhsValue;
  cudf::column_device_view rhs;
  cudf::mutable_column_device_view out;
  __int128_t rescaleFactor;
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(cudf::size_type idx, int32_t& localStatus) const {
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    if (rhs.element<InColT>(idx) == 0) {
      markError(localStatus, DecimalBinaryOpStatus::kDivisionByZero);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhsValue, rhs.element<InColT>(idx), rescaleFactor, localStatus);
  }
};

template <typename InColT, typename OutT>
struct DivideRhsScalarFunctor {
  cudf::column_device_view lhs;
  __int128_t rhsValue;
  cudf::mutable_column_device_view out;
  __int128_t rescaleFactor;
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(cudf::size_type idx, int32_t& localStatus) const {
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    if (rhsValue == 0) {
      markError(localStatus, DecimalBinaryOpStatus::kDivisionByZero);
      return;
    }
    out.element<OutT>(idx) = decimalDivideImpl<OutT>(
        lhs.element<InColT>(idx), rhsValue, rescaleFactor, localStatus);
  }
};

// Grid-stride loop: each thread ORs row errors into a register, then performs
// at most one global atomicOr when finished.
template <typename RowOp>
__global__ void overflowCheckedKernel(
    cudf::size_type size,
    RowOp rowOp,
    int32_t* overflowFlag) {
  int32_t localStatus = 0;
  auto idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (; idx < static_cast<cudf::thread_index_type>(size); idx += stride) {
    rowOp(static_cast<cudf::size_type>(idx), localStatus);
  }
  if (localStatus != 0) {
    atomicOr(overflowFlag, localStatus);
  }
}

// Runs buildOp() across [0, size) rows behind a single per-launch overflowFlag
// and reports the raw flag bits. Shared by the divide and ADD/SUB/MUL/MOD
// kernels; buildOp returns the per-row functor. Returns the raw overflowFlag
// bits (kDecimal*Bit).
template <typename BuildOp>
int32_t launchOverflowChecked(
    cudf::size_type size,
    BuildOp buildOp,
    rmm::cuda_stream_view stream) {
  if (size == 0) {
    return 0;
  }
  cudf::detail::device_scalar<int32_t> overflowFlag{0, stream};
  auto op = buildOp();
  cudf::detail::grid_1d const grid{size, kOverflowCheckedBlockSize};
  overflowCheckedKernel<<<
      grid.num_blocks,
      kOverflowCheckedBlockSize,
      0,
      stream.value()>>>(size, op, overflowFlag.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  return overflowFlag.value(stream);
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
  DecimalBinaryOpStatus operator()() const {
    auto lhsDev = cudf::column_device_view::create(lhs, stream);
    auto rhsDev = cudf::column_device_view::create(rhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return toDecimalBinaryOpStatus(launchOverflowChecked(
        lhs.size(),
        [&]() {
          return DivideFunctor<InT, OutT>{
              *lhsDev, *rhsDev, *outDev, rescaleFactor, out.null_mask()};
        },
        stream));
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  DecimalBinaryOpStatus operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return DecimalBinaryOpStatus::kOverflow;
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
  DecimalBinaryOpStatus operator()() const {
    auto lhsDev = cudf::column_device_view::create(lhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return toDecimalBinaryOpStatus(launchOverflowChecked(
        lhs.size(),
        [&]() {
          return DivideRhsScalarFunctor<InT, OutT>{
              *lhsDev, rhsValue, *outDev, rescaleFactor, out.null_mask()};
        },
        stream));
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  DecimalBinaryOpStatus operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return DecimalBinaryOpStatus::kOverflow;
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
  DecimalBinaryOpStatus operator()() const {
    auto rhsDev = cudf::column_device_view::create(rhs, stream);
    auto outDev = cudf::mutable_column_device_view::create(out, stream);
    return toDecimalBinaryOpStatus(launchOverflowChecked(
        rhs.size(),
        [&]() {
          return DivideLhsScalarFunctor<InT, OutT>{
              lhsValue, *rhsDev, *outDev, rescaleFactor, out.null_mask()};
        },
        stream));
  }

  template <typename InT, typename OutT>
    requires(!ValidDecimalDivideStorageTypes<InT, OutT>)
  DecimalBinaryOpStatus operator()() const {
    CUDF_FAIL("Invalid types for decimal divide");
    return DecimalBinaryOpStatus::kOverflow;
  }
};

DecimalBinaryOpStatus decimalDivideColumnColumn(
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

DecimalBinaryOpStatus decimalDivideColumnScalar(
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

DecimalBinaryOpStatus decimalDivideScalarColumn(
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
// Decimal DIV uses the detail:: kernels above (including division-by-zero);
// this path covers the remaining fixed-point arithmetic. Failures are tracked
// with a thread-local status register flushed once per thread via atomicOr to
// match the fail-fast semantics of Presto / Velox CPU decimal arithmetic.
// ---------------------------------------------------------------------------
namespace {

using errc = cudf::errc;

// Velox maps a decimal precision of at most 18 to a short decimal (DECIMAL64)
// and anything wider to a long decimal (DECIMAL128), so a storage width implies
// the largest precision it can hold.
template <typename Rep>
constexpr int32_t kMaxPrecisionFor = std::is_same_v<Rep, int64_t> ? 18 : 38;

// Storage width of the arithmetic performed by evalDecimalBinaryRow, given the
// three storage widths involved. int64_t is only safe when all three are
// DECIMAL64: the operand rescale, the op itself and check_precision all run at
// this type, so a narrower type than the output would report a false overflow
// for a result the output was widened to hold, and would evaluate
// ipow<Rep, BASE_10>(outPrecision) past its own range.
template <typename LhsRep, typename RhsRep, typename OutRep>
using ComputeRepFor = std::conditional_t<
    std::is_same_v<LhsRep, int64_t> && std::is_same_v<RhsRep, int64_t> &&
        std::is_same_v<OutRep, int64_t>,
    int64_t,
    __int128_t>;

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
// operators. On overflow it ORs the thread-local status and writes a
// well-defined 0 (the value is irrelevant because the host fails the whole
// batch).
template <typename Rep, typename OutRep>
__device__ void evalDecimalBinaryRow(
    numeric::decimal<Rep> lhsDec,
    numeric::decimal<Rep> rhsDec,
    cudf::binary_operator op,
    numeric::scale_type outScale,
    int32_t outPrecision,
    OutRep* out,
    int32_t idx,
    int32_t& localStatus) {
  // Modulo by a zero divisor is a distinct error, checked before the operand
  // rescale so it takes precedence over a later overflow (matching Velox CPU,
  // which validates the divisor before rescaling). rhs is always the divisor
  // for MOD across all operand orderings.
  if (op == cudf::binary_operator::MOD && rhsDec.value() == Rep{0}) {
    markError(localStatus, DecimalBinaryOpStatus::kDivisionByZero);
    out[idx] = OutRep{0};
    return;
  }

  // ADD/SUB/MOD require both operands at a common scale (the output scale).
  // cuDF's *_overflow operators reach that scale via fixed_point::rescaled,
  // whose widening multiply is unchecked and can silently overflow Rep for
  // mixed-scale operands. Pre-rescale the operands here with overflow
  // detection so the conversion is covered by the same fail-fast path. MUL is
  // excluded: it adds operand scales and its product is rescaled to outScale
  // by the checkedRescale below.
  if (op == cudf::binary_operator::ADD || op == cudf::binary_operator::SUB ||
      op == cudf::binary_operator::MOD) {
    auto lhsRescaled = checkedRescale<Rep>(lhsDec, outScale);
    auto rhsRescaled = checkedRescale<Rep>(rhsDec, outScale);
    if (!lhsRescaled.has_value() || !rhsRescaled.has_value()) {
      markError(localStatus, DecimalBinaryOpStatus::kOverflow);
      out[idx] = OutRep{0};
      return;
    }
    lhsDec = lhsRescaled.value();
    rhsDec = rhsRescaled.value();
  }

  // Modulo by a divisor of magnitude 1 always yields remainder 0. Special-case
  // a -1 divisor to match Velox CPU: cuDF's mod_overflow forms the quotient
  // a / b, and INT128_MIN / -1 overflows int128 (2^127 is unrepresentable), so
  // it would report a false overflow even though INT128_MIN % -1 == 0. The CPU
  // path computes the remainder in unsigned magnitude space and never hits
  // this.
  if (op == cudf::binary_operator::MOD && rhsDec.value() == Rep{-1}) {
    out[idx] = OutRep{0};
    return;
  }

  auto opResult = applyCheckedBinOp<Rep>(op, lhsDec, rhsDec);
  if (!opResult.has_value()) {
    if (opResult.error() == errc::DIVISION_BY_ZERO) {
      markError(localStatus, DecimalBinaryOpStatus::kDivisionByZero);
    } else {
      markError(localStatus, DecimalBinaryOpStatus::kOverflow);
    }
    out[idx] = OutRep{0};
    return;
  }

  auto rescaled = checkedRescale<Rep>(opResult.value(), outScale);
  if (!rescaled.has_value()) {
    markError(localStatus, DecimalBinaryOpStatus::kOverflow);
    out[idx] = OutRep{0};
    return;
  }

  auto precisionChecked =
      cudf::detail::ops::check_precision(rescaled.value(), outPrecision);
  if (!precisionChecked.has_value()) {
    markError(localStatus, DecimalBinaryOpStatus::kOverflow);
    out[idx] = OutRep{0};
    return;
  }
  out[idx] = static_cast<OutRep>(precisionChecked.value().value());
}

// Operands are loaded at their own storage width and widened to the compute
// type. Loading both through one representation would reinterpret a DECIMAL64
// operand at the wrong width and stride whenever the other operand or the
// output is DECIMAL128.
template <typename LhsRep, typename RhsRep, typename OutRep>
struct DecimalBinaryColColFunctor {
  using Rep = ComputeRepFor<LhsRep, RhsRep, OutRep>;

  const LhsRep* lhs;
  const RhsRep* rhs;
  OutRep* out;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(int32_t idx, int32_t& localStatus) const {
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(lhs[idx]), lhsScale}},
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(rhs[idx]), rhsScale}},
        op,
        outScale,
        outPrecision,
        out,
        idx,
        localStatus);
  }
};

template <typename LhsRep, typename RhsRep, typename OutRep>
struct DecimalBinaryLhsScalarFunctor {
  using Rep = ComputeRepFor<LhsRep, RhsRep, OutRep>;

  LhsRep lhsValue;
  const RhsRep* rhs;
  OutRep* out;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(int32_t idx, int32_t& localStatus) const {
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(lhsValue), lhsScale}},
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(rhs[idx]), rhsScale}},
        op,
        outScale,
        outPrecision,
        out,
        idx,
        localStatus);
  }
};

template <typename LhsRep, typename RhsRep, typename OutRep>
struct DecimalBinaryRhsScalarFunctor {
  using Rep = ComputeRepFor<LhsRep, RhsRep, OutRep>;

  const LhsRep* lhs;
  RhsRep rhsValue;
  OutRep* out;
  numeric::scale_type lhsScale;
  numeric::scale_type rhsScale;
  numeric::scale_type outScale;
  int32_t outPrecision;
  cudf::binary_operator op;
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(int32_t idx, int32_t& localStatus) const {
    if (!isRowActive(nullMask, idx)) {
      return;
    }
    evalDecimalBinaryRow<Rep, OutRep>(
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(lhs[idx]), lhsScale}},
        numeric::decimal<Rep>{
            numeric::scaled_integer<Rep>{static_cast<Rep>(rhsValue), rhsScale}},
        op,
        outScale,
        outPrecision,
        out,
        idx,
        localStatus);
  }
};

// The output store narrows the compute type to OutRep behind check_precision,
// which only bounds the value to 10^outPrecision. A precision wider than OutRep
// can hold would therefore truncate silently.
template <typename OutRep>
void checkDecimalOutputPrecision(int32_t outPrecision) {
  CUDF_EXPECTS(
      outPrecision > 0 && outPrecision <= kMaxPrecisionFor<OutRep>,
      "Decimal binop output precision must fit the output storage width");
}

// data<T>() reinterprets column memory without checking T against the runtime
// type, so a kernel instantiated at the wrong width would stride over the
// elements with no diagnostic. The dispatch derives these types from the same
// columns and cannot disagree; this states the contract for callers that
// instantiate the launch directly.
template <typename Rep>
void checkDecimalStorageWidth(cudf::data_type type) {
  CUDF_EXPECTS(
      cudf::size_of(type) == sizeof(Rep),
      "Decimal binop storage width must match its kernel type");
}

template <typename LhsRep, typename RhsRep, typename OutRep>
int32_t launchDecimalBinaryColColKernel(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  checkDecimalStorageWidth<LhsRep>(lhs.type());
  checkDecimalStorageWidth<RhsRep>(rhs.type());
  checkDecimalStorageWidth<OutRep>(out.type());
  checkDecimalOutputPrecision<OutRep>(outPrecision);
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      lhs.size(),
      [&]() {
        return DecimalBinaryColColFunctor<LhsRep, RhsRep, OutRep>{
            lhs.data<LhsRep>(),
            rhs.data<RhsRep>(),
            out.data<OutRep>(),
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask};
      },
      stream);
}

template <typename LhsRep, typename RhsRep, typename OutRep>
int32_t launchDecimalBinaryRhsScalarKernel(
    cudf::column_view const& lhs,
    RhsRep rhsValue,
    numeric::scale_type rhsScale,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  // rhsValue arrives already decoded at RhsRep, so it needs no width check.
  checkDecimalStorageWidth<LhsRep>(lhs.type());
  checkDecimalStorageWidth<OutRep>(out.type());
  checkDecimalOutputPrecision<OutRep>(outPrecision);
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      lhs.size(),
      [&]() {
        return DecimalBinaryRhsScalarFunctor<LhsRep, RhsRep, OutRep>{
            lhs.data<LhsRep>(),
            rhsValue,
            out.data<OutRep>(),
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask};
      },
      stream);
}

template <typename LhsRep, typename RhsRep, typename OutRep>
int32_t launchDecimalBinaryLhsScalarKernel(
    LhsRep lhsValue,
    numeric::scale_type lhsScale,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  // lhsValue arrives already decoded at LhsRep, so it needs no width check.
  checkDecimalStorageWidth<RhsRep>(rhs.type());
  checkDecimalStorageWidth<OutRep>(out.type());
  checkDecimalOutputPrecision<OutRep>(outPrecision);
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  return launchOverflowChecked(
      rhs.size(),
      [&]() {
        return DecimalBinaryLhsScalarFunctor<LhsRep, RhsRep, OutRep>{
            lhsValue,
            rhs.data<RhsRep>(),
            out.data<OutRep>(),
            lhsScale,
            rhsScale,
            outScale,
            outPrecision,
            op,
            nullMask};
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

template <typename LhsRep, typename RhsRep, typename OutRep>
std::pair<std::unique_ptr<cudf::column>, int32_t>
decimalBinaryOperationColColImpl(
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

  int32_t const statusFlag =
      launchDecimalBinaryColColKernel<LhsRep, RhsRep, OutRep>(
          lhs, rhs, result->mutable_view(), op, outputPrecision, stream);
  return {std::move(result), statusFlag};
}

// Invokes fn with a value of the storage representation behind `type`, so a
// runtime decimal width selects a kernel template argument.
template <typename Fn>
auto dispatchDecimalRep(cudf::data_type type, Fn&& fn)
    -> decltype(fn(int64_t{})) {
  switch (type.id()) {
    case cudf::type_id::DECIMAL64:
      return fn(int64_t{});
    case cudf::type_id::DECIMAL128:
      return fn(__int128_t{});
    default:
      CUDF_FAIL(
          "Unsupported decimal storage type for overflow-checked execution");
  }
}

// The lhs, rhs and output storage widths vary independently, so all three are
// dispatched on. Velox's modulo result precision is
// min(p1 - s1, p2 - s2) + max(s1, s2), which unlike ADD/SUB/MUL can be narrower
// than an operand, and ExpressionEvaluator only widens operands to the result
// storage when that result is DECIMAL128. A DECIMAL128 operand feeding a
// DECIMAL64 result is therefore reachable, e.g. decimal(20,0) % decimal(5,2).
std::pair<std::unique_ptr<cudf::column>, int32_t>
dispatchDecimalBinaryOperationColCol(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return dispatchDecimalRep(lhs.type(), [&](auto lhsRep) {
    return dispatchDecimalRep(rhs.type(), [&](auto rhsRep) {
      return dispatchDecimalRep(outputType, [&](auto outRep) {
        return decimalBinaryOperationColColImpl<
            decltype(lhsRep),
            decltype(rhsRep),
            decltype(outRep)>(
            lhs, rhs, op, outputType, outputPrecision, stream, mr);
      });
    });
  });
}

// Decoding at the scalar's own storage width keeps the narrowing from the
// __int128_t payload exact; the kernel widens to the compute type.
template <typename ScalarRep>
ScalarRep getTypedDecimalScalarValue(
    const cudf::scalar& s,
    rmm::cuda_stream_view stream) {
  return static_cast<ScalarRep>(detail::getDecimalScalarValue(s, stream));
}

int32_t dispatchDecimalBinaryOperationColScalar(
    cudf::column_view const& lhs,
    cudf::scalar const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream) {
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  return dispatchDecimalRep(lhs.type(), [&](auto lhsRep) {
    return dispatchDecimalRep(rhs.type(), [&](auto rhsRep) {
      return dispatchDecimalRep(out.type(), [&](auto outRep) {
        using RhsRep = decltype(rhsRep);
        return launchDecimalBinaryRhsScalarKernel<
            decltype(lhsRep),
            RhsRep,
            decltype(outRep)>(
            lhs,
            getTypedDecimalScalarValue<RhsRep>(rhs, stream),
            rhsScale,
            out,
            op,
            outputPrecision,
            stream);
      });
    });
  });
}

int32_t dispatchDecimalBinaryOperationScalarCol(
    cudf::scalar const& lhs,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    cudf::binary_operator op,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream) {
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  return dispatchDecimalRep(lhs.type(), [&](auto lhsRep) {
    return dispatchDecimalRep(rhs.type(), [&](auto rhsRep) {
      return dispatchDecimalRep(out.type(), [&](auto outRep) {
        using LhsRep = decltype(lhsRep);
        return launchDecimalBinaryLhsScalarKernel<
            LhsRep,
            decltype(rhsRep),
            decltype(outRep)>(
            getTypedDecimalScalarValue<LhsRep>(lhs, stream),
            lhsScale,
            rhs,
            out,
            op,
            outputPrecision,
            stream);
      });
    });
  });
}

} // namespace

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
decimalBinaryOperationWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::binary_operator op,
    cudf::data_type outputType,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateDecimalBinaryOp(op);
  auto [result, statusFlag] = dispatchDecimalBinaryOperationColCol(
      lhs, rhs, op, outputType, outputPrecision, stream, mr);
  return {std::move(result), toDecimalBinaryOpStatus(statusFlag)};
}

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
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
    return {std::move(result), DecimalBinaryOpStatus::kOk};
  }
  auto nullMask = cudf::copy_bitmask(lhs, stream, mr);
  auto result = makeResultColumn(
      lhs.size(),
      outputType,
      std::move(nullMask),
      lhs.null_count(),
      stream,
      mr);

  int32_t const statusFlag = dispatchDecimalBinaryOperationColScalar(
      lhs, rhs, result->mutable_view(), op, outputPrecision, stream);
  return {std::move(result), toDecimalBinaryOpStatus(statusFlag)};
}

std::pair<std::unique_ptr<cudf::column>, DecimalBinaryOpStatus>
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
    return {std::move(result), DecimalBinaryOpStatus::kOk};
  }
  auto nullMask = cudf::copy_bitmask(rhs, stream, mr);
  auto result = makeResultColumn(
      rhs.size(),
      outputType,
      std::move(nullMask),
      rhs.null_count(),
      stream,
      mr);

  int32_t const statusFlag = dispatchDecimalBinaryOperationScalarCol(
      lhs, rhs, result->mutable_view(), op, outputPrecision, stream);
  return {std::move(result), toDecimalBinaryOpStatus(statusFlag)};
}

} // namespace facebook::velox::cudf_velox
