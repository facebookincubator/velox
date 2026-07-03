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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
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

#include <rmm/device_scalar.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/std/limits>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

using errc = cudf::errc;

// Sets the single per-launch overflow flag. Because Presto / Velox CPU
// semantics are fail-fast (any overflow fails the whole expression), every
// thread that detects an overflow simply ORs into one shared device flag.
// No per-row (O(n)) overflow column is required.
__device__ inline void flagOverflow(int32_t* overflowFlag) {
  atomicOr(overflowFlag, 1);
}

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
  auto opResult = applyCheckedBinOp<Rep>(op, lhsDec, rhsDec);
  if (!opResult.has_value()) {
    flagOverflow(overflowFlag);
    out[idx] = OutRep{0};
    return;
  }

  auto rescaled = checkedRescale<Rep>(opResult.value(), outScale);
  if (!rescaled.has_value()) {
    flagOverflow(overflowFlag);
    out[idx] = OutRep{0};
    return;
  }

  auto precisionChecked =
      cudf::detail::ops::check_precision(rescaled.value(), outPrecision);
  if (!precisionChecked.has_value()) {
    flagOverflow(overflowFlag);
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
void launchDecimalBinaryColColKernel(
    cudf::column_view const& lhs,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    int32_t* overflowFlag,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DecimalBinaryColColFunctor<InRep, OutRep> functor{
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
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      lhs.size(),
      functor,
      stream.value());
}

template <typename InRep, typename OutRep>
void launchDecimalBinaryRhsScalarKernel(
    cudf::column_view const& lhs,
    InRep rhsValue,
    numeric::scale_type rhsScale,
    cudf::mutable_column_view out,
    int32_t* overflowFlag,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DecimalBinaryRhsScalarFunctor<InRep, OutRep> functor{
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
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      lhs.size(),
      functor,
      stream.value());
}

template <typename InRep, typename OutRep>
void launchDecimalBinaryLhsScalarKernel(
    InRep lhsValue,
    numeric::scale_type lhsScale,
    cudf::column_view const& rhs,
    cudf::mutable_column_view out,
    int32_t* overflowFlag,
    cudf::binary_operator op,
    int32_t outPrecision,
    rmm::cuda_stream_view stream) {
  if (rhs.size() == 0) {
    return;
  }
  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DecimalBinaryLhsScalarFunctor<InRep, OutRep> functor{
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
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      rhs.size(),
      functor,
      stream.value());
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

  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);
  launchDecimalBinaryColColKernel<InRep, OutRep>(
      lhs,
      rhs,
      result->mutable_view(),
      overflowFlag.data(),
      op,
      outputPrecision,
      stream);
  bool const didOverflow = overflowFlag.value(stream) != 0;
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

template <typename OutT>
__device__ OutT decimalDivideImplChecked(
    __int128_t numerator,
    __int128_t denom,
    __int128_t scale,
    bool* overflow) {
  if (denom == 0) {
    *overflow = false;
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

  if (numeric::multiplication_overflow<__int128_t>(numerator, scale)) {
    *overflow = true;
    return OutT{0};
  }
  __int128_t const scaled = numerator * scale;

  __int128_t quotient = scaled / denom;
  __int128_t remainder = scaled % denom;
  if (remainder * 2 >= denom) {
    if (quotient == cuda::std::numeric_limits<__int128_t>::max()) {
      *overflow = true;
      return OutT{0};
    }
    ++quotient;
  }
  if (sign < 0) {
    quotient = -quotient;
  }
  *overflow = false;
  return static_cast<OutT>(quotient);
}

template <typename OutT>
__device__ void evalDecimalDivideRow(
    __int128_t numerator,
    __int128_t denom,
    __int128_t scale,
    numeric::scale_type outScale,
    int32_t outPrecision,
    OutT* out,
    int32_t idx,
    int32_t* overflowFlag) {
  bool divideOverflow = false;
  auto const quotient =
      decimalDivideImplChecked<OutT>(numerator, denom, scale, &divideOverflow);
  if (divideOverflow) {
    flagOverflow(overflowFlag);
    out[idx] = OutT{0};
    return;
  }

  using Rep =
      std::conditional_t<std::is_same_v<OutT, int64_t>, int64_t, __int128_t>;
  auto const resultDec = makeDecimal<Rep>(static_cast<Rep>(quotient), outScale);
  auto const precisionChecked =
      cudf::detail::ops::check_precision(resultDec, outPrecision);
  if (!precisionChecked.has_value()) {
    flagOverflow(overflowFlag);
    out[idx] = OutT{0};
    return;
  }
  out[idx] = quotient;
}

template <typename InT, typename OutT>
struct DivideFunctor {
  const InT* lhs;
  const InT* rhs;
  OutT* out;
  int32_t* overflowFlag;
  __int128_t scale;
  int32_t outPrecision;
  numeric::scale_type outScale;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalDivideRow<OutT>(
        static_cast<__int128_t>(lhs[idx]),
        static_cast<__int128_t>(rhs[idx]),
        scale,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

template <typename InColT, typename OutT>
struct DivideLhsScalarFunctor {
  __int128_t lhsValue;
  const InColT* rhs;
  OutT* out;
  int32_t* overflowFlag;
  __int128_t scale;
  int32_t outPrecision;
  numeric::scale_type outScale;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalDivideRow<OutT>(
        lhsValue,
        static_cast<__int128_t>(rhs[idx]),
        scale,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

template <typename InColT, typename OutT>
struct DivideRhsScalarFunctor {
  const InColT* lhs;
  __int128_t rhsValue;
  OutT* out;
  int32_t* overflowFlag;
  __int128_t scale;
  int32_t outPrecision;
  numeric::scale_type outScale;
  cudf::bitmask_type const* nullMask;
  bool hasNullMask;

  __device__ void operator()(int32_t idx) const {
    bool const rowActive = isRowActive(nullMask, hasNullMask, idx);
    if (!rowActive) {
      return;
    }
    evalDecimalDivideRow<OutT>(
        static_cast<__int128_t>(lhs[idx]),
        rhsValue,
        scale,
        outScale,
        outPrecision,
        out,
        idx,
        overflowFlag);
  }
};

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
    int32_t* overflowFlag,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DivideFunctor<InT, OutT> op{
      lhs.data<InT>(),
      rhs.data<InT>(),
      out.data<OutT>(),
      overflowFlag,
      pow10Int128(aRescale),
      outputPrecision,
      outScale,
      nullMask,
      nullMask != nullptr};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), lhs.size(), op, stream.value());
}

template <typename InColT, typename OutT>
void launchDivideKernelLhsScalar(
    __int128_t lhsValue,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t* overflowFlag,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream) {
  if (rhs.size() == 0) {
    return;
  }
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DivideLhsScalarFunctor<InColT, OutT> op{
      lhsValue,
      rhs.data<InColT>(),
      out.data<OutT>(),
      overflowFlag,
      pow10Int128(aRescale),
      outputPrecision,
      outScale,
      nullMask,
      nullMask != nullptr};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), rhs.size(), op, stream.value());
}

template <typename InColT, typename OutT>
void launchDivideKernelRhsScalar(
    const cudf::column_view& lhs,
    __int128_t rhsValue,
    cudf::mutable_column_view out,
    int32_t* overflowFlag,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  auto const outScale = numeric::scale_type{out.type().scale()};
  auto const* nullMask = out.null_mask();
  DivideRhsScalarFunctor<InColT, OutT> op{
      lhs.data<InColT>(),
      rhsValue,
      out.data<OutT>(),
      overflowFlag,
      pow10Int128(aRescale),
      outputPrecision,
      outScale,
      nullMask,
      nullMask != nullptr};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), lhs.size(), op, stream.value());
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

template <typename InRep>
InRep getTypedDecimalScalarValue(
    const cudf::scalar& s,
    rmm::cuda_stream_view stream) {
  if constexpr (std::is_same_v<InRep, int64_t>) {
    return static_cast<int64_t>(getDecimalScalarValue(s, stream));
  }
  return static_cast<InRep>(getDecimalScalarValue(s, stream));
}

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
  auto nullMask = cudf::copy_bitmask(lhs, stream, mr);
  auto result = makeResultColumn(
      lhs.size(),
      outputType,
      std::move(nullMask),
      lhs.null_count(),
      stream,
      mr);
  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);

  auto const rhsScale = numeric::scale_type{rhs.type().scale()};
  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    auto const rhsValue = getTypedDecimalScalarValue<int64_t>(rhs, stream);
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDecimalBinaryRhsScalarKernel<int64_t, int64_t>(
          lhs,
          rhsValue,
          rhsScale,
          result->mutable_view(),
          overflowFlag.data(),
          op,
          outputPrecision,
          stream);
    } else {
      launchDecimalBinaryRhsScalarKernel<int64_t, __int128_t>(
          lhs,
          rhsValue,
          rhsScale,
          result->mutable_view(),
          overflowFlag.data(),
          op,
          outputPrecision,
          stream);
    }
  } else {
    auto const rhsValue = getTypedDecimalScalarValue<__int128_t>(rhs, stream);
    launchDecimalBinaryRhsScalarKernel<__int128_t, __int128_t>(
        lhs,
        rhsValue,
        rhsScale,
        result->mutable_view(),
        overflowFlag.data(),
        op,
        outputPrecision,
        stream);
  }
  bool const didOverflow = overflowFlag.value(stream) != 0;
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
  auto nullMask = cudf::copy_bitmask(rhs, stream, mr);
  auto result = makeResultColumn(
      rhs.size(),
      outputType,
      std::move(nullMask),
      rhs.null_count(),
      stream,
      mr);
  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);

  auto const lhsScale = numeric::scale_type{lhs.type().scale()};
  if (rhs.type().id() == cudf::type_id::DECIMAL64) {
    auto const lhsValue = getTypedDecimalScalarValue<int64_t>(lhs, stream);
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDecimalBinaryLhsScalarKernel<int64_t, int64_t>(
          lhsValue,
          lhsScale,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          op,
          outputPrecision,
          stream);
    } else {
      launchDecimalBinaryLhsScalarKernel<int64_t, __int128_t>(
          lhsValue,
          lhsScale,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          op,
          outputPrecision,
          stream);
    }
  } else {
    auto const lhsValue = getTypedDecimalScalarValue<__int128_t>(lhs, stream);
    launchDecimalBinaryLhsScalarKernel<__int128_t, __int128_t>(
        lhsValue,
        lhsScale,
        rhs,
        result->mutable_view(),
        overflowFlag.data(),
        op,
        outputPrecision,
        stream);
  }
  bool const didOverflow = overflowFlag.value(stream) != 0;
  return {std::move(result), didOverflow};
}

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Decimal divide requires equal sizes");
  CUDF_EXPECTS(
      lhs.type().id() == rhs.type().id(),
      "Decimal divide requires matching input types");
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  auto [nullMask, nullCount] =
      cudf::bitmask_and(cudf::table_view({lhs, rhs}), stream, mr);
  auto result = makeResultColumn(
      lhs.size(), outputType, std::move(nullMask), nullCount, stream, mr);
  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);

  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernel<int64_t, int64_t>(
          lhs,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    } else {
      launchDivideKernel<int64_t, __int128_t>(
          lhs,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    }
  } else {
    launchDivideKernel<__int128_t, __int128_t>(
        lhs,
        rhs,
        result->mutable_view(),
        overflowFlag.data(),
        aRescale,
        outputPrecision,
        stream);
  }
  bool const didOverflow = overflowFlag.value(stream) != 0;
  return {std::move(result), didOverflow};
}

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::column_view& lhs,
    const cudf::scalar& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  if (!rhs.is_valid(stream)) {
    return {
        makeAllNullDecimalColumn(outputType, lhs.size(), stream, mr), false};
  }

  auto nullMask = cudf::copy_bitmask(lhs, stream, mr);
  auto result = makeResultColumn(
      lhs.size(),
      outputType,
      std::move(nullMask),
      lhs.null_count(),
      stream,
      mr);
  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);
  auto rhsValue = getDecimalScalarValue(rhs, stream);

  if (lhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernelRhsScalar<int64_t, int64_t>(
          lhs,
          rhsValue,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    } else {
      launchDivideKernelRhsScalar<int64_t, __int128_t>(
          lhs,
          rhsValue,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    }
  } else {
    launchDivideKernelRhsScalar<__int128_t, __int128_t>(
        lhs,
        rhsValue,
        result->mutable_view(),
        overflowFlag.data(),
        aRescale,
        outputPrecision,
        stream);
  }
  bool const didOverflow = overflowFlag.value(stream) != 0;
  return {std::move(result), didOverflow};
}

std::pair<std::unique_ptr<cudf::column>, bool> decimalDivideWithOverflow(
    const cudf::scalar& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    int32_t outputPrecision,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  if (!lhs.is_valid(stream)) {
    return {
        makeAllNullDecimalColumn(outputType, rhs.size(), stream, mr), false};
  }

  auto nullMask = cudf::copy_bitmask(rhs, stream, mr);
  auto result = makeResultColumn(
      rhs.size(),
      outputType,
      std::move(nullMask),
      rhs.null_count(),
      stream,
      mr);
  rmm::device_scalar<int32_t> overflowFlag(0, stream, mr);
  auto lhsValue = getDecimalScalarValue(lhs, stream);

  if (rhs.type().id() == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernelLhsScalar<int64_t, int64_t>(
          lhsValue,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    } else {
      launchDivideKernelLhsScalar<int64_t, __int128_t>(
          lhsValue,
          rhs,
          result->mutable_view(),
          overflowFlag.data(),
          aRescale,
          outputPrecision,
          stream);
    }
  } else {
    launchDivideKernelLhsScalar<__int128_t, __int128_t>(
        lhsValue,
        rhs,
        result->mutable_view(),
        overflowFlag.data(),
        aRescale,
        outputPrecision,
        stream);
  }
  bool const didOverflow = overflowFlag.value(stream) != 0;
  return {std::move(result), didOverflow};
}

} // namespace facebook::velox::cudf_velox
