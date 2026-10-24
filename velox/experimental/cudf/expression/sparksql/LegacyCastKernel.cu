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

#include "velox/experimental/cudf/expression/sparksql/FloatingPointToDecimal.cuh"
#include "velox/experimental/cudf/expression/sparksql/LegacyCastKernel.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

using UInt128 = unsigned __int128;
using Int128 = __int128;

constexpr int32_t kBlockSize = 256;
constexpr int32_t kWarpSize = 32;

__device__ UInt128 powerOfTen(int32_t exponent) {
  UInt128 result = 1;
  for (int32_t i = 0; i < exponent; ++i) {
    result *= 10;
  }
  return result;
}

__device__ UInt128 magnitude(Int128 value) {
  // Unsigned negation also handles INT128_MIN without signed overflow.
  const auto unsignedValue = static_cast<UInt128>(value);
  return value < 0 ? -unsignedValue : unsignedValue;
}

// Apply HALF_UP rescaling in unsigned space. Checking against the declared
// precision before multiplying prevents both storage and intermediate overflow.
__device__ bool rescaleMagnitude(
    UInt128& value,
    int32_t sourceScale,
    int32_t targetScale,
    UInt128 maxValue) {
  const int32_t delta = targetScale - sourceScale;
  if (delta > 0) {
    const UInt128 multiplier = powerOfTen(delta);
    if (value > maxValue / multiplier) {
      return false;
    }
    value *= multiplier;
  } else if (delta < 0) {
    const UInt128 divisor = powerOfTen(-delta);
    const UInt128 remainder = value % divisor;
    value = value / divisor + (remainder >= divisor / 2);
  }
  return value <= maxValue;
}

template <typename Input, typename Output>
struct DecimalCast {
  int32_t sourceScale;
  int32_t targetScale;
  UInt128 maxValue;
  int32_t targetPrecision;

  __device__ bool operator()(Input input, Output& output) const {
    if constexpr (std::is_floating_point_v<Input>) {
      Int128 value;
      // Spark promotes REAL to DOUBLE before constructing its decimal. Use
      // the decimal representation of that double, not a 7/15-digit binary
      // normalization that can move values across decimal halfway points.
      if (!detail::floatingPointToDecimal(
              static_cast<double>(input),
              targetPrecision,
              targetScale,
              value)) {
        return false;
      }
      output = static_cast<Output>(value);
      return true;
    } else {
      UInt128 value = magnitude(static_cast<Int128>(input));
      if (!rescaleMagnitude(value, sourceScale, targetScale, maxValue)) {
        return false;
      }
      const Int128 signedValue = static_cast<Int128>(value);
      output = static_cast<Output>(input < 0 ? -signedValue : signedValue);
      return true;
    }
  }
};

template <typename Input, typename Output>
struct FloatingToIntegralCast {
  __device__ bool operator()(Input input, Output& output) const {
    // Java narrows float/double to byte and short through int. Saturating to
    // byte/short limits directly would incorrectly turn 129.9 into 127.
    using Intermediate =
        std::conditional_t<(sizeof(Output) < sizeof(int32_t)), int32_t, Output>;
    constexpr auto kMin = cuda::std::numeric_limits<Intermediate>::min();
    constexpr auto kMax = cuda::std::numeric_limits<Intermediate>::max();
    Intermediate value;
    if (isnan(input)) {
      value = 0;
    } else if (input >= static_cast<Input>(kMax)) {
      value = kMax;
    } else if (input <= static_cast<Input>(kMin)) {
      value = kMin;
    } else {
      value = static_cast<Intermediate>(input);
    }
    output = static_cast<Output>(value);
    return true;
  }
};

template <typename Input, typename Output, typename Cast>
__global__ void castKernel(
    const Input* input,
    const cudf::bitmask_type* inputMask,
    cudf::size_type inputOffset,
    Output* output,
    cudf::bitmask_type* outputMask,
    cudf::size_type size,
    Cast cast) {
  const auto row =
      static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  bool valid = row < size &&
      (inputMask == nullptr || cudf::bit_is_set(inputMask, row + inputOffset));
  if (valid) {
    valid = cast(input[row], output[row]);
  }
  // One lane writes each complete mask word. All lanes participate, including
  // the tail beyond size, so no atomics or initialized mask are needed.
  const auto validity = __ballot_sync(0xffffffffU, valid);
  if (threadIdx.x % kWarpSize == 0 && row < size) {
    outputMask[row / kWarpSize] = validity;
  }
}

template <typename Input, typename Output, typename Cast>
void launchCast(
    cudf::column_view input,
    cudf::mutable_column_view output,
    Cast cast,
    cuda::stream_ref stream) {
  const auto blocks =
      (static_cast<int64_t>(input.size()) + kBlockSize - 1) / kBlockSize;
  castKernel<<<blocks, kBlockSize, 0, stream.get()>>>(
      input.data<Input>(),
      input.null_mask(),
      input.offset(),
      output.data<Output>(),
      output.null_mask(),
      input.size(),
      cast);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Input>
void dispatchIntegralOutput(
    cudf::column_view input,
    cudf::mutable_column_view output,
    cuda::stream_ref stream) {
  switch (output.type().id()) {
    case cudf::type_id::INT8:
      return launchCast<Input, int8_t>(
          input, output, FloatingToIntegralCast<Input, int8_t>{}, stream);
    case cudf::type_id::INT16:
      return launchCast<Input, int16_t>(
          input, output, FloatingToIntegralCast<Input, int16_t>{}, stream);
    case cudf::type_id::INT32:
      return launchCast<Input, int32_t>(
          input, output, FloatingToIntegralCast<Input, int32_t>{}, stream);
    case cudf::type_id::INT64:
      return launchCast<Input, int64_t>(
          input, output, FloatingToIntegralCast<Input, int64_t>{}, stream);
    default:
      CUDF_FAIL("Unsupported Spark legacy integral cast target");
  }
}

template <typename Input>
void dispatchDecimalOutput(
    cudf::column_view input,
    cudf::mutable_column_view output,
    int32_t targetPrecision,
    cuda::stream_ref stream) {
  const bool sourceIsDecimal = input.type().id() == cudf::type_id::DECIMAL64 ||
      input.type().id() == cudf::type_id::DECIMAL128;
  const int32_t sourceScale = sourceIsDecimal ? -input.type().scale() : 0;
  const int32_t targetScale = -output.type().scale();
  UInt128 maxValue = 1;
  for (int32_t i = 0; i < targetPrecision; ++i) {
    maxValue *= 10;
  }
  --maxValue;

  if (output.type().id() == cudf::type_id::DECIMAL64) {
    launchCast<Input, int64_t>(
        input,
        output,
        DecimalCast<Input, int64_t>{
            sourceScale, targetScale, maxValue, targetPrecision},
        stream);
  } else {
    launchCast<Input, Int128>(
        input,
        output,
        DecimalCast<Input, Int128>{
            sourceScale, targetScale, maxValue, targetPrecision},
        stream);
  }
}

void dispatchDecimalInput(
    cudf::column_view input,
    cudf::mutable_column_view output,
    int32_t targetPrecision,
    cuda::stream_ref stream) {
  switch (input.type().id()) {
    case cudf::type_id::BOOL8:
      return dispatchDecimalOutput<bool>(
          input, output, targetPrecision, stream);
    case cudf::type_id::INT8:
      return dispatchDecimalOutput<int8_t>(
          input, output, targetPrecision, stream);
    case cudf::type_id::INT16:
      return dispatchDecimalOutput<int16_t>(
          input, output, targetPrecision, stream);
    case cudf::type_id::INT32:
      return dispatchDecimalOutput<int32_t>(
          input, output, targetPrecision, stream);
    case cudf::type_id::INT64:
    case cudf::type_id::DECIMAL64:
      return dispatchDecimalOutput<int64_t>(
          input, output, targetPrecision, stream);
    case cudf::type_id::DECIMAL128:
      return dispatchDecimalOutput<Int128>(
          input, output, targetPrecision, stream);
    case cudf::type_id::FLOAT32:
      return dispatchDecimalOutput<float>(
          input, output, targetPrecision, stream);
    case cudf::type_id::FLOAT64:
      return dispatchDecimalOutput<double>(
          input, output, targetPrecision, stream);
    default:
      CUDF_FAIL("Unsupported Spark legacy decimal cast source");
  }
}

} // namespace

std::unique_ptr<cudf::column> legacyCastNumeric(
    cudf::column_view input,
    cudf::data_type targetType,
    int32_t targetPrecision,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const bool targetIsDecimal = targetType.id() == cudf::type_id::DECIMAL64 ||
      targetType.id() == cudf::type_id::DECIMAL128;
  if (targetIsDecimal) {
    const int32_t maxPrecision =
        targetType.id() == cudf::type_id::DECIMAL64 ? 18 : 38;
    CUDF_EXPECTS(
        targetPrecision >= 1 && targetPrecision <= maxPrecision &&
            targetType.scale() <= 0 && -targetType.scale() <= targetPrecision,
        "Invalid Spark decimal cast precision or scale");
    if (input.type().id() == cudf::type_id::DECIMAL64 ||
        input.type().id() == cudf::type_id::DECIMAL128) {
      CUDF_EXPECTS(
          input.type().scale() >= -38 && input.type().scale() <= 0,
          "Invalid Spark decimal cast source scale");
    }
  } else {
    CUDF_EXPECTS(
        input.type().id() == cudf::type_id::FLOAT32 ||
            input.type().id() == cudf::type_id::FLOAT64,
        "Spark legacy integral kernel requires floating point input");
  }

  auto result = cudf::make_fixed_width_column(
      targetType, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);
  if (input.is_empty()) {
    result->set_null_count(0);
    return result;
  }

  if (targetIsDecimal) {
    dispatchDecimalInput(
        input, result->mutable_view(), targetPrecision, stream);
  } else if (input.type().id() == cudf::type_id::FLOAT32) {
    dispatchIntegralOutput<float>(input, result->mutable_view(), stream);
  } else {
    dispatchIntegralOutput<double>(input, result->mutable_view(), stream);
  }
  result->set_null_count(
      cudf::null_count(result->view().null_mask(), 0, input.size(), stream));
  return result;
}

} // namespace facebook::velox::cudf_velox::sparksql
