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
#pragma once

#include "velox/experimental/cudf/functions/GpuColumnIO.cuh"
#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <type_traits>

namespace facebook::velox::gpu {

namespace detail {

template <typename FnType, typename ReturnType, typename... ArgTypes>
using CallReturnType = decltype(std::declval<FnType>().call(
    std::declval<ReturnType&>(),
    std::declval<ArgTypes>()...));

template <typename FnType, typename ReturnType, typename... ArgTypes>
inline constexpr bool kCallReturnsBool =
    std::is_same_v<CallReturnType<FnType, ReturnType, ArgTypes...>, bool>;

} // namespace detail

// Default-null kernel: skips rows where any input is null or row is inactive.
// For each active non-null row, calls FnType::call(result, args...).
// Handles both void-returning call() and bool-returning call() (nullable
// result). When call() returns false the output row is marked null.
template <typename FnType, typename ReturnType, typename... ArgTypes>
__global__ void gpuSimpleFunctionKernel(
    GpuColumnWriter<ReturnType> output,
    GpuColumnReader<ArgTypes>... inputs,
    const cudf::bitmask_type* combinedNullBitmask,
    const cudf::bitmask_type* activeRows,
    cudf::size_type numRows) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numRows)
    return;

  if (activeRows && !cudf::bit_is_set(activeRows, row)) {
    return;
  }

  if (combinedNullBitmask && !cudf::bit_is_set(combinedNullBitmask, row)) {
    output.setNull(row);
    return;
  }

  FnType fn;
  ReturnType result{};
  if constexpr (detail::kCallReturnsBool<FnType, ReturnType, ArgTypes...>) {
    bool notNull = fn.call(result, inputs.read(row)...);
    if (notNull) {
      output.write(row, result);
      output.setValid(row);
    } else {
      output.setNull(row);
    }
  } else {
    fn.call(result, inputs.read(row)...);
    output.write(row, result);
    output.setValid(row);
  }
}

// Host-side adapter that launches the kernel for a given function type.
template <typename FnType, typename ReturnType, typename... ArgTypes>
struct GpuSimpleFunctionAdapter {
  static constexpr int kBlockSize = 256;

  static void apply(
      GpuColumnWriter<ReturnType> output,
      GpuColumnReader<ArgTypes>... inputs,
      const cudf::bitmask_type* combinedNullBitmask,
      const cudf::bitmask_type* activeRows,
      cudf::size_type numRows,
      cudaStream_t stream = 0) {
    if (numRows == 0)
      return;

    int blocks = (numRows + kBlockSize - 1) / kBlockSize;
    gpuSimpleFunctionKernel<FnType, ReturnType, ArgTypes...>
        <<<blocks, kBlockSize, 0, stream>>>(
            output,
            inputs...,
            combinedNullBitmask,
            activeRows,
            numRows);
  }
};

} // namespace facebook::velox::gpu
