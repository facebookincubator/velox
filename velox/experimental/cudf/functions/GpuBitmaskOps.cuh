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

// GPU bitmask operations for expression evaluation: AND, OR, NOT on
// cudf::bitmask_type arrays. Also provides special form kernels for
// IF/SWITCH/COALESCE that operate on cuDF columns.
#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

namespace facebook::velox::gpu {

__global__ void bitmaskAnd(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* lhs,
    const cudf::bitmask_type* rhs,
    cudf::size_type numWords) {
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numWords)
    return;
  output[i] = lhs[i] & rhs[i];
}

__global__ void bitmaskOr(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* lhs,
    const cudf::bitmask_type* rhs,
    cudf::size_type numWords) {
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numWords)
    return;
  output[i] = lhs[i] | rhs[i];
}

__global__ void bitmaskNot(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* input,
    cudf::size_type numWords) {
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numWords)
    return;
  output[i] = ~input[i];
}

__global__ void bitmaskAndNot(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* lhs,
    const cudf::bitmask_type* rhs,
    cudf::size_type numWords) {
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numWords)
    return;
  output[i] = lhs[i] & ~rhs[i];
}

// IF special form: for each row, selects thenCol or elseCol based on
// conditionBool (a bool column). Output nullity is derived from the
// selected branch's null mask.
template <typename T>
__global__ void gpuIfKernel(
    T* output,
    cudf::bitmask_type* outNull,
    const bool* conditionBool,
    const cudf::bitmask_type* condNull,
    const T* thenData,
    const cudf::bitmask_type* thenNull,
    const T* elseData,
    const cudf::bitmask_type* elseNull,
    cudf::size_type numRows) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numRows)
    return;

  bool condIsNull = condNull && !cudf::bit_is_set(condNull, row);
  if (condIsNull) {
    // Null condition → pick else branch
    output[row] = elseData[row];
    if (elseNull && !cudf::bit_is_set(elseNull, row)) {
      cudf::clear_bit(outNull, row);
    } else {
      cudf::set_bit(outNull, row);
    }
    return;
  }

  if (conditionBool[row]) {
    output[row] = thenData[row];
    if (thenNull && !cudf::bit_is_set(thenNull, row)) {
      cudf::clear_bit(outNull, row);
    } else {
      cudf::set_bit(outNull, row);
    }
  } else {
    output[row] = elseData[row];
    if (elseNull && !cudf::bit_is_set(elseNull, row)) {
      cudf::clear_bit(outNull, row);
    } else {
      cudf::set_bit(outNull, row);
    }
  }
}

// COALESCE: returns the first non-null value from a list of columns.
// columns[0..numCols-1] and nullMasks[0..numCols-1] are device arrays.
template <typename T>
__global__ void gpuCoalesceKernel(
    T* output,
    cudf::bitmask_type* outNull,
    const T* const* columns,
    const cudf::bitmask_type* const* nullMasks,
    int numCols,
    cudf::size_type numRows) {
  cudf::size_type row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numRows)
    return;

  for (int c = 0; c < numCols; ++c) {
    if (!nullMasks[c] || cudf::bit_is_set(nullMasks[c], row)) {
      output[row] = columns[c][row];
      cudf::set_bit(outNull, row);
      return;
    }
  }
  cudf::clear_bit(outNull, row);
}

inline void launchBitmaskAnd(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* lhs,
    const cudf::bitmask_type* rhs,
    cudf::size_type numRows,
    cudaStream_t stream = 0) {
  cudf::size_type words = (numRows + 31) / 32;
  int blocks = (words + 255) / 256;
  bitmaskAnd<<<blocks, 256, 0, stream>>>(output, lhs, rhs, words);
}

inline void launchBitmaskOr(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* lhs,
    const cudf::bitmask_type* rhs,
    cudf::size_type numRows,
    cudaStream_t stream = 0) {
  cudf::size_type words = (numRows + 31) / 32;
  int blocks = (words + 255) / 256;
  bitmaskOr<<<blocks, 256, 0, stream>>>(output, lhs, rhs, words);
}

inline void launchBitmaskNot(
    cudf::bitmask_type* output,
    const cudf::bitmask_type* input,
    cudf::size_type numRows,
    cudaStream_t stream = 0) {
  cudf::size_type words = (numRows + 31) / 32;
  int blocks = (words + 255) / 256;
  bitmaskNot<<<blocks, 256, 0, stream>>>(output, input, words);
}

} // namespace facebook::velox::gpu
