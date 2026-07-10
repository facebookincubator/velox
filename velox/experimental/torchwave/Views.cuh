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

#include "velox/experimental/torchwave/KernelParams.h"

namespace torch::wave {

#ifdef __CUDACC__

// On a view/slice error, alias the output to the input (same storage, shape,
// strides, and element count) so downstream reads stay within mapped memory
// while the error is reported to the host via blockInfo.
__device__ inline void viewErrorFallback(const Tensor* input, Tensor* output) {
  output->storage = input->storage;
  output->rank = input->rank;
  for (int i = 0; i < kMaxDims; ++i) {
    output->dims[i] = input->dims[i];
    output->strides[i] = input->strides[i];
  }
  output->numEl = input->numEl;
  output->contiguous = input->contiguous;
}

// Records a view/slice error in 'block' (line + two diagnostic values + a short
// message). errorString() on the host treats a non-zero line as an error. This
// mirrors the out-of-range index error path in Elementwise.cuh without
// depending on Core.cuh's SET_MSG macro.
__device__ inline void setViewError(
    BlockInfo& block,
    int line,
    int64_t a,
    int64_t b,
    const char* msg) {
  if (!block.debugInfo) {
    return;
  }
  block.debugInfo->line = line;
  block.debugInfo->extra[0] = a;
  block.debugInfo->extra[1] = b;
  for (int i = 0; i < (int)sizeof(block.debugInfo->message); ++i) {
    char c = msg[i];
    block.debugInfo->message[i] = c;
    if (c == '\0') {
      break;
    }
  }
}

// Computes a view of 'input' with the shape given by 'shape' and writes the
// result into 'output'. The output shares storage with the input. A -1 in
// shape.data means infer that dimension from the input's total element count.
// On an invalid view (rank out of range, non-contiguous input, element-count
// mismatch, or more than one inferred dim) the output is left aliasing the
// input and the error is recorded in 'block'. Uses Tensor::synchronized so that
// concurrent blocks calling this on the same output safely execute only once.
__device__ inline void tw_view(
    Tensor* input,
    const ScalarList& shape,
    Tensor* output,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    output->synchronized([&]() {
      uint32_t totalNumEl = 1;
      for (int i = 0; i < input->rank; ++i) {
        totalNumEl *= input->dims[i];
      }

      // Validate the requested shape before writing any output metadata.
      bool ok = shape.size > 0 && shape.size <= kMaxDims && input->contiguous;
      int32_t inferIdx = -1;
      int64_t knownProduct = 1;
      for (int i = 0; ok && i < shape.size; ++i) {
        int64_t d = shape.data[i];
        if (d == -1) {
          if (inferIdx >= 0) {
            ok = false; // at most one inferred dim
          }
          inferIdx = i;
        } else if (d < 0) {
          ok = false; // negative dim other than -1
        } else {
          knownProduct *= d;
        }
      }
      if (ok) {
        if (inferIdx >= 0) {
          ok = knownProduct != 0 && (totalNumEl % knownProduct) == 0;
        } else {
          ok = static_cast<int64_t>(totalNumEl) == knownProduct;
        }
      }

      if (!ok) {
        viewErrorFallback(input, output);
        setViewError(block, __LINE__, input->rank, shape.size, "Bad view");
        return;
      }

      output->storage = input->storage;
      output->rank = shape.size;
      for (int i = 0; i < shape.size; ++i) {
        output->dims[i] = (i == inferIdx)
            ? static_cast<int32_t>(totalNumEl / knownProduct)
            : static_cast<int32_t>(shape.data[i]);
      }

      // Contiguous strides: innermost dim has stride 1.
      output->strides[shape.size - 1] = 1;
      for (int i = shape.size - 2; i >= 0; --i) {
        output->strides[i] = output->strides[i + 1] * output->dims[i + 1];
      }

      output->numEl = totalNumEl;
      output->contiguous = true;
    });
  }
  __syncthreads();
}

// Slices 'input' along dimension 'dim' to the half-open range [start, end) with
// the given 'step', producing a strided view that shares storage with the input
// (aten.slice.Tensor semantics: negative and out-of-range start/end are
// clamped). On an invalid slice (dim out of range or step <= 0) the output is
// left aliasing the input and the error is recorded in 'block'.
template <typename T>
__device__ inline void tw_slice(
    Tensor* input,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step,
    Tensor* output,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    output->synchronized([&]() {
      int64_t d = dim < 0 ? dim + input->rank : dim;
      if (d < 0 || d >= input->rank || step <= 0) {
        viewErrorFallback(input, output);
        setViewError(block, __LINE__, dim, step, "Bad slice");
        return;
      }

      const int64_t dimSize = input->dims[d];
      int64_t s = start < 0 ? start + dimSize : start;
      s = s < 0 ? 0 : (s > dimSize ? dimSize : s);
      int64_t e = end < 0 ? end + dimSize : end;
      e = e < 0 ? 0 : (e > dimSize ? dimSize : e);
      const int64_t length = e > s ? (e - s + step - 1) / step : 0;

      output->storage = reinterpret_cast<char*>(
          static_cast<T*>(input->storage) +
          s * static_cast<int64_t>(input->strides[d]));
      output->rank = input->rank;
      for (int i = 0; i < input->rank; ++i) {
        output->dims[i] = input->dims[i];
        output->strides[i] = input->strides[i];
      }
      output->dims[d] = static_cast<int32_t>(length);
      output->strides[d] = input->strides[d] * static_cast<int32_t>(step);

      uint32_t n = 1;
      for (int i = 0; i < output->rank; ++i) {
        n *= output->dims[i];
      }
      output->numEl = n;
      output->contiguous = output->isContiguous();
    });
  }
  __syncthreads();
}

// Selects a single element along dimension 'dim' at position 'index',
// reducing the rank by one. The output shares storage with the input.
template <typename T>
__device__ inline void tw_select(
    Tensor* input,
    int64_t dim,
    int64_t index,
    Tensor* output,
    BlockInfo& /*block*/) {
  if (threadIdx.x == 0) {
    output->synchronized([&]() {
      if (dim < 0) {
        dim += input->rank;
      }
      if (index < 0) {
        index += input->dims[dim];
      }
      output->storage = reinterpret_cast<char*>(
          static_cast<T*>(input->storage) +
          static_cast<int64_t>(index) * input->strides[dim]);
      output->rank = input->rank - 1;
      int out_i = 0;
      for (int i = 0; i < input->rank; ++i) {
        if (i == dim) {
          continue;
        }
        output->dims[out_i] = input->dims[i];
        output->strides[out_i] = input->strides[i];
        ++out_i;
      }
      uint32_t n = 1;
      for (int i = 0; i < output->rank; ++i) {
        n *= output->dims[i];
      }
      output->numEl = n;
      output->contiguous = output->isContiguous();
    });
  }
  __syncthreads();
}

// Makes 'dest' a 1D view of 'src' starting at 'elementOffset' elements.
// 'elementSize' is sizeof the element type (e.g. 4 for float).
__device__ inline void
__view(Tensor& src, int32_t elementOffset, int32_t elementSize, Tensor& dest) {
  dest.storage = static_cast<char*>(src.storage) +
      static_cast<int64_t>(elementOffset) * elementSize;
  dest.status = Tensor::kInited;
}

#endif

} // namespace torch::wave
