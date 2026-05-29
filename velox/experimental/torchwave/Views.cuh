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

// Computes a view of 'input' with the shape given by 'shape' and writes the
// result into 'output'. The output shares storage with the input. A -1 in
// shape.data means infer that dimension from the input's total element count.
// Uses Tensor::synchronized so that concurrent blocks calling this on the same
// output safely execute only once.
__device__ inline void tw_view(
    Tensor* input,
    const ScalarList& shape,
    Tensor* output,
    BlockInfo& /*block*/) {
  if (threadIdx.x == 0) {
    output->synchronized([&]() {
      output->storage = input->storage;
      output->rank = shape.size;

      uint32_t totalNumEl = 1;
      for (int i = 0; i < input->rank; ++i) {
        totalNumEl *= input->dims[i];
      }

      int32_t inferIdx = -1;
      int32_t knownProduct = 1;
      for (int i = 0; i < shape.size; ++i) {
        if (shape.data[i] == -1) {
          inferIdx = i;
        } else {
          output->dims[i] = shape.data[i];
          knownProduct *= shape.data[i];
        }
      }
      if (inferIdx >= 0) {
        output->dims[inferIdx] =
            knownProduct == 0 ? 0 : totalNumEl / knownProduct;
      }

      // Contiguous strides: innermost dim has stride 1.
      if (shape.size > 0) {
        output->strides[shape.size - 1] = 1;
        for (int i = shape.size - 2; i >= 0; --i) {
          output->strides[i] = output->strides[i + 1] * output->dims[i + 1];
        }
      }

      output->numEl = totalNumEl;
      output->contiguous = true;
    });
  }
  __syncthreads();
}

// Reshape is the same operation as view at the kernel level: both produce a
// contiguous view with new dims and strides computed from the shape.
__device__ inline void tw_reshape(
    Tensor* input,
    const ScalarList& shape,
    Tensor* output,
    BlockInfo& block) {
  tw_view(input, shape, output, block);
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
