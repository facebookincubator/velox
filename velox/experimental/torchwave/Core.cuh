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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "velox/experimental/torchwave/KernelParams.h"

namespace torch::wave {

template <typename T>
__device__ inline T* param(const BlockInfo& block, int32_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(block.params) + offset);
}

__device__ inline uint32_t numEl(const Tensor& tensor) {
  uint32_t size = 1;
  for (auto i = 0; i < tensor.rank; ++i) {
    size = size * tensor.dims[i];
  }
  return size;
}

template <typename T>
__device__ inline T* storage(const Tensor* tensor) {
  return reinterpret_cast<T*>(tensor->storage);
}

template <typename T>
__device__ inline T elementRef(Tensor* t, uint32_t idx, uint32_t size) {
  return idx < size ? storage<T>(t)[idx] : T();
}

template <typename T>
__device__ void __copy(Tensor* source, T* dest, BlockInfo& block) {
  // An empty (size-0) concat operand arrives as an undefined tensor with null
  // storage, yet Tensor::init gives such a rank-0 operand numEl==1 (the empty
  // product). Guard here so the loop below does not dereference null: an empty
  // operand contributes no elements to the concatenation.
  if (source->storage == nullptr) {
    return;
  }
  auto n = source->numEl;
  uint32_t start = block.blockInOp * blockDim.x + threadIdx.x;
  uint32_t stride = block.numBlocksInOp * blockDim.x;
  if (source->contiguous) {
    auto* src = storage<T>(source);
    for (uint32_t i = start; i < n; i += stride) {
      dest[i] = src[i];
    }
  } else {
    for (uint32_t i = start; i < n; i += stride) {
      dest[i] = storage<T>(source)[source->indexToOffset(i)];
    }
  }
}

// Like __copy, but value-converts each element from SrcT to DstT (e.g. a cat of
// mixed dtypes, where torch promotes an int64 element into a float output).
template <typename SrcT, typename DstT>
__device__ void __copyConvert(Tensor* source, DstT* dest, BlockInfo& block) {
  // See __copy: an empty concat operand has null storage but init gives it
  // numEl==1, so skip it here to avoid dereferencing null.
  if (source->storage == nullptr) {
    return;
  }
  auto n = source->numEl;
  uint32_t start = block.blockInOp * blockDim.x + threadIdx.x;
  uint32_t stride = block.numBlocksInOp * blockDim.x;
  if (source->contiguous) {
    auto* src = storage<SrcT>(source);
    for (uint32_t i = start; i < n; i += stride) {
      dest[i] = static_cast<DstT>(src[i]);
    }
  } else {
    for (uint32_t i = start; i < n; i += stride) {
      dest[i] =
          static_cast<DstT>(storage<SrcT>(source)[source->indexToOffset(i)]);
    }
  }
}

__device__ inline void copyTensorHead(const Tensor* in, Tensor* out) {
  out->storage = in->storage;
  out->rank = in->rank;
  for (int i = 0; i < kMaxDims; ++i) {
    out->dims[i] = in->dims[i];
    out->strides[i] = in->strides[i];
  }
}

__device__ inline uint32_t
complexIdx(bool fast, const Tensor* t, uint32_t idx) {
  if (!fast) {
    return t->indexToOffset(idx);
  }
  return idx;
}

struct Int32X32 {
  uint64_t data[32];
  __host__ __device__ operator void*() {
    return static_cast<void*>(data);
  }
  __host__ __device__ operator void*() const {
    return const_cast<void*>(static_cast<const void*>(data));
  }
};

// Thread 0 loads BlockInfo from params (inlined in TorchWaveParams or from a
// separate device allocation) into shared memory; all threads sync before use.
// LEAVE records timing into the per-block DebugInfo.
#define ENTRY                                                                  \
  __shared__ BlockInfo blockInfo;                                              \
  if (threadIdx.x == 0) {                                                      \
    blockInfo =                                                                \
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x]; \
    blockInfo.debugInfo =                                                      \
        params.debugInfo ? &params.debugInfo[blockIdx.x] : nullptr;            \
    blockInfo.barrierClocks = 0;                                               \
    if (blockInfo.debugInfo && blockInfo.op != kDebugNoOp) {                   \
      blockInfo.start = clock64();                                             \
      memset(blockInfo.debugInfo, 0, sizeof(DebugInfo));                       \
    }                                                                          \
  }                                                                            \
  __syncthreads();

#define LEAVE()                                                                \
  __syncthreads();                                                             \
  if (threadIdx.x == 0 && blockInfo.debugInfo && blockInfo.op != kDebugNoOp) { \
    blockInfo.debugInfo->clocks = clock64() - blockInfo.start;                 \
    blockInfo.debugInfo->barrierClocks = blockInfo.barrierClocks;              \
    blockInfo.debugInfo->op = blockInfo.op;                                    \
  }

#define SET_MSG(info, str)                                   \
  do {                                                       \
    static const __device__ char __msg[] __align__(8) = str; \
    *reinterpret_cast<int64_t*>((info)->message) =           \
        *reinterpret_cast<const int64_t*>(__msg);            \
  } while (0)

// NVIDIA GPU warp size. All scan/reduce templates assume this value.
constexpr int32_t kWarpThreads = 32;

template <typename T, typename U>
__host__ __device__ constexpr inline T roundUp(T value, U factor) {
  return (value + (factor - 1)) / factor * factor;
}

/// Rounds 'x' up to the next multiple of 'y', where y must be a power of two.
template <typename T, typename U>
__host__ __device__ constexpr inline T roundUpPwr2(T x, U y) {
  return (x + (y - 1)) & ~(y - 1);
}

__device__ inline void traceTensor(const BlockInfo& block, int32_t offset) {
  auto* t = param<Tensor>(block, offset);
  printf(
      "offset=%d storage=%p rank=%d dims=[%d,%d,%d] strides=[%d,%d,%d]\n",
      offset,
      t->storage,
      t->rank,
      t->dims[0],
      t->dims[1],
      t->dims[2],
      t->strides[0],
      t->strides[1],
      t->strides[2]);
}

#define TRACE0(__x)         \
  do {                      \
    if (threadIdx.x == 0) { \
      __x;                  \
    }                       \
  } while (0)

// Each counter offset must be used at most once per kernel invocation.
// The counter is not reset after use, so reusing the same offset would
// cause the barrier to pass immediately.
__device__ inline void opBarrier(BlockInfo& info, int32_t counterOffset) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    auto barrierStart = clock64();
    auto* counter = param<int32_t>(info, counterOffset);
    atomicAdd(counter, 1);
    volatile int32_t* vc = reinterpret_cast<volatile int32_t*>(counter);
    while (*vc < info.numBlocksInOp) {
      __nanosleep(100);
    }
    info.barrierClocks += clock64() - barrierStart;
  }
  __syncthreads();
  // Acquire fence: the leading __threadfence() only provides the release side
  // (a block's writes are visible before it signals arrival). Without a
  // matching acquire on the consumer side, a block that has observed all
  // producers arrive may still read stale, cached global memory written by the
  // other blocks. 'volatile' only forces re-reading the counter, not the data
  // the counter guards. Every thread must acquire here, so this fence is
  // outside the threadIdx.x == 0 block.
  __threadfence();
}

// Copies all elements from source to dest using grid-strided loop.
template <typename T>
__device__ void __copyTensor(Tensor* source, Tensor* dest, BlockInfo& block) {
  __copy<T>(source, storage<T>(dest), block);
}

// Computes linear offset from scalar index values in registers.
__device__ inline int32_t indexOffset(Tensor* tensor, int32_t index0) {
  return index0 * tensor->strides[0];
}

__device__ inline int32_t
indexOffset(Tensor* tensor, int32_t index0, int32_t index1) {
  return index0 * tensor->strides[0] + index1 * tensor->strides[1];
}

__device__ inline int32_t
indexOffset(Tensor* tensor, int32_t index0, int32_t index1, int32_t index2) {
  return index0 * tensor->strides[0] + index1 * tensor->strides[1] +
      index2 * tensor->strides[2];
}

// Scatter values into dest where mask is true. Supports broadcast: mask
// or source of length 1 broadcast to dest size.
template <typename T>
__device__ void __masked_put(
    Tensor* dest,
    Tensor* mask,
    Tensor* source,
    Tensor* /*output*/,
    uint32_t& size,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    auto destN = dest->numEl;
    auto maskN = mask->numEl;
    if (maskN != destN && maskN != 1) {
      size = 0;
      if (block.debugInfo) {
        block.debugInfo->line = __LINE__;
        block.debugInfo->extra[0] = destN;
        block.debugInfo->extra[1] = maskN;
        SET_MSG(block.debugInfo, "sz msmt\0");
      }
    } else {
      size = destN;
    }
  }
  __syncthreads();
  if (size == 0) {
    return;
  }
  auto* dst = storage<T>(dest);
  auto* msk = storage<bool>(mask);
  auto* src = storage<T>(source);
  bool broadcastMask = mask->numEl == 1;
  bool broadcastSrc = source->numEl == 1;
  for (uint32_t i = block.blockInOp * blockDim.x + threadIdx.x; i < size;
       i += block.numBlocksInOp * blockDim.x) {
    if (msk[broadcastMask ? 0 : i]) {
      dst[i] = src[broadcastSrc ? 0 : i];
    }
  }
}

} // namespace torch::wave
