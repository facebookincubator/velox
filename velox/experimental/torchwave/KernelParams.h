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

#include <cstdint>

namespace torch::wave {

constexpr int32_t kDebugNoOp = -1;

/// Header for host to torch::wave kernel communication. Included in both host
/// and device code.

constexpr int kMaxDims = 4;

// ScalarType constants matching c10::ScalarType enum values. Defined here so
// device code can switch on element types without including c10 headers.
constexpr uint8_t kScalarTypeByte = 0;
constexpr uint8_t kScalarTypeChar = 1;
constexpr uint8_t kScalarTypeShort = 2;
constexpr uint8_t kScalarTypeInt = 3;
constexpr uint8_t kScalarTypeLong = 4;
constexpr uint8_t kScalarTypeHalf = 5;
constexpr uint8_t kScalarTypeFloat = 6;
constexpr uint8_t kScalarTypeDouble = 7;
constexpr uint8_t kScalarTypeBool = 11;
constexpr uint8_t kScalarTypeBFloat16 = 15;

// Fast integer division by multiplication using a precomputed magic number.
// Adapted from PyTorch ATen IntDivider. The dividend must be at most
// INT32_MAX and the divisor must be in [1, INT32_MAX].
struct IntDivider {
  unsigned int divisor;
  unsigned int m1; // magic multiplier
  unsigned int shift;

  IntDivider() = default;

#ifdef __CUDACC__
  /// Computes magic number and shift for fast division by 'd'.
  __device__ void init(unsigned int d) {
    divisor = d;
    for (shift = 0; shift < 32; shift++) {
      if ((1U << shift) >= divisor) {
        break;
      }
    }
    uint64_t one = 1;
    uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
    m1 = magic;
  }

  __device__ inline unsigned int div(unsigned int n) const {
    unsigned int t = __umulhi(n, m1);
    return (t + n) >> shift;
  }

  __device__ inline unsigned int mod(unsigned int n) const {
    return n - div(n) * divisor;
  }

  __device__ inline void
  divmod(unsigned int n, unsigned int& q, unsigned int& r) const {
    q = div(n);
    r = n - q * divisor;
  }
#endif
};

/// Device-side tensor descriptor passed in kernel parameters. Shared between
/// host and device.
struct Tensor {
  static constexpr uint32_t kUninited = 0;
  static constexpr uint32_t kIniting = 1;
  static constexpr uint32_t kInited = 2;

  void* storage{nullptr};
  int8_t rank{0};
  uint8_t elementSize{0};
  uint8_t elementType{0};
  int32_t dims[kMaxDims]{};
  int32_t strides[kMaxDims]{};
  uint32_t numEl{0};
  uint32_t status{kUninited};
  bool contiguous{false};

  /// Index calculator dividers, initialized on device via
  /// initIndexCalculator().
  IntDivider sizes[kMaxDims]{};

#ifdef __CUDACC__
  /// Initializes IntDivider magic numbers. If 'output' is provided, uses
  /// output dims for broadcast index decomposition while keeping this
  /// tensor's strides.
  // sizes[] are stored innermost-first: sizes[0] = dims[rank-1], etc.
  // indexToOffset peels off the innermost dim first via repeated divmod,
  // matching row-major linear index decomposition.
  __device__ void initIndexCalculator(const Tensor* output = nullptr) {
    const int32_t* d = output ? output->dims : dims;
    int32_t dimOffset = output ? output->rank - rank : 0;
    for (int i = 0; i < rank; ++i) {
      sizes[i].init(d[rank - 1 - i + dimOffset]);
    }
  }

  /// Returns the byte offset for element 'linearIdx' in this strided tensor.
  __device__ inline int32_t indexToOffset(uint32_t linearIdx) const {
    int32_t offset = 0;
#pragma unroll
    for (int dim = 0; dim < kMaxDims; ++dim) {
      if (dim == rank) {
        break;
      }
      unsigned int q, r;
      sizes[dim].divmod(linearIdx, q, r);
      linearIdx = q;
      // A size-1 dimension broadcasts: every output index along it maps to
      // element 0, so its stride contributes nothing.  (When the output dim
      // is also 1, r is already 0, so this is harmless.)
      int32_t stride = dims[rank - 1 - dim] == 1 ? 0 : strides[rank - 1 - dim];
      offset += r * stride;
    }
    return offset;
  }

  /// Checks if tensor is contiguous in row-major order (fastest dim at rank-1).
  __device__ bool isContiguous() const {
    if (rank == 0) {
      return true;
    }
    if (strides[rank - 1] != 1) {
      return false;
    }
    for (int i = rank - 2; i >= 0; --i) {
      if (strides[i] != strides[i + 1] * dims[i + 1]) {
        return false;
      }
    }
    return true;
  }

  /// Ensures exactly one thread executes 'func', others wait until complete.
  template <typename Func>
  __device__ void synchronized(Func func) {
    if (atomicAdd(&status, 0u) == kInited) {
      return;
    }
    uint32_t old = atomicCAS(&status, kUninited, kIniting);
    if (old != kUninited) {
      unsigned int waitNano = 50;
      while (atomicAdd(&status, 0u) != kInited) {
        __nanosleep(waitNano);
        waitNano += threadIdx.x & 31;
      }
      return;
    }
    func();
    __threadfence();
    atomicExch(&status, kInited);
  }

  /// Initializes contiguity flag, numEl, and index calculator if needed.
  /// If 'output' is non-null, uses output dims for the index calculator
  /// (broadcast support) and sets numEl from output dims.
  template <bool kConcurrent>
  __device__ void init(const Tensor* output = nullptr) {
    if ((!output || output->dims[0] == dims[0]) && rank == 1 &&
        strides[0] == 1) {
      numEl = dims[0];
      contiguous = true;
      return;
    }
    auto doInit = [&]() {
      contiguous = isContiguous();
      initIndexCalculator(output);
      uint32_t n = 1;
      for (int i = 0; i < rank; ++i) {
        n *= dims[i];
      }
      numEl = n;
    };
    if constexpr (kConcurrent) {
      synchronized(doInit);
    } else {
      if (status == kInited) {
        return;
      }
      doInit();
      status = kInited;
    }
  }
#endif
};

/// Device-side list-of-tensors descriptor for kernel parameters.
struct TensorList {
  int64_t size;
  Tensor** tensors;
};

/// Device-side list-of-scalars descriptor for kernel parameters.
struct ScalarList {
  int64_t size;
  int64_t* data;
};

/// Struct for returning errors to host. Each block has one. These may be
/// checked at a delay, so return status that requires action on host side must
/// be sent in returnStatus, not here.
struct DebugInfo {
  int64_t clocks{0};
  int64_t barrierClocks{0};
  int32_t op{0};
  int32_t line{0};
  int64_t extra[2] = {};
  char message[20] = {};
};

/// Each TB fetches its instructions from BlockInfo at blockIdx.x. Copied at
/// offset 0 in dynamic shared memory of the block.
struct BlockInfo {
  /// kernel dependent op code for this block.
  int32_t op;

  /// Index of this block within the blocks with for the same op.
  int32_t blockInOp;

  /// Number of blocks for this op.
  int32_t numBlocksInOp;

  /// Pointer to per-op params, format depends on 'op'.
  void* params;

  DebugInfo* debugInfo;

  /// clock64() at start of block.
  int64_t start;
  int64_t barrierClocks;
};

/// Top-level kernel parameter struct containing BlockInfo array and debug
/// pointers.
struct TorchWaveParams {
  /// Pointer to BlockInfo for blockIdx.x 0. gridDim.x consecutive BlockInfos.
  /// If nullptr the BlockInfos are in inlineInfo.
  BlockInfo* info;
  DebugInfo* debugInfo;
  /// If the grid fits in kMaxInlineBlocks BlockInfos, they are inlined here
  /// to avoid a separate device allocation for the block info array.
  static constexpr int32_t kMaxInlineBlocks = 100;
  BlockInfo inlineInfo[kMaxInlineBlocks];
};

} // namespace torch::wave
