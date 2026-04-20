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

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "velox/experimental/gpu/Common.h"

namespace torch::wave {
constexpr int32_t kBlockSize = 256;
}

#include "velox/experimental/torchwave/Core.cuh"

namespace torch::wave {
namespace {

using facebook::velox::gpu::CudaPtr;

template <typename T>
CudaPtr<T[]> allocateManagedArray(size_t count) {
  T* ptr;
  CUDA_CHECK_FATAL(cudaMallocManaged(&ptr, count * sizeof(T)));
  return CudaPtr<T[]>(ptr);
}

void fillTensorParam(const at::Tensor& tensor, Tensor* t) {
  memset(t, 0, sizeof(Tensor));
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < kMaxDims; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
  // numEl and contiguous are set by init() on device; leave status as kUninited.
}

constexpr int kNumTensors = 6;
constexpr int kLanesPerTensor = 20;
constexpr int kNumNonContiguous = 3;
constexpr int kFirstNonContiguous = kNumTensors - kNumNonContiguous;
constexpr int kIndicesPerLane = 8;
constexpr int kTotalIndices =
    kNumNonContiguous * kLanesPerTensor * kIndicesPerLane;

// Params layout passed to the kernel via BlockInfo.params:
//   offset 0:                          Tensor[kNumTensors]
//   offset kNumTensors*sizeof(Tensor): int32_t* linearIndices (device ptr)
//   offset + sizeof(ptr):              int32_t* offsets       (device ptr)
struct TensorTestPtrs {
  int32_t* linearIndices;
  int32_t* offsets;
};

__global__ void tensorInitKernel(TorchWaveParams params) {
  ENTRY

  Tensor* tensors = param<Tensor>(blockInfo, 0);
  auto* ptrs =
      param<TensorTestPtrs>(blockInfo, kNumTensors * sizeof(Tensor));

  int32_t tensorIdx = threadIdx.x / kLanesPerTensor;
  int32_t laneIdx = threadIdx.x % kLanesPerTensor;

  // Each of the 20 lanes per tensor calls init concurrently.
  if (tensorIdx < kNumTensors) {
    tensors[tensorIdx].init<true>();
  }
  __syncthreads();

  // Every lane verifies its tensor reached kInited.
  if (tensorIdx < kNumTensors) {
    if (tensors[tensorIdx].status != Tensor::kInited) {
      if (blockInfo.debugInfo) {
        blockInfo.debugInfo->line = __LINE__;
        blockInfo.debugInfo->extra[0] = tensorIdx;
        blockInfo.debugInfo->extra[1] = tensors[tensorIdx].status;
      }
    }
  }

  // Index calculator test: non-contiguous tensors only.
  if (tensorIdx >= kFirstNonContiguous && tensorIdx < kNumTensors) {
    int ncIdx = tensorIdx - kFirstNonContiguous;
    auto& tensor = tensors[tensorIdx];
    int base = (ncIdx * kLanesPerTensor + laneIdx) * kIndicesPerLane;
    for (int i = 0; i < kIndicesPerLane; ++i) {
      int32_t linearIdx = ptrs->linearIndices[base + i];
      ptrs->offsets[base + i] = tensor.indexToOffset(linearIdx);
    }
  }

  LEAVE()
}

// Host-side reference: same column-major decomposition as indexToOffset.
int32_t computeExpectedOffset(const Tensor& tensor, int32_t linearIdx) {
  int32_t offset = 0;
  for (int dim = 0; dim < tensor.rank; ++dim) {
    int32_t r = linearIdx % tensor.dims[dim];
    linearIdx /= tensor.dims[dim];
    offset += r * tensor.strides[dim];
  }
  return offset;
}

class TensorTest : public ::testing::Test {};

TEST_F(TensorTest, initAndIndexCalculator) {
  auto opts = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

  // Contiguous tensors: 1D, 2D, 3D.
  auto cont1d = at::arange(100, opts);
  auto cont2d = at::arange(200, opts).reshape({10, 20});
  auto cont3d = at::arange(120, opts).reshape({4, 5, 6});

  // Non-contiguous tensors: 1D (stride-2 slice), 2D (transpose), 3D (permute).
  auto noncont1d = at::arange(100, opts).slice(0, 0, 100, 2);
  auto noncont2d = at::arange(200, opts).reshape({10, 20}).t();
  auto noncont3d = at::arange(120, opts).reshape({4, 5, 6}).permute({2, 0, 1});

  ASSERT_TRUE(cont1d.is_contiguous());
  ASSERT_TRUE(cont2d.is_contiguous());
  ASSERT_TRUE(cont3d.is_contiguous());
  ASSERT_FALSE(noncont1d.is_contiguous());
  ASSERT_FALSE(noncont2d.is_contiguous());
  ASSERT_FALSE(noncont3d.is_contiguous());

  at::Tensor allTensors[kNumTensors] = {
      cont1d, cont2d, cont3d, noncont1d, noncont2d, noncont3d};

  // Allocate params buffer: Tensor[kNumTensors] + TensorTestPtrs.
  constexpr size_t kPtrsOffset = kNumTensors * sizeof(Tensor);
  constexpr size_t kParamsSize = kPtrsOffset + sizeof(TensorTestPtrs);
  auto paramsBuffer = allocateManagedArray<char>(kParamsSize);
  memset(paramsBuffer.get(), 0, kParamsSize);

  auto* tensors = reinterpret_cast<Tensor*>(paramsBuffer.get());
  auto* ptrs =
      reinterpret_cast<TensorTestPtrs*>(paramsBuffer.get() + kPtrsOffset);

  for (int i = 0; i < kNumTensors; ++i) {
    fillTensorParam(allTensors[i], &tensors[i]);
  }

  // Allocate index test data as managed memory.
  auto linearIndices = allocateManagedArray<int32_t>(kTotalIndices);
  auto offsets = allocateManagedArray<int32_t>(kTotalIndices);
  memset(offsets.get(), 0, kTotalIndices * sizeof(int32_t));

  // Fill linear indices: each lane gets a unique set.
  for (int nc = 0; nc < kNumNonContiguous; ++nc) {
    int tensorIdx = kFirstNonContiguous + nc;
    uint32_t tensorNumEl = allTensors[tensorIdx].numel();
    for (int lane = 0; lane < kLanesPerTensor; ++lane) {
      int base = (nc * kLanesPerTensor + lane) * kIndicesPerLane;
      for (int i = 0; i < kIndicesPerLane; ++i) {
        linearIndices[base + i] = static_cast<int32_t>(
            (static_cast<uint32_t>(base + i) * 7 + 3) % tensorNumEl);
      }
    }
  }

  ptrs->linearIndices = linearIndices.get();
  ptrs->offsets = offsets.get();

  // Allocate DebugInfo for ENTRY/LEAVE timing.
  auto debugInfo = allocateManagedArray<DebugInfo>(1);
  memset(debugInfo.get(), 0, sizeof(DebugInfo));

  // Prefetch all managed memory to device before launching the kernel.
  int device;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(paramsBuffer.get(), kParamsSize, device);
  cudaMemPrefetchAsync(
      linearIndices.get(), kTotalIndices * sizeof(int32_t), device);
  cudaMemPrefetchAsync(offsets.get(), kTotalIndices * sizeof(int32_t), device);
  cudaMemPrefetchAsync(debugInfo.get(), sizeof(DebugInfo), device);
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // Set up TorchWaveParams with a single block.
  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  twParams.debugInfo = debugInfo.get();
  auto& bi = twParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = paramsBuffer.get();

  tensorInitKernel<<<1, kBlockSize>>>(twParams);
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // Check DebugInfo for kernel-side assertion failures.
  EXPECT_EQ(debugInfo[0].line, 0)
      << "Kernel error at line " << debugInfo[0].line
      << " tensorIdx=" << debugInfo[0].extra[0]
      << " status=" << debugInfo[0].extra[1];

  // Verify every tensor reached kInited with correct metadata.
  for (int i = 0; i < kNumTensors; ++i) {
    EXPECT_EQ(tensors[i].status, Tensor::kInited)
        << "Tensor " << i << " not inited";

    bool expectedContiguous = allTensors[i].is_contiguous();
    EXPECT_EQ(tensors[i].contiguous, expectedContiguous)
        << "Tensor " << i << " contiguous flag mismatch";

    uint32_t expectedNumEl = allTensors[i].numel();
    EXPECT_EQ(tensors[i].numEl, expectedNumEl)
        << "Tensor " << i << " numEl mismatch";
  }

  // Verify index calculator offsets for each non-contiguous tensor.
  for (int nc = 0; nc < kNumNonContiguous; ++nc) {
    int tensorIdx = kFirstNonContiguous + nc;
    for (int lane = 0; lane < kLanesPerTensor; ++lane) {
      int base = (nc * kLanesPerTensor + lane) * kIndicesPerLane;
      for (int i = 0; i < kIndicesPerLane; ++i) {
        int32_t linearIdx = linearIndices[base + i];
        int32_t expected = computeExpectedOffset(tensors[tensorIdx], linearIdx);
        EXPECT_EQ(offsets[base + i], expected)
            << "IndexCalculator mismatch: tensor=" << tensorIdx
            << " lane=" << lane << " linearIdx=" << linearIdx;
      }
    }
  }

  // Print timing from DebugInfo.
  printf(
      "tensorInitKernel: %ld GPU clocks\n",
      static_cast<long>(debugInfo[0].clocks));
}

} // namespace
} // namespace torch::wave
