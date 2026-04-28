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
    t->strides[i] =
        i < tensor.dim() ? (tensor.size(i) == 1 ? 0 : tensor.stride(i)) : 0;
  }
  // numEl and contiguous are set by init() on device; leave status as
  // kUninited.
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
  auto* ptrs = param<TensorTestPtrs>(blockInfo, kNumTensors * sizeof(Tensor));

  constexpr int32_t kT = sizeof(Tensor);
  {
    static int32_t paramOffsets[] = {0, kT, 2 * kT, 3 * kT, 4 * kT, 5 * kT};
    static int32_t outputOffsets[] = {0, kT, 2 * kT, 3 * kT, 4 * kT, 5 * kT};
    static int32_t altOffsets[] = {-1, -1, -1, -1, -1, -1};
    for (auto i = threadIdx.x;
         i < sizeof(paramOffsets) / sizeof(paramOffsets[0]);
         i += blockDim.x) {
      if (altOffsets[i] != -1) {
        copyTensorHead(
            param<Tensor>(blockInfo, paramOffsets[i]),
            param<Tensor>(blockInfo, altOffsets[i]));
        param<Tensor>(blockInfo, altOffsets[i])
            ->init<true>(param<Tensor>(blockInfo, outputOffsets[i]));
      } else {
        param<Tensor>(blockInfo, paramOffsets[i])
            ->init<true>(
                outputOffsets[i] != paramOffsets[i]
                    ? param<Tensor>(blockInfo, outputOffsets[i])
                    : nullptr);
      }
    }
  }
  __syncthreads();

  int32_t tensorIdx = threadIdx.x / kLanesPerTensor;
  int32_t laneIdx = threadIdx.x % kLanesPerTensor;

  // Verify non-fast-path tensors reached kInited (1D contiguous tensors
  // take a fast path that does not set status).
  if (tensorIdx < kNumTensors && tensors[tensorIdx].rank > 1) {
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

// Host-side reference: row-major decomposition matching indexToOffset.
// Peels off the innermost (last) dim first.
int32_t computeExpectedOffset(const Tensor& tensor, int32_t linearIdx) {
  int32_t offset = 0;
  for (int dim = tensor.rank - 1; dim >= 0; --dim) {
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

  // Verify metadata. 1D contiguous tensors take a fast path that does not
  // set status to kInited.
  for (int i = 0; i < kNumTensors; ++i) {
    if (allTensors[i].dim() > 1 || !allTensors[i].is_contiguous()) {
      EXPECT_EQ(tensors[i].status, Tensor::kInited)
          << "Tensor " << i << " not inited";
    }

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

// Broadcast test: tensors[0] is the output (full shape), tensors[1..N] are
// broadcast inputs initialized with init(output). The kernel reads each
// broadcast input at every output index and writes the offset to the result.
constexpr int kBroadcastOutputIdx = 0;
constexpr int kBroadcastInputs = 4;
constexpr int kBroadcastTensors = 1 + kBroadcastInputs;
constexpr int kBroadcastElements = 60; // 3*4*5

struct BroadcastTestPtrs {
  int32_t* offsets; // [kBroadcastInputs][kBroadcastElements]
};

__global__ void broadcastInitKernel(TorchWaveParams params) {
  ENTRY

  Tensor* tensors = param<Tensor>(blockInfo, 0);
  auto* ptrs =
      param<BroadcastTestPtrs>(blockInfo, kBroadcastTensors * sizeof(Tensor));

  // Init the output tensor normally.
  if (threadIdx.x == 0) {
    tensors[kBroadcastOutputIdx].init<false>();
  }
  __syncthreads();

  // Init broadcast inputs with the output tensor as reference.
  int inputIdx = threadIdx.x;
  if (inputIdx < kBroadcastInputs) {
    tensors[1 + inputIdx].init<false>(&tensors[kBroadcastOutputIdx]);
  }
  __syncthreads();

  // Compute offsets for each broadcast input at every output linear index.
  for (int inp = 0; inp < kBroadcastInputs; ++inp) {
    auto& tensor = tensors[1 + inp];
    for (uint32_t idx = threadIdx.x; idx < kBroadcastElements;
         idx += blockDim.x) {
      ptrs->offsets[inp * kBroadcastElements + idx] = tensor.indexToOffset(idx);
    }
  }

  LEAVE()
}

// Host reference for broadcast offset: row-major decomposition matching
// indexToOffset. Peels off the innermost (last) dim first.
int32_t computeBroadcastOffset(
    const Tensor& input,
    const int32_t* outputDims,
    int32_t rank,
    int32_t linearIdx) {
  int32_t offset = 0;
  for (int dim = rank - 1; dim >= 0; --dim) {
    int32_t r = linearIdx % outputDims[dim];
    linearIdx /= outputDims[dim];
    offset += r * input.strides[dim];
  }
  return offset;
}

TEST_F(TensorTest, broadcastIndexCalculator) {
  auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

  // Output shape: [3, 4, 5].
  auto output = at::arange(60, opts.dtype(at::kFloat)).reshape({3, 4, 5});

  // Broadcast inputs: different dims are 1 (stride 0).
  auto bc_dim0 = at::randn({1, 4, 5}, opts); // broadcast dim 0
  auto bc_dim1 = at::randn({3, 1, 5}, opts); // broadcast dim 1
  auto bc_dim2 = at::randn({3, 4, 1}, opts); // broadcast dim 2
  auto bc_all = at::randn({1, 1, 1}, opts); // broadcast all

  at::Tensor allTensors[kBroadcastTensors] = {
      output, bc_dim0, bc_dim1, bc_dim2, bc_all};

  constexpr size_t kPtrsOffset = kBroadcastTensors * sizeof(Tensor);
  constexpr size_t kParamsSize = kPtrsOffset + sizeof(BroadcastTestPtrs);
  auto paramsBuffer = allocateManagedArray<char>(kParamsSize);
  memset(paramsBuffer.get(), 0, kParamsSize);

  auto* tensors = reinterpret_cast<Tensor*>(paramsBuffer.get());
  auto* ptrs =
      reinterpret_cast<BroadcastTestPtrs*>(paramsBuffer.get() + kPtrsOffset);

  for (int i = 0; i < kBroadcastTensors; ++i) {
    fillTensorParam(allTensors[i], &tensors[i]);
  }

  auto offsets =
      allocateManagedArray<int32_t>(kBroadcastInputs * kBroadcastElements);
  memset(
      offsets.get(),
      0xff,
      kBroadcastInputs * kBroadcastElements * sizeof(int32_t));
  ptrs->offsets = offsets.get();

  auto debugInfo = allocateManagedArray<DebugInfo>(1);
  memset(debugInfo.get(), 0, sizeof(DebugInfo));

  int device;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(paramsBuffer.get(), kParamsSize, device);
  cudaMemPrefetchAsync(
      offsets.get(),
      kBroadcastInputs * kBroadcastElements * sizeof(int32_t),
      device);
  cudaMemPrefetchAsync(debugInfo.get(), sizeof(DebugInfo), device);
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  twParams.debugInfo = debugInfo.get();
  auto& bi = twParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = paramsBuffer.get();

  broadcastInitKernel<<<1, kBlockSize>>>(twParams);
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  EXPECT_EQ(debugInfo[0].line, 0)
      << "Kernel error at line " << debugInfo[0].line;

  // The output tensor should have correct numEl.
  EXPECT_EQ(tensors[0].numEl, 60u);
  EXPECT_TRUE(tensors[0].contiguous);

  // Each broadcast input has numEl from its own dims.
  at::Tensor bcTensors[kBroadcastInputs] = {bc_dim0, bc_dim1, bc_dim2, bc_all};
  for (int inp = 0; inp < kBroadcastInputs; ++inp) {
    EXPECT_EQ(
        tensors[1 + inp].numEl, static_cast<uint32_t>(bcTensors[inp].numel()))
        << "Broadcast input " << inp << " numEl mismatch";
  }

  // Verify offsets against host reference.
  int32_t outputDims[kMaxDims] = {3, 4, 5};
  for (int inp = 0; inp < kBroadcastInputs; ++inp) {
    for (int32_t idx = 0; idx < kBroadcastElements; ++idx) {
      int32_t expected =
          computeBroadcastOffset(tensors[1 + inp], outputDims, 3, idx);
      int32_t actual = offsets[inp * kBroadcastElements + idx];
      EXPECT_EQ(actual, expected)
          << "Broadcast offset mismatch: input=" << inp << " linearIdx=" << idx
          << " expected=" << expected << " actual=" << actual;
    }
  }

  // Spot-check: for bc_dim1 [3, 1, 5] broadcast to [3, 4, 5]:
  // linearIdx 5 = (0, 1, 0) in output → input offset should be
  // 0 * stride[0] + 0 * stride[1] + 0 * stride[2] = 0
  // since stride[1] = 0 (broadcast), row 0 col 0.
  // linearIdx 10 = (0, 2, 0) → same as above = 0.
  // linearIdx 20 = (1, 0, 0) → 1 * 5 + 0 + 0 = 5.
  int bc1_idx = 1; // bc_dim1 is input index 1
  EXPECT_EQ(offsets[bc1_idx * kBroadcastElements + 0], 0)
      << "bc_dim1[0] should be 0";
  EXPECT_EQ(offsets[bc1_idx * kBroadcastElements + 5], 0)
      << "bc_dim1[5] (dim1 changes) should still be 0 (stride 0)";
  EXPECT_EQ(offsets[bc1_idx * kBroadcastElements + 20], 5)
      << "bc_dim1[20] (dim0=1) should be 5";
}

} // namespace
} // namespace torch::wave
