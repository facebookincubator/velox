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

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "velox/experimental/gpu/Common.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace torch::wave {
constexpr int32_t kBlockSize = 256;
}

#include "velox/experimental/torchwave/Core.cuh"
#include "velox/experimental/torchwave/Hash.cuh"
#include "velox/experimental/torchwave/Scan.cuh"

DEFINE_bool(
    scan_benchmark,
    false,
    "Run the prefix sum and stream compaction benchmarks. The benchmark case is DISABLED_, so pass --gtest_also_run_disabled_tests too");

DEFINE_int64(
    scan_bytes,
    200 << 20,
    "Size of the scanned/compacted int64 data in bytes");

DEFINE_double(
    scan_selectivity,
    0.8,
    "Fraction of set flags in the stream compaction benchmark");

DEFINE_int32(
    scan_repeats,
    5,
    "Number of timed repetitions. The shortest time is reported");

DEFINE_bool(
    scan_check,
    true,
    "Verify every variant against a CPU reference before timing it");

DEFINE_bool(
    scan_single_block,
    false,
    "Include the single-block variants, which are two orders of magnitude slower than the rest at the default size");

namespace torch::wave {
namespace {

using facebook::velox::gpu::allocateDeviceMemory;
using facebook::velox::gpu::allocateManagedMemory;
using facebook::velox::gpu::CudaPtr;

template <typename T>
CudaPtr<T[]> allocateManagedArray(size_t count) {
  T* ptr;
  CUDA_CHECK_FATAL(cudaMallocManaged(&ptr, count * sizeof(T)));
  return CudaPtr<T[]>(ptr);
}

void fillTensorParam(const at::Tensor& tensor, Tensor* t) {
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Tensors with more than 3 dims not supported, got ",
      tensor.dim());
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  t->elementSize = tensor.element_size();
  t->elementType = static_cast<uint8_t>(tensor.scalar_type());
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
  t->numEl = tensor.numel();
  t->status = Tensor::kUninited;
  // The kernels reach non-contiguous inputs through indexToOffset, which reads
  // the on-device index calculator no standalone kernel here initializes. The
  // flag is not defaulted because the params come from raw device memory, so
  // set it explicitly.
  t->contiguous = tensor.is_contiguous();
}

__global__ void maskedSelectKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ uint32_t size;
  __shared__ uint32_t counter;
  __shared__ Int32X32 temp;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  size = numEl(*param<Tensor>(blockInfo, 0));
  __syncthreads();

  uint32_t roundedSize = roundUpPwr2(size, (uint32_t)kBlockSize);
  for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x;
       idx < roundedSize;
       idx += blockInfo.numBlocksInOp * blockDim.x) {
    masked_select<kBlockSize, int32_t>(
        param<Tensor>(blockInfo, 0),
        param<Tensor>(blockInfo, sizeof(Tensor)),
        param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
        (void*)temp,
        counter,
        idx,
        size,
        blockInfo);
  }
}

class KernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(KernelTest, maskedSelect) {
  constexpr int32_t kSize = 100000;

  auto input = at::randint(
      0, 1000, {kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto mask =
      at::rand({kSize}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
          .lt(0.9)
          .to(at::kBool);
  auto output =
      at::zeros({kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  auto params = allocateManagedArray<Tensor>(3);
  fillTensorParam(input, &params[0]);
  fillTensorParam(mask, &params[1]);
  fillTensorParam(output, &params[2]);

  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  auto& bi = twParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = params.get();

  maskedSelectKernel<<<1, kBlockSize>>>(twParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  auto inputCpu = input.cpu();
  auto maskCpu = mask.cpu();
  auto outputCpu = output.cpu();

  auto* inputData = inputCpu.data_ptr<int32_t>();
  auto* maskData = maskCpu.data_ptr<bool>();
  auto* outputData = outputCpu.data_ptr<int32_t>();

  std::vector<int32_t> expected;
  for (int i = 0; i < kSize; ++i) {
    if (maskData[i]) {
      expected.push_back(inputData[i]);
    }
  }

  int32_t resultSize = params[2].dims[0];
  EXPECT_EQ(resultSize, static_cast<int32_t>(expected.size()));
  for (int i = 0;
       i < std::min(resultSize, static_cast<int32_t>(expected.size()));
       ++i) {
    EXPECT_EQ(outputData[i], expected[i]) << "Mismatch at index " << i;
  }
}

__global__ void maskedSelectHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  masked_select_head<kBlockSize, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void addSizesKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ uint32_t counter;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    auto* t = param<Tensor>(blockInfo, 0);
    int32_t n = numEl(*t);
    int32_t* data = storage<int32_t>(t);
    printf("addSizes input: n=%d values:", n);
    for (int i = 0; i < n && i < 20; ++i) {
      printf(" %d", data[i]);
    }
    if (n > 20)
      printf(" ...");
    printf("\n");
  }
  __syncthreads();

  add_sizes<kBlockSize>(
      param<Tensor>(blockInfo, 0),
      param<int32_t>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);

  if (threadIdx.x == 0) {
    auto* t = param<Tensor>(blockInfo, 0);
    int32_t n = numEl(*t);
    int32_t* data = storage<int32_t>(t);
    printf(
        "addSizes output: total=%d prefix:",
        *param<int32_t>(blockInfo, sizeof(Tensor)));
    for (int i = 0; i < n && i < 20; ++i) {
      printf(" %d", data[i]);
    }
    if (n > 20)
      printf(" ...");
    printf("\n");
  }
}

__global__ void maskedSelectFinalKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  constexpr int32_t kOutputTensorOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  masked_select_final<kBlockSize, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      param<int32_t>(blockInfo, 3 * sizeof(Tensor)),
      param<Tensor>(blockInfo, kOutputTensorOffset),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

TEST_F(KernelTest, maskedSelectThreeKernel) {
  constexpr int32_t kSize = 100000;
  constexpr int32_t kNumBlocks = (kSize + kBlockSize - 1) / kBlockSize;
  // Each block does 2 iterations.
  constexpr int32_t kGridBlocks = (kNumBlocks + 1) / 2;

  auto input = at::randint(
      0, 1000, {kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto mask =
      at::rand({kSize}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
          .lt(0.9)
          .to(at::kBool);
  auto counts = at::zeros(
      {kNumBlocks}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto output =
      at::zeros({kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  auto maskCpu = mask.cpu();
  auto* maskData = maskCpu.data_ptr<bool>();

  // --- Kernel 1: maskedSelectHead ---
  auto headParams = allocateManagedArray<Tensor>(3);
  fillTensorParam(input, &headParams[0]);
  fillTensorParam(mask, &headParams[1]);
  fillTensorParam(counts, &headParams[2]);

  auto headBlockInfo = allocateManagedArray<BlockInfo>(kGridBlocks);
  for (int i = 0; i < kGridBlocks; ++i) {
    memset(&headBlockInfo[i], 0, sizeof(BlockInfo));
    headBlockInfo[i].blockInOp = i;
    headBlockInfo[i].numBlocksInOp = kGridBlocks;
    headBlockInfo[i].params = headParams.get();
  }

  auto headBlockInfoDev = allocateDeviceMemory<BlockInfo>(kGridBlocks);
  CUDA_CHECK_FATAL(cudaMemcpy(
      headBlockInfoDev.get(),
      headBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice));

  TorchWaveParams headTwParams;
  memset(&headTwParams, 0, sizeof(headTwParams));
  headTwParams.info = headBlockInfoDev.get();

  maskedSelectHeadKernel<<<kGridBlocks, kBlockSize>>>(headTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // --- Verify head kernel: per-block counts ---
  auto countsCpu = counts.cpu();
  auto* countData = countsCpu.data_ptr<int32_t>();

  std::vector<int32_t> expectedCounts(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    int32_t cnt = 0;
    for (int i = b * kBlockSize; i < std::min((b + 1) * kBlockSize, kSize);
         ++i) {
      if (maskData[i]) {
        ++cnt;
      }
    }
    expectedCounts[b] = cnt;
    EXPECT_EQ(countData[b], cnt)
        << "Head kernel: block " << b << " count mismatch";
  }

  // --- Kernel 2: addSizes ---
  auto addSizesParams =
      allocateManagedArray<char>(sizeof(Tensor) + sizeof(int32_t));
  fillTensorParam(counts, reinterpret_cast<Tensor*>(addSizesParams.get()));
  // The kernel writes the total directly at params + sizeof(Tensor) as int32_t.
  *reinterpret_cast<int32_t*>(addSizesParams.get() + sizeof(Tensor)) = 0;

  TorchWaveParams addSizesTwParams;
  memset(&addSizesTwParams, 0, sizeof(addSizesTwParams));
  auto& addBi = addSizesTwParams.inlineInfo[0];
  addBi.numBlocksInOp = 1;
  addBi.params = addSizesParams.get();

  addSizesKernel<<<1, kBlockSize>>>(addSizesTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  int32_t totalCount =
      *reinterpret_cast<int32_t*>(addSizesParams.get() + sizeof(Tensor));

  // --- Verify addSizes: inclusive prefix sum of counts ---
  auto prefixCpu = counts.cpu();
  auto* prefixData = prefixCpu.data_ptr<int32_t>();

  int32_t runningSum = 0;
  for (int b = 0; b < kNumBlocks; ++b) {
    runningSum += expectedCounts[b];
    EXPECT_EQ(prefixData[b], runningSum)
        << "addSizes: prefix sum mismatch at block " << b;
  }
  EXPECT_EQ(totalCount, runningSum) << "addSizes: total mismatch";

  // --- Kernel 3: maskedSelectFinal ---
  constexpr auto kFinalOutputOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  auto finalParams =
      allocateManagedArray<char>(kFinalOutputOffset + sizeof(Tensor));
  fillTensorParam(input, reinterpret_cast<Tensor*>(finalParams.get()));
  fillTensorParam(
      mask, reinterpret_cast<Tensor*>(finalParams.get() + sizeof(Tensor)));
  fillTensorParam(
      counts,
      reinterpret_cast<Tensor*>(finalParams.get() + 2 * sizeof(Tensor)));
  *reinterpret_cast<int32_t*>(finalParams.get() + 3 * sizeof(Tensor)) =
      totalCount;
  fillTensorParam(
      output,
      reinterpret_cast<Tensor*>(finalParams.get() + kFinalOutputOffset));

  auto finalBlockInfo = allocateManagedArray<BlockInfo>(kGridBlocks);
  for (int i = 0; i < kGridBlocks; ++i) {
    memset(&finalBlockInfo[i], 0, sizeof(BlockInfo));
    finalBlockInfo[i].blockInOp = i;
    finalBlockInfo[i].numBlocksInOp = kGridBlocks;
    finalBlockInfo[i].params = finalParams.get();
  }

  auto finalBlockInfoDev = allocateDeviceMemory<BlockInfo>(kGridBlocks);
  CUDA_CHECK_FATAL(cudaMemcpy(
      finalBlockInfoDev.get(),
      finalBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice));

  TorchWaveParams finalTwParams;
  memset(&finalTwParams, 0, sizeof(finalTwParams));
  finalTwParams.info = finalBlockInfoDev.get();

  maskedSelectFinalKernel<<<kGridBlocks, kBlockSize>>>(finalTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // --- Verify final output ---
  auto inputCpu = input.cpu();
  auto outputCpu = output.cpu();

  auto* inputData = inputCpu.data_ptr<int32_t>();
  auto* outputData = outputCpu.data_ptr<int32_t>();

  std::vector<int32_t> expected;
  for (int i = 0; i < kSize; ++i) {
    if (maskData[i]) {
      expected.push_back(inputData[i]);
    }
  }

  EXPECT_EQ(totalCount, static_cast<int32_t>(expected.size()));
  for (int i = 0;
       i < std::min(totalCount, static_cast<int32_t>(expected.size()));
       ++i) {
    EXPECT_EQ(outputData[i], expected[i]) << "Mismatch at index " << i;
  }
}

// --- Cumsum kernels ---

__global__ void cumsumKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ int32_t counter;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  cumsum<kBlockSize, int32_t, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);
}

__global__ void cumsumHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  cumsum_head<kBlockSize, int32_t, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void cumsumFinalKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  cumsum_final<kBlockSize, int32_t, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void sumHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ Int32X32 temp;

  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  tw_sum_head<kBlockSize, int64_t, int64_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

// An empty input still has to leave a per-block partial behind. The host
// reserves one slot per block whatever the input length (numBlocksShape in
// Builtins.cpp) and the final stage reduces over all of them, so a slot the
// head skips feeds whatever the buffer already held into the result. The
// partials buffer starts poisoned here because a fresh, zeroed allocation
// gives the right answer either way -- which is exactly what hides this in a
// whole-graph test, where it only shows up on a second execution.
TEST_F(KernelTest, reduceHeadEmptyInput) {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
  auto input = at::empty({0}, options);
  auto partials = at::full({1}, 44, options);

  auto params = allocateManagedArray<Tensor>(2);
  fillTensorParam(input, &params[0]);
  fillTensorParam(partials, &params[1]);

  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  auto& bi = twParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = params.get();

  sumHeadKernel<<<1, kBlockSize>>>(twParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  EXPECT_EQ(partials.cpu().data_ptr<int64_t>()[0], 0)
      << "empty input: the head left its partial slot at its old value";
}

TEST_F(KernelTest, cumsumSingleBlock) {
  constexpr int32_t kSize = 200;

  auto input = at::randint(
      0, 10, {kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto output =
      at::zeros({kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  auto params = allocateManagedArray<Tensor>(2);
  fillTensorParam(input, &params[0]);
  fillTensorParam(output, &params[1]);

  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  auto& bi = twParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = params.get();

  cumsumKernel<<<1, kBlockSize>>>(twParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  auto inputCpu = input.cpu();
  auto outputCpu = output.cpu();
  auto* inData = inputCpu.data_ptr<int32_t>();
  auto* outData = outputCpu.data_ptr<int32_t>();

  int32_t sum = 0;
  for (int i = 0; i < kSize; ++i) {
    sum += inData[i];
    EXPECT_EQ(outData[i], sum) << "Inclusive mismatch at " << i;
  }
}

TEST_F(KernelTest, cumsumThreeStage) {
  constexpr int32_t kSize = 100000;
  constexpr int32_t kNumBlocks = (kSize + kBlockSize - 1) / kBlockSize;
  constexpr int32_t kGridBlocks = (kNumBlocks + 1) / 2;

  auto input = at::randint(
      0, 10, {kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto counts = at::zeros(
      {kNumBlocks}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto output =
      at::zeros({kSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  auto inputCpu = input.cpu();
  auto* inData = inputCpu.data_ptr<int32_t>();

  // --- Stage 1: cumsum_head ---
  auto headParams = allocateManagedArray<Tensor>(2);
  fillTensorParam(input, &headParams[0]);
  fillTensorParam(counts, &headParams[1]);

  auto headBlockInfo = allocateManagedArray<BlockInfo>(kGridBlocks);
  for (int i = 0; i < kGridBlocks; ++i) {
    memset(&headBlockInfo[i], 0, sizeof(BlockInfo));
    headBlockInfo[i].blockInOp = i;
    headBlockInfo[i].numBlocksInOp = kGridBlocks;
    headBlockInfo[i].params = headParams.get();
  }

  auto headBlockInfoDev = allocateDeviceMemory<BlockInfo>(kGridBlocks);
  CUDA_CHECK_FATAL(cudaMemcpy(
      headBlockInfoDev.get(),
      headBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice));

  TorchWaveParams headTwParams;
  memset(&headTwParams, 0, sizeof(headTwParams));
  headTwParams.info = headBlockInfoDev.get();

  cumsumHeadKernel<<<kGridBlocks, kBlockSize>>>(headTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // Verify per-block sums.
  auto countsCpu = counts.cpu();
  auto* countData = countsCpu.data_ptr<int32_t>();
  std::vector<int32_t> expectedBlockSums(kNumBlocks);
  for (int b = 0; b < kNumBlocks; ++b) {
    int32_t s = 0;
    for (int i = b * kBlockSize; i < std::min((b + 1) * kBlockSize, kSize);
         ++i) {
      s += inData[i];
    }
    expectedBlockSums[b] = s;
    EXPECT_EQ(countData[b], s) << "Head: block " << b << " sum mismatch";
  }

  // --- Stage 2: add_sizes ---
  auto addSizesParams =
      allocateManagedArray<char>(sizeof(Tensor) + sizeof(int32_t));
  fillTensorParam(counts, reinterpret_cast<Tensor*>(addSizesParams.get()));
  *reinterpret_cast<int32_t*>(addSizesParams.get() + sizeof(Tensor)) = 0;

  TorchWaveParams addSizesTwParams;
  memset(&addSizesTwParams, 0, sizeof(addSizesTwParams));
  auto& addBi = addSizesTwParams.inlineInfo[0];
  addBi.numBlocksInOp = 1;
  addBi.params = addSizesParams.get();

  addSizesKernel<<<1, kBlockSize>>>(addSizesTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // --- Stage 3: cumsum_final ---
  auto finalParams = allocateManagedArray<Tensor>(3);
  fillTensorParam(input, &finalParams[0]);
  fillTensorParam(counts, &finalParams[1]);
  fillTensorParam(output, &finalParams[2]);

  auto finalBlockInfo = allocateManagedArray<BlockInfo>(kGridBlocks);
  for (int i = 0; i < kGridBlocks; ++i) {
    memset(&finalBlockInfo[i], 0, sizeof(BlockInfo));
    finalBlockInfo[i].blockInOp = i;
    finalBlockInfo[i].numBlocksInOp = kGridBlocks;
    finalBlockInfo[i].params = finalParams.get();
  }

  auto finalBlockInfoDev = allocateDeviceMemory<BlockInfo>(kGridBlocks);
  CUDA_CHECK_FATAL(cudaMemcpy(
      finalBlockInfoDev.get(),
      finalBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice));

  TorchWaveParams finalTwParams;
  memset(&finalTwParams, 0, sizeof(finalTwParams));
  finalTwParams.info = finalBlockInfoDev.get();

  cumsumFinalKernel<<<kGridBlocks, kBlockSize>>>(finalTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  auto outputCpu = output.cpu();
  auto* outData = outputCpu.data_ptr<int32_t>();
  int32_t sum = 0;
  for (int i = 0; i < kSize; ++i) {
    sum += inData[i];
    EXPECT_EQ(outData[i], sum) << "3-stage mismatch at " << i;
  }
}

// --- Isin kernels ---

__global__ void isinHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  if (threadIdx.x == 0) {
    blockInfo =
        params.info ? params.info[blockIdx.x] : params.inlineInfo[blockIdx.x];
  }
  __syncthreads();

  tw_isin_head<int64_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      blockInfo);
}

__global__ void isinFinalKernel(
    const int64_t* elements,
    const Tensor* hashTable,
    bool* output,
    uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __isin_final<int64_t>(
        idx, size, elements[idx], const_cast<Tensor*>(hashTable), false);
  }
}

TEST_F(KernelTest, isin) {
  // Build a set of values: odd numbers 1..999 plus 0.
  std::vector<int64_t> setVals;
  setVals.push_back(0);
  for (int64_t i = 1; i < 1000; i += 2) {
    setVals.push_back(i);
  }
  auto setTensor =
      at::from_blob(
          setVals.data(), {static_cast<int64_t>(setVals.size())}, at::kLong)
          .clone()
          .cuda();

  // Hash table size: next power of 2 >= 2*n, plus 1 for the kEmpty flag.
  int64_t n = setVals.size();
  int64_t tableSize = 1;
  while (tableSize < n * 2) {
    tableSize *= 2;
  }
  tableSize += 1;
  auto hashTable = at::zeros(
      {tableSize}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  // Run isin_head.
  auto headParams = allocateManagedArray<Tensor>(2);
  fillTensorParam(setTensor, &headParams[0]);
  fillTensorParam(hashTable, &headParams[1]);

  TorchWaveParams headTwParams;
  memset(&headTwParams, 0, sizeof(headTwParams));
  auto& bi = headTwParams.inlineInfo[0];
  bi.numBlocksInOp = 1;
  bi.params = headParams.get();

  isinHeadKernel<<<1, kBlockSize>>>(headTwParams);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // Elements to query: 0..999.
  constexpr int32_t kQuerySize = 1000;
  auto elements = at::arange(
      0, kQuerySize, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  auto output = at::zeros(
      {kQuerySize}, at::TensorOptions().dtype(at::kBool).device(at::kCUDA));

  // We need a managed Tensor param for the hash table so the kernel can
  // read dims/storage.
  auto finalTableParam = allocateManagedArray<Tensor>(1);
  fillTensorParam(hashTable, &finalTableParam[0]);

  constexpr int32_t kFinalGrid = (kQuerySize + kBlockSize - 1) / kBlockSize;
  isinFinalKernel<<<kFinalGrid, kBlockSize>>>(
      elements.data_ptr<int64_t>(),
      finalTableParam.get(),
      output.data_ptr<bool>(),
      kQuerySize);
  CUDA_CHECK_FATAL(cudaGetLastError());
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());

  // Verify.
  auto outputCpu = output.cpu();
  auto* outData = outputCpu.data_ptr<bool>();

  std::unordered_set<int64_t> refSet(setVals.begin(), setVals.end());
  for (int i = 0; i < kQuerySize; ++i) {
    bool expected = refSet.count(i) > 0;
    EXPECT_EQ(outData[i], expected) << "Mismatch at element " << i;
  }
}

// ---------------------------------------------------------------------------
// Benchmark of the prefix sum and stream compaction variants.
//
// Measures every form of the two ops against each other and against cub, on an
// inclusive prefix sum over int64 data and a stream compaction of the same data
// behind a one byte mask. The setup follows the Wave block scan benchmark
// (D114313208) -- same sizes, same selectivity, same per-variant check against
// a CPU reference before timing -- but drives the torchwave kernels through
// their real signatures, so a row measures the op as the executor launches it:
// tensor parameters read out of a BlockInfo, a grid-strided partition of the
// input, and opBarrier for the cross-block ordering.
//
// The variants differ in how the cross-block dependency is resolved:
//   single   one block walks the whole input, carrying the running sum
//   3-kernel per-tile totals, a scan of the totals, then the pass proper
//   cg       the same three stages in one launch, separated by opBarriers
//   1-pass   decoupled look-back: the input is read once
//   1-pass-s look-back plus shared memory staging of the compacted output
//   device   cub::DeviceScan / cub::DeviceSelect as a reference point
// ---------------------------------------------------------------------------

using BenchType = int64_t;

// Assembles the parameter block a torchwave device function reads through
// BlockInfo::params: the op's Tensor and scalar arguments at fixed offsets,
// followed by the int32 counters opBarrier spins on. The block lives in managed
// memory so the host can read back an output length the kernel set on device.
class ParamBlock {
 public:
  int32_t addTensor(const at::Tensor& tensor) {
    const auto offset = reserve(sizeof(Tensor), alignof(Tensor));
    fillTensorParam(tensor, reinterpret_cast<Tensor*>(&bytes_[offset]));
    return offset;
  }

  template <typename T>
  int32_t addScalar(T value) {
    const auto offset = reserve(sizeof(T), alignof(T));
    *reinterpret_cast<T*>(&bytes_[offset]) = value;
    return offset;
  }

  int32_t addBarrier() {
    const auto offset = addScalar<int32_t>(0);
    barriers_.push_back(offset);
    return offset;
  }

  // Copies the host image to managed memory. Must be called after the last add.
  void upload() {
    params_ = allocateManagedArray<char>(bytes_.size());
    memcpy(params_.get(), bytes_.data(), bytes_.size());
  }

  char* params() const {
    return params_.get();
  }

  const Tensor& tensorAt(int32_t offset) const {
    return *reinterpret_cast<const Tensor*>(params_.get() + offset);
  }

  template <typename T>
  T scalarAt(int32_t offset) const {
    return *reinterpret_cast<const T*>(params_.get() + offset);
  }

  // Clears the barrier counters. opBarrier never resets them, so a relaunch of
  // the same block would pass its barriers immediately.
  void resetBarriers() const {
    for (auto offset : barriers_) {
      CUDA_CHECK_FATAL(
          cudaMemsetAsync(params_.get() + offset, 0, sizeof(int32_t)));
    }
  }

 private:
  int32_t reserve(size_t size, size_t alignment) {
    const auto offset = roundUp(bytes_.size(), alignment);
    bytes_.resize(offset + size, 0);
    return static_cast<int32_t>(offset);
  }

  std::vector<char> bytes_;
  std::vector<int32_t> barriers_;
  CudaPtr<char[]> params_;
};

// gridBlocks BlockInfos, all pointing at 'params', numbered within the op.
CudaPtr<BlockInfo[]> makeBlockInfo(
    const ParamBlock& params,
    int32_t gridBlocks) {
  auto info = allocateManagedArray<BlockInfo>(gridBlocks);
  for (int32_t i = 0; i < gridBlocks; ++i) {
    memset(&info[i], 0, sizeof(BlockInfo));
    info[i].blockInOp = i;
    info[i].numBlocksInOp = gridBlocks;
    info[i].params = params.params();
  }
  return info;
}

TorchWaveParams twParams(const CudaPtr<BlockInfo[]>& info) {
  TorchWaveParams params;
  memset(&params, 0, sizeof(params));
  params.info = info.get();
  return params;
}

// Runs 'body' once as a warmup and then FLAGS_scan_repeats times, returning the
// shortest elapsed time in milliseconds.
template <typename Body>
float timeBest(Body body) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK_FATAL(cudaEventCreate(&start));
  CUDA_CHECK_FATAL(cudaEventCreate(&stop));
  body();
  CUDA_CHECK_FATAL(cudaDeviceSynchronize());
  float best = std::numeric_limits<float>::max();
  for (int32_t i = 0; i < FLAGS_scan_repeats; ++i) {
    CUDA_CHECK_FATAL(cudaEventRecord(start));
    body();
    CUDA_CHECK_FATAL(cudaEventRecord(stop));
    CUDA_CHECK_FATAL(cudaEventSynchronize(stop));
    float milliseconds;
    CUDA_CHECK_FATAL(cudaEventElapsedTime(&milliseconds, start, stop));
    best = std::min(best, milliseconds);
  }
  CUDA_CHECK_FATAL(cudaEventDestroy(start));
  CUDA_CHECK_FATAL(cudaEventDestroy(stop));
  return best;
}

int32_t numMultiProcessors() {
  int32_t device;
  CUDA_CHECK_FATAL(cudaGetDevice(&device));
  int32_t value;
  CUDA_CHECK_FATAL(
      cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
  return value;
}

struct KernelResources {
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
};

template <typename Kernel>
KernelResources kernelResources(Kernel kernel, int32_t dynamicShared = 0) {
  KernelResources resources;
  CUDA_CHECK_FATAL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resources.blocksPerSm, kernel, kBlockSize, dynamicShared));
  cudaFuncAttributes attributes;
  CUDA_CHECK_FATAL(cudaFuncGetAttributes(&attributes, kernel));
  resources.numRegisters = attributes.numRegs;
  resources.sharedBytes =
      static_cast<int32_t>(attributes.sharedSizeBytes) + dynamicShared;
  return resources;
}

struct Measurement {
  std::string variant;
  int32_t blocksPerSm{0};
  int32_t numRegisters{0};
  int32_t sharedBytes{0};
  int32_t numBlocks{0};
  // Per stage; a single-launch variant leaves the last two at zero.
  float firstMs{0};
  float secondMs{0};
  float thirdMs{0};
  bool correct{true};

  float totalMs() const {
    return firstMs + secondMs + thirdMs;
  }
};

std::string optional(int32_t value) {
  return value == 0 ? std::string("-") : std::to_string(value);
}

std::string optional(float milliseconds) {
  return milliseconds == 0 ? std::string("-")
                           : fmt::format("{:.3f}", milliseconds);
}

// 'movedBytes' is what a single pass over the data must read and write, so the
// rate says how close a variant gets to just moving the data once.
void printMeasurements(
    const std::string& title,
    int64_t movedBytes,
    std::vector<Measurement> measurements) {
  // Fastest first. The order the variants were run in carries no information.
  std::sort(
      measurements.begin(),
      measurements.end(),
      [](const Measurement& left, const Measurement& right) {
        return left.totalMs() < right.totalMs();
      });
  std::cout << "\n" << title << "\n";
  std::cout << fmt::format(
      "{:>11} {:>7} {:>7} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>6}\n",
      "variant",
      "blocks",
      "per SM",
      "regs",
      "smem",
      "st1 ms",
      "st2 ms",
      "st3 ms",
      "total ms",
      "GB/s",
      "check");
  for (const auto& measurement : measurements) {
    std::cout << fmt::format(
        "{:>11} {:>7} {:>7} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9.3f} {:>9.1f} "
        "{:>6}\n",
        measurement.variant,
        optional(measurement.numBlocks),
        optional(measurement.blocksPerSm),
        optional(measurement.numRegisters),
        optional(measurement.sharedBytes),
        optional(measurement.firstMs),
        optional(measurement.secondMs),
        optional(measurement.thirdMs),
        measurement.totalMs(),
        static_cast<double>(movedBytes) / (measurement.totalMs() * 1e-3) / 1e9,
        measurement.correct ? "ok" : "FAIL");
  }
}

// --- Benchmark kernels. Each wraps one Scan.cuh entry point in the prologue
// the generated kernels use: thread 0 loads the block's BlockInfo, everyone
// syncs, then the device function reads its arguments out of it. ---

__global__ void benchCumsumKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ BenchType counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  cumsum<kBlockSize, BenchType, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);
}

__global__ void benchCumsumHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  cumsum_head<kBlockSize, BenchType, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void benchAddSizesKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ BenchType counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  add_sizes<kBlockSize, BenchType>(
      param<Tensor>(blockInfo, 0),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);
}

__global__ void benchCumsumFinalKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  cumsum_final<kBlockSize, BenchType, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void benchCumsumCgKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ BenchType counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  constexpr int32_t kBar0 = 3 * sizeof(Tensor);
  cumsum_cg<kBlockSize, BenchType, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      kBar0,
      kBar0 + static_cast<int32_t>(sizeof(int32_t)),
      blockInfo);
}

template <int32_t kItemsPerThread>
__global__ void benchCumsum1passKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t numTiles;
  __shared__ BenchType runTotal;
  __shared__ BenchType tileExclusive;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  cumsum_1pass<kBlockSize, BenchType, BenchType, 0, kItemsPerThread>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      numTiles,
      runTotal,
      tileExclusive,
      3 * static_cast<int32_t>(sizeof(Tensor)),
      3 * static_cast<int32_t>(sizeof(Tensor)) +
          static_cast<int32_t>(sizeof(int32_t)),
      blockInfo);
}

__global__ void benchMaskedSelectKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    size = numEl(*param<Tensor>(blockInfo, 0));
  }
  __syncthreads();
  const uint32_t rounded = roundUpPwr2(size, (uint32_t)kBlockSize);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    masked_select<kBlockSize, BenchType>(
        param<Tensor>(blockInfo, 0),
        param<Tensor>(blockInfo, sizeof(Tensor)),
        param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
        (void*)temp,
        counter,
        idx,
        size,
        blockInfo);
  }
}

__global__ void benchMaskedSelectHeadKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  masked_select_head<kBlockSize, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void benchAddSizesInt32Kernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ uint32_t counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  add_sizes<kBlockSize>(
      param<Tensor>(blockInfo, 0),
      param<int32_t>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);
}

__global__ void benchMaskedSelectFinalKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  constexpr int32_t kOutputOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  masked_select_final<kBlockSize, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      param<int32_t>(blockInfo, 3 * sizeof(Tensor)),
      param<Tensor>(blockInfo, kOutputOffset),
      (void*)temp,
      size,
      rounded,
      blockInfo);
}

__global__ void benchMaskedSelectCgKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ uint32_t counter;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  constexpr int32_t kBar0 = 4 * sizeof(Tensor);
  constexpr int32_t kBarStride = sizeof(int32_t);
  masked_select_cg<kBlockSize, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      param<Tensor>(blockInfo, 3 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      kBar0,
      kBar0 + kBarStride,
      kBar0 + 2 * kBarStride,
      blockInfo);
}

__global__ void benchMaskedSelect1passKernel(TorchWaveParams params) {
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t numTiles;
  __shared__ int32_t tileTotal;
  __shared__ int32_t tileExclusive;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  masked_select_1pass_lane<kBlockSize, BenchType>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      param<Tensor>(blockInfo, 3 * sizeof(Tensor)),
      (void*)temp,
      size,
      numTiles,
      tileTotal,
      tileExclusive,
      4 * static_cast<int32_t>(sizeof(Tensor)),
      blockInfo);
}

template <int32_t kItemsPerThread>
__global__ void benchMaskedSelect1passSharedKernel(TorchWaveParams params) {
  extern __shared__ __align__(16) char staging[];
  __shared__ BlockInfo blockInfo;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t numTiles;
  __shared__ int32_t tileTotal;
  __shared__ int32_t tileExclusive;
  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();
  masked_select_1pass_shared<kBlockSize, BenchType, kItemsPerThread>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      param<Tensor>(blockInfo, 3 * sizeof(Tensor)),
      reinterpret_cast<BenchType*>(staging),
      (void*)temp,
      size,
      numTiles,
      tileTotal,
      tileExclusive,
      4 * static_cast<int32_t>(sizeof(Tensor)),
      blockInfo);
}

// --- Host side ---

// Blocks to launch for a variant: as many as stay resident, capped by the
// tiles there are to process. The look-back and opBarrier variants both need
// every block resident, so this is a correctness requirement for them, not
// only a performance one.
int32_t residentBlocks(const KernelResources& resources, int64_t numTiles) {
  return static_cast<int32_t>(std::min<int64_t>(
      static_cast<int64_t>(resources.blocksPerSm) * numMultiProcessors(),
      std::max<int64_t>(numTiles, 1)));
}

int64_t numTilesFor(int64_t numElements, int32_t tileSize) {
  return (numElements + tileSize - 1) / tileSize;
}

std::vector<BenchType> inclusivePrefixSum(const std::vector<BenchType>& input) {
  std::vector<BenchType> expected(input.size());
  BenchType running = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    running += input[i];
    expected[i] = running;
  }
  return expected;
}

bool matches(const at::Tensor& actual, const std::vector<BenchType>& expected) {
  auto host = actual.cpu();
  const auto* data = host.data_ptr<BenchType>();
  return std::equal(expected.begin(), expected.end(), data);
}

// --- Prefix sum ---

Measurement runCumsumSingleBlock(
    const at::Tensor& input,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  ParamBlock params;
  params.addTensor(input);
  params.addTensor(output);
  params.upload();
  auto info = makeBlockInfo(params, 1);
  auto kernelParams = twParams(info);

  Measurement measurement;
  measurement.variant = "single";
  const auto resources = kernelResources(benchCumsumKernel);
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = 1;

  auto pass = [&]() { benchCumsumKernel<<<1, kBlockSize>>>(kernelParams); };
  if (FLAGS_scan_check) {
    output.fill_(-1);
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = matches(output, expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runCumsumThreeKernel(
    const at::Tensor& input,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles = numTilesFor(input.numel(), kBlockSize);
  auto counts = at::zeros(
      {numTiles}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  ParamBlock headParams;
  headParams.addTensor(input);
  headParams.addTensor(counts);
  headParams.upload();

  ParamBlock addParams;
  addParams.addTensor(counts);
  addParams.upload();

  ParamBlock finalParams;
  finalParams.addTensor(input);
  finalParams.addTensor(counts);
  finalParams.addTensor(output);
  finalParams.upload();

  Measurement measurement;
  measurement.variant = "3-kernel";
  const auto resources = kernelResources(benchCumsumFinalKernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto headInfo = makeBlockInfo(headParams, measurement.numBlocks);
  auto addInfo = makeBlockInfo(addParams, 1);
  auto finalInfo = makeBlockInfo(finalParams, measurement.numBlocks);
  auto headKernelParams = twParams(headInfo);
  auto addKernelParams = twParams(addInfo);
  auto finalKernelParams = twParams(finalInfo);

  auto headPass = [&]() {
    benchCumsumHeadKernel<<<measurement.numBlocks, kBlockSize>>>(
        headKernelParams);
  };
  auto addPass = [&]() {
    benchAddSizesKernel<<<1, kBlockSize>>>(addKernelParams);
  };
  auto finalPass = [&]() {
    benchCumsumFinalKernel<<<measurement.numBlocks, kBlockSize>>>(
        finalKernelParams);
  };

  if (FLAGS_scan_check) {
    output.fill_(-1);
    headPass();
    addPass();
    finalPass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = matches(output, expected);
  }
  // add_sizes scans the per-tile totals in place, so a repeat of the head pass
  // must precede every repeat of it; time the three separately in that order.
  measurement.firstMs = timeBest(headPass);
  measurement.secondMs = std::max(
      0.0f,
      timeBest([&]() {
        headPass();
        addPass();
      }) - measurement.firstMs);
  measurement.thirdMs = timeBest(finalPass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runCumsumCg(
    const at::Tensor& input,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles = numTilesFor(input.numel(), kBlockSize);
  auto counts = at::zeros(
      {numTiles}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  ParamBlock params;
  params.addTensor(input);
  params.addTensor(output);
  params.addTensor(counts);
  params.addBarrier();
  params.addBarrier();
  params.upload();

  Measurement measurement;
  measurement.variant = "cg";
  const auto resources = kernelResources(benchCumsumCgKernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto info = makeBlockInfo(params, measurement.numBlocks);
  auto kernelParams = twParams(info);
  auto pass = [&]() {
    params.resetBarriers();
    benchCumsumCgKernel<<<measurement.numBlocks, kBlockSize>>>(kernelParams);
  };

  if (FLAGS_scan_check) {
    output.fill_(-1);
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = matches(output, expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

template <int32_t kItemsPerThread>
Measurement runCumsum1pass(
    const at::Tensor& input,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles =
      numTilesFor(input.numel(), kBlockSize * kItemsPerThread);
  auto state = at::empty(
      {LookbackState<BenchType>::numWords(static_cast<int32_t>(numTiles))},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  ParamBlock params;
  params.addTensor(input);
  params.addTensor(output);
  params.addTensor(state);
  params.addBarrier();
  params.addBarrier();
  params.upload();

  auto* kernel = benchCumsum1passKernel<kItemsPerThread>;
  Measurement measurement;
  measurement.variant = fmt::format("1-pass/{}", kItemsPerThread);
  const auto resources = kernelResources(kernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto info = makeBlockInfo(params, measurement.numBlocks);
  auto kernelParams = twParams(info);
  auto pass = [&]() {
    params.resetBarriers();
    kernel<<<measurement.numBlocks, kBlockSize>>>(kernelParams);
  };

  if (FLAGS_scan_check) {
    output.fill_(-1);
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = matches(output, expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runCumsumDevice(
    const at::Tensor& input,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const auto numElements = static_cast<int32_t>(input.numel());
  const auto* in = input.data_ptr<BenchType>();
  auto* out = output.data_ptr<BenchType>();
  size_t tempBytes = 0;
  CUDA_CHECK_FATAL(
      cub::DeviceScan::InclusiveSum(nullptr, tempBytes, in, out, numElements));
  auto temp = at::empty(
      {static_cast<int64_t>(tempBytes)},
      at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  auto pass = [&]() {
    size_t bytes = tempBytes;
    CUDA_CHECK_FATAL(
        cub::DeviceScan::InclusiveSum(
            temp.data_ptr(), bytes, in, out, numElements));
  };

  Measurement measurement;
  measurement.variant = "device";
  if (FLAGS_scan_check) {
    output.fill_(-1);
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = matches(output, expected);
  }
  measurement.firstMs = timeBest(pass);
  return measurement;
}

void runPrefixSumBenchmark(int64_t numElements) {
  std::vector<BenchType> hostInput(numElements);
  std::mt19937_64 rng(1);
  for (int64_t i = 0; i < numElements; ++i) {
    // Small values keep the int64 sum exact and far from overflow.
    hostInput[i] = static_cast<BenchType>(rng() & 1023);
  }
  const auto expected = inclusivePrefixSum(hostInput);

  auto input =
      at::from_blob(
          hostInput.data(), {numElements}, at::TensorOptions().dtype(at::kLong))
          .to(at::kCUDA);
  auto output = at::empty(
      {numElements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  std::vector<Measurement> measurements;
  if (FLAGS_scan_single_block) {
    measurements.push_back(runCumsumSingleBlock(input, output, expected));
  }
  measurements.push_back(runCumsumThreeKernel(input, output, expected));
  measurements.push_back(runCumsumCg(input, output, expected));
  measurements.push_back(runCumsum1pass<1>(input, output, expected));
  measurements.push_back(runCumsum1pass<2>(input, output, expected));
  measurements.push_back(runCumsum1pass<4>(input, output, expected));
  measurements.push_back(runCumsum1pass<8>(input, output, expected));
  measurements.push_back(runCumsum1pass<16>(input, output, expected));
  measurements.push_back(runCumsumDevice(input, output, expected));

  const int64_t movedBytes = 2 * numElements * sizeof(BenchType);
  printMeasurements(
      fmt::format(
          "Inclusive prefix sum of {} int64 ({:.1f} MB). Rates are over the "
          "{:.1f} MB a single pass reads plus writes.",
          numElements,
          numElements * sizeof(BenchType) / static_cast<double>(1 << 20),
          movedBytes / static_cast<double>(1 << 20)),
      movedBytes,
      std::move(measurements));
}

// --- Stream compaction ---

// Output length a masked select variant set on device, read back from the
// output tensor descriptor in the managed parameter block.
int32_t selectedCount(const ParamBlock& params, int32_t outputOffset) {
  return params.tensorAt(outputOffset).dims[0];
}

bool selectMatches(
    const at::Tensor& output,
    int32_t count,
    const std::vector<BenchType>& expected) {
  if (count != static_cast<int32_t>(expected.size())) {
    return false;
  }
  auto host = output.slice(0, 0, count).cpu();
  const auto* data = host.data_ptr<BenchType>();
  return std::equal(expected.begin(), expected.end(), data);
}

Measurement runSelectSingleBlock(
    const at::Tensor& values,
    const at::Tensor& mask,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  ParamBlock params;
  params.addTensor(values);
  params.addTensor(mask);
  const auto outputOffset = params.addTensor(output);
  params.upload();
  auto info = makeBlockInfo(params, 1);
  auto kernelParams = twParams(info);

  Measurement measurement;
  measurement.variant = "single";
  const auto resources = kernelResources(benchMaskedSelectKernel);
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = 1;

  auto pass = [&]() {
    benchMaskedSelectKernel<<<1, kBlockSize>>>(kernelParams);
  };
  if (FLAGS_scan_check) {
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct =
        selectMatches(output, selectedCount(params, outputOffset), expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runSelectThreeKernel(
    const at::Tensor& values,
    const at::Tensor& mask,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles = numTilesFor(values.numel(), kBlockSize);
  auto counts = at::zeros(
      {numTiles}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  ParamBlock headParams;
  headParams.addTensor(values);
  headParams.addTensor(mask);
  headParams.addTensor(counts);
  headParams.upload();

  ParamBlock addParams;
  addParams.addTensor(counts);
  const auto totalOffset = addParams.addScalar<int32_t>(0);
  addParams.upload();

  ParamBlock finalParams;
  finalParams.addTensor(values);
  finalParams.addTensor(mask);
  finalParams.addTensor(counts);
  finalParams.addScalar<int32_t>(0);
  const auto outputOffset = finalParams.addTensor(output);
  finalParams.upload();
  // benchMaskedSelectFinalKernel hardcodes the same offset; the two layouts
  // must agree or the kernel reads the output descriptor from padding.
  TORCH_CHECK(
      outputOffset ==
      static_cast<int32_t>(
          roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor))));

  Measurement measurement;
  measurement.variant = "3-kernel";
  const auto resources = kernelResources(benchMaskedSelectFinalKernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto headInfo = makeBlockInfo(headParams, measurement.numBlocks);
  auto addInfo = makeBlockInfo(addParams, 1);
  auto finalInfo = makeBlockInfo(finalParams, measurement.numBlocks);
  auto headKernelParams = twParams(headInfo);
  auto addKernelParams = twParams(addInfo);
  auto finalKernelParams = twParams(finalInfo);

  auto headPass = [&]() {
    benchMaskedSelectHeadKernel<<<measurement.numBlocks, kBlockSize>>>(
        headKernelParams);
  };
  auto addPass = [&]() {
    benchAddSizesInt32Kernel<<<1, kBlockSize>>>(addKernelParams);
  };
  auto finalPass = [&]() {
    benchMaskedSelectFinalKernel<<<measurement.numBlocks, kBlockSize>>>(
        finalKernelParams);
  };

  if (FLAGS_scan_check) {
    headPass();
    addPass();
    finalPass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct =
        addParams.scalarAt<int32_t>(totalOffset) ==
            static_cast<int32_t>(expected.size()) &&
        selectMatches(
            output, selectedCount(finalParams, outputOffset), expected);
  }
  // add_sizes scans the per-tile counts in place, so a repeat of it alone would
  // scan an already scanned array; time it as the difference against the head.
  measurement.firstMs = timeBest(headPass);
  measurement.secondMs = std::max(
      0.0f,
      timeBest([&]() {
        headPass();
        addPass();
      }) - measurement.firstMs);
  measurement.thirdMs = timeBest(finalPass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runSelectCg(
    const at::Tensor& values,
    const at::Tensor& mask,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles = numTilesFor(values.numel(), kBlockSize);
  auto counts = at::zeros(
      {numTiles}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  ParamBlock params;
  params.addTensor(values);
  params.addTensor(mask);
  const auto outputOffset = params.addTensor(output);
  params.addTensor(counts);
  params.addBarrier();
  params.addBarrier();
  params.addBarrier();
  params.upload();

  Measurement measurement;
  measurement.variant = "cg";
  const auto resources = kernelResources(benchMaskedSelectCgKernel);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto info = makeBlockInfo(params, measurement.numBlocks);
  auto kernelParams = twParams(info);
  auto pass = [&]() {
    params.resetBarriers();
    benchMaskedSelectCgKernel<<<measurement.numBlocks, kBlockSize>>>(
        kernelParams);
  };

  if (FLAGS_scan_check) {
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct =
        selectMatches(output, selectedCount(params, outputOffset), expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

template <typename Kernel>
Measurement runSelect1pass(
    const char* variant,
    Kernel kernel,
    int32_t itemsPerThread,
    int32_t dynamicShared,
    const at::Tensor& values,
    const at::Tensor& mask,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const int64_t numTiles =
      numTilesFor(values.numel(), kBlockSize * itemsPerThread);
  auto state = at::empty(
      {LookbackState<int32_t>::numWords(static_cast<int32_t>(numTiles))},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

  ParamBlock params;
  params.addTensor(values);
  params.addTensor(mask);
  const auto outputOffset = params.addTensor(output);
  params.addTensor(state);
  params.addBarrier();
  params.upload();

  Measurement measurement;
  measurement.variant = variant;
  const auto resources = kernelResources(kernel, dynamicShared);
  measurement.blocksPerSm = resources.blocksPerSm;
  measurement.numRegisters = resources.numRegisters;
  measurement.sharedBytes = resources.sharedBytes;
  measurement.numBlocks = residentBlocks(resources, numTiles);

  auto info = makeBlockInfo(params, measurement.numBlocks);
  auto kernelParams = twParams(info);
  auto pass = [&]() {
    params.resetBarriers();
    kernel<<<measurement.numBlocks, kBlockSize, dynamicShared>>>(kernelParams);
  };

  if (FLAGS_scan_check) {
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct =
        selectMatches(output, selectedCount(params, outputOffset), expected);
  }
  measurement.firstMs = timeBest(pass);
  CUDA_CHECK_FATAL(cudaGetLastError());
  return measurement;
}

Measurement runSelectDevice(
    const at::Tensor& values,
    const at::Tensor& mask,
    const at::Tensor& output,
    const std::vector<BenchType>& expected) {
  const auto numElements = static_cast<int32_t>(values.numel());
  const auto* in = values.data_ptr<BenchType>();
  const auto* flags = mask.data_ptr<bool>();
  auto* out = output.data_ptr<BenchType>();
  auto numSelected =
      at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
  auto* count = numSelected.data_ptr<int32_t>();
  size_t tempBytes = 0;
  CUDA_CHECK_FATAL(
      cub::DeviceSelect::Flagged(
          nullptr, tempBytes, in, flags, out, count, numElements));
  auto temp = at::empty(
      {static_cast<int64_t>(tempBytes)},
      at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  auto pass = [&]() {
    size_t bytes = tempBytes;
    CUDA_CHECK_FATAL(
        cub::DeviceSelect::Flagged(
            temp.data_ptr(), bytes, in, flags, out, count, numElements));
  };

  Measurement measurement;
  measurement.variant = "device";
  if (FLAGS_scan_check) {
    pass();
    CUDA_CHECK_FATAL(cudaDeviceSynchronize());
    measurement.correct = selectMatches(
        output, numSelected.cpu().data_ptr<int32_t>()[0], expected);
  }
  measurement.firstMs = timeBest(pass);
  return measurement;
}

void runCompactionBenchmark(int64_t numElements) {
  std::vector<BenchType> hostValues(numElements);
  std::vector<uint8_t> hostFlags(numElements);
  std::mt19937_64 rng(2);
  const auto threshold = static_cast<uint64_t>(
      FLAGS_scan_selectivity *
      static_cast<double>(std::numeric_limits<uint64_t>::max()));
  for (int64_t i = 0; i < numElements; ++i) {
    hostValues[i] = static_cast<BenchType>(i);
    hostFlags[i] = rng() < threshold;
  }
  std::vector<BenchType> expected;
  expected.reserve(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    if (hostFlags[i]) {
      expected.push_back(hostValues[i]);
    }
  }

  auto values = at::from_blob(
                    hostValues.data(),
                    {numElements},
                    at::TensorOptions().dtype(at::kLong))
                    .to(at::kCUDA);
  auto mask =
      at::from_blob(
          hostFlags.data(), {numElements}, at::TensorOptions().dtype(at::kBool))
          .to(at::kCUDA);
  auto output = at::empty(
      {numElements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  std::vector<Measurement> measurements;
  if (FLAGS_scan_single_block) {
    measurements.push_back(
        runSelectSingleBlock(values, mask, output, expected));
  }
  measurements.push_back(runSelectThreeKernel(values, mask, output, expected));
  measurements.push_back(runSelectCg(values, mask, output, expected));
  measurements.push_back(runSelect1pass(
      "1-pass",
      benchMaskedSelect1passKernel,
      1,
      0,
      values,
      mask,
      output,
      expected));
  const auto shared = [&](auto items) {
    constexpr int32_t kItems = decltype(items)::value;
    measurements.push_back(runSelect1pass(
        fmt::format("1-pass-s/{}", kItems).c_str(),
        benchMaskedSelect1passSharedKernel<kItems>,
        kItems,
        kBlockSize * kItems * static_cast<int32_t>(sizeof(BenchType)),
        values,
        mask,
        output,
        expected));
  };
  shared(std::integral_constant<int32_t, 2>{});
  shared(std::integral_constant<int32_t, 4>{});
  shared(std::integral_constant<int32_t, 8>{});
  shared(std::integral_constant<int32_t, 16>{});
  measurements.push_back(runSelectDevice(values, mask, output, expected));

  // The mask is read once, the values once, and the selected values written.
  const int64_t movedBytes = numElements +
      numElements * static_cast<int64_t>(sizeof(BenchType)) +
      static_cast<int64_t>(expected.size() * sizeof(BenchType));
  printMeasurements(
      fmt::format(
          "Stream compaction of {} int64 ({:.1f} MB) behind a one byte mask, "
          "{} selected ({:.1f}%). Rates are over the {:.1f} MB read plus "
          "written.",
          numElements,
          numElements * sizeof(BenchType) / static_cast<double>(1 << 20),
          expected.size(),
          100.0 * static_cast<double>(expected.size()) /
              static_cast<double>(numElements),
          movedBytes / static_cast<double>(1 << 20)),
      movedBytes,
      std::move(measurements));
}

// DISABLED_ so the test framework does not collect this case at all. It is a
// benchmark, so it only ever runs when asked for, and a case that always skips
// reports no signal to CI, which is land-blocking. The flag below still gates
// it, so running the suite with --gtest_also_run_disabled_tests does not start
// a multi-second benchmark by accident.
TEST_F(KernelTest, DISABLED_scanBenchmark) {
  if (!FLAGS_scan_benchmark) {
    GTEST_SKIP() << "Pass --gtest_also_run_disabled_tests --scan_benchmark";
  }
  int32_t device;
  CUDA_CHECK_FATAL(cudaGetDevice(&device));
  cudaDeviceProp properties;
  CUDA_CHECK_FATAL(cudaGetDeviceProperties(&properties, device));
  std::cout << fmt::format(
      "{} sm_{}{} with {} SMs, block size {}\n",
      properties.name,
      properties.major,
      properties.minor,
      properties.multiProcessorCount,
      kBlockSize);

  const int64_t numElements = FLAGS_scan_bytes / sizeof(BenchType);
  ASSERT_GT(numElements, 0) << "--scan_bytes must cover at least one int64";
  runPrefixSumBenchmark(numElements);
  runCompactionBenchmark(numElements);
}

} // namespace
} // namespace torch::wave

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv};
  return RUN_ALL_TESTS();
}
