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
#include "velox/experimental/torchwave/Scan.cuh"

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
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
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
        temp,
        counter,
        idx,
        size,
        blockInfo);
  }
}

class KernelTest : public ::testing::Test {};

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
  cudaDeviceSynchronize();

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
      temp,
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
      temp,
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
      temp,
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
  cudaMemcpy(
      headBlockInfoDev.get(),
      headBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice);

  TorchWaveParams headTwParams;
  memset(&headTwParams, 0, sizeof(headTwParams));
  headTwParams.info = headBlockInfoDev.get();

  maskedSelectHeadKernel<<<kGridBlocks, kBlockSize>>>(headTwParams);
  cudaDeviceSynchronize();

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
  cudaDeviceSynchronize();

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
  cudaMemcpy(
      finalBlockInfoDev.get(),
      finalBlockInfo.get(),
      kGridBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice);

  TorchWaveParams finalTwParams;
  memset(&finalTwParams, 0, sizeof(finalTwParams));
  finalTwParams.info = finalBlockInfoDev.get();

  maskedSelectFinalKernel<<<kGridBlocks, kBlockSize>>>(finalTwParams);
  cudaDeviceSynchronize();

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

} // namespace
} // namespace torch::wave
