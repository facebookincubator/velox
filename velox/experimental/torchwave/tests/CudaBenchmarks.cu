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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "velox/experimental/gpu/Common.h"

namespace torch::wave {
constexpr int32_t kBlockSize = 256;
} // namespace torch::wave

#include "velox/experimental/torchwave/Core.cuh"
#include "velox/experimental/torchwave/Elementwise.cuh"

namespace torch::wave {

__global__ void torchwave0(TorchWaveParams params) {
  ENTRY;
  __shared__ uint32_t size;
  __shared__ uint32_t isFastPath0;
  switch (blockInfo.op) {
    case 3: {
      if (threadIdx.x == 0) {
        isFastPath0 = 0;
        Tensor* temp = param<Tensor>(blockInfo, 0);
        size = numEl(*temp);
        isFastPath0 |= isFastPathTensor(*temp);
        uint32_t size2;
        temp = param<Tensor>(blockInfo, 40);
        size2 = numEl(*temp);
        isFastPath0 |= isFastPathTensor(*temp) << 1;
        if (size2 != size) {
          if (size2 > size) {
            isFastPath0 &= ~((1 << 1) - 1);
            size = size2;
          } else {
            isFastPath0 &= ~(1 << 1);
          }
        }
      }
      __syncthreads();
      int64_t* b0 = storage<int64_t>(param<Tensor>(blockInfo, 0));
      int64_t* b1 = storage<int64_t>(param<Tensor>(blockInfo, 40));
      int64_t* b2 = storage<int64_t>(param<Tensor>(blockInfo, 80));
      int64_t attr120 = *param<int64_t>(blockInfo, 120);
      if (isFastPath0 == 0x3) {
        for (uint32_t idx =
                 blockInfo.blockInOp * blockDim.x + threadIdx.x;
             idx < size;
             idx += blockInfo.numBlocksInOp * blockDim.x) {
          int64_t result0 = __add(b0[idx], b1[idx], attr120);
          b2[idx] = result0;
        }
      } else {
        printf(
            "Unimplemented slow path %d isFastPath0=%u\n",
            __LINE__,
            isFastPath0);
        __trap();
      }
      break;
    }
  }
  LEAVE();
}

} // namespace torch::wave

using namespace torch::wave;
using facebook::velox::gpu::allocateDeviceMemory;
using facebook::velox::gpu::CudaPtr;

template <typename T>
static CudaPtr<T[]> allocateManagedArray(size_t count) {
  T* ptr;
  CUDA_CHECK_FATAL(cudaMallocManaged(&ptr, count * sizeof(T)));
  return CudaPtr<T[]>(ptr);
}

static void fillTensorParam(const at::Tensor& tensor, Tensor* t) {
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
}

struct Trial {
  int32_t dataSize;
  int32_t numBlocks;
  double micros;
};

int main() {
  constexpr int32_t kMaxSize = 1 << 24;
  constexpr int32_t kElementSize = sizeof(int64_t); // 8

  // Base tensors: 2 inputs + 1 output, each 1<<20 int64_t elements on GPU.
  auto b0 = at::randint(
      0,
      100,
      {kMaxSize},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  auto b1 = at::randint(
      0,
      100,
      {kMaxSize},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  auto b2 = at::zeros(
      {kMaxSize}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));

  // Param block: 3 Tensors (offsets 0, 40, 80) + 2 int64_t attrs at 120, 128.
  // sizeof(Tensor) == 40. Total: 3*40 + 2*8 = 136 bytes.
  constexpr int32_t kParamSize = 3 * 40 + 2 * 8;
  auto paramBlock = allocateManagedArray<char>(kParamSize);
  memset(paramBlock.get(), 0, kParamSize);

  // Both attrs set to 1.
  *reinterpret_cast<int64_t*>(paramBlock.get() + 120) = 1;
  *reinterpret_cast<int64_t*>(paramBlock.get() + 128) = 1;

  // Pre-allocate device BlockInfo for max block count.
  int32_t maxBlocks = (kMaxSize + kBlockSize - 1) / kBlockSize;
  auto blockInfoHost = allocateManagedArray<BlockInfo>(maxBlocks);
  auto blockInfoDev = allocateDeviceMemory<BlockInfo>(maxBlocks);

  std::vector<Trial> trials;

  // Warmup: one kernel launch to initialize CUDA runtime.
  {
    auto v0 = b0.slice(0, 0, kBlockSize);
    auto v1 = b1.slice(0, 0, kBlockSize);
    auto v2 = b2.slice(0, 0, kBlockSize);
    fillTensorParam(v0, reinterpret_cast<Tensor*>(paramBlock.get() + 0));
    fillTensorParam(v1, reinterpret_cast<Tensor*>(paramBlock.get() + 40));
    fillTensorParam(v2, reinterpret_cast<Tensor*>(paramBlock.get() + 80));

    memset(&blockInfoHost[0], 0, sizeof(BlockInfo));
    blockInfoHost[0].op = 3;
    blockInfoHost[0].blockInOp = 0;
    blockInfoHost[0].numBlocksInOp = 1;
    blockInfoHost[0].params = paramBlock.get();

    cudaMemcpy(
        blockInfoDev.get(),
        blockInfoHost.get(),
        sizeof(BlockInfo),
        cudaMemcpyHostToDevice);

    TorchWaveParams twParams;
    memset(&twParams, 0, sizeof(twParams));
    twParams.info = blockInfoDev.get();
    torchwave0<<<1, kBlockSize>>>(twParams);
    cudaDeviceSynchronize();
  }

  // Benchmark: data sizes from 256 to 1<<20 in powers of two.
  for (int32_t dataSize = 256; dataSize <= kMaxSize; dataSize *= 2) {
    // Create views over the base tensors.
    auto v0 = b0.slice(0, 0, dataSize);
    auto v1 = b1.slice(0, 0, dataSize);
    auto v2 = b2.slice(0, 0, dataSize);

    // Fill tensor params for this data size.
    fillTensorParam(v0, reinterpret_cast<Tensor*>(paramBlock.get() + 0));
    fillTensorParam(v1, reinterpret_cast<Tensor*>(paramBlock.get() + 40));
    fillTensorParam(v2, reinterpret_cast<Tensor*>(paramBlock.get() + 80));

    int32_t startBlocks = (dataSize + kBlockSize - 1) / kBlockSize;

    for (int32_t numBlocks = startBlocks; numBlocks >= 1; numBlocks /= 2) {
      // Fill BlockInfo: all blocks have op=3, same params.
      for (int32_t i = 0; i < numBlocks; ++i) {
        memset(&blockInfoHost[i], 0, sizeof(BlockInfo));
        blockInfoHost[i].op = 3;
        blockInfoHost[i].blockInOp = i;
        blockInfoHost[i].numBlocksInOp = numBlocks;
        blockInfoHost[i].params = paramBlock.get();
      }

      cudaMemcpy(
          blockInfoDev.get(),
          blockInfoHost.get(),
          numBlocks * sizeof(BlockInfo),
          cudaMemcpyHostToDevice);

      TorchWaveParams twParams;
      memset(&twParams, 0, sizeof(twParams));
      twParams.info = blockInfoDev.get();

      // Timed run: includes launch + sync.
      auto start = std::chrono::high_resolution_clock::now();
      torchwave0<<<numBlocks, kBlockSize>>>(twParams);
      cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();

      double micros =
          std::chrono::duration<double, std::micro>(end - start).count();

      printf(
          "DataSize: %7d  Blocks: %4d  Time: %8.1f us\n",
          dataSize,
          numBlocks,
          micros);

      trials.push_back({dataSize, numBlocks, micros});

      if (numBlocks == 1) {
        break;
      }
    }
  }

  // Find minimum run time.
  double minMicros = trials[0].micros;
  int minIdx = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].micros < minMicros) {
      minMicros = trials[i].micros;
      minIdx = i;
    }
  }

  // Find maximum throughput: tensor_size * element_size / time, in GB/s.
  double maxGBps = 0;
  int maxIdx = 0;
  for (size_t i = 0; i < trials.size(); ++i) {
    double gbps = static_cast<double>(trials[i].dataSize) * kElementSize /
        (trials[i].micros * 1000.0);
    if (gbps > maxGBps) {
      maxGBps = gbps;
      maxIdx = i;
    }
  }

  printf("\n--- Summary ---\n");
  printf(
      "Min time: %.1f us (DataSize: %d, Blocks: %d)\n",
      trials[minIdx].micros,
      trials[minIdx].dataSize,
      trials[minIdx].numBlocks);

  double maxGBpsVal = static_cast<double>(trials[maxIdx].dataSize) *
      kElementSize / (trials[maxIdx].micros * 1000.0);
  printf(
      "Max throughput: %.2f GB/s (DataSize: %d, Blocks: %d)\n",
      maxGBpsVal,
      trials[maxIdx].dataSize,
      trials[maxIdx].numBlocks);

  return 0;
}
