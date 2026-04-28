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
#include <gflags/gflags.h>
#include "velox/experimental/gpu/Common.h"

DEFINE_int32(
    complex_path,
    0,
    "0: simple kernel, 1: complex kernel with contiguous inputs, "
    "2: complex kernel with non-contiguous inputs");

namespace torch::wave {
constexpr int32_t kBlockSize = 256;
} // namespace torch::wave

#include "velox/experimental/torchwave/Core.cuh"
#include "velox/experimental/torchwave/Elementwise.cuh"

namespace torch::wave {

// Simple kernel: no init, host sets numEl/contiguous. Has complexIdx slow path.
__global__ void torchwave0(TorchWaveParams params) {
  ENTRY;
  __shared__ uint32_t size;
  __shared__ uint32_t isFastPath0;
  constexpr int32_t kT = sizeof(Tensor);
  switch (blockInfo.op) {
    case 3: {
      {
        if (threadIdx.x == 0) {
          isFastPath0 = 0;
          Tensor* temp = param<Tensor>(blockInfo, 0);
          size = temp->numEl;
          isFastPath0 |= temp->contiguous;
          uint32_t size2;
          temp = param<Tensor>(blockInfo, kT);
          size2 = temp->numEl;
          isFastPath0 |= (uint32_t)temp->contiguous << 1;
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
        int64_t* b1 = storage<int64_t>(param<Tensor>(blockInfo, kT));
        int64_t* b2 = storage<int64_t>(param<Tensor>(blockInfo, 2 * kT));
        int64_t attr = *param<int64_t>(blockInfo, 3 * kT);
        if (isFastPath0 == 0x3) {
          for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x;
               idx < size;
               idx += blockInfo.numBlocksInOp * blockDim.x) {
            int64_t result0 = __add(b0[idx], b1[idx], attr);
            b2[idx] = result0;
          }
        } else {
          for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x;
               idx < size;
               idx += blockInfo.numBlocksInOp * blockDim.x) {
            int64_t result0 = __add(
                b0[complexIdx(
                    isFastPath0 & (1 << 0), param<Tensor>(blockInfo, 0), idx)],
                b1[complexIdx(
                    isFastPath0 & (1 << 1), param<Tensor>(blockInfo, kT), idx)],
                attr);
            b2[idx] = result0;
          }
        }
      }
      break;
    }
  }
  LEAVE();
}

// Complex path kernel: init<true>() with outputOffsets, has complexIdx slow
// path.
__global__ void torchwave1(TorchWaveParams params) {
  ENTRY;
  __shared__ uint32_t size;
  __shared__ uint32_t isFastPath0;
  constexpr int32_t kT = sizeof(Tensor);
  switch (blockInfo.op) {
    case 3: {
      {
        static int32_t paramOffsets[] = {0, kT, 2 * kT};
        static int32_t outputOffsets[] = {2 * kT, 2 * kT, 2 * kT};
        static int32_t altOffsets[] = {-1, -1, -1};
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
      {
        if (threadIdx.x == 0) {
          isFastPath0 = 0;
          Tensor* temp = param<Tensor>(blockInfo, 0);
          size = temp->numEl;
          isFastPath0 |= temp->contiguous;
          uint32_t size2;
          temp = param<Tensor>(blockInfo, kT);
          size2 = temp->numEl;
          isFastPath0 |= (uint32_t)temp->contiguous << 1;
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
        int64_t* b1 = storage<int64_t>(param<Tensor>(blockInfo, kT));
        int64_t* b2 = storage<int64_t>(param<Tensor>(blockInfo, 2 * kT));
        int64_t attr = *param<int64_t>(blockInfo, 3 * kT);
        if (isFastPath0 == 0x3) {
          for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x;
               idx < size;
               idx += blockInfo.numBlocksInOp * blockDim.x) {
            int64_t result0 = __add(b0[idx], b1[idx], attr);
            b2[idx] = result0;
          }
        } else {
          for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x;
               idx < size;
               idx += blockInfo.numBlocksInOp * blockDim.x) {
            int64_t result0 = __add(
                b0[complexIdx(
                    isFastPath0 & (1 << 0), param<Tensor>(blockInfo, 0), idx)],
                b1[complexIdx(
                    isFastPath0 & (1 << 1), param<Tensor>(blockInfo, kT), idx)],
                attr);
            b2[idx] = result0;
          }
        }
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
  t->numEl = tensor.numel();
  t->contiguous = true;
  t->status = Tensor::kUninited;
}

// Fills a Tensor param for the complex path kernel. For nonContiguous, creates
// a 2D tensor with reversed strides (column-major): dims=[dataSize/16, 16],
// strides=[1, dataSize/16].
static void fillTensorParamComplex(
    void* storagePtr,
    Tensor* t,
    int32_t dataSize,
    bool nonContiguous) {
  memset(t, 0, sizeof(Tensor));
  t->storage = storagePtr;
  if (nonContiguous) {
    t->rank = 2;
    t->dims[0] = dataSize / 16;
    t->dims[1] = 16;
    t->strides[0] = 1;
    t->strides[1] = dataSize / 16;
    t->status = Tensor::kUninited;
  } else {
    t->rank = 1;
    t->dims[0] = dataSize;
    t->strides[0] = 1;
    t->numEl = dataSize;
    t->contiguous = true;
  }
}

struct Trial {
  int32_t dataSize;
  int32_t numBlocks;
  double micros;
};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  constexpr int32_t kMaxSize = 1 << 24;
  constexpr int32_t kElementSize = sizeof(int64_t);
  const bool useComplexKernel = FLAGS_complex_path > 0;
  const bool nonContiguous = FLAGS_complex_path == 2;
  constexpr int32_t tensorStride = sizeof(Tensor);

  // Base tensors: 2 inputs + 1 output, each kMaxSize int64_t elements on GPU.
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

  // Param block: 3 Tensors + 1 attr (int64_t).
  int32_t kParamSize = 3 * tensorStride + 1 * 8;
  auto paramBlock = allocateManagedArray<char>(kParamSize);
  memset(paramBlock.get(), 0, kParamSize);

  // Attr set to 1.
  *reinterpret_cast<int64_t*>(paramBlock.get() + 3 * tensorStride) = 1;

  // Pre-allocate device BlockInfo for max block count.
  int32_t maxBlocks = (kMaxSize + kBlockSize - 1) / kBlockSize;
  auto blockInfoHost = allocateManagedArray<BlockInfo>(maxBlocks);
  auto blockInfoDev = allocateDeviceMemory<BlockInfo>(maxBlocks);

  // Select kernel function pointer.
  auto kernelFunc = useComplexKernel ? torchwave1 : torchwave0;

  std::vector<Trial> trials;

  printf(
      "Mode: complex_path=%d (%s kernel, %s inputs)\n",
      FLAGS_complex_path,
      useComplexKernel ? "complex" : "simple",
      nonContiguous ? "non-contiguous" : "contiguous");

  // Warmup: one kernel launch to initialize CUDA runtime.
  {
    int32_t warmupSize = kBlockSize;
    auto v0 = b0.slice(0, 0, warmupSize);
    auto v1 = b1.slice(0, 0, warmupSize);
    auto v2 = b2.slice(0, 0, warmupSize);

    if (useComplexKernel) {
      fillTensorParamComplex(
          v0.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 0 * tensorStride),
          warmupSize,
          nonContiguous);
      fillTensorParamComplex(
          v1.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 1 * tensorStride),
          warmupSize,
          nonContiguous);
      fillTensorParamComplex(
          v2.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 2 * tensorStride),
          warmupSize,
          false);
    } else {
      fillTensorParam(
          v0, reinterpret_cast<Tensor*>(paramBlock.get() + 0 * tensorStride));
      fillTensorParam(
          v1, reinterpret_cast<Tensor*>(paramBlock.get() + 1 * tensorStride));
      fillTensorParam(
          v2, reinterpret_cast<Tensor*>(paramBlock.get() + 2 * tensorStride));
    }

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
    kernelFunc<<<1, kBlockSize>>>(twParams);
    cudaDeviceSynchronize();
  }

  // Benchmark: data sizes from 256 to kMaxSize in powers of two.
  for (int32_t dataSize = 256; dataSize <= kMaxSize; dataSize *= 2) {
    auto v0 = b0.slice(0, 0, dataSize);
    auto v1 = b1.slice(0, 0, dataSize);
    auto v2 = b2.slice(0, 0, dataSize);

    if (useComplexKernel) {
      fillTensorParamComplex(
          v0.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 0 * tensorStride),
          dataSize,
          nonContiguous);
      fillTensorParamComplex(
          v1.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 1 * tensorStride),
          dataSize,
          nonContiguous);
      fillTensorParamComplex(
          v2.data_ptr(),
          reinterpret_cast<Tensor*>(paramBlock.get() + 2 * tensorStride),
          dataSize,
          false);
    } else {
      fillTensorParam(
          v0, reinterpret_cast<Tensor*>(paramBlock.get() + 0 * tensorStride));
      fillTensorParam(
          v1, reinterpret_cast<Tensor*>(paramBlock.get() + 1 * tensorStride));
      fillTensorParam(
          v2, reinterpret_cast<Tensor*>(paramBlock.get() + 2 * tensorStride));
    }

    int32_t startBlocks = (dataSize + kBlockSize - 1) / kBlockSize;

    for (int32_t numBlocks = startBlocks; numBlocks >= 1; numBlocks /= 2) {
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

      auto start = std::chrono::high_resolution_clock::now();
      kernelFunc<<<numBlocks, kBlockSize>>>(twParams);
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

  // Print kernel info.
  cudaFuncAttributes attrs;
  cudaFuncGetAttributes(&attrs, kernelFunc);
  printf(
      "\nKernel: %s  regs=%d  sharedMem=%zu  localMem=%zu  maxThreads=%d\n",
      useComplexKernel ? "torchwave1 (complex)" : "torchwave0 (simple)",
      attrs.numRegs,
      attrs.sharedSizeBytes,
      attrs.localSizeBytes,
      attrs.maxThreadsPerBlock);

  return 0;
}
