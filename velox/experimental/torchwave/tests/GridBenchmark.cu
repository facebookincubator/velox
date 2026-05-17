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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "velox/experimental/gpu/Common.h"

DEFINE_int32(num_inputs, 8, "Number of parallel masked_select inputs");
DEFINE_int32(input_size, 100000, "Number of elements per input");
DEFINE_int32(repeats, 10, "Number of launch repetitions for timing");

namespace torch::wave {
constexpr int32_t kBlockSize = 256;
} // namespace torch::wave

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
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
  t->numEl = tensor.numel();
  t->status = Tensor::kUninited;
}

// Single-block masked_select kernel: each block handles one complete input.
__global__ void maskedSelectSingleKernel(TorchWaveParams params) {
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

// Three-kernel masked_select: head, addSizes, final.
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

  add_sizes<kBlockSize>(
      param<Tensor>(blockInfo, 0),
      param<int32_t>(blockInfo, sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      counter,
      blockInfo);
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

struct InputSet {
  at::Tensor input;
  at::Tensor mask;
  at::Tensor output;
  at::Tensor counts;
};

class GridBenchmark : public ::testing::Test {
 protected:
  std::vector<InputSet> makeInputs(int32_t numInputs, int32_t inputSize) {
    std::vector<InputSet> result;
    result.reserve(numInputs);
    for (int i = 0; i < numInputs; ++i) {
      InputSet s;
      s.input = at::randint(
          0,
          1000,
          {inputSize},
          at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
      s.mask = at::rand(
                   {inputSize},
                   at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
                   .lt(0.9)
                   .to(at::kBool);
      s.output = at::zeros(
          {inputSize}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
      int32_t numBlocks = (inputSize + kBlockSize - 1) / kBlockSize;
      s.counts = at::zeros(
          {numBlocks}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
      result.push_back(std::move(s));
    }
    return result;
  }

  void verify(const std::vector<InputSet>& inputs) {
    for (size_t s = 0; s < inputs.size(); ++s) {
      auto inputCpu = inputs[s].input.cpu();
      auto maskCpu = inputs[s].mask.cpu();
      auto outputCpu = inputs[s].output.cpu();
      auto* inputData = inputCpu.data_ptr<int32_t>();
      auto* maskData = maskCpu.data_ptr<bool>();
      auto* outputData = outputCpu.data_ptr<int32_t>();
      std::vector<int32_t> expected;
      for (int i = 0; i < inputCpu.numel(); ++i) {
        if (maskData[i]) {
          expected.push_back(inputData[i]);
        }
      }
      for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(outputData[i], expected[i])
            << "Input " << s << " mismatch at index " << i;
      }
    }
  }
};

TEST_F(GridBenchmark, singleBlock) {
  auto numInputs = FLAGS_num_inputs;
  auto inputSize = FLAGS_input_size;
  auto inputs = makeInputs(numInputs, inputSize);

  // Each input gets 1 block (single-block masked_select).
  int32_t totalBlocks = numInputs;

  // Allocate per-input param buffers and a single BlockInfo array.
  std::vector<CudaPtr<Tensor[]>> paramPtrs;
  for (int i = 0; i < numInputs; ++i) {
    auto p = allocateManagedArray<Tensor>(3);
    fillTensorParam(inputs[i].input, &p[0]);
    fillTensorParam(inputs[i].mask, &p[1]);
    fillTensorParam(inputs[i].output, &p[2]);
    paramPtrs.push_back(std::move(p));
  }

  auto blockInfoHost = allocateManagedArray<BlockInfo>(totalBlocks);
  for (int i = 0; i < numInputs; ++i) {
    memset(&blockInfoHost[i], 0, sizeof(BlockInfo));
    blockInfoHost[i].blockInOp = 0;
    blockInfoHost[i].numBlocksInOp = 1;
    blockInfoHost[i].params = paramPtrs[i].get();
  }

  auto blockInfoDev = allocateDeviceMemory<BlockInfo>(totalBlocks);
  cudaMemcpy(
      blockInfoDev.get(),
      blockInfoHost.get(),
      totalBlocks * sizeof(BlockInfo),
      cudaMemcpyHostToDevice);

  TorchWaveParams twParams;
  memset(&twParams, 0, sizeof(twParams));
  twParams.info = blockInfoDev.get();

  // Warmup.
  maskedSelectSingleKernel<<<totalBlocks, kBlockSize>>>(twParams);
  cudaDeviceSynchronize();

  verify(inputs);

  // Timed runs.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int r = 0; r < FLAGS_repeats; ++r) {
    // Reset output tensor status for re-runs.
    for (int i = 0; i < numInputs; ++i) {
      paramPtrs[i][2].status = Tensor::kUninited;
    }
    cudaMemcpy(
        blockInfoDev.get(),
        blockInfoHost.get(),
        totalBlocks * sizeof(BlockInfo),
        cudaMemcpyHostToDevice);
    maskedSelectSingleKernel<<<totalBlocks, kBlockSize>>>(twParams);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf(
      "singleBlock: %d inputs x %d elements, %d repeats, %.2f ms total, "
      "%.2f ms/launch\n",
      numInputs,
      inputSize,
      FLAGS_repeats,
      ms,
      ms / FLAGS_repeats);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

TEST_F(GridBenchmark, threeKernel) {
  auto numInputs = FLAGS_num_inputs;
  auto inputSize = FLAGS_input_size;
  auto inputs = makeInputs(numInputs, inputSize);

  int32_t numBlocks = (inputSize + kBlockSize - 1) / kBlockSize;
  // Each input gets half as many grid blocks (each block does ~2 iterations).
  int32_t blocksPerInput = (numBlocks + 1) / 2;

  // --- Head params: one per input, each with input+mask+counts tensors ---
  std::vector<CudaPtr<Tensor[]>> headParamPtrs;
  for (int i = 0; i < numInputs; ++i) {
    auto p = allocateManagedArray<Tensor>(3);
    fillTensorParam(inputs[i].input, &p[0]);
    fillTensorParam(inputs[i].mask, &p[1]);
    fillTensorParam(inputs[i].counts, &p[2]);
    headParamPtrs.push_back(std::move(p));
  }

  int32_t headTotalBlocks = numInputs * blocksPerInput;
  auto headBlockInfoHost = allocateManagedArray<BlockInfo>(headTotalBlocks);
  for (int i = 0; i < numInputs; ++i) {
    for (int b = 0; b < blocksPerInput; ++b) {
      auto& bi = headBlockInfoHost[i * blocksPerInput + b];
      memset(&bi, 0, sizeof(BlockInfo));
      bi.blockInOp = b;
      bi.numBlocksInOp = blocksPerInput;
      bi.params = headParamPtrs[i].get();
    }
  }
  auto headBlockInfoDev = allocateDeviceMemory<BlockInfo>(headTotalBlocks);

  // --- AddSizes params: one per input, counts tensor + int32_t total ---
  std::vector<CudaPtr<char[]>> addSizesParamPtrs;
  for (int i = 0; i < numInputs; ++i) {
    auto p = allocateManagedArray<char>(sizeof(Tensor) + sizeof(int32_t));
    addSizesParamPtrs.push_back(std::move(p));
  }

  auto addSizesBlockInfoHost = allocateManagedArray<BlockInfo>(numInputs);

  // --- Final params: one per input ---
  constexpr auto kFinalOutputOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  std::vector<CudaPtr<char[]>> finalParamPtrs;
  for (int i = 0; i < numInputs; ++i) {
    auto p = allocateManagedArray<char>(kFinalOutputOffset + sizeof(Tensor));
    finalParamPtrs.push_back(std::move(p));
  }

  int32_t finalTotalBlocks = numInputs * blocksPerInput;
  auto finalBlockInfoHost = allocateManagedArray<BlockInfo>(finalTotalBlocks);
  auto finalBlockInfoDev = allocateDeviceMemory<BlockInfo>(finalTotalBlocks);

  auto setupAndRun = [&]() {
    // Head launch.
    cudaMemcpy(
        headBlockInfoDev.get(),
        headBlockInfoHost.get(),
        headTotalBlocks * sizeof(BlockInfo),
        cudaMemcpyHostToDevice);
    TorchWaveParams headTw;
    memset(&headTw, 0, sizeof(headTw));
    headTw.info = headBlockInfoDev.get();
    maskedSelectHeadKernel<<<headTotalBlocks, kBlockSize>>>(headTw);

    // AddSizes launch: one block per input.
    for (int i = 0; i < numInputs; ++i) {
      fillTensorParam(
          inputs[i].counts,
          reinterpret_cast<Tensor*>(addSizesParamPtrs[i].get()));
      *reinterpret_cast<int32_t*>(addSizesParamPtrs[i].get() + sizeof(Tensor)) =
          0;
      auto& bi = addSizesBlockInfoHost[i];
      memset(&bi, 0, sizeof(BlockInfo));
      bi.numBlocksInOp = 1;
      bi.params = addSizesParamPtrs[i].get();
    }
    auto addSizesBlockInfoDev = allocateDeviceMemory<BlockInfo>(numInputs);
    cudaMemcpy(
        addSizesBlockInfoDev.get(),
        addSizesBlockInfoHost.get(),
        numInputs * sizeof(BlockInfo),
        cudaMemcpyHostToDevice);
    TorchWaveParams addSizesTw;
    memset(&addSizesTw, 0, sizeof(addSizesTw));
    addSizesTw.info = addSizesBlockInfoDev.get();
    addSizesKernel<<<numInputs, kBlockSize>>>(addSizesTw);
    cudaDeviceSynchronize();

    // Final launch.
    for (int i = 0; i < numInputs; ++i) {
      int32_t totalCount = *reinterpret_cast<int32_t*>(
          addSizesParamPtrs[i].get() + sizeof(Tensor));
      fillTensorParam(
          inputs[i].input, reinterpret_cast<Tensor*>(finalParamPtrs[i].get()));
      fillTensorParam(
          inputs[i].mask,
          reinterpret_cast<Tensor*>(finalParamPtrs[i].get() + sizeof(Tensor)));
      fillTensorParam(
          inputs[i].counts,
          reinterpret_cast<Tensor*>(
              finalParamPtrs[i].get() + 2 * sizeof(Tensor)));
      *reinterpret_cast<int32_t*>(
          finalParamPtrs[i].get() + 3 * sizeof(Tensor)) = totalCount;
      fillTensorParam(
          inputs[i].output,
          reinterpret_cast<Tensor*>(
              finalParamPtrs[i].get() + kFinalOutputOffset));
      for (int b = 0; b < blocksPerInput; ++b) {
        auto& bi = finalBlockInfoHost[i * blocksPerInput + b];
        memset(&bi, 0, sizeof(BlockInfo));
        bi.blockInOp = b;
        bi.numBlocksInOp = blocksPerInput;
        bi.params = finalParamPtrs[i].get();
      }
    }
    cudaMemcpy(
        finalBlockInfoDev.get(),
        finalBlockInfoHost.get(),
        finalTotalBlocks * sizeof(BlockInfo),
        cudaMemcpyHostToDevice);
    TorchWaveParams finalTw;
    memset(&finalTw, 0, sizeof(finalTw));
    finalTw.info = finalBlockInfoDev.get();
    maskedSelectFinalKernel<<<finalTotalBlocks, kBlockSize>>>(finalTw);
    cudaDeviceSynchronize();
  };

  // Warmup + verify.
  setupAndRun();
  verify(inputs);

  // Timed runs.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int r = 0; r < FLAGS_repeats; ++r) {
    // Reset counts and output status.
    for (int i = 0; i < numInputs; ++i) {
      inputs[i].counts.zero_();
      headParamPtrs[i][2].status = Tensor::kUninited;
    }
    setupAndRun();
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf(
      "threeKernel: %d inputs x %d elements, %d repeats, %.2f ms total, "
      "%.2f ms/launch\n",
      numInputs,
      inputSize,
      FLAGS_repeats,
      ms,
      ms / FLAGS_repeats);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Per-input barrier using a global memory counter. All blocks for the same
// input call this with the same counter. Spins until all have arrived.
__device__ void inputBarrier(int32_t* counter, int32_t expected) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(counter, 1);
    volatile int32_t* vc = reinterpret_cast<volatile int32_t*>(counter);
    while (*vc < expected) {
      __nanosleep(100);
    }
  }
  __syncthreads();
}

// Cooperative masked_select: runs head, addSizes, and final in a single
// kernel launch. Each input has 'blocksPerInput' blocks that synchronize
// via per-input atomic counters in 'barriers'.
//
// Params layout per input (same as the three-kernel final params):
//   0: Tensor input
//   sizeof(Tensor): Tensor mask
//   2*sizeof(Tensor): Tensor counts
//   3*sizeof(Tensor): int32_t total
//   kOutputOffset: Tensor output
//
// barriers: numInputs * 2 int32_t counters in device memory, inited to 0.
__global__ void maskedSelectCoopKernel(
    TorchWaveParams params,
    int32_t blocksPerInput,
    int32_t* barriers) {
  __shared__ BlockInfo blockInfo;
  __shared__ BlockInfo addSizesBi;
  __shared__ Int32X32 temp;
  __shared__ uint32_t size;
  __shared__ uint32_t rounded;
  __shared__ uint32_t counter;
  __shared__ uint32_t addSize;
  __shared__ uint32_t addRounded;
  __shared__ uint32_t addCounter;

  if (threadIdx.x == 0) {
    blockInfo = params.info[blockIdx.x];
  }
  __syncthreads();

  int32_t localBlock = blockInfo.blockInOp;
  int32_t inputIdx = blockIdx.x / blocksPerInput;
  int32_t* barrier0 = &barriers[inputIdx * 2];
  int32_t* barrier1 = &barriers[inputIdx * 2 + 1];

  // Stage 1: head — all blocks for this input.
  masked_select_head<kBlockSize, int32_t>(
      param<Tensor>(blockInfo, 0),
      param<Tensor>(blockInfo, sizeof(Tensor)),
      param<Tensor>(blockInfo, 2 * sizeof(Tensor)),
      (void*)temp,
      size,
      rounded,
      blockInfo);
  __threadfence();

  inputBarrier(barrier0, blocksPerInput);

  // Stage 2: addSizes — only local block 0.
  if (localBlock == 0) {
    if (threadIdx.x == 0) {
      addSizesBi.blockInOp = 0;
      addSizesBi.numBlocksInOp = 1;
      addSizesBi.params =
          reinterpret_cast<char*>(blockInfo.params) + 2 * sizeof(Tensor);
    }
    __syncthreads();
    add_sizes<kBlockSize>(
        param<Tensor>(addSizesBi, 0),
        param<int32_t>(addSizesBi, sizeof(Tensor)),
        (void*)temp,
        addSize,
        addRounded,
        addCounter,
        addSizesBi);
    __threadfence();
  }

  inputBarrier(barrier1, blocksPerInput);

  // Stage 3: final — all blocks for this input.
  constexpr int32_t kOutputOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  masked_select_final<kBlockSize, int32_t>(
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

TEST_F(GridBenchmark, cooperative) {
  auto numInputs = FLAGS_num_inputs;
  auto inputSize = FLAGS_input_size;
  auto inputs = makeInputs(numInputs, inputSize);

  // Query device for max cooperative blocks. cudaLaunchCooperativeKernel
  // guarantees all blocks are resident simultaneously.
  int device;
  cudaGetDevice(&device);
  int maxBlocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, maskedSelectCoopKernel, kBlockSize, 0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  maxBlocks *= prop.multiProcessorCount;

  int32_t numBlocks = (inputSize + kBlockSize - 1) / kBlockSize;
  int32_t blocksPerInput = (numBlocks + 1) / 2;
  // Cap so total blocks fit on the device.
  blocksPerInput = std::min(blocksPerInput, maxBlocks / numInputs);
  ASSERT_GT(blocksPerInput, 0) << "Too many inputs for device capacity";
  int32_t totalBlocks = numInputs * blocksPerInput;

  // Params layout: input, mask, counts, total, output (same as final).
  constexpr auto kOutputOffset =
      roundUp(3 * sizeof(Tensor) + sizeof(int32_t), alignof(Tensor));
  constexpr auto kParamSize = kOutputOffset + sizeof(Tensor);

  std::vector<CudaPtr<char[]>> paramPtrs;
  for (int i = 0; i < numInputs; ++i) {
    paramPtrs.push_back(allocateManagedArray<char>(kParamSize));
  }

  auto blockInfoHost = allocateManagedArray<BlockInfo>(totalBlocks);
  auto blockInfoDev = allocateDeviceMemory<BlockInfo>(totalBlocks);

  // Barrier counters: 2 per input, in device memory.
  auto barriersDev = allocateDeviceMemory<int32_t>(numInputs * 2);

  auto setupAndRun = [&]() {
    // Zero barrier counters.
    cudaMemset(barriersDev.get(), 0, numInputs * 2 * sizeof(int32_t));

    // Fill params and block info.
    for (int i = 0; i < numInputs; ++i) {
      auto* p = paramPtrs[i].get();
      fillTensorParam(inputs[i].input, reinterpret_cast<Tensor*>(p));
      fillTensorParam(
          inputs[i].mask, reinterpret_cast<Tensor*>(p + sizeof(Tensor)));
      fillTensorParam(
          inputs[i].counts, reinterpret_cast<Tensor*>(p + 2 * sizeof(Tensor)));
      *reinterpret_cast<int32_t*>(p + 3 * sizeof(Tensor)) = 0;
      fillTensorParam(
          inputs[i].output, reinterpret_cast<Tensor*>(p + kOutputOffset));

      for (int b = 0; b < blocksPerInput; ++b) {
        auto& bi = blockInfoHost[i * blocksPerInput + b];
        memset(&bi, 0, sizeof(BlockInfo));
        bi.blockInOp = b;
        bi.numBlocksInOp = blocksPerInput;
        bi.params = p;
      }
    }

    cudaMemcpy(
        blockInfoDev.get(),
        blockInfoHost.get(),
        totalBlocks * sizeof(BlockInfo),
        cudaMemcpyHostToDevice);

    TorchWaveParams tw;
    memset(&tw, 0, sizeof(tw));
    tw.info = blockInfoDev.get();

    int32_t* barriersPtr = barriersDev.get();
    void* args[] = {&tw, &blocksPerInput, &barriersPtr};
    auto err = cudaLaunchCooperativeKernel(
        (void*)maskedSelectCoopKernel, totalBlocks, kBlockSize, args);
    TORCH_CHECK(
        err == cudaSuccess,
        "cudaLaunchCooperativeKernel failed: ",
        cudaGetErrorString(err));
    cudaDeviceSynchronize();
  };

  // Warmup + verify.
  setupAndRun();

  verify(inputs);

  // Timed runs.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int r = 0; r < FLAGS_repeats; ++r) {
    for (int i = 0; i < numInputs; ++i) {
      inputs[i].counts.zero_();
      // Reset Tensor::status for counts and output in managed params.
      reinterpret_cast<Tensor*>(paramPtrs[i].get() + 2 * sizeof(Tensor))
          ->status = Tensor::kUninited;
      reinterpret_cast<Tensor*>(paramPtrs[i].get() + kOutputOffset)->status =
          Tensor::kUninited;
    }
    setupAndRun();
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf(
      "cooperative: %d inputs x %d elements, %d blocks/input, "
      "%d repeats, %.2f ms total, %.2f ms/launch\n",
      numInputs,
      inputSize,
      blocksPerInput,
      FLAGS_repeats,
      ms,
      ms / FLAGS_repeats);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

} // namespace
} // namespace torch::wave
