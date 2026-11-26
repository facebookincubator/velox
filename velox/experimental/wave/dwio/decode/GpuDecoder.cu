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

#include "velox/experimental/wave/common/Bits.cuh"
#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/common/GpuArena.h"
#include "velox/experimental/wave/dwio/decode/DecodeStep.h"
#include "velox/experimental/wave/dwio/decode/ReaderKernel.h"

namespace facebook::velox::wave {

int32_t GpuDecode::tempSize() const {
  // 1 int32 per lane as an upper limit.
  return std::max<int32_t>(kBlockSize * sizeof(int32_t), sizeof(NonNullState));
}

int32_t GpuDecode::sharedMemorySize() const {
  return decodeSharedMemorySize(step, kBlockSize);
}

int32_t decodeSharedMemorySize(DecodeStep step, int32_t blockSize) {
  // See detail::kScanSmemSlots.
  const int32_t scanSize =
      (blockSize / kWarpThreads + 2) * static_cast<int32_t>(sizeof(int64_t));
  switch (step) {
    case DecodeStep::kRleTotalLength:
    case DecodeStep::kMainlyConstant:
    case DecodeStep::kRleBool:
    case DecodeStep::kVarint:
    case DecodeStep::kMakeScatterIndices:
    case DecodeStep::kLengthToOffset:
      return scanSize;
    case DecodeStep::kRle:
      // Scan temp followed by an offset per thread.
      return scanSize + blockSize * static_cast<int32_t>(sizeof(int32_t));
    default:
      return 0;
  }
}

namespace {

// Sets the opcode of each op and returns the kernel for the ops.
ReaderKernel compiledKernel(const DecodePrograms& programs, int32_t& shared) {
  std::vector<GpuDecode*> ops;
  for (auto& program : programs.programs) {
    for (auto& op : program) {
      ops.push_back(op.get());
    }
  }
  auto key = makeReaderKernelKey(ops, kBlockSize);
  shared = key.sharedMemorySize();
  return getReaderKernel(key);
}

// Calls 'launch' and returns its wall time in microseconds.
template <typename Launch>
int64_t timedLaunch(Launch launch) {
  const auto start = std::chrono::steady_clock::now();
  launch();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

DecodeLaunchInfo launchDecode(
    const DecodePrograms& programs,
    LaunchParams& launchParams,
    Stream* stream) {
  int32_t numBlocks = programs.programs.size();
  int32_t numOps = 0;
  bool allSingle = true;
  for (auto& program : programs.programs) {
    if (program.size() != 1) {
      allSingle = false;
    }
    numOps += program.size();
  }
  int32_t shared = 0;
  auto readerKernel = compiledKernel(programs, shared);
  auto& kernel = readerKernel.kernel;
  DecodeLaunchInfo info;
  info.numBlocks = numBlocks;
  info.isNewKernel = readerKernel.isNew;
  GpuDecodeParams localParams;
  GpuDecodeParams* params = &localParams;
  char* host = nullptr;
  char* device = nullptr;
  if (numOps > GpuDecodeParams::kMaxInlineOps || allSingle) {
    auto pair = launchParams.setup(
        (numOps + 1) * (sizeof(GpuDecode) + sizeof(int32_t)) + 16);
    host = pair.first;
    device = pair.second;
    uintptr_t aligned = roundUp(reinterpret_cast<uintptr_t>(host), 16);
    params = reinterpret_cast<GpuDecodeParams*>(aligned);
  }
  GpuDecode* decodes =
      reinterpret_cast<GpuDecode*>(&params->ends[0] + roundUp(numBlocks, 4));
  uintptr_t decodeOffset = reinterpret_cast<char*>(decodes) - host;
  int32_t fill = 0;
  for (auto i = 0; i < programs.programs.size(); ++i) {
    params->ends[i] =
        (i == 0 ? 0 : params->ends[i - 1]) + programs.programs[i].size();
    for (auto& op : programs.programs[i]) {
      decodes[fill++] = *op;
    }
  }
  if (allSingle) {
    launchParams.transfer(*stream);
    auto* deviceOps = reinterpret_cast<GpuDecode*>(device + decodeOffset);
    void* args[] = {&deviceOps};
    info.launchMicros = timedLaunch([&]() {
      kernel->launch(
          static_cast<int32_t>(ReaderEntry::kSingle),
          numBlocks,
          kBlockSize,
          shared,
          stream,
          args);
    });
    programs.result.transfer(*stream);
    return info;
  }
  if (launchParams.device) {
    localParams.external = reinterpret_cast<GpuDecodeParams*>(device);
    launchParams.transfer(*stream);
  }
  void* args[] = {&localParams};
  info.launchMicros = timedLaunch([&]() {
    kernel->launch(
        static_cast<int32_t>(ReaderEntry::kPrograms),
        numBlocks,
        kBlockSize,
        shared,
        stream,
        args);
  });
  programs.result.transfer(*stream);
  return info;
}

void launchDecodeOps(
    GpuDecode* ops,
    int32_t numOps,
    int32_t blockSize,
    Stream* stream) {
  std::vector<GpuDecode*> opPointers;
  opPointers.reserve(numOps);
  for (auto i = 0; i < numOps; ++i) {
    opPointers.push_back(&ops[i]);
  }
  auto key = makeReaderKernelKey(opPointers, blockSize);
  auto kernel = getReaderKernel(key).kernel;
  void* args[] = {&ops};
  kernel->launch(
      static_cast<int32_t>(ReaderEntry::kSingle),
      numOps,
      blockSize,
      key.sharedMemorySize(),
      stream,
      args);
}

} // namespace facebook::velox::wave
