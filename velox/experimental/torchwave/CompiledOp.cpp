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

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GroupAlloc.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Standalones.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/wave/common/KernelFsCache.h"

#include <ATen/ATen.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/StringUtil.h>
#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <type_traits>
#include <unordered_set>

#include "velox/experimental/wave/common/GpuArena.h"

// debug_single_ops is now WaveConfig::debugSingleOps
DEFINE_string(
    debug_kernel_dir,
    "",
    "If non-empty, read kernel code from this directory instead of generating it");
DEFINE_bool(
    compile_meter,
    false,
    "Compile each case of a composite kernel individually to measure per-case nvrtc time");

// Forward declaration of the CUDA runtime call used to synchronize the default
// stream. This translation unit is built in a CPU-configured target without the
// CUDA headers; the symbol resolves from the CUDA runtime linked into the final
// binary. PyTorch dispatches eager standalone ops to the default stream.
extern "C" int cudaStreamSynchronize(void* stream);

// Forward-declared (not via <c10/cuda/...>) for the same reason as
// cudaStreamSynchronize above: this TU is CPU-configured and has no CUDA
// headers. current_device() is a non-inline C10_CUDA_API symbol resolved at
// final link. Allocator stats go through the CPU-safe, device-agnostic
// c10::getDeviceAllocator(CUDA) base interface
// (<c10/core/CachingDeviceAllocator.h>), whose getDeviceStats is a virtual
// dispatched to libc10_cuda's registered allocator, so no CUDA-header (or new
// build) dependency is needed.
namespace c10::cuda {
c10::DeviceIndex current_device();
} // namespace c10::cuda

namespace torch::wave {

namespace {

// Synchronizes the CUDA default stream (stream 0), where eager ATen standalone
// ops are dispatched. Used to order them against wave-stream work before a
// composite invocation returns.
void syncTorchDefaultStream() {
  cudaStreamSynchronize(nullptr);
}

// Bytes currently held in live tensors by the torch CUDA caching allocator on
// the active device. Sampled per step for the kTiming trace's "GPU RAM" field.
int64_t currentAllocatedBytes() {
  auto* allocator = c10::getDeviceAllocator(c10::DeviceType::CUDA);
  auto stats = allocator->getDeviceStats(c10::cuda::current_device());
  return stats
      .allocated_bytes[static_cast<size_t>(
          c10::CachingAllocator::StatType::AGGREGATE)]
      .current;
}

facebook::velox::wave::CompiledKernel& patchOpcodesKernel() {
  static std::unique_ptr<facebook::velox::wave::CompiledKernel> kernel;
  static std::once_flag flag; // @lint-ignore facebook-hte-std::once_flag
  std::call_once(flag, [] { // @lint-ignore facebook-hte-std::call_once
    kernel =
        facebook::velox::wave::CompiledKernel::getKernel("patchOpcodes", [] {
          facebook::velox::wave::KernelSpec spec;
          spec.code = R"(
struct BlockInfo {
  int op;
  int blockInOp;
  int numBlocksInOp;
  void* params;
  void* debugInfo;
  long long start;
  long long barrierClocks;
};

extern "C" __global__ void patchOpcodes(
    BlockInfo* blocks,
    int startBlock,
    int count,
    int opcode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    blocks[startBlock + idx].op = opcode;
  }
}
)";
          spec.entryPoints = {"patchOpcodes"};
          return spec;
        });
  });
  return *kernel;
}

void setOpCodes(
    BlockInfo* deviceBlocks,
    int32_t startBlock,
    int32_t count,
    int32_t opcode,
    facebook::velox::wave::Stream* stream) {
  auto& kernel = patchOpcodesKernel();
  int32_t numThreads = 256;
  int32_t numBlocks = (count + numThreads - 1) / numThreads;
  void* args[] = {&deviceBlocks, &startBlock, &count, &opcode};
  kernel.launch(0, numBlocks, numThreads, 0, stream, args);
}

} // namespace

void fillEmptyTensorParam(void* dest) {
  memset(dest, 0, sizeof(Tensor));
}

namespace {

std::string dumpOpParams(
    const KernelOperation& op,
    uint8_t* paramBase,
    const OpInvocation* invocation) {
  std::stringstream ss;
  const auto& inputs = op.orderedInputs();
  auto numInputs = op.numInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto offset = op.paramOffset(inputs[i]);
    bool isOutput = static_cast<int32_t>(i) >= numInputs;
    // The generated code refers to operands by param offset (b0 = param(1040)).
    // Print the formal and actual value ids alongside it so a dumped kernel
    // line can be tied back to a graph value (the actual id is what
    // --trace_values takes).
    auto formalId = inputs[i]->id();
    auto actualId = formalId;
    if (invocation != nullptr) {
      auto it = invocation->bindings().find(formalId);
      if (it != invocation->bindings().end()) {
        actualId = it->second;
      }
    }
    ss << "  " << (isOutput ? "output" : "input") << "[" << i
       << "] offset=" << offset << " %" << formalId << " -> %" << actualId
       << " (" << inputs[i]->name() << ")";
    if (inputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
      auto* t = reinterpret_cast<Tensor*>(paramBase + offset);
      ss << " " << tensorToString(*t);
    } else {
      ss << " scalar=" << *reinterpret_cast<int64_t*>(paramBase + offset);
    }
    ss << "\n";
  }
  auto constantOffset = op.constantAreaOffset();
  for (int32_t i = 0; i < op.numConstants(); ++i) {
    ss << "  const[" << i << "] offset=" << constantOffset
       << " value=" << *reinterpret_cast<int64_t*>(paramBase + constantOffset)
       << "\n";
    constantOffset += 8;
  }
  return ss.str();
}

} // namespace

void fillScalarParam(
    const c10::IValue& ivalue,
    void* dest,
    nativert::ValueId valueId) {
  if (ivalue.isInt()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toInt();
  } else if (ivalue.isDouble()) {
    *reinterpret_cast<double*>(dest) = ivalue.toDouble();
  } else if (ivalue.isBool()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toBool() ? 1 : 0;
  } else if (ivalue.isNone()) {
    *reinterpret_cast<int64_t*>(dest) = 0;
  } else {
    // A None here usually means the value's producer (e.g. a view/slice fed
    // into a prim.ListPack) was never executed, so its scalar (e.g. sym_size)
    // was never set in the frame. Report the value id to trace the producer.
    TORCH_CHECK(
        false,
        "Unsupported IValue type for kernel param: ",
        ivalue.tagKind(),
        " for value %",
        valueId);
  }
}

// Elements processed per thread for a cheap op: a block covers
// kMaxElementsPerThread * blockSize inputs, which caps the number of blocks so
// cheap ops do not pay launch overhead for tiny per-block work. Expensive ops
// drop toward one element per thread (more blocks, more parallelism) -- see
// elementsPerThreadForCost.
constexpr int32_t kMaxElementsPerThread = 4;

// Adjusted per-input cost (unitCost * costAdjustFactor) at and below which a
// block covers kMaxElementsPerThread elements per thread, and at and above
// which it covers exactly one. Between the two it interpolates linearly.
constexpr float kLowCostPerInput = 100.0f;
constexpr float kHighCostPerInput = 500.0f;

// Elements per thread for an op with the given adjusted per-input cost. Falls
// from kMaxElementsPerThread (cheap) to 1 (expensive) as the cost rises from
// kLowCostPerInput to kHighCostPerInput, then stays at 1. A higher cost thus
// allows more blocks (up to numElements / blockSize) for more parallelism.
int32_t elementsPerThreadForCost(float costPerInput) {
  if (costPerInput <= kLowCostPerInput) {
    return kMaxElementsPerThread;
  }
  if (costPerInput >= kHighCostPerInput) {
    return 1;
  }
  float frac = (costPerInput - kLowCostPerInput) /
      (kHighCostPerInput - kLowCostPerInput);
  int32_t ept = kMaxElementsPerThread -
      static_cast<int32_t>(frac * (kMaxElementsPerThread - 1) + 0.5f);
  return ept < 1 ? 1 : ept;
}

// Default SM count when device info is unavailable.
constexpr int32_t kDefaultNumSMs = 100;

// Default blocks per SM when occupancy info is unavailable.
constexpr int32_t kDefaultBlocksPerSM = 4;

// Shared memory per SM assumed when device info is unavailable (A100).
constexpr int32_t kDefaultSharedMemPerSM = 164 * 1024;

namespace {
// Dynamic shared memory a launch of 'launches' at 'blockSize' threads needs:
// the max over the ops, since the ops share one kernel launch.
int32_t dynamicSharedBytes(
    const std::vector<LaunchData>& launches,
    int32_t blockSize) {
  int64_t bytes = 0;
  for (const auto& launch : launches) {
    if (launch.launch && launch.launch->op) {
      bytes = std::max(bytes, launch.launch->op->dynamicSharedBytes(blockSize));
    }
  }
  return static_cast<int32_t>(bytes);
}
} // namespace

// Blocks a step's grid aims to fill: the device's SM count times the blocks of
// this kernel that stay resident on one SM. makeGrid distributes this many
// across the step's launches, and layoutParamSlots bounds the BlockInfo
// reservation by it, so the two must derive it the same way -- a reservation
// computed from a larger figure than makeGrid uses would be too small.
//
// 'maxBlocksPerSM' is the kernel's occupancy at zero dynamic shared memory. If
// an op in the step needs some, fewer blocks fit on an SM, and a cooperative
// launch of more blocks than fit fails outright. A caller that only wants an
// upper bound can pass dynSharedPerBlock == 0, which skips that reduction and
// so can only over-estimate.
GridDevice currentGridDevice(
    int32_t maxBlocksPerSM,
    int32_t staticSharedPerBlock,
    std::function<int32_t(int32_t)> occupancyFor = nullptr) {
  auto* device = facebook::velox::wave::currentDevice();
  int32_t numSMs = WaveConfig::get().numSms;
  if (numSMs == 0) {
    numSMs = device ? device->numSM : kDefaultNumSMs;
  }
  return {
      .numSMs = numSMs,
      .maxBlocksPerSM =
          maxBlocksPerSM > 0 ? maxBlocksPerSM : kDefaultBlocksPerSM,
      .sharedPerSM = device ? device->sharedMemPerSM : kDefaultSharedMemPerSM,
      .staticSharedPerBlock = staticSharedPerBlock,
      .occupancyFor = std::move(occupancyFor)};
}

// Asks the driver for the kernel's occupancy, memoized per dynamic shared size.
// The packer calls it once per op while grouping, and a step's ops fall into a
// handful of distinct shared-memory sizes, so the driver is asked a handful of
// times rather than once per op.
std::function<int32_t(int32_t)> occupancyQuery(
    const CompositeKernel* kernel,
    int32_t entry) {
  if (kernel == nullptr) {
    return nullptr;
  }
  auto cache = std::make_shared<folly::F14FastMap<int32_t, int32_t>>();
  return [kernel, entry, cache](int32_t dynamicShared) -> int32_t {
    auto it = cache->find(dynamicShared);
    if (it != cache->end()) {
      return it->second;
    }
    const int32_t blocks = kernel->occupancy(entry, dynamicShared);
    cache->emplace(dynamicShared, blocks);
    return blocks;
  };
}

int32_t targetBlockCount(
    int32_t maxBlocksPerSM,
    int32_t dynSharedPerBlock,
    int32_t staticSharedPerBlock) {
  const auto device = currentGridDevice(maxBlocksPerSM, staticSharedPerBlock);
  return device.numSMs * blocksPerSM(device, dynSharedPerBlock);
}

namespace {

// The step's ops as the launch layout sees them: cost and block cap from the
// pass above, occupancy and splittability from the op itself.
std::vector<GridOp> describeGridOps(
    const std::vector<LaunchData>& launches,
    const StepVectors& sv) {
  std::vector<GridOp> ops(launches.size());
  for (size_t i = 0; i < launches.size(); ++i) {
    auto* kernelOp = launches[i].launch->op;
    ops.at(i).cost = sv.costs.at(i);
    ops.at(i).maxBlocks = sv.maxBlocks.at(i);
    ops.at(i).dynamicShared = kernelOp
        ? static_cast<int32_t>(kernelOp->dynamicSharedBytes(sv.blockSize))
        : 0;
    ops.at(i).alwaysSingleBlock = kernelOp && kernelOp->alwaysSingleBlock();
    ops.at(i).hasBarrier = kernelOp && !kernelOp->barrierCounters().empty();
    ops.at(i).blocksPerSm = launches.at(i).launch->blocksPerSm;
  }
  return ops;
}

// WaveConfig::minBlockUs in the cost units the block quantum is expressed in,
// from the clocks per unit cost this step's previous execution measured. 0
// when nothing has been measured yet, which leaves the quantum at an even
// division of the step over one wave.
float minBlockCost(const StepVectors& sv) {
  const float targetUs = WaveConfig::get().minBlockUs;
  if (targetUs <= 0 || sv.clocksPerCost <= 0) {
    return 0;
  }
  const double clocks = targetUs * 1000.0 / kNsPerBlockClock;
  return static_cast<float>(clocks / sv.clocksPerCost);
}

// Splits the step into several launches when its single launch is skewed or
// occupancy-starved enough to be worth the serialization.
//
// A step with barrier ops is not refused. opBarrier needs every block of ITS
// op co-resident, which the packer already guarantees -- a barrier op is never
// split across launches and no launch is wider than a wave -- and each segment
// carries its own cooperative flag, exactly as the whole-step launch derives
// one today. This is the case that matters on the real graph: the cg trim
// below reduces a step with more ops than a wave has blocks to one block per
// op, whatever the costs say, and then launches it non-cooperatively anyway.
PartitionParams partitionParams(const StepVectors& sv) {
  const auto& config = WaveConfig::get();
  return PartitionParams{
      .maxWaves = config.maxLaunchWaves,
      .skewThreshold = config.launchSkewThreshold,
      .minSavingUs = config.launchSplitMinUs,
      .measuredIdleUs = sv.splitRejected || sv.measureUnsplit ? 0.0
          : sv.measuredMaxClocks > 0
          ? static_cast<double>(sv.measuredMaxClocks) *
              (1.0 - sv.measuredUtil) * kNsPerBlockClock / 1000.0
          : -1.0,
      .minBlockCost = minBlockCost(sv),
      .measuredUtil = sv.measuredUtil,
      .measuredBlocks = sv.measuredBlocks};
}

std::optional<LaunchPlan> maybePartition(
    const std::vector<GridOp>& ops,
    const GridDevice& device,
    const StepVectors& sv) {
  const auto& config = WaveConfig::get();
  if (config.debugSingleOps) {
    return std::nullopt;
  }
  const auto params = partitionParams(sv);
  // Sizing every step by quantum needs no skew gate: the question it answers is
  // how many blocks the work is worth, which does not depend on the step being
  // skewed. Presized, so a step that packs into one launch still gets the
  // quantum's counts rather than falling back to the pro-rata ones.
  if (config.quantumGrid) {
    const auto want = sizeByQuantum(ops, device, params);
    return partitionLaunches(ops, device, sv.gridStats, params, &want);
  }
  // A step that would launch cooperatively wider than the device holds
  // co-resident has to be split whatever the skew gate says: the driver
  // refuses such a launch outright, so leaving it whole is not a slower plan
  // but a broken one. The one-block-per-op floor alone reaches this on a step
  // with more ops than a wave has blocks -- the trim in makeGrid cannot go
  // below one block per op -- which is how a 631-op step came to ask for 631
  // blocks against a 540-block cooperative capacity.
  const bool mustSplit =
      exceedsCooperativeCapacity(ops, sv.numBlocksPerLaunch, device);
  if (!config.partitionLaunches) {
    return std::nullopt;
  }
  if (!mustSplit && !shouldPartition(sv.gridStats, params)) {
    return std::nullopt;
  }
  return partitionLaunches(ops, device, sv.gridStats, params);
}

// Turns the laid-out grid into the BlockInfo array the kernel reads. An op
// split across launches keeps one blockInOp series over the whole step, so a
// block finds its slice the same way wherever it was launched from.
void emitGrid(
    const std::vector<LaunchData>& launches,
    const LaunchPlan& plan,
    StepVectors& sv) {
  sv.blocks.resize(plan.blocks.size());
  sv.launchIndices.resize(plan.blocks.size());
  for (size_t b = 0; b < plan.blocks.size(); ++b) {
    const auto& gridBlock = plan.blocks[b];
    auto& info = sv.blocks[b];
    info.op = launches[gridBlock.op].launch->op->opCode();
    info.blockInOp = gridBlock.blockInOp;
    info.numBlocksInOp = plan.blocksPerOp.at(gridBlock.op);
    info.params = nullptr;
    sv.launchIndices[b] = gridBlock.op;
  }
}

// Prints one line per step under WaveConfig::kGrid: what the block allocator
// was working with and how balanced the grid came out.
void traceGrid(int32_t sequenceNumber, int32_t stepIdx, const StepVectors& sv) {
  const auto& stats = sv.gridStats;
  std::string text = fmt::format(
      "  grid {}.{}: ops={} target={} blocks={} cost={:.4g} skew={:.2f} starved={} occ={}/{} poison={:.2f}",
      sequenceNumber,
      stepIdx,
      stats.numOps,
      stats.targetBlocks,
      stats.totalBlocks,
      stats.totalCost,
      stats.skew,
      stats.numStarved,
      stats.occupancy,
      stats.bestOccupancy,
      stats.poison);
  if (sv.segments.size() > 1) {
    text += fmt::format(" launches={} [", sv.segments.size());
    for (size_t i = 0; i < sv.segments.size(); ++i) {
      text += fmt::format(
          "{}{}b/{}KB",
          i == 0 ? "" : " ",
          sv.segments[i].numBlocks,
          sv.segments[i].dynamicShared / 1024);
    }
    text += "]";
  }
  std::cout << text << std::endl;
}

} // namespace

int32_t makeGrid(
    std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t blockSize,
    int32_t maxBlocksPerSM,
    int32_t staticSharedPerBlock,
    const std::function<int32_t(int32_t)>& occupancyFor) {
  auto unitCost = [&](size_t i) {
    return WaveConfig::get().uniformCost ? 1.0f
                                         : launches[i].launch->op->unitCost();
  };

  // Compute cost per launch: numElements * unitCost * costAdjustFactor.
  sv.costs.resize(launches.size());
  float totalCost = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    float adjust =
        launches[i].costAdjustFactor > 0 ? launches[i].costAdjustFactor : 1.0f;
    sv.costs[i] =
        static_cast<float>(launches[i].numElements) * unitCost(i) * adjust;
    totalCost += sv.costs[i];
  }

  // Max blocks each launch could use. The block's elements-per-thread shrinks
  // from kMaxElementsPerThread to 1 as the op's per-input cost rises, so an
  // expensive op may use up to numElements / blockSize blocks while a cheap one
  // is capped at numElements / (kMaxElementsPerThread * blockSize).
  sv.maxBlocks.resize(launches.size());
  for (size_t i = 0; i < launches.size(); ++i) {
    // alwaysSingleBlock ops fold their cross-block barriers into __syncthreads
    // and are only correct when run as a single block. Cap maxBlocks at 1 so
    // neither the pro-rata assignment nor the latency-balancing pass below can
    // grow them past one block.
    if (launches[i].launch->op && launches[i].launch->op->alwaysSingleBlock()) {
      sv.maxBlocks[i] = 1;
      continue;
    }
    float adjust =
        launches[i].costAdjustFactor > 0 ? launches[i].costAdjustFactor : 1.0f;
    float costPerInput = unitCost(i) * adjust;
    int32_t elementsPerBlock =
        blockSize * elementsPerThreadForCost(costPerInput);
    sv.maxBlocks[i] = static_cast<int32_t>(
        (launches[i].numElements + elementsPerBlock - 1) / elementsPerBlock);
    if (sv.maxBlocks[i] < 1) {
      sv.maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count and kernel occupancy.
  const auto budgetDevice =
      currentGridDevice(maxBlocksPerSM, staticSharedPerBlock, occupancyFor);
  int32_t maxBlocks = budgetDevice.numSMs *
      blocksPerSM(budgetDevice, dynamicSharedBytes(launches, blockSize));
  int32_t targetBlocks = maxBlocks;

  // An op that asked for a width gets no more than numSMs times it.
  // partitionLaunchPlan caps a launch's capacity the same way, but it declines
  // to reshape a step that is already a single launch -- which is exactly the
  // shape a lone op has, and a lone expensive op is the case the preference
  // exists for -- so the cap has to be applied to the op's block ceiling here
  // as well, where every path goes through it.
  for (size_t i = 0; i < launches.size(); ++i) {
    const int32_t prefer = launches[i].launch->blocksPerSm;
    if (prefer > 0) {
      sv.maxBlocks[i] =
          std::max(1, std::min(sv.maxBlocks[i], budgetDevice.numSMs * prefer));
    }
  }

  // Assign blocks pro rata by cost, at least 1 per launch, capped by
  // maxBlocks.
  sv.numBlocksPerLaunch.resize(launches.size());
  int32_t totalAssigned = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    // Operations marked alwaysSingleBlock always get exactly one block.
    if (launches[i].launch->op && launches[i].launch->op->alwaysSingleBlock()) {
      sv.numBlocksPerLaunch[i] = 1;
      totalAssigned += 1;
      continue;
    }
    float fraction = (totalCost > 0)
        ? sv.costs[i] / totalCost
        : 1.0f / static_cast<float>(launches.size());
    int32_t assigned = std::max(
        1,
        static_cast<int32_t>(
            fraction * static_cast<float>(targetBlocks) + 0.5f));
    assigned = std::min(assigned, sv.maxBlocks[i]);
    // Round down blocks so all but the last process a multiple of blockSize
    // elements.
    if (assigned > 1) {
      auto elemsPerBlock = (launches[i].numElements + assigned - 1) / assigned;
      auto alignedElems =
          roundUp(elemsPerBlock, static_cast<int64_t>(blockSize));
      assigned = std::max(
          1,
          static_cast<int32_t>(
              (launches[i].numElements + alignedElems - 1) / alignedElems));
    }
    sv.numBlocksPerLaunch[i] = assigned;
    totalAssigned += assigned;
  }

  // A width is a property of a LAUNCH -- blocks resident per SM -- not of one
  // op's share of it, so the cap is on the total blocks of the ops that asked
  // for the same width, not on each one. One opcode can be several launches in
  // a step: op 107 on the ROO graph is five launches of one KernelOperation,
  // each well under numSMs times its preference and together well over it.
  //
  // Done here rather than only in partitionLaunchPlan because that declines to
  // reshape a step it would leave as a single launch, which is the common case
  // for exactly these ops, and its capacity is a packing hint for an ordinary
  // launch rather than a bound.
  if (!WaveConfig::get().preferBlocksPerSm.empty()) {
    std::map<int32_t, int32_t> assignedPerWidth;
    for (size_t i = 0; i < launches.size(); ++i) {
      const int32_t prefer = launches[i].launch->blocksPerSm;
      if (prefer > 0) {
        assignedPerWidth[prefer] += sv.numBlocksPerLaunch[i];
      }
    }
    for (const auto& [prefer, assigned] : assignedPerWidth) {
      const int32_t budget = std::max(1, budgetDevice.numSMs * prefer);
      if (assigned <= budget) {
        continue;
      }
      // Scale the members down together, then trim the largest until the sum
      // fits: proportional scaling alone cannot land exactly, and rounding up
      // anywhere would leave the launch wider than was asked for.
      const float scale =
          static_cast<float>(budget) / static_cast<float>(assigned);
      int32_t sum = 0;
      for (size_t i = 0; i < launches.size(); ++i) {
        if (launches[i].launch->blocksPerSm != prefer) {
          continue;
        }
        auto& blocks = sv.numBlocksPerLaunch[i];
        totalAssigned -= blocks;
        blocks = std::max(
            1, static_cast<int32_t>(static_cast<float>(blocks) * scale));
        sum += blocks;
      }
      while (sum > budget) {
        int32_t widest = -1;
        for (size_t i = 0; i < launches.size(); ++i) {
          if (launches[i].launch->blocksPerSm == prefer &&
              sv.numBlocksPerLaunch[i] > 1 &&
              (widest < 0 ||
               sv.numBlocksPerLaunch[i] > sv.numBlocksPerLaunch[widest])) {
            widest = static_cast<int32_t>(i);
          }
        }
        if (widest < 0) {
          // Every member is down to one block; the width cannot be honoured
          // without dropping an op, so leave it at one apiece.
          break;
        }
        --sv.numBlocksPerLaunch[widest];
        --sum;
      }
      totalAssigned += sum;
    }
  }

  // For cooperative grids, cap total blocks at what the GPU can run
  // concurrently.
  if (sv.isCgGrid && totalAssigned > targetBlocks) {
    // Trim one block at a time from the launch that loses least by it: the one
    // whose per-block cost stays lowest with a block fewer. Trimming the
    // launches with the most blocks instead levels them regardless of cost; on
    // a step where 251 one-block copies took half the wave, that cut a 3.9M
    // element gather to the 34 blocks of ops with a third of its work.
    while (totalAssigned > targetBlocks) {
      int32_t best = -1;
      float bestLat = 0;
      for (size_t i = 0; i < launches.size(); ++i) {
        const int32_t blocks = sv.numBlocksPerLaunch[i];
        if (blocks <= 1) {
          continue;
        }
        const float lat = sv.costs[i] / static_cast<float>(blocks - 1);
        if (best < 0 || lat < bestLat) {
          best = static_cast<int32_t>(i);
          bestLat = lat;
        }
      }
      if (best < 0) {
        break;
      }
      --sv.numBlocksPerLaunch[best];
      --totalAssigned;
    }
  }

  // Balance projected latency: move blocks from the largest-blocked op to the
  // highest-latency op when the highest-latency op has fewer blocks.
  if (launches.size() > 1) {
    for (int32_t pass = 0; pass < 20; ++pass) {
      int32_t highLatIdx = -1;
      float highLat = 0;
      int32_t donorIdx = -1;
      float donorLat = 0;
      int32_t donorBlocks = 0;
      for (size_t i = 0; i < launches.size(); ++i) {
        if (sv.numBlocksPerLaunch[i] <= 0) {
          continue;
        }
        float lat = sv.costs[i] / static_cast<float>(sv.numBlocksPerLaunch[i]);
        if (lat > highLat) {
          highLat = lat;
          highLatIdx = static_cast<int32_t>(i);
        }
      }
      if (highLatIdx < 0) {
        break;
      }
      for (size_t i = 0; i < launches.size(); ++i) {
        if (static_cast<int32_t>(i) == highLatIdx ||
            sv.numBlocksPerLaunch[i] <= 1) {
          continue;
        }
        float lat = sv.costs[i] / static_cast<float>(sv.numBlocksPerLaunch[i]);
        if (sv.numBlocksPerLaunch[i] > donorBlocks ||
            (sv.numBlocksPerLaunch[i] == donorBlocks && lat < donorLat)) {
          donorIdx = static_cast<int32_t>(i);
          donorLat = lat;
          donorBlocks = sv.numBlocksPerLaunch[i];
        }
      }
      if (donorIdx < 0 || donorLat >= highLat) {
        break;
      }
      // Check if moving a block actually helps: the donor's new latency
      // must stay below the receiver's new latency.
      float newHighLat = sv.costs[highLatIdx] /
          static_cast<float>(sv.numBlocksPerLaunch[highLatIdx] + 1);
      float newDonorLat = sv.costs[donorIdx] /
          static_cast<float>(sv.numBlocksPerLaunch[donorIdx] - 1);
      if (newDonorLat >= highLat || newHighLat >= highLat * 0.95f) {
        break;
      }
      if (sv.numBlocksPerLaunch[highLatIdx] >= sv.maxBlocks[highLatIdx]) {
        break;
      }
      ++sv.numBlocksPerLaunch[highLatIdx];
      --sv.numBlocksPerLaunch[donorIdx];
    }
  }

  // Record expected fraction for cost adjustment feedback.
  for (size_t i = 0; i < launches.size(); ++i) {
    launches[i].expectedFraction = totalCost > 0 ? sv.costs[i] / totalCost : 0;
  }

  const auto device =
      currentGridDevice(maxBlocksPerSM, staticSharedPerBlock, occupancyFor);
  const auto gridOps = describeGridOps(launches, sv);
  // Measured on the grid above, i.e. on the single launch the step would run
  // as today. That is the opportunity the partitioning gate reads, so it must
  // not describe the split the gate went on to make.
  sv.gridStats = gridStats(gridOps, sv.numBlocksPerLaunch, device);

  auto plan = maybePartition(gridOps, device, sv);
  if (plan.has_value()) {
    sv.numBlocksPerLaunch = plan->blocksPerOp;
    // With no gate to feed, the reported grid should describe what ran rather
    // than the pro-rata split that was discarded. util and starved are how the
    // sizing is judged, so they have to be measured on it.
    if (WaveConfig::get().quantumGrid) {
      const auto segments = sv.gridStats.numSegments;
      sv.gridStats = gridStats(gridOps, sv.numBlocksPerLaunch, device);
      sv.gridStats.numSegments = segments;
    }
  } else {
    plan = singleLaunchPlan(
        gridOps, sv.numBlocksPerLaunch, WaveConfig::get().orderBlocksByCost);
  }
  emitGrid(launches, *plan, sv);
  sv.segments = std::move(plan->segments);
  sv.gridStats.numSegments = static_cast<int32_t>(sv.segments.size());
  return blockSize;
}

// --- Launch ---

namespace {

// Marks every launch in 'sv' with LaunchData::d2hProducer: the step whose
// device-to-host transfer produced a value it reads, i.e. the step whose
// transfer has to have landed before this launch can be sized, allocated or
// have its params filled. Runs on every step, not just under kTiming -- the
// marks are what a deferred D2H wait is driven by; the kTiming counters are
// tallied from them separately.
//
// Shortcuts have to be followed transitively. A metadata-only op (a view, a
// slice, a ListPack) runs on the host inside this step and its result goes
// into a fused kernel's param descriptor, so a kernel can reach a returned
// scalar without naming it: returned size -> shortcut -> param. Shortcuts are
// walked first, in execution order, and a dependent one taints its own outputs
// so the chain is followed however long it is.
void markD2hDependencies(ExecutionState& state, StepVectors& sv) {
  // LaunchData is pooled across runs, so a mark left by the previous run would
  // otherwise stand in for this one.
  for (auto* launches :
       {&sv.shortcutStandalones, &sv.kernels, &sv.standalones}) {
    for (auto& data : *launches) {
      data.d2hProducer = -1;
    }
  }
  if (state.returnedIds.empty()) {
    return;
  }
  // Producing step of the nearest returned value an op reads, or -1.
  auto producerOf = [&](const std::vector<nativert::ValueId>& ids) {
    int32_t best = -1;
    for (auto id : ids) {
      best = std::max(best, state.returnedAtStep(id));
    }
    return best;
  };

  // Shortcuts first: they run before the kernel and feed its params.
  folly::F14FastMap<nativert::ValueId, int32_t> tainted;
  for (auto& data : sv.shortcutStandalones) {
    auto producer = producerOf(data.actualInputs);
    for (auto id : data.actualInputs) {
      auto it = tainted.find(id);
      if (it != tainted.end() && it->second > producer) {
        producer = it->second;
      }
    }
    if (producer < 0) {
      continue;
    }
    data.d2hProducer = producer;
    for (auto id : data.actualOutputs) {
      auto& slot = tainted[id];
      slot = std::max(slot, producer);
    }
  }

  auto opProducer = [&](const LaunchData& data) {
    int32_t producer = producerOf(data.actualInputs);
    producer = std::max(producer, producerOf(data.tensorsInFrame));
    producer = std::max(producer, producerOf(data.scalarsInFrame));
    for (const auto& ids :
         {data.actualInputs, data.tensorsInFrame, data.scalarsInFrame}) {
      for (auto id : ids) {
        auto it = tainted.find(id);
        if (it != tainted.end() && it->second > producer) {
          producer = it->second;
        }
      }
    }
    for (auto id : data.viewReadIds) {
      producer = std::max(producer, state.returnedAtStep(id));
      auto tt = tainted.find(id);
      if (tt != tainted.end() && tt->second > producer) {
        producer = tt->second;
      }
    }
    return producer;
  };
  for (auto* launches : {&sv.kernels, &sv.standalones}) {
    for (auto& data : *launches) {
      data.d2hProducer = opProducer(data);
    }
  }
}

// Tallies the marks markD2hDependencies left into the per-step report counters:
// how much of the step is blocked on a transfer, and how far back the nearest
// producer is. kTiming only.
void countD2hDependencies(ExecutionState& state, StepVectors& sv) {
  sv.d2hDepFused = 0;
  sv.d2hDepStandalone = 0;
  sv.d2hDepShortcut = 0;
  sv.d2hDepOnPrevStep = 0;
  sv.d2hNearestProducer = -1;
  sv.viewNodeDescs = 0;
  auto note = [&](int32_t producer) {
    auto distance = state.executedSteps - producer;
    if (sv.d2hNearestProducer < 0 || distance < sv.d2hNearestProducer) {
      sv.d2hNearestProducer = distance;
    }
    if (distance == 1) {
      ++sv.d2hDepOnPrevStep;
    }
  };
  auto tally = [&](const std::vector<LaunchData>& launches, int32_t& counter) {
    for (const auto& data : launches) {
      if (data.d2hProducer >= 0) {
        ++counter;
        note(data.d2hProducer);
      }
    }
  };
  tally(sv.shortcutStandalones, sv.d2hDepShortcut);
  tally(sv.kernels, sv.d2hDepFused);
  tally(sv.standalones, sv.d2hDepStandalone);
  for (const auto& data : sv.kernels) {
    for (const auto& desc : data.actualOutputDescs) {
      if (desc.viewNode != nullptr) {
        ++sv.viewNodeDescs;
      }
    }
  }
}

// Registers the values this step sends back, so later steps can report that
// they depend on them.
void recordD2hProducers(ExecutionState& state, const StepVectors& sv) {
  auto record = [&](nativert::ValueId id) {
    if (static_cast<size_t>(id) >= state.returnedAt.size()) {
      state.returnedAt.resize(id + 1, -1);
    }
    if (state.returnedAt[id] < 0) {
      state.returnedIds.push_back(id);
    }
    state.returnedAt[id] = state.executedSteps;
  };
  for (const auto& data : sv.kernels) {
    for (auto id : data.returnValues) {
      record(id);
    }
  }
  for (const auto& copy : sv.hostCopies) {
    record(copy.id);
  }
}

} // namespace

void setViewReadIds(LaunchData& data) {
  data.viewReadIds.clear();
  for (const auto& desc : data.actualOutputDescs) {
    if (desc.viewNode == nullptr) {
      continue;
    }
    const auto* viewMeta = Registry::metadata(desc.viewNode->target());
    int32_t baseOrdinal = (viewMeta && viewMeta->viewOfArg.has_value())
        ? *viewMeta->viewOfArg
        : -1;
    const auto& viewInputs = desc.viewNode->inputs();
    for (size_t k = 0; k < viewInputs.size(); ++k) {
      if (static_cast<int32_t>(k) != baseOrdinal &&
          viewInputs[k].value != nullptr) {
        data.viewReadIds.push_back(viewInputs[k].value->id());
      }
    }
  }
}

namespace {

// Whether 'node' is a new_* factory whose constant size has no elements. It
// needs no kernel: its output is an empty tensor of the right dtype.
bool isEmptyFactory(NodeCP node) {
  static const folly::F14FastSet<std::string_view> kFactories{
      "torch.ops.aten.new_ones.default",
      "torch.ops.aten.new_zeros.default",
      "torch.ops.aten.new_full.default",
      "torch.ops.aten.new_empty.default",
  };
  if (kFactories.count(node->target()) == 0) {
    return false;
  }
  const auto* size = node->tryGetAttribute("size");
  if (size == nullptr ||
      !std::holds_alternative<std::vector<int64_t>>(size->value)) {
    return false;
  }
  const auto& dims = std::get<std::vector<int64_t>>(size->value);
  return std::find(dims.begin(), dims.end(), 0) != dims.end();
}

// Maps a metadata-only standalone op's target to its host-side shortcut.
StandaloneShortcut standaloneShortcutForTarget(std::string_view target) {
  if (target == "prim.ListPack") {
    return StandaloneShortcut::kListPack;
  }
  if (target == "torch.ops.aten.view.default") {
    return StandaloneShortcut::kView;
  }
  if (target == "torch.ops.aten.slice.Tensor") {
    return StandaloneShortcut::kSlice;
  }
  if (target == "torch.ops.aten.select.int") {
    return StandaloneShortcut::kSelectInt;
  }
  if (target == "torch.ops.aten.unsqueeze.default") {
    return StandaloneShortcut::kUnsqueeze;
  }
  if (target == "torch.ops.aten.transpose.int") {
    return StandaloneShortcut::kTranspose;
  }
  if (target == "torch.ops.aten.narrow.default") {
    return StandaloneShortcut::kNarrow;
  }
  if (target == "prim.ListUnpack") {
    return StandaloneShortcut::kListUnpack;
  }
  if (target == "torch.ops.aten.unbind.int") {
    return StandaloneShortcut::kUnbind;
  }
  if (target == "torch.ops.aten.split_with_sizes.default") {
    return StandaloneShortcut::kSplitWithSizes;
  }
  if (target == "torch.ops.aten.squeeze.dim") {
    return StandaloneShortcut::kSqueezeDim;
  }
  if (target == "torch.ops.aten.expand.default") {
    return StandaloneShortcut::kExpand;
  }
  if (target == "torch.ops.aten.sym_size.int") {
    return StandaloneShortcut::kSymSize;
  }
  if (target == "torch.ops.aten.sym_numel.default") {
    return StandaloneShortcut::kSymNumel;
  }
  return StandaloneShortcut::kNone;
}
} // namespace

Launch::Launch(
    NodeCP standaloneNode,
    const ValueTypes& types,
    WaveGraph& waveGraph)
    : standalone(standaloneNode) {
  standaloneShortcut = standaloneShortcutForTarget(standaloneNode->target());
  // prim.ListPack is metadata-only only when it builds a TensorList; a SymInt /
  // int list packs scalars, which the kListPack shortcut cannot handle, so
  // leave those on the generic path. The same holds for the unpack: its input
  // is the list, so the type to check is the operand's, not the output's.
  if (standaloneShortcut == StandaloneShortcut::kListPack &&
      (standaloneNode->outputs().empty() ||
       standaloneNode->outputs()[0]->type().kind() !=
           nativert::Type::Kind::TensorList)) {
    standaloneShortcut = StandaloneShortcut::kNone;
  }
  if (standaloneShortcut == StandaloneShortcut::kListUnpack &&
      (standaloneNode->inputs().empty() ||
       standaloneNode->inputs()[0].value->type().kind() !=
           nativert::Type::Kind::TensorList)) {
    standaloneShortcut = StandaloneShortcut::kNone;
  }
  if (isEmptyFactory(standaloneNode)) {
    standaloneShortcut = StandaloneShortcut::kEmptyFactory;
    metadataOnly = true;
    return;
  }
  auto* meta = Registry::metadata(standaloneNode->target());
  // prim.ListPack has no registry entry but is metadata-only by definition;
  // every other op's metadata-only status comes from its Metadata.
  metadataOnly = meta ? meta->metadataOnly
                      : (standaloneNode->target() == "prim.ListPack" ||
                         standaloneNode->target() == "prim.ListUnpack");
  if (!meta || meta->argumentMeta.empty()) {
    return;
  }
  for (size_t i = 0;
       i < meta->argumentMeta.size() && i < standaloneNode->inputs().size();
       ++i) {
    if (!meta->argumentMeta[i].cpuOnly ||
        standaloneNode->inputs()[i].value->type().kind() !=
            nativert::Type::Kind::Tensor) {
      continue;
    }
    auto* deviceValue = standaloneNode->inputs()[i].value;
    auto dtype = c10::ScalarType::Long;
    auto id = deviceValue->id();
    if (static_cast<size_t>(id) < types.types.size() && types.types[id]) {
      dtype = types.types[id]->dtype();
    }
    auto* cpuValue = waveGraph.newTensorValue(
        waveGraph.placeholderNode(), waveGraph.uniqueName("cpu_copy"), dtype);
    argOnDevice.push_back(deviceValue);
    argOnCpu.push_back(cpuValue);
  }
}

std::string Launch::toString(Listing mode) const {
  if (op) {
    return "kernel: " + op->toString();
  }
  if (standalone) {
    Subgraph sg;
    sg.root = standalone;
    sg.inputs = inputValues(standalone);
    return "standalone " + sg.toString(mode);
  }
  return "";
}

// --- ProjectOperation ---

ProjectOperation::ProjectOperation(const Subgraph& sg) : subgraph_(sg) {}

// --- CompositeInvocation ---

CompositeInvocation::CompositeInvocation(
    std::unique_ptr<CompositeKernel> kernel,
    std::vector<OpInvocation> ops,
    std::deque<c10::IValue> ivalueStorage,
    int32_t sequenceNumber,
    std::vector<nativert::ValueId> lastUseIds,
    std::vector<std::vector<int32_t>> lastUseReaderOps,
    std::vector<nativert::ValueId> reusableIds,
    std::vector<Launch> prePassStandalones,
    std::vector<std::pair<nativert::ValueId, int32_t>> elidedCloneInputs)
    : kernel_(std::move(kernel)),
      ops_(std::move(ops)),
      ivalueStorage_(std::move(ivalueStorage)),
      sequenceNumber_(sequenceNumber),
      lastUseIds_(std::move(lastUseIds)),
      lastUseReaderOps_(std::move(lastUseReaderOps)),
      reusableIds_(reusableIds.begin(), reusableIds.end()),
      prePassStandalones_(std::move(prePassStandalones)),
      elidedCloneInputs_(std::move(elidedCloneInputs)) {}

CompositeInvocation::~CompositeInvocation() = default;

namespace {

void printLaunchGrid(
    std::stringstream& ss,
    const LaunchGrid& grid,
    const char* heading,
    Listing /*mode*/) {
  if (heading) {
    ss << heading << "\n";
  }
  for (size_t step = 0; step < grid.size(); ++step) {
    ss << "Step" << step << "\n";
    for (size_t lane = 0; lane < grid[step].size(); ++lane) {
      ss << "  Lane " << lane << ": ";
      const auto& launch = grid[step][lane];
      if (launch.op) {
        ss << launch.op->toString();
        ss << "    Params: (";
        for (size_t i = 0; i < launch.values.size(); ++i) {
          if (i > 0) {
            ss << ", ";
          }
          ss << "%%" << launch.values[i]->id();
        }
        ss << ")\n";
        if (!launch.constantIndices.empty()) {
          ss << "    ConstantIndices: (";
          for (size_t i = 0; i < launch.constantIndices.size(); ++i) {
            if (i > 0) {
              ss << ", ";
            }
            ss << launch.constantIndices[i];
          }
          ss << ")\n";
        }
      } else if (launch.standalone) {
        ss << "Standalone: " << standaloneToString(launch.standalone) << "\n";
      }
    }
  }
}

} // namespace

std::string ProjectOperation::toString(Listing mode) const {
  std::stringstream ss;
  printLaunchGrid(ss, grid_, nullptr, mode);
  if (!singleBlockGrid_.empty()) {
    printLaunchGrid(ss, singleBlockGrid_, "Single Block Variant", mode);
  }
  return ss.str();
}

namespace {

// Walks formal and actual subgraphs in parallel, calling 'visitor' for each
// matched (formal, actual) node pair. Handles view producers at subgraph
// boundaries.
template <typename Visitor>
void walkSubgraphPairs(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    Visitor&& visitor) {
  std::unordered_set<ValueCP> formalInputSet(
      formalSg.inputs.begin(), formalSg.inputs.end());
  std::unordered_set<NodeCP> visited;
  std::function<void(NodeCP, NodeCP)> walk = [&](NodeCP formalNode,
                                                 NodeCP actualNode) {
    if (!visited.insert(formalNode).second) {
      return;
    }
    visitor(formalNode, actualNode);
    const auto& fi = formalNode->inputs();
    const auto& ai = actualNode->inputs();
    for (size_t i = 0; i < fi.size(); ++i) {
      if (formalInputSet.count(fi[i].value)) {
        auto* fp = fi[i].value->producer();
        if (fp) {
          auto* meta = Registry::metadata(fp->target());
          if (meta && meta->isView()) {
            auto* ap = ai[i].value->producer();
            TORCH_CHECK(
                ap && Registry::metadata(ap->target()) &&
                    Registry::metadata(ap->target())->isView(),
                "Formal input has view producer but actual does not");
            visitor(fp, ap);
          }
        }
        continue;
      }
      auto* fp = fi[i].value->producer();
      auto* ap = ai[i].value->producer();
      if (fp && ap) {
        walk(fp, ap);
      }
    }
  };
  if (formalSg.root && actualSg.root) {
    walk(formalSg.root, actualSg.root);
  }
}

void makeNodeMap(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    NodeMap& nodeMap) {
  walkSubgraphPairs(formalSg, actualSg, [&](NodeCP formal, NodeCP actual) {
    nodeMap.emplace(formal, actual);
  });
}

FormalToActual makeSubgraphBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg) {
  TORCH_CHECK(
      formalSg.inputs.size() == actualSg.inputs.size(),
      "Input count mismatch: formal=",
      formalSg.inputs.size(),
      " actual=",
      actualSg.inputs.size());
  FormalToActual bindings;
  for (size_t i = 0; i < formalSg.inputs.size(); ++i) {
    bindings[formalSg.inputs[i]->id()] = actualSg.inputs[i]->id();
  }
  walkSubgraphPairs(
      formalSg, actualSg, [&](NodeCP formalNode, NodeCP actualNode) {
        const auto& fo = formalNode->outputs();
        const auto& ao = actualNode->outputs();
        TORCH_CHECK(
            fo.size() == ao.size(),
            "Output count mismatch at node ",
            formalNode->target());
        for (size_t i = 0; i < fo.size(); ++i) {
          bindings[fo[i]->id()] = ao[i]->id();
          if (fo[i]->type().kind() == nativert::Type::Kind::TensorList) {
            auto formalElems = fo[i]->getListElements();
            auto actualElems = ao[i]->getListElements();
            for (size_t j = 0; j < formalElems.size() && j < actualElems.size();
                 ++j) {
              bindings[formalElems[j]->id()] = actualElems[j]->id();
            }
          }
        }
      });
  return bindings;
}

} // namespace

OpInvocation::OpInvocation(
    ProjectOperation* projectOp,
    const Subgraph& sg,
    std::deque<c10::IValue>& storage)
    : projectOp_{projectOp} {
  const auto& formalSg = projectOp->subgraph();
  bindings_ = makeSubgraphBindings(formalSg, sg);
  constants_ = listConstants(sg, storage);
  makeNodeMap(formalSg, sg, nodeMap_);
}

std::string printNodeMap(const NodeMap& nodeMap) {
  std::stringstream ss;
  for (auto& [formal, actual] : nodeMap) {
    ss << "  " << static_cast<const void*>(formal) << " "
       << standaloneToString(formal) << "\n    -> "
       << static_cast<const void*>(actual) << " " << standaloneToString(actual)
       << "\n";
  }
  return ss.str();
}

std::string OpInvocation::toString() const {
  std::stringstream ss;
  ss << "OpInvocation bindings:\n";
  std::vector<std::pair<int32_t, int32_t>> sortedBindings(
      bindings_.begin(), bindings_.end());
  std::sort(sortedBindings.begin(), sortedBindings.end());
  for (auto& [formal, actual] : sortedBindings) {
    ss << "  %" << formal << " -> %" << actual << "\n";
  }
  ss << "OpInvocation nodeMap:\n" << printNodeMap(nodeMap_);
  return ss.str();
}

// --- LaunchData ---

namespace {
// Collects a metadata shortcut op's operands from 'node' into 'data' in c10
// schema order (first-to-last for prim.ListPack, which has no schema). A value
// operand goes in args; an integer constant goes in intArgs at the same
// position with a nullptr in args; an all-integer list operand (e.g. aten.view
// size) goes in intList for direct pass-through.
void collectShortcutArgs(NodeCP node, LaunchData& data) {
  auto pushValue = [&](ValueCP v) {
    data.args.push_back(v);
    data.intArgs.push_back(0);
  };
  auto pushInt = [&](int64_t c) {
    data.args.push_back(nullptr);
    data.intArgs.push_back(c);
  };
  const auto* meta = Registry::metadata(node->target());
  if (meta != nullptr && meta->functionSchema != nullptr) {
    for (const auto& arg : meta->functionSchema->arguments()) {
      if (const auto* in = node->tryGetInput(arg.name())) {
        pushValue(in->value);
      } else if (const auto* attr = node->tryGetAttribute(arg.name())) {
        if (std::holds_alternative<int64_t>(attr->value)) {
          pushInt(std::get<int64_t>(attr->value));
        } else if (std::holds_alternative<std::vector<int64_t>>(attr->value)) {
          const auto& vec = std::get<std::vector<int64_t>>(attr->value);
          data.intList.assign(vec.begin(), vec.end());
          pushValue(nullptr);
        } else {
          pushValue(nullptr);
        }
      } else {
        pushValue(nullptr);
      }
    }
  } else {
    for (auto& input : node->inputs()) {
      pushValue(input.value);
    }
  }
}

} // namespace

// The view recipe for 'node', or null when no shortcut covers its op or one of
// its outputs is not a tensor.
std::shared_ptr<const ViewRecipe> makeViewRecipe(NodeCP node) {
  const auto kind = standaloneShortcutForTarget(node->target());
  if (kind == StandaloneShortcut::kNone ||
      kind == StandaloneShortcut::kListPack ||
      kind == StandaloneShortcut::kListUnpack ||
      kind == StandaloneShortcut::kSymSize ||
      kind == StandaloneShortcut::kSymNumel || node->outputs().size() != 1 ||
      node->outputs()[0]->type().kind() != nativert::Type::Kind::Tensor) {
    return nullptr;
  }
  auto recipe = std::make_shared<ViewRecipe>();
  recipe->launch.standalone = node;
  recipe->launch.standaloneShortcut = kind;
  recipe->launch.metadataOnly = true;
  auto& data = recipe->data;
  data.launch = &recipe->launch;
  data.standalone = node;
  for (auto& input : node->inputs()) {
    data.actualInputs.push_back(input.value->id());
  }
  data.actualOutputs.push_back(node->outputs()[0]->id());
  collectShortcutArgs(node, data);
  return recipe;
}

LaunchData::LaunchData(
    const Launch& launch,
    OpInvocation& op,
    const IdToValueMap& idToValue)
    : launch(&launch), invocation(&op), numElements(0) {
  const auto& bindings = op.bindings();

  auto translateId = [&](ValueCP formal) -> nativert::ValueId {
    auto it = bindings.find(formal->id());
    auto actualId = it != bindings.end() ? it->second : formal->id();
    return actualId;
  };

  if (!launch.op) {
    // Standalone: translate node via nodeMap, inputs and outputs via bindings.
    auto nodeIt = op.nodeMap().find(launch.standalone);
    standalone =
        nodeIt != op.nodeMap().end() ? nodeIt->second : launch.standalone;
    for (auto& input : launch.standalone->inputs()) {
      actualInputs.push_back(translateId(input.value));
    }
    for (auto* output : launch.standalone->outputs()) {
      actualOutputs.push_back(translateId(output));
    }
    // For a metadata-only shortcut op, collect its operands from the actual
    // node in c10 schema order (first-to-last for prim.ListPack, which has no
    // schema). A value operand goes in args; an integer constant goes in
    // intArgs at the same position with a nullptr in args; an all-integer list
    // operand (e.g. aten.view size) goes in intList for direct pass-through.
    if (launch.standaloneShortcut != StandaloneShortcut::kNone) {
      collectShortcutArgs(standalone, *this);
    }
  } else {
    // Kernel op: translate sizeExpr, inputs, outputs, and output descs.
    auto* kernelOp = launch.op;

    // launch.values stands in for the op's own parameters, position by
    // position. They are the same for all but one case: the copies that fill a
    // wide concat's bands all run one kernel op, generated once, and each
    // launch names the source it reads and the band it writes. Falls back to
    // the op's own when a launch did not set them.
    const auto& opInputs = kernelOp->orderedInputs();
    const auto& orderedInputs =
        launch.values.size() == opInputs.size() ? launch.values : opInputs;
    auto nInputs = kernelOp->numInputs();

    // The size expression names the op's formals, and every tensor leaf of one
    // IS an input: makeDeepSizeExpr either walks the elementwise subgraph as
    // far as its inputs, or lists orderedInputs outright. So it has to be
    // translated through whatever the inputs were -- the launch's own values
    // where it set them, not the invocation's bindings alone. The two agree for
    // every op that is not shared across launches with different parameters,
    // which is why the bindings sufficed until the concat copies arrived.
    //
    // Getting this wrong is silent, not fatal: the formal is a real graph value
    // with a live frame entry, so the grid comes out sized for whichever
    // operand the op was GENERATED against. The kernel still copies the right
    // data -- the device loop runs to the true size -- on a grid that can be
    // orders of magnitude too small. On the ROO graph it sized 414 copies at
    // 256 elements against 96.8M actual, one of them 2.87M on a single block.
    bool perLaunchValues = launch.values.size() == opInputs.size();
    if (perLaunchValues) {
      perLaunchValues = false;
      for (size_t i = 0; i < opInputs.size(); ++i) {
        if (launch.values[i] != opInputs[i]) {
          perLaunchValues = true;
          break;
        }
      }
    }
    // Built only when a launch actually renamed something: copying the
    // invocation's bindings for every launch of every op would cost more than
    // the concat copies it exists for.
    FormalToActual perLaunchBindings;
    if (perLaunchValues) {
      perLaunchBindings = bindings;
      for (size_t i = 0; i < opInputs.size(); ++i) {
        perLaunchBindings[opInputs[i]->id()] = translateId(launch.values[i]);
      }
    }
    const FormalToActual& sizeBindings =
        perLaunchValues ? perLaunchBindings : bindings;

    sizeExpr = kernelOp->sizeExpr().toActual(sizeBindings, idToValue);
    for (int32_t i = 0; i < nInputs; ++i) {
      actualInputs.push_back(translateId(orderedInputs[i]));
    }
    for (size_t i = nInputs; i < orderedInputs.size(); ++i) {
      actualOutputs.push_back(translateId(orderedInputs[i]));
    }

    const auto& outputDescs = kernelOp->outputDescs();
    for (size_t i = 0; i < outputDescs.size(); ++i) {
      const auto& desc = outputDescs[i];
      OutputDesc actualDesc = desc;
      actualDesc.sizeExpr = desc.sizeExpr.toActual(sizeBindings, idToValue);
      if (desc.viewNode) {
        auto viewIt = op.nodeMap().find(desc.viewNode);
        TORCH_CHECK(
            viewIt != op.nodeMap().end(), "View node not found in nodeMap");
        actualDesc.viewNode = viewIt->second;
      }
      if (desc.aliasSelfId) {
        auto it = bindings.find(*desc.aliasSelfId);
        actualDesc.aliasSelfId =
            it != bindings.end() ? it->second : *desc.aliasSelfId;
      }
      // Non-tensor outputs (scalars, SymInt, etc.) must be read back to host.
      auto outputValueId = actualOutputs[i];
      auto outputValueIt = idToValue.find(outputValueId);
      TORCH_CHECK(
          outputValueIt != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputValueId);
      if (outputValueIt->second->type().kind() !=
          nativert::Type::Kind::Tensor) {
        actualDesc.neededOnHost = true;
      }
      if (actualDesc.shapeSetOnDevice || actualDesc.neededOnHost) {
        returnValues.push_back(actualOutputs[i]);
        returnTypes.push_back(outputValueIt->second->type().kind());
      }
      actualOutputDescs.push_back(std::move(actualDesc));
    }

    // Record the type kind from each output Value.
    for (size_t i = 0; i < actualOutputs.size(); ++i) {
      auto outputId = actualOutputs[i];
      auto it = idToValue.find(outputId);
      TORCH_CHECK(
          it != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputId);
      actualOutputTypes.push_back(it->second->type().kind());
    }
  }
}

// --- CompositeKernel ---

CompositeKernel::CompositeKernel(
    std::vector<std::unique_ptr<ProjectOperation>>&& ops,
    std::vector<std::unique_ptr<KernelOperation>>&& kernelOps,
    const std::unordered_set<std::string>& includes,
    int32_t kernelId)
    : ops_(std::move(ops)), kernelOpStorage_(std::move(kernelOps)) {
  auto kernelName = "torchwave" + std::to_string(kernelId);
  const auto entryPoint = "torch::wave::" + kernelName;
  // Each op's case.
  std::vector<std::string> caseTexts;
  {
    WithPrintOptions printGuard("D3,L4,S");
    for (const auto& kop : kernelOpStorage_) {
      std::stringstream cs;
      cs << "    // " << kop->label() << "\n";
      auto opStr = kop->toString();
      std::istringstream lines(opStr);
      std::string line;
      while (std::getline(lines, line)) {
        cs << "    // " << line << "\n";
      }
      cs << "    case " << kop->opCode() << ": {\n";

      // Build parallel paramOffsets/outputOffsets/altOffsets arrays from
      // ElementExprs and remaining orderedInputs.
      std::vector<int32_t> paramOffs;
      std::vector<int32_t> outputOffs;
      std::vector<int32_t> altOffs;
      std::unordered_set<ValueCP> inElementExpr;

      for (const auto& ee : kop->elementExprs()) {
        int32_t outputOff = kop->paramOffset(ee.output);
        for (auto* v : ee.inputs) {
          if (v->type().kind() != nativert::Type::Kind::Tensor) {
            continue;
          }
          inElementExpr.insert(v);
          const auto leafOff = kop->paramOffset(v);
          const auto refIt = kop->broadcastRefs().find(leafOff);
          paramOffs.push_back(leafOff);
          outputOffs.push_back(
              refIt != kop->broadcastRefs().end() ? refIt->second : outputOff);
          auto ait = ee.altParamOffset.find(v);
          altOffs.push_back(ait != ee.altParamOffset.end() ? ait->second : -1);
        }
        if (ee.output->type().kind() == nativert::Type::Kind::Tensor) {
          inElementExpr.insert(ee.output);
          paramOffs.push_back(outputOff);
          outputOffs.push_back(outputOff);
          altOffs.push_back(-1);
        }
      }

      for (const auto* value : kop->orderedInputs()) {
        if (value->type().kind() == nativert::Type::Kind::Tensor &&
            !inElementExpr.count(value)) {
          auto off = kop->paramOffset(value);
          paramOffs.push_back(off);
          outputOffs.push_back(off);
          altOffs.push_back(-1);
        }
      }

      // Gather-op tensors (index_select/repeat output and whole-tensor
      // operands) need their own-dims index calculators, not the broadcast
      // ones init<true>() would compute; both share Tensor::status, so keep
      // these offsets out of the broadcast loop and force-init them separately.
      const auto& ownDims = kop->ownDimsCalcOffsets();
      std::vector<int32_t> normalParam, normalOutput, normalAlt, gatherParam;
      std::unordered_set<int32_t> seenGather;
      for (size_t i = 0; i < paramOffs.size(); ++i) {
        bool isOwnDims = ownDims.count(paramOffs[i]) > 0;
        if (isOwnDims && seenGather.insert(paramOffs[i]).second) {
          gatherParam.push_back(paramOffs[i]);
        }
        // An entry with an alt slot broadcast-inits that copy, not the primary,
        // and the two do not share 'status', so an own-dims primary is no
        // reason to skip it. Skipping left the alt Tensor at the host's
        // kUninited zero fill -- null storage, and the expression reading
        // through it faulted.
        if (altOffs.at(i) == -1 && isOwnDims) {
          continue;
        }
        normalParam.push_back(paramOffs[i]);
        normalOutput.push_back(outputOffs.at(i));
        normalAlt.push_back(altOffs.at(i));
      }

      if (!normalParam.empty() || !gatherParam.empty()) {
        auto emitArray = [&](const char* name,
                             const std::vector<int32_t>& arr) {
          cs << "    static int32_t " << name << "[] = {";
          for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) {
              cs << ", ";
            }
            cs << arr[i];
          }
          cs << "};\n";
        };
        if (!normalParam.empty()) {
          cs << "  {\n";
          emitArray("paramOffsets", normalParam);
          emitArray("outputOffsets", normalOutput);
          emitArray("altOffsets", normalAlt);
          cs << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
             << "      if (altOffsets[i] != -1) {\n"
             << "        copyTensorHead(param<Tensor>(blockInfo, paramOffsets[i]), param<Tensor>(blockInfo, altOffsets[i]));\n"
             << "        param<Tensor>(blockInfo, altOffsets[i])->init<true>(param<Tensor>(blockInfo, outputOffsets[i]));\n"
             << "      } else {\n"
             << "        param<Tensor>(blockInfo, paramOffsets[i])->init<true>(outputOffsets[i] != paramOffsets[i] ? param<Tensor>(blockInfo, outputOffsets[i]) : nullptr);\n"
             << "      }\n"
             << "    }\n"
             << "  }\n";
        }
        if (!gatherParam.empty()) {
          cs << "  {\n";
          emitArray("gatherOffsets", gatherParam);
          cs << "    for (auto i = threadIdx.x; i < sizeof(gatherOffsets) / sizeof(gatherOffsets[0]); i += blockDim.x) {\n"
             << "      param<Tensor>(blockInfo, gatherOffsets[i])->ensureIndexCalculator();\n"
             << "    }\n"
             << "  }\n";
        }
        cs << "  __syncthreads();\n";
      }
      cs << kop->code() << "      break;\n"
         << "    }\n";
      caseTexts.push_back(cs.str());
    }
  } // WithPrintOptions scope

  // Kernel ids are per-construction (not globally unique), so the debug copies
  // in /tmp carry a process-wide ordinal to keep filenames distinct when
  // several graphs compile concurrently.
  static std::atomic<int64_t> debugDumpOrdinal{0};
  entryPoint_ = entryPoint;

  if (FLAGS_compile_meter && facebook::velox::wave::currentDevice()) {
    // Build the include header common to all single-case kernels.
    std::stringstream includeHeader;
    includeHeader << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
    for (const auto& inc : includes) {
      includeHeader << "#include \"" << inc << "\"\n";
    }
    auto headerStr = includeHeader.str();

    for (const auto& kop : kernelOpStorage_) {
      auto caseLabel = kop->label();
      // Replace dots and spaces with underscores for a valid identifier.
      std::string safeName = caseLabel;
      for (auto& ch : safeName) {
        if (ch == '.' || ch == ' ') {
          ch = '_';
        }
      }
      auto trialName = "torchwave" + std::to_string(kernelId) + "_" + safeName;
      auto trialEntry = "torch::wave::" + trialName;
      auto trialFile =
          "/tmp/kernel" + std::to_string(kernelId) + "-" + caseLabel + ".cu";

      std::stringstream ts;
      ts << headerStr << "\nnamespace torch::wave {\n\n";
      if (!kop->helperCode().empty()) {
        ts << kop->helperCode();
      }
      ts << "#define " << kBlockSizeMacro << " "
         << std::max(WaveConfig::get().blockSize, kop->blockSizePreference())
         << "\n";
      ts << "__global__ void " << trialName << "(TorchWaveParams params) {\n"
         << "  ENTRY;\n";
      for (const auto& decl : kop->sharedDeclarations()) {
        ts << decl;
      }
      ts << "  switch (blockInfo.op) {\n"
         << "    case " << kop->opCode() << ": {\n"
         << kop->code() << "      break;\n"
         << "    }\n"
         << "  }\n"
         << "  LEAVE();\n"
         << "}\n\n"
         << "} // namespace torch::wave\n";
      auto trialCode = ts.str();

      {
        std::ofstream out(trialFile);
        out << trialCode;
      }

      facebook::velox::wave::KernelSpec spec;
      spec.code = trialCode;
      spec.entryPoints = {trialEntry};
      spec.filePath = trialFile;
      spec.numHeaders = 0;
      spec.headers = nullptr;
      facebook::velox::wave::CompiledModule::create(spec);
      LOG(INFO) << "compile_meter " << caseLabel
                << ": code=" << kop->code().size()
                << " helpers=" << kop->helperCode().size()
                << " total=" << trialCode.size()
                << " inputs=" << kop->numInputs() << " file=" << trialFile;
    }
  }

  // Diagnostic: one single-case kernel per op, queued alongside the composite
  // below so the two compile concurrently. Each is the composite's preamble
  // with a switch holding only this op's case, so its register / shared /
  // local-memory footprint is that op's alone. Nothing launches them for
  // results; WaveGraph warms them up once at the end of graph construction and
  // logs their occupancy next to the composite's.
  if (WaveConfig::get().configPerOp && facebook::velox::wave::currentDevice()) {
    std::stringstream includeHeader;
    includeHeader << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
    for (const auto& inc : includes) {
      includeHeader << "#include \"" << inc << "\"\n";
    }
    auto headerStr = includeHeader.str();

    for (const auto& kop : kernelOpStorage_) {
      auto opName = kernelName + "_op_" + std::to_string(kop->opCode());
      auto opEntry = "torch::wave::" + opName;
      auto opFile = "/tmp/" + opName + ".cu";

      std::stringstream os;
      os << headerStr << "\nnamespace torch::wave {\n\n";
      if (!kop->helperCode().empty()) {
        os << kop->helperCode();
      }
      os << "#define " << kBlockSizeMacro << " "
         << std::max(WaveConfig::get().blockSize, kop->blockSizePreference())
         << "\n";
      os << "__global__ void " << opName << "(TorchWaveParams params) {\n"
         << "  ENTRY;\n";
      for (const auto& decl : kop->sharedDeclarations()) {
        os << decl;
      }
      os << "  switch (blockInfo.op) {\n"
         << "    case " << kop->opCode() << ": {\n"
         << kop->code() << "      break;\n"
         << "    }\n"
         << "  }\n"
         << "  LEAVE();\n"
         << "}\n\n"
         << "} // namespace torch::wave\n";
      auto opText = os.str();
      {
        std::ofstream out(opFile);
        out << opText;
      }

      PerOpKernel perOp;
      perOp.opCode = kop->opCode();
      perOp.entryPoint = opEntry;
      // Deliberately not the KernelFsCache: these are throwaway diagnostics and
      // must not compete with the composite for the on-disk cache.
      perOp.kernel = facebook::velox::wave::CompiledKernel::getKernel(
          opText,
          [code = opText,
           opEntry,
           opFile]() -> facebook::velox::wave::KernelSpec {
            facebook::velox::wave::KernelSpec spec;
            spec.code = code;
            spec.entryPoints = {opEntry};
            spec.filePath = opFile;
            spec.numHeaders = 0;
            spec.headers = nullptr;
            return spec;
          });
      perOpKernels_.push_back(std::move(perOp));
    }
  }

  // One module per distinct set of ops that share a step of the grid the run
  // uses: a step runs with the registers its own ops need, and the modules
  // compile in parallel.
  std::vector<std::string> moduleTexts;
  std::vector<std::vector<std::string>> moduleEntryPoints;
  std::vector<std::string> moduleFiles;
  {
    folly::F14FastMap<const KernelOperation*, size_t> opIndex;
    for (size_t i = 0; i < kernelOpStorage_.size(); ++i) {
      opIndex[kernelOpStorage_[i].get()] = i;
    }
    std::vector<std::vector<size_t>> stepOps;
    for (const auto& projectOp : ops_) {
      const auto& grid = allocGroupGrid(*projectOp);
      if (stepOps.size() < grid.size()) {
        stepOps.resize(grid.size());
      }
      for (size_t step = 0; step < grid.size(); ++step) {
        for (const auto& launch : grid[step]) {
          if (launch.op != nullptr) {
            auto it = opIndex.find(launch.op);
            TORCH_CHECK(
                it != opIndex.end(), "A step launches an op of another kernel");
            stepOps[step].push_back(it->second);
          }
        }
      }
    }
    std::map<std::vector<size_t>, int32_t> moduleOf;
    for (auto& opsOfStep : stepOps) {
      std::sort(opsOfStep.begin(), opsOfStep.end());
      opsOfStep.erase(
          std::unique(opsOfStep.begin(), opsOfStep.end()), opsOfStep.end());
      if (opsOfStep.empty()) {
        stepModule_.push_back(-1);
        continue;
      }
      auto [it, added] = moduleOf.try_emplace(
          opsOfStep, static_cast<int32_t>(moduleOf.size()));
      stepModule_.push_back(it->second);
      if (!added) {
        continue;
      }
      const int32_t module = it->second;
      std::stringstream text;
      text << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
      for (const auto& inc : includes) {
        text << "#include \"" << inc << "\"\n";
      }
      text << "\nnamespace torch::wave {\n\n";
      int32_t minBlocks = 0;
      std::vector<int32_t> sizes = {WaveConfig::get().blockSize};
      for (auto i : opsOfStep) {
        const auto& kop = kernelOpStorage_[i];
        if (!kop->helperCode().empty()) {
          text << kop->helperCode();
        }
        minBlocks = std::max(minBlocks, kop->minBlocksPerSm());
        const int32_t threads = kop->blockSizePreference();
        if (threads > WaveConfig::get().blockSize &&
            std::find(sizes.begin(), sizes.end(), threads) == sizes.end()) {
          sizes.push_back(threads);
        }
      }
      std::sort(sizes.begin(), sizes.end());
      std::stringstream body;
      body << "(TorchWaveParams params) {\n"
           << "  ENTRY;\n";
      eltTrace(
          body,
          "\"entry blockIdx %d op %d blockInOp %d\\n\", blockIdx.x, blockInfo.op, blockInfo.blockInOp");
      std::unordered_set<std::string> decls;
      for (auto i : opsOfStep) {
        for (const auto& decl : kernelOpStorage_[i]->sharedDeclarations()) {
          if (decls.insert(decl).second) {
            body << decl;
          }
        }
      }
      body << "  switch (blockInfo.op) {\n";
      for (auto i : opsOfStep) {
        body << caseTexts[i];
      }
      body << "  }\n"
           << "  LEAVE();\n"
           << "}\n\n";
      std::vector<std::string> points;
      for (const int32_t threads : sizes) {
        const auto name = kernelName + "_s" + std::to_string(module) +
            (threads == WaveConfig::get().blockSize
                 ? std::string()
                 : "_b" + std::to_string(threads));
        points.push_back("torch::wave::" + name);
        text << "#define " << kBlockSizeMacro << " " << threads << "\n"
             << "__global__ ";
        if (minBlocks > 0) {
          text << "__launch_bounds__(" << threads << ", "
               << std::min(minBlocks, 2048 / threads) << ") ";
        } else if (threads > WaveConfig::get().blockSize) {
          text << "__launch_bounds__(" << threads << ") ";
        }
        text << "void " << name << body.str() << "#undef " << kBlockSizeMacro
             << "\n\n";
        entryBlockSize_.push_back(threads);
        entryModule_.push_back(module);
        entryLocal_.push_back(static_cast<int32_t>(points.size() - 1));
      }
      text << "} // namespace torch::wave\n";
      std::string code = text.str();
      if (!FLAGS_debug_kernel_dir.empty()) {
        auto debugFile = fmt::format("{}_s{}.cu", kernelName, module);
        auto debugPath =
            fmt::format("{}/{}", FLAGS_debug_kernel_dir, debugFile);
        std::ifstream in(debugPath);
        if (in.good()) {
          code = std::string(
              std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
          LOG(INFO) << "Reading code of " << debugFile << " from " << debugPath;
        } else {
          LOG(INFO) << "No debug version for " << debugFile;
        }
      }
      moduleTexts.push_back(std::move(code));
      moduleEntryPoints.push_back(std::move(points));
      moduleFiles.push_back(
          "/tmp/kernel" + std::to_string(kernelId) + "_s" +
          std::to_string(module) + "_" + std::to_string(debugDumpOrdinal++) +
          ".cu");
      std::ofstream(moduleFiles.back()) << moduleTexts.back();
    }
  }

  // Only compile the kernel if a GPU is available. The one-time
  // NVRTC/system-header initialization (CompiledKernel::initialize()) is run
  // eagerly on the main thread by torch::wave::initialize() before any kernel
  // is compiled, so the async compile enqueued below never triggers it lazily
  // on a Wave compile-pool thread (which deadlocks warmup() in heavyweight
  // NCCL/Thrift/folly hosts -- T275179010).
  if (facebook::velox::wave::currentDevice()) {
    for (size_t m = 0; m < moduleTexts.size(); ++m) {
      auto genFunc =
          [code = moduleTexts[m],
           points = moduleEntryPoints[m],
           file = moduleFiles[m]]() -> facebook::velox::wave::KernelSpec {
        facebook::velox::wave::KernelSpec spec;
        spec.code = code;
        spec.entryPoints = points;
        spec.filePath = file;
        spec.numHeaders = 0;
        spec.headers = nullptr;
        return spec;
      };
      if (!WaveConfig::get().kernelCacheDir.empty()) {
        static facebook::velox::wave::KernelFsCache cache(
            WaveConfig::get().kernelCacheDir);
        modules_.push_back(cache.getKernel(moduleTexts[m], std::move(genFunc)));
      } else {
        modules_.push_back(
            facebook::velox::wave::CompiledKernel::getKernel(
                moduleTexts[m], std::move(genFunc)));
      }
    }
  }
}

void CompositeKernel::warmup() {
  TorchWaveParams params{};
  memset(&params, 0, sizeof(params));
  params.info = nullptr;
  params.debugInfo = nullptr;
  params.inlineInfo[0].op = kDebugNoOp;
  void* args[] = {&params};
  facebook::velox::wave::Stream stream;
  for (auto& module : modules_) {
    module->launch(0, 1, 1, 0, &stream, args);
  }
  stream.wait();
}

std::vector<std::pair<std::string, facebook::velox::wave::KernelInfo>>
CompositeKernel::perOpKernelInfo() {
  std::vector<std::pair<std::string, facebook::velox::wave::KernelInfo>> result;
  result.reserve(perOpKernels_.size());
  for (auto& perOp : perOpKernels_) {
    if (!perOp.kernel) {
      continue;
    }
    // The launch is the sync point with the queued compile, exactly as
    // warmup() is for the composite. blockInfo.op is kDebugNoOp, which matches
    // no case, so the body does nothing.
    TorchWaveParams params{};
    memset(&params, 0, sizeof(params));
    params.info = nullptr;
    params.debugInfo = nullptr;
    params.inlineInfo[0].op = kDebugNoOp;
    void* args[] = {&params};
    facebook::velox::wave::Stream stream;
    perOp.kernel->launch(0, 1, 1, 0, &stream, args);
    stream.wait();
    result.emplace_back(perOp.entryPoint, perOp.kernel->info(0));
  }
  return result;
}

std::string CompositeKernel::entryInfoString(int32_t entry) const {
  return fmt::format(
      "{} module {} at {} threads: {}",
      entryPoint_,
      entryModule_.at(entry),
      entryBlockSize_.at(entry),
      kernelInfo(entry).toString());
}

int32_t CompositeKernel::entryFor(int32_t blockSize, int32_t stepIdx) const {
  TORCH_CHECK(
      stepIdx >= 0 && static_cast<size_t>(stepIdx) < stepModule_.size() &&
          stepModule_[stepIdx] >= 0,
      entryPoint_,
      " has no module for step ",
      stepIdx);
  for (size_t i = 0; i < entryModule_.size(); ++i) {
    if (entryModule_[i] == stepModule_[stepIdx] &&
        entryBlockSize_[i] == blockSize) {
      return static_cast<int32_t>(i);
    }
  }
  TORCH_CHECK(
      false,
      entryPoint_,
      " step ",
      stepIdx,
      " has no entry point for ",
      blockSize,
      " threads per block");
}

int32_t CompositeKernel::maxOccupancy(int32_t entry) const {
  return occupancy(entry, 0);
}

int32_t CompositeKernel::maxOccupancyOfEntries() const {
  int32_t blocks = 0;
  for (size_t i = 0; i < entryBlockSize_.size(); ++i) {
    blocks = std::max(blocks, maxOccupancy(static_cast<int32_t>(i)));
  }
  return blocks;
}

int32_t CompositeKernel::occupancy(int32_t entry, int32_t dynamicSharedBytes)
    const {
  if (modules_.empty()) {
    return 0;
  }
  return modules_[entryModule_.at(entry)]->occupancy(
      entryLocal_.at(entry), entryBlockSize_.at(entry), dynamicSharedBytes);
}

facebook::velox::wave::KernelInfo CompositeKernel::kernelInfo(
    int32_t entry) const {
  if (!modules_.empty()) {
    return modules_[entryModule_.at(entry)]->info(entryLocal_.at(entry));
  }
  return {};
}

void CompositeKernel::launch(
    int32_t entry,
    int32_t numBlocks,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  modules_[entryModule_.at(entry)]->launch(
      entryLocal_.at(entry),
      numBlocks,
      entryBlockSize_.at(entry),
      sharedMemory,
      stream,
      args);
}

void CompositeKernel::launchCooperative(
    int32_t entry,
    int32_t numBlocks,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  modules_[entryModule_.at(entry)]->launchCooperative(
      entryLocal_.at(entry),
      numBlocks,
      entryBlockSize_.at(entry),
      sharedMemory,
      stream,
      args);
}

std::string CompositeKernel::toString(Listing /*mode*/) const {
  std::stringstream ss;
  for (const auto& kop : kernelOpStorage_) {
    ss << kop->toString();
  }
  if (!modules_.empty()) {
    for (int32_t entry = 0; entry < numEntries(); ++entry) {
      ss << "entry=" << entryInfoString(entry) << "\n";
    }
  }
  return ss.str();
}

namespace {

void traceTensor(
    nativert::ValueId actualId,
    c10::IntArrayRef dims,
    const char* action) {
  if (!(WaveConfig::get().trace & WaveConfig::kTensors)) {
    return;
  }
  auto t =
      at::empty(dims, at::TensorOptions().dtype(at::kFloat).device(at::kMeta));
  std::cout << "  tensor %" << actualId << " " << action << " "
            << traceIValue(c10::IValue(t)) << std::endl;
}

// Elementwise in-place reuse counters (per thread). Snapshotted per node in
// CompositeInvocation::execute() under WaveConfig::kTiming. Each reuse turns an
// elementwise output allocation into an in-place write over a reusable input.
thread_local int64_t gElementwiseReuseCount = 0;
thread_local int64_t gElementwiseReuseBytes = 0;

} // namespace

void ensureCudaTensor(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId actualId,
    c10::IntArrayRef dims,
    const std::string& sizeKey) {
  ScopedSetupPhase phase(SetupPhase::kTensor);
  auto& existing = frame.getIValue(actualId);
  // Allocation groups carve every member out of one buffer once the whole
  // group has been sized, so a member only reports its shape here; its frame
  // slot is written later, by materializeAllocGroup. Checked ahead of the
  // resize/keep path below because a pooled frame can still hold the previous
  // execution's tensor for this value, and keeping that would leave the member
  // outside the group its lifetime was planned around.
  //
  // Not, however, when the slot already holds a tensor a second frame slot
  // shares: that is the output of an in-place op whose reserve function has
  // just aliased it to the argument it mutates (tw.masked_put_ and the other
  // scatters). Its buffer is deliberately somebody else's, and carving it a
  // slot of its own would drop the mutation on the floor. Left out of the
  // group, which materializeAllocGroup tolerates. Same ownership test the
  // freeing path uses to tell an alias from a solely-owned buffer.
  const bool sharedWithAnotherValue = existing.isTensor() &&
      existing.toTensor().defined() && existing.toTensor().use_count() > 1;
  // The frame already holds a tensor of exactly this shape, so the ordinary
  // path below would keep it rather than allocate. Then this is not this
  // launch's buffer -- something else produced the value and the sizing only
  // checks it -- and carving it a slot would replace what that something wrote.
  const bool wouldKeep = existing.isTensor() && existing.toTensor().defined() &&
      existing.toTensor().is_cuda() && existing.toTensor().sizes() == dims;
  if (auto* collector = currentAllocCollector();
      !sharedWithAnotherValue && !wouldKeep) {
    if (collector != nullptr && collector->capture(actualId, dims)) {
      // The group's buffer does not exist until every member has been sized,
      // but the rest of the step's sizing reads shapes out of the frame -- size
      // expressions, reserve functions, the grid choice. Leave a shape-only
      // tensor behind so all of those see the shape they need;
      // materializeAllocGroup replaces it with the real slot. Reaching for the
      // data rather than the shape throws on a meta tensor, which is the
      // failure worth having.
      const auto* typeMeta = types.types.at(actualId);
      const auto dtype =
          typeMeta != nullptr ? typeMeta->dtype() : c10::ScalarType::Float;
      writeFrame(
          frame,
          actualId,
          at::empty(dims, at::TensorOptions().dtype(dtype).device(at::kMeta)));
      return;
    }
  }
  // Bytes the output needs, for the allocation trace. Computed only when the
  // trace is on: it is a multiply per dim in the per-op path otherwise.
  auto requestedBytes = [&](c10::ScalarType dtype) {
    int64_t numel = 1;
    for (auto d : dims) {
      numel *= d;
    }
    return numel * static_cast<int64_t>(c10::elementSize(dtype));
  };
  if (existing.isTensor() && existing.toTensor().is_cuda()) {
    auto& tensor = existing.toTensor();
    if (tensor.sizes() != dims) {
      traceTensor(actualId, dims, "resize");
      if (allocTraceEnabled()) {
        logKeyedAllocEvent(
            "resize", actualId, requestedBytes(tensor.scalar_type()), sizeKey);
      }
      {
        ScopedAllocCall timed;
        tensor.resize_(dims);
      }
    } else {
      traceTensor(actualId, dims, "keep");
      if (allocTraceEnabled()) {
        logKeyedAllocEvent(
            "keep", actualId, requestedBytes(tensor.scalar_type()), sizeKey);
      }
    }
    publishFromFrame(frame, actualId);
  } else {
    auto* meta = types.types.at(actualId);
    if (!meta) {
      return;
    }
    traceTensor(actualId, dims, "alloc");
    if (allocTraceEnabled()) {
      logKeyedAllocEvent(
          "alloc", actualId, requestedBytes(meta->dtype()), sizeKey);
    }
    at::Tensor tensor;
    {
      ScopedAllocCall timed;
      tensor = at::empty(
          dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
    }
    writeFrame(frame, actualId, std::move(tensor));
  }
}

namespace {

// True for the list kinds whose elements are tensors, i.e. the cases where a
// pack/unpack only moves handles.
bool isTensorListKind(nativert::Type::Kind kind) {
  return kind == nativert::Type::Kind::TensorList ||
      kind == nativert::Type::Kind::NestedTensorList ||
      kind == nativert::Type::Kind::OptionalTensorList;
}

} // namespace

namespace {
thread_local ExecutionState* tlsExecutionState = nullptr;
} // namespace

void setCurrentExecutionState(ExecutionState* state) {
  tlsExecutionState = state;
}

ExecutionState* currentExecutionState() {
  return tlsExecutionState;
}

// Runs a fused output's host-side view as a metadata shortcut. Returns false
// when the view has no recipe, which leaves it to the op's kernel.
bool runViewRecipe(const OutputDesc& desc, nativert::ExecutionFrame& frame) {
  ScopedSetupPhase phase(SetupPhase::kView);
  if (!desc.viewRecipeTried) {
    desc.viewRecipe = makeViewRecipe(desc.viewNode);
    desc.viewRecipeTried = true;
  }
  if (desc.viewRecipe == nullptr) {
    return false;
  }
  runStandaloneShortcut(desc.viewRecipe->data, frame);
  return true;
}

namespace {} // namespace

facebook::velox::wave::WaveBufferPtr& getOrAllocateBuffer(
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>& buffers,
    int32_t sequenceNumber,
    int32_t stepIdx,
    int64_t requiredBytes,
    facebook::velox::wave::GpuArena* arena,
    const std::function<void(void*, int64_t)>& initFunc) {
  if (static_cast<int32_t>(buffers.size()) <= sequenceNumber) {
    buffers.resize(sequenceNumber + 1);
  }
  auto& steps = buffers.at(sequenceNumber);
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  auto& buffer = steps.at(stepIdx);
  if (!buffer || buffer->capacity() < static_cast<size_t>(requiredBytes)) {
    buffer = arena->allocateBytes(requiredBytes);
    if (initFunc) {
      initFunc(buffer->as<void>(), requiredBytes);
    }
  }
  return buffer;
}

namespace {} // namespace

StepVectors& getStepVectors(
    std::vector<std::vector<StepVectors>>& allSteps,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  if (static_cast<int32_t>(allSteps.size()) <= sequenceNumber) {
    allSteps.resize(sequenceNumber + 1);
  }
  auto& steps = allSteps.at(sequenceNumber);
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  return steps.at(stepIdx);
}

namespace {

// Returns true if every launch's numElements falls within the cached
// [sizesLower, sizesUpper] bounds.
bool gridSizesMatch(
    const std::vector<LaunchData>& launches,
    const StepVectors& sv) {
  if (!sv.hasGridCache || launches.size() != sv.sizesLower.size()) {
    return false;
  }
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    if (n < sv.sizesLower.at(i) || n > sv.sizesUpper.at(i)) {
      return false;
    }
  }
  return true;
}

// Updates the cached size bounds to [size - size/8, size + size/8] for each
// launch.
void updateGridSizeBounds(
    const std::vector<LaunchData>& launches,
    StepVectors& sv) {
  sv.sizesLower.resize(launches.size());
  sv.sizesUpper.resize(launches.size());
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    auto margin = n / 8;
    sv.sizesLower[i] = n - margin;
    sv.sizesUpper[i] = n + margin;
  }
  sv.hasGridCache = true;
}

} // namespace

// --- CompositeInvocation ---

void CompositeInvocation::layoutParamSlots(int32_t stepIdx, StepVectors& sv) {
  if (!sv.opSlotBegin.empty()) {
    return;
  }
  sv.opSlotBegin.reserve(ops_.size() + 1);
  int64_t cursor = 0;
  for (auto& op : ops_) {
    sv.opSlotBegin.push_back(static_cast<int32_t>(sv.slotOffsets.size()));
    auto* projectOp = op.projectOp();
    // Per kernel slot, the largest param block any variant would put there. A
    // variant that has fewer launches at this step simply leaves the tail
    // slots unused.
    std::vector<int32_t> slotSizes;
    for (const auto* grid :
         {&projectOp->grid(),
          &projectOp->singleBlockGrid(),
          &projectOp->cgGrid()}) {
      if (stepIdx >= static_cast<int32_t>(grid->size())) {
        continue;
      }
      size_t slot = 0;
      for (const auto& launch : (*grid)[stepIdx]) {
        if (launch.op == nullptr) {
          continue;
        }
        if (slot >= slotSizes.size()) {
          slotSizes.resize(slot + 1, 0);
        }
        // Grown to slot + 1 just above.
        // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
        auto& slotSize = slotSizes[slot];
        slotSize = std::max(slotSize, launch.op->altParamOffset());
        ++slot;
      }
    }
    for (auto size : slotSizes) {
      sv.slotOffsets.push_back(cursor);
      cursor += size;
    }
  }
  sv.opSlotBegin.push_back(static_cast<int32_t>(sv.slotOffsets.size()));
  sv.paramRegionBytes = cursor;

  // Everything this step can read, unioned over the same three variants. Same
  // sources and same reason as Compile.cpp's opReadSet: orderingInputs is what
  // the scheduler orders a launch against, so it already covers the operands
  // that reach the host only through a reserve / sizeExpr at sizing time and
  // the host-side view operands that are inputs of no fused node. The subgraph
  // leaves are added for any op present at this step, since a variant can reach
  // a boundary input without naming it in a launch.
  sv.opReadIds.resize(ops_.size());
  sv.opReadSignature.assign(ops_.size(), 0);
  sv.opKernelCount.assign(ops_.size(), 0);
  sv.opStandaloneCount.assign(ops_.size(), 0);
  sv.opShortcutCount.assign(ops_.size(), 0);
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    auto& opIds = sv.opReadIds[i];
    const auto& bindings = op.bindings();
    auto toActual = [&](nativert::ValueId formalId) {
      auto it = bindings.find(formalId);
      return it != bindings.end() ? it->second : formalId;
    };
    auto* projectOp = op.projectOp();
    bool present = false;
    for (const auto* grid :
         {&projectOp->grid(),
          &projectOp->singleBlockGrid(),
          &projectOp->cgGrid()}) {
      if (stepIdx >= static_cast<int32_t>(grid->size())) {
        continue;
      }
      present = true;
      for (const auto& launch : (*grid)[stepIdx]) {
        if (launch.op != nullptr) {
          for (auto id : launch.op->orderingInputs()) {
            opIds.insert(toActual(id));
          }
        } else if (launch.standalone != nullptr) {
          // prim.ListPack / prim.ListUnpack only move tensor handles between
          // the frame and a list; they never read an element's contents or its
          // shape. Charging them their inputs' ids would defer them behind a
          // pending transfer and drag the host into a waveDone wait they do not
          // need. Consumers of the packed/unpacked values carry their own read
          // ids, so the real dependency is still enforced where it matters.
          // Only when the list is a TensorList: then the op merely moves
          // tensor handles and never reads an element's contents or shape, so
          // charging it its inputs' ids would defer it behind a pending
          // transfer and drag the host into a waveDone wait it does not need.
          // A SymInt / int list is the opposite -- the scalars ARE the data,
          // and packing one before its transfer lands would store a stale
          // value. Same distinction Launch::Launch draws for the kListPack
          // shortcut.
          const auto target = launch.standalone->target();
          const bool isPack = target == "prim.ListPack";
          const bool isUnpack = target == "prim.ListUnpack";
          if (isPack || isUnpack) {
            ValueCP listValue = nullptr;
            if (isPack) {
              listValue = launch.standalone->outputs().empty()
                  ? nullptr
                  : launch.standalone->outputs()[0];
            } else if (!launch.standalone->inputs().empty()) {
              listValue = launch.standalone->inputs()[0].value;
            }
            if (listValue != nullptr &&
                isTensorListKind(listValue->type().kind())) {
              continue;
            }
          }
          for (const auto& input : launch.standalone->inputs()) {
            if (input.value != nullptr) {
              opIds.insert(toActual(input.value->id()));
            }
          }
        }
      }
    }
    if (present) {
      for (const auto* input : projectOp->subgraph().inputs) {
        opIds.insert(toActual(input->id()));
      }
    }
    uint64_t signature = 0;
    for (auto id : opIds) {
      signature |= uint64_t{1} << (static_cast<uint64_t>(id) & 63);
    }
    // Assigned ops_.size() entries above; i runs over the same bound.
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    sv.opReadSignature[i] = signature;
    sv.readIds.insert(opIds.begin(), opIds.end());
  }

  // Bound on what makeGrid can assign here. It gives each launch
  // max(1, round(fraction * targetBlocks)) blocks, then only ever reduces that
  // (the per-launch cap, the round-down to whole blockSize chunks) and the
  // rebalancing pass conserves the total. So the sum cannot exceed
  // targetBlocks plus 1.5 per launch; take 2 per launch.
  //
  // Partitioning a step deliberately emits more than one wave -- that is the
  // point of it -- so the reservation grows by the same cap the packer bounds
  // itself with. Without the flag the figure is exactly what it always was.
  const auto numSlots = static_cast<int32_t>(sv.slotOffsets.size());
  const int32_t waves = WaveConfig::get().partitionLaunches
      ? std::max(1, WaveConfig::get().maxLaunchWaves)
      : 1;
  sv.blockCapacity = 2 * numSlots +
      waves *
          targetBlockCount(
              kernel_->maxOccupancyOfEntries(),
              // A bound, so skip the shared-memory reduction: it only ever
              // lowers the count makeGrid will actually use, and a step's
              // dynamic shared-memory need is not known until its launches are
              // gathered.
              /*dynSharedPerBlock=*/0,
              /*staticSharedPerBlock=*/0);
}

namespace {

// Writes the step's BlockInfo array into its pinned buffer, unless the same
// grid was already written there for the same device buffer.
void writeBlockInfos(
    StepVectors& sv,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    DebugInfo* deviceDebugBase) {
  const auto numBlocks = sv.blocks.size();
  if (!WaveConfig::get().debugSingleOps && sv.blockInfoPinned == pinnedBase &&
      sv.blockInfoDevice == deviceBase &&
      sv.blockInfoGridVersion == sv.gridVersion &&
      sv.blockInfoCount == numBlocks) {
    return;
  }
  auto* pinnedBlocks =
      reinterpret_cast<BlockInfo*>(pinnedBase + sv.blockInfoOffset);
  if (numBlocks > 0) {
    memcpy(pinnedBlocks, sv.blocks.data(), numBlocks * sizeof(BlockInfo));
  }
  for (size_t b = 0; b < numBlocks; ++b) {
    auto idx = sv.launchIndices[b];
    pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
    pinnedBlocks[b].debugInfo = deviceDebugBase + b;
  }
  sv.blockInfoPinned = pinnedBase;
  sv.blockInfoDevice = deviceBase;
  sv.blockInfoGridVersion = sv.gridVersion;
  sv.blockInfoCount = numBlocks;
}

} // namespace

namespace {

void verifyAgainstReference(
    const std::vector<LaunchData>& launches,
    nativert::ExecutionFrame& frame,
    ExecutionState& state) {
  auto* ref = WaveConfig::get().referenceFrame;
  if (!ref) {
    return;
  }
  // This checks both fused outputs (produced on the wave stream) and standalone
  // outputs (produced by eager ops on the default stream), so sync both: the
  // wave stream and the default stream where eager standalones run.
  syncWaveStream(state);
  syncTorchDefaultStream();
  // The reference stores scalars and scalar lists as 1-D tensors; fold the
  // actual frame value into a tensor the same way so it can be compared
  // element-wise against the recorded tensor.
  auto asTensor = [](const c10::IValue& iv) -> std::optional<at::Tensor> {
    if (iv.isTensor()) {
      return iv.toTensor();
    }
    return scalarLikeToTensor(iv);
  };
  // The input of an elided clone is written in place by the writer that used to
  // read the clone, so its buffer holds the post-mutation value while the
  // reference recorded the pre-mutation one. The divergence is intended, so
  // exclude these values from every reference comparison below.
  auto isElidedCloneInput = [&](nativert::ValueId id) {
    return state.waveGraph != nullptr &&
        state.waveGraph->isElidedCloneInput(id);
  };
  // An output no node reads is never written by anyone: the op declares it
  // because the eager schema has it (an exported graph names these
  // '<op>_unused_N'), but nothing consumes the data so no kernel fills it. Its
  // buffer holds whatever the allocator last left there, so comparing it tests
  // allocation history rather than correctness -- it passes or fails depending
  // on which buffer the value happened to get. Only ever skips values with no
  // users at all; a graph output has the output node as a user, and a stale
  // user entry keeps the value compared, so this cannot hide a real reader.
  auto hasNoReader = [&](nativert::ValueId id) {
    if (state.waveGraph == nullptr) {
      return false;
    }
    const auto& idToValue = state.waveGraph->idToValue();
    auto it = idToValue.find(id);
    return it != idToValue.end() && it->second != nullptr &&
        it->second->users().empty();
  };
  int32_t numMismatches = 0;
  std::string passedIds;
  // Which values failed, not just which passed: the detail goes to LOG(ERROR),
  // and a host that drops glog output leaves the message that does come out
  // naming everything except the thing that went wrong.
  std::string failedIds;
  // The first difference of the first failure, which otherwise only exists in
  // the LOG(ERROR) below. A host that drops glog output is left with a message
  // that names the value and says nothing about how it is wrong.
  std::string firstFailure;
  auto addFailed = [&failedIds](nativert::ValueId actualId) {
    if (!failedIds.empty()) {
      failedIds += " ";
    }
    failedIds += "%" + std::to_string(actualId);
  };
  int32_t numPassed = 0;
  for (const auto& data : launches) {
    bool nodeChecked = false;
    for (size_t oi = 0; oi < data.actualOutputs.size(); ++oi) {
      auto actualId = data.actualOutputs[oi];
      if (isElidedCloneInput(actualId) || hasNoReader(actualId)) {
        continue;
      }
      auto refIt = ref->find(actualId);
      if (refIt == ref->end()) {
        continue;
      }
      if (!refIt->second.isTensor()) {
        continue;
      }
      // Skip scalar/symint outputs.  The reference stores SymInt/SymFloat/
      // SymBool as 1-D tensors, but wave computes them as register scalars --
      // frequently consumed internally for shapes/bounds (e.g. sym_numel used
      // as a clamp max) and not materialized into a frame tensor.  Their
      // correctness is covered indirectly: a metadata scalar (sym_numel/
      // sym_size) derives from a tensor that IS verified, and any wrong symint
      // produces a wrong downstream tensor shape that surfaces as a mismatch on
      // that tensor.
      if (oi < data.actualOutputTypes.size() &&
          data.actualOutputTypes[oi] != nativert::Type::Kind::Tensor &&
          data.actualOutputTypes[oi] != nativert::Type::Kind::TensorList) {
        continue;
      }
      auto actualOpt = asTensor(debugValue(frame, actualId));
      if (!actualOpt) {
        continue;
      }
      const at::Tensor& actualTensor = *actualOpt;
      const auto& refTensor = refIt->second.toTensor();
      if (actualTensor.numel() == 0) {
        continue;
      }
      // A meta tensor carries no data. An intentional shape-only output (e.g.
      // an index a composite consumes internally and exposes only for
      // downstream shape inference, like a gather index) has nothing to compare
      // -- its correctness is covered by verifying its data-consumer's output.
      // A meta output that is NOT shape-only is unexpected (a materialization
      // bug): surface it as a mismatch rather than silently skipping, so we do
      // not lose a correctness signal.
      if (actualTensor.is_meta()) {
        bool isShapeOnly = oi < data.actualOutputDescs.size() &&
            data.actualOutputDescs[oi].shapeOnly;
        if (!isShapeOnly) {
          ++numMismatches;
          addFailed(actualId);
          LOG(ERROR) << "Value %" << actualId
                     << " is a meta tensor (no data) but is not a shape-only "
                        "output; cannot verify (unexpected materialization).";
        }
        continue;
      }
      if (state.numRefTensorsChecked) {
        ++*state.numRefTensorsChecked;
      }
      nodeChecked = true;
      if (!tensorsMatch(actualTensor, refTensor)) {
        ++numMismatches;
        addFailed(actualId);
        if (firstFailure.empty()) {
          firstFailure = fmt::format(
              "%{}: {}", actualId, firstDifference(actualTensor, refTensor));
        }
        auto limit = WaveConfig::get().tensorPrintElementLimit;
        LOG(ERROR) << "Reference mismatch for value %" << actualId << "\n  "
                   << firstDifference(actualTensor, refTensor)
                   << "\n  expected: " << tensorDebugString(refTensor, limit)
                   << "\n  actual:   "
                   << tensorDebugString(actualTensor, limit);
      } else {
        ++numPassed;
        if (!passedIds.empty()) {
          passedIds += " ";
        }
        passedIds += "%" + std::to_string(actualId);
      }
    }
    if (nodeChecked && state.numRefNodesChecked) {
      ++*state.numRefNodesChecked;
    }
  }
  // Record newly passed ids for reverification.
  if (WaveConfig::get().reverify) {
    for (const auto& data : launches) {
      for (auto actualId : data.actualOutputs) {
        if (isElidedCloneInput(actualId)) {
          continue;
        }
        auto refIt = ref->find(actualId);
        if (refIt != ref->end() && refIt->second.isTensor()) {
          auto actualOpt = asTensor(debugValue(frame, actualId));
          if (actualOpt && tensorsMatch(*actualOpt, refIt->second.toTensor())) {
            state.verifiedIds.push_back(actualId);
          }
        }
      }
    }
  }

  // Re-verify all previously passed values to detect corruption.
  int32_t numCorrupted = 0;
  if (WaveConfig::get().reverify) {
    // Check inputs of current launches for corruption.
    for (const auto& data : launches) {
      for (auto actualId : data.actualInputs) {
        if (isElidedCloneInput(actualId)) {
          continue;
        }
        auto refIt = ref->find(actualId);
        if (refIt == ref->end() || !refIt->second.isTensor()) {
          continue;
        }
        auto actualOpt = asTensor(debugValue(frame, actualId));
        if (!actualOpt) {
          continue;
        }
        const at::Tensor& actualTensor = *actualOpt;
        if (!tensorsMatch(actualTensor, refIt->second.toTensor())) {
          ++numCorrupted;
          auto limit = WaveConfig::get().tensorPrintElementLimit;
          LOG(ERROR) << "INPUT CORRUPTION: value %" << actualId
                     << " no longer matches reference\n  "
                     << firstDifference(actualTensor, refIt->second.toTensor())
                     << "\n  expected: "
                     << tensorDebugString(refIt->second.toTensor(), limit)
                     << "\n  actual:   "
                     << tensorDebugString(actualTensor, limit);
        }
      }
    }
    // Re-verify previously passed outputs.
    for (auto prevId : state.verifiedIds) {
      auto refIt = ref->find(prevId);
      if (refIt == ref->end() || !refIt->second.isTensor()) {
        continue;
      }
      auto actualOpt = asTensor(frame.getIValue(prevId));
      if (!actualOpt) {
        continue;
      }
      const at::Tensor& actualTensor = *actualOpt;
      if (!tensorsMatch(actualTensor, refIt->second.toTensor())) {
        ++numCorrupted;
        auto limit = WaveConfig::get().tensorPrintElementLimit;
        LOG(ERROR) << "CORRUPTION: previously passed value %" << prevId
                   << " no longer matches reference\n  "
                   << firstDifference(actualTensor, refIt->second.toTensor())
                   << "\n  expected: "
                   << tensorDebugString(refIt->second.toTensor(), limit)
                   << "\n  actual:   "
                   << tensorDebugString(actualTensor, limit);
      }
    }
  }

  if (WaveConfig::get().trace & WaveConfig::kTensors) {
    if (!passedIds.empty()) {
      std::cout << "  Passed: " << passedIds << std::endl;
    }
  }
  if (numMismatches > 0 || numCorrupted > 0) {
    auto msg = fmt::format(
        "{} reference mismatches ({}), {} corrupted, {} passed ({})\n  first: {}",
        numMismatches,
        failedIds,
        numCorrupted,
        numPassed,
        passedIds,
        firstFailure);
    if (WaveConfig::get().continueAfterMismatch) {
      LOG(ERROR) << msg;
    } else {
      TORCH_CHECK(false, msg);
    }
  }
}

} // namespace

// Bound path: a returned value that lives in the table alone takes the dims
// the device set, with contiguous strides as resize_ would give it, and keeps
// its storage. Returns false for a value with a frame tensor.
bool readShapeIntoTable(
    nativert::ValueId id,
    const Tensor& returned,
    bool frameHoldsTensor) {
  auto* state = currentExecutionState();
  if (state == nullptr || !state->tableOnlyActive || id < 0 ||
      static_cast<size_t>(id) >= state->tableOnly.size()) {
    return false;
  }
  const bool inTableAlone = state->tableOnly[id] != 0 ||
      (!frameHoldsTensor && state->valueTable.valid(id) &&
       state->valueTable.isTensor(id));
  if (!inTableAlone) {
    return false;
  }
  auto& entry = state->valueEntries[id];
  auto& t = entry.t;
  const auto owner = state->valueCold[id].owner;
  if (owner >= 0 && t.elementSize > 0) {
    const auto& buffer = state->allocations[owner].buffer;
    const auto* end = static_cast<const char*>(buffer.storage().data()) +
        buffer.storage().nbytes();
    const auto capacity =
        (end - static_cast<const char*>(t.storage)) / t.elementSize;
    TORCH_CHECK(
        static_cast<int64_t>(returned.numEl) <= capacity,
        "Value ",
        id,
        " came back from the device with ",
        returned.numEl,
        " elements, past the end of the ",
        capacity,
        " element buffer it was carved from");
  }
  t.rank = returned.rank;
  int64_t stride = 1;
  for (int32_t d = kMaxDims - 1; d >= 0; --d) {
    if (d >= returned.rank) {
      t.dims[d] = 0;
      t.strides[d] = 0;
      continue;
    }
    t.dims[d] = returned.dims[d];
    t.strides[d] = returned.dims[d] == 1 ? 0 : static_cast<int32_t>(stride);
    stride *= returned.dims[d];
  }
  t.numEl = static_cast<uint32_t>(stride);
  return true;
}

// Bound path: a returned integer that lives in the table alone goes there and
// nowhere else.
bool readScalarIntoTable(nativert::ValueId id, int64_t value) {
  auto* state = currentExecutionState();
  if (state == nullptr || !state->tableOnlyActive || id < 0 ||
      static_cast<size_t>(id) >= state->tableOnly.size() ||
      state->tableOnly[id] == 0) {
    return false;
  }
  auto& entry = state->valueEntries[id];
  entry.scalar = value;
  entry.kind = ValueEntry::kInt;
  entry.generation = state->tableGeneration;
  return true;
}

// Resizes 'tensor' in 'frame' to match the dims returned by the device kernel.
void resizeTensorFromDevice(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    const uint8_t* pinnedBase,
    int64_t absOffset,
    bool trace) {
  auto* t = reinterpret_cast<const Tensor*>(pinnedBase + absOffset);
  if (readShapeIntoTable(id, *t, frame.getIValue(id).isTensor())) {
    return;
  }
  TORCH_CHECK(
      frame.getIValue(id).isTensor(),
      "Value %",
      id,
      " came back from the device but has no frame tensor or table entry");
  auto& tensor = frame.getIValue(id).toTensor();
  std::vector<int64_t> newDims(t->rank);
  int64_t newNumel = 1;
  for (int d = 0; d < t->rank; ++d) {
    newDims[d] = t->dims[d];
    newNumel *= newDims[d];
  }
  // An allocation-group slot is a view into a buffer it shares with the rest of
  // its group. resize_ past the end of that buffer reallocates the storage,
  // which would move it out from under every other slot -- so a grow is refused
  // for a tensor that does not own its storage outright. The device reports a
  // count at or below the reserved one, so this can only trip on a kernel that
  // has already overrun its output.
  if (tensor.has_storage() && newNumel > tensor.numel()) {
    const int64_t capacity =
        static_cast<int64_t>(tensor.storage().nbytes()) / tensor.element_size();
    TORCH_CHECK(
        (tensor.storage_offset() == 0 && capacity == tensor.numel()) ||
            tensor.storage_offset() + newNumel <= capacity,
        "Value ",
        id,
        " came back from the device with ",
        newNumel,
        " elements, past the end of the shared buffer its ",
        tensor.numel(),
        " element slot was carved from");
  }
  tensor.resize_(newDims);
  publishFromFrame(frame, id);
  if (trace) {
    std::cout << "  D2H: %" << id << " = " << traceIValue(c10::IValue(tensor))
              << std::endl;
  }
}

// Reads a scalar of type T from pinned memory and sets it in the frame.
template <typename T>
void readScalarFromDevice(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    const uint8_t* pinnedBase,
    int64_t absOffset,
    bool trace) {
  auto val = *reinterpret_cast<const T*>(pinnedBase + absOffset);
  if constexpr (std::is_same_v<T, int64_t>) {
    if (readScalarIntoTable(id, val)) {
      return;
    }
  }
  c10::IValue ival;
  if constexpr (std::is_same_v<T, double>) {
    ival = c10::IValue(val);
  } else if constexpr (std::is_same_v<T, bool>) {
    ival = c10::IValue(val);
  } else {
    ival = c10::IValue(static_cast<int64_t>(val));
  }
  writeFrame(frame, id, ival);
  if (trace) {
    if constexpr (std::is_same_v<T, bool>) {
      std::cout << "  D2H: %" << id << " = " << (val ? "true" : "false")
                << std::endl;
    } else {
      std::cout << "  D2H: %" << id << " = " << val << std::endl;
    }
  }
}

void prepareHostCopies(ExecutionState& state, StepVectors& sv) {
  for (auto& copy : sv.hostCopies) {
    copy.src = nullptr;
    copy.bytes = 0;
    if (!state.valueTable.valid(copy.id) ||
        !state.valueTable.isTensor(copy.id)) {
      continue;
    }
    const auto& t = state.valueTable.tensor(copy.id);
    copy.src = t.storage;
    copy.bytes = static_cast<int64_t>(t.numEl) * t.elementSize;
    if (copy.bytes > 0 &&
        (!copy.buffer ||
         copy.buffer->capacity() < static_cast<size_t>(copy.bytes))) {
      copy.buffer = state.pinnedArena->allocateBytes(copy.bytes);
    }
  }
}

void enqueueHostCopies(
    const StepVectors& sv,
    facebook::velox::wave::Stream& stream) {
  for (const auto& copy : sv.hostCopies) {
    if (copy.bytes > 0) {
      stream.deviceToHostAsync(
          copy.buffer->as<uint8_t>(), copy.src, copy.bytes);
    }
  }
}

void processReturnData(
    StepVectors& sv,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
  if (!sv.hostCopies.empty()) {
    if (auto* state = currentExecutionState()) {
      for (const auto& copy : sv.hostCopies) {
        auto& entry = state->valueCold[copy.id];
        entry.hostData = copy.bytes > 0 ? copy.buffer->as<uint8_t>() : nullptr;
        entry.hostGeneration = state->tableGeneration;
      }
    }
  }
  bool trace = WaveConfig::get().trace & WaveConfig::kTensors;
  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    auto& data = sv.kernels[i];
    if (data.returnValues.empty()) {
      continue;
    }
    for (size_t j = 0; j < data.returnValues.size(); ++j) {
      auto actualId = data.returnValues[j];
      auto absOffset = sv.paramOffsets.at(i) + data.returnOffsets.at(j);
      auto typeKind = data.returnTypes[j];
      if (typeKind == nativert::Type::Kind::Tensor) {
        resizeTensorFromDevice(frame, actualId, pinnedBase, absOffset, trace);
      } else if (typeKind == nativert::Type::Kind::TensorList) {
        for (const auto& tlp : data.tensorLists) {
          if (tlp.listOffset != data.returnOffsets[j]) {
            continue;
          }
          for (size_t k = 0; k < tlp.elementOffsets.size(); ++k) {
            auto elemAbsOffset = sv.paramOffsets[i] + tlp.elementOffsets[k];
            resizeTensorFromDevice(
                frame, tlp.elementIds[k], pinnedBase, elemAbsOffset, trace);
          }
          break;
        }
      } else if (typeKind == nativert::Type::Kind::SymFloat) {
        readScalarFromDevice<double>(
            frame, actualId, pinnedBase, absOffset, trace);
      } else if (typeKind == nativert::Type::Kind::SymBool) {
        auto val = *reinterpret_cast<int64_t*>(pinnedBase + absOffset) != 0;
        writeFrame(frame, actualId, c10::IValue(val));
        if (trace) {
          std::cout << "  D2H: %" << actualId << " = "
                    << (val ? "true" : "false") << std::endl;
        }
      } else {
        readScalarFromDevice<int64_t>(
            frame, actualId, pinnedBase, absOffset, trace);
      }
    }
  }
}

namespace {} // namespace

// Under WaveConfig::syncEachStep, drains both streams before a step allocates,
// so everything the preceding steps freed is back in the allocator first.
// Without it the frees trail the host by however far it has run ahead and the
// peak measures that lag rather than the release schedule.
//
// Draining BEFORE the step rather than after it also keeps a node's last step
// out of kSynced until execute() has stamped the node's leftover last-use
// values onto it: advanceSyncedStages only ever visits a kAllocated step, so
// values stamped onto one already swept would never be freed at all.
void drainBeforeStepIfRequested(ExecutionState& state) {
  if (!WaveConfig::get().syncEachStep) {
    return;
  }
  syncTorchDefaultStream();
  syncWaveStream(state);
}

namespace {} // namespace

// Latest executed step among the pending transfers whose returned values this
// step can read, or -1 when it reads none of them -- in which case the step
// runs start to finish while every one of them is still in flight. sv.readIds
// is a union over the step's grid variants, so this does not depend on a
// variant choice.
int32_t neededPendingStep(
    const ExecutionState& state,
    const folly::F14FastSet<nativert::ValueId>& readIds) {
  // Newest first: the pending list is in issue order, so the first hit is the
  // latest transfer this step needs, and resolving through it resolves every
  // earlier one too.
  for (auto it = state.pendingReturns.rbegin();
       it != state.pendingReturns.rend();
       ++it) {
    const auto& producer =
        state.stepVectors.at(it->sequenceNumber).at(it->stepIdx);
    for (const auto& data : producer.kernels) {
      for (auto id : data.returnValues) {
        if (readIds.count(id) != 0) {
          return it->executedStep;
        }
      }
    }
    for (const auto& copy : producer.hostCopies) {
      if (readIds.count(copy.id) != 0) {
        return it->executedStep;
      }
    }
  }
  return -1;
}

int32_t neededPendingStep(
    const ExecutionState& state,
    const int64_t* producers,
    int32_t numProducers) {
  if (numProducers == 0) {
    return -1;
  }
  for (auto it = state.pendingReturns.rbegin();
       it != state.pendingReturns.rend();
       ++it) {
    if (std::binary_search(
            producers,
            producers + numProducers,
            stepKey(it->sequenceNumber, it->stepIdx))) {
      return it->executedStep;
    }
  }
  return -1;
}

namespace {

// Fails if any op in 'sv' reads a value a still-pending transfer brings back.
// sv.readIds -- the union over grid variants that drove the deferral -- is a
// superset of the read paths markD2hDependencies walks here, so a hit is not a
// marginal disagreement between the two: it means the read set missed a path
// and the step has already sized and allocated against a frame value that has
// not landed. Cheap enough to run unconditionally, and the alternative symptom
// is a wrong result far downstream.
void checkNoPendingReads(
    const ExecutionState& state,
    const StepVectors& sv,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  if (state.pendingReturns.empty()) {
    return;
  }
  const int32_t oldest = state.pendingReturns.front().executedStep;
  for (const auto* launches :
       {&sv.shortcutStandalones, &sv.kernels, &sv.standalones}) {
    for (const auto& data : *launches) {
      TORCH_CHECK(
          data.d2hProducer < oldest,
          "defer_d2h: node ",
          sequenceNumber,
          " step ",
          stepIdx,
          " reads a value returned at executed step ",
          data.d2hProducer,
          " whose transfer is still pending (oldest pending step ",
          oldest,
          ")");
    }
  }
}

// Reports any op in 'sv' that reads a value already stamped for release. Such a
// read means the reader analysis behind releaseLastUseAtStep missed a reader
// and the buffer is freed while still live -- a corruption whose only other
// symptom is a wrong result far downstream. kFrame only; the scan is the same
// shape as the D2H dependency one and just as unfit for a timed run.
void checkNoReleasedReads(
    const ExecutionState& state,
    const StepVectors& sv,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  auto report = [&](nativert::ValueId id, const char* kind) {
    auto it = state.releasedAtStep.find(id);
    if (it != state.releasedAtStep.end()) {
      LOG(ERROR) << "LASTUSE-RELEASED-TOO-EARLY %" << id << " released at step "
                 << it->second << " but read by a " << kind << " of node "
                 << sequenceNumber << " step " << stepIdx;
    }
  };
  for (const auto& data : sv.shortcutStandalones) {
    for (auto id : data.actualInputs) {
      report(id, "shortcut");
    }
  }
  for (const auto* launches : {&sv.kernels, &sv.standalones}) {
    for (const auto& data : *launches) {
      for (const auto& ids :
           {data.actualInputs, data.tensorsInFrame, data.scalarsInFrame}) {
        for (auto id : ids) {
          report(id, "fused or standalone op");
        }
      }
      for (const auto& desc : data.actualOutputDescs) {
        if (desc.viewNode == nullptr) {
          continue;
        }
        for (const auto& input : desc.viewNode->inputs()) {
          if (input.value != nullptr) {
            report(input.value->id(), "host-side view operand");
          }
        }
      }
    }
  }
}

} // namespace

bool CompositeInvocation::stepLevelRelease() const {
  // Releasing a value at the last step that reads it, rather than at the node's
  // last step, needs a reader set per value. Without one -- or with a pre-pass
  // standalone, which runs outside the grids the reader set is built from --
  // every value falls back to the node's last step.
  return WaveConfig::get().freeIntermediates && WaveConfig::get().stepLastUse &&
      !lastUseIds_.empty() && lastUseReaderOps_.size() == lastUseIds_.size() &&
      prePassStandalones_.empty();
}

void CompositeInvocation::setAllocGroupPlan(
    std::unique_ptr<AllocGroupPlan> plan) {
  allocGroupPlan_ = std::move(plan);
}

void CompositeInvocation::bindLastUseSteps(
    const std::vector<GridChoice>& grids,
    int32_t numSteps) {
  // An op with a grid of L steps does its last work at step L-1, so once every
  // reader of a value is past that, no later step of this node can read it.
  // The grid is fixed at bind, so the step is too.
  lastUseByStep_.assign(numSteps, {});
  if (lastUseReaderOps_.size() != lastUseIds_.size()) {
    return;
  }
  for (size_t i = 0; i < lastUseIds_.size(); ++i) {
    const auto& readers = lastUseReaderOps_[i];
    if (readers.empty()) {
      continue;
    }
    int32_t step = 0;
    for (auto op : readers) {
      step =
          std::max(step, static_cast<int32_t>(grids.at(op).grid->size()) - 1);
    }
    if (step < numSteps) {
      lastUseByStep_[step].push_back(static_cast<int32_t>(i));
    }
  }
}

void CompositeInvocation::releaseLastUseAtStep(
    ExecutionState& state,
    int32_t stepIdx,
    StepVectors& sv,
    std::vector<int32_t>& releaseStep) {
  if (stepIdx >= static_cast<int32_t>(lastUseByStep_.size())) {
    return;
  }
  const bool traceFrame = (WaveConfig::get().trace & WaveConfig::kFrame) != 0;
  for (auto i : lastUseByStep_[stepIdx]) {
    // releaseStep is sized to lastUseIds_.size() by the only caller, which
    // also gates the call on stepLastUse; i indexes lastUseIds_.
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    if (releaseStep[i] >= 0) {
      continue;
    }
    releaseStep[i] = stepIdx;
    addLastUseId(state, sv, lastUseIds_[i]);
    if (traceFrame) {
      state.releasedAtStep[lastUseIds_[i]] = state.executedSteps;
    }
  }
}

void CompositeInvocation::execute(ExecutionState& state) {
  executeBound(state);
}

void CompositeInvocation::materializeStepGroups(
    ExecutionState& state,
    AllocGroupCollector& collector,
    StepVectors& sv,
    bool afterWait,
    bool doTiming) {
  auto& frame = *state.frame;
  for (size_t g = 0; g < collector.numGroups(); ++g) {
    if (collector.materialized(g) ||
        (!afterWait && (!collector.complete(g) || collector.needsSync(g)))) {
      continue;
    }
    // A concat group is laid out by the shape of the result rather than by
    // packing slots, and its members are the regions of that result. A
    // member the sizing pass never reached is not dropped the way an
    // ordinary group's is: the layout still has to account for its extent,
    // so it is measured from the frame and simply not carved.
    const auto* concat = collector.concatLayout(g);
    if (concat != nullptr) {
      sizeConcatReadsAhead(*concat, state);
    }
    // An operand written by a launch after the step's wait is sized only then,
    // so a concat group whose layout needs it waits for the second pass.
    if (!afterWait && concat != nullptr &&
        !concatGroupMeasurable(*concat, collector.sizedMask(g), state)) {
      continue;
    }
    int32_t numCarved = 0;
    AllocGroupBuffer buffer;
    const uint64_t tAlloc = doTiming ? folly::hardware_timestamp() : 0;
    const int64_t allocCallBefore = threadAllocCallUs();
    if (concat != nullptr) {
      buffer = materializeConcatGroupTable(
          *concat,
          collector.requests(g),
          collector.sizedMask(g),
          state,
          numCarved);
    } else {
      // A complete group carves every request; only a partial one needs the
      // copy of the sized ones.
      std::vector<AllocRequest> partial;
      if (!collector.complete(g)) {
        partial = collector.sizedRequests(g);
      }
      const auto& requests =
          collector.complete(g) ? collector.requests(g) : partial;
      if (requests.empty()) {
        collector.markMaterialized(g);
        continue;
      }
      buffer = state.tableOnlyActive
          ? materializeAllocGroupTable(requests, frame, state)
          : materializeAllocGroup(requests, frame);
      numCarved = static_cast<int32_t>(requests.size());
    }
    collector.markMaterialized(g);
    ++sv.allocGroups;
    sv.allocGroupTensors += numCarved;
    if (doTiming) {
      sv.allocUs += static_cast<int64_t>(
          static_cast<double>(folly::hardware_timestamp() - tAlloc) /
          tscTicksPerMicro());
      sv.allocCallUs += threadAllocCallUs() - allocCallBefore;
    }
    if (allocTraceEnabled()) {
      logAllocEvent("allocgroup", -1, buffer.totalBytes);
    }
  }
}

CompositeInvocation::NodeRun CompositeInvocation::startNodeRun(
    ExecutionState& state) {
  NodeRun run{.state = state};
  run.reuseCount0 = gElementwiseReuseCount;
  run.reuseBytes0 = gElementwiseReuseBytes;
  run.doTiming = WaveConfig::get().printTiming ||
      (WaveConfig::get().trace & WaveConfig::kTiming);
  run.doEventTiming = (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  run.countElidedClones = (WaveConfig::get().trace & WaveConfig::kTiming) &&
      !elidedCloneInputs_.empty();
  run.elidedCounted.assign(
      run.countElidedClones ? elidedCloneInputs_.size() : 0, false);
  run.stepLastUse = stepLevelRelease();
  run.lastUseReleaseStep.assign(run.stepLastUse ? lastUseIds_.size() : 0, -1);
  return run;
}

void CompositeInvocation::addElidedClones(NodeRun& run, StepVectors& sv) {
  const auto& frame = *run.state.frame;
  for (size_t i = 0; i < elidedCloneInputs_.size(); ++i) {
    if (run.elidedCounted[i]) {
      continue;
    }
    const auto& [valueId, numClones] = elidedCloneInputs_[i];
    const auto& ivalue = frame.getIValue(valueId);
    if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    sv.elidedCloneBytes += tensor.numel() * tensor.element_size() * numClones;
    run.elidedCounted[i] = true;
  }
}

void CompositeInvocation::finishStep(
    NodeRun& run,
    int32_t stepIdx,
    StepVectors& sv,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t totalPinnedBytes,
    int32_t returnBegin,
    int32_t returnEnd,
    const SetupCounters& setupBefore,
    std::vector<GridChoice>& currentGridChoices) {
  auto& state = run.state;
  auto& frame = *state.frame;
  using Clock = std::chrono::high_resolution_clock;
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - start)
        .count();
  };
  const bool doTiming = run.doTiming;
  const bool doEventTiming = run.doEventTiming;
  bool& ranStandalones = run.ranStandalones;
  const bool countElidedClones = run.countElidedClones;
  auto addElidedCloneBytes = [&](StepVectors& stepVectors) {
    addElidedClones(run, stepVectors);
  };
  const bool stepLastUse = run.stepLastUse;
  auto& lastUseReleaseStep = run.lastUseReleaseStep;
  int32_t& blockSize = run.blockSize;
  // Every launch's parameter block, for diffing the two step loops.
  static const bool dumpParams = getenv("TW_DUMP_PARAMS") != nullptr;
  if (dumpParams) {
    for (size_t k = 0; k < sv.kernels.size(); ++k) {
      auto* kernelOp = sv.kernels[k].launch->op;
      std::cout << "TWP node " << sequenceNumber_ << " step " << stepIdx
                << " launch " << k << " n=" << sv.kernels[k].numElements
                << " off=" << sv.paramOffsets[k] << " "
                << kernelOp->toString(sv.kernels[k].invocation) << "\n"
                << dumpOpParams(
                       *kernelOp,
                       pinnedBase + sv.paramOffsets[k],
                       sv.kernels[k].invocation)
                << std::endl;
    }
  }
  markD2hDependencies(state, sv);
  if (!state.pendingReturns.empty()) {
    checkNoPendingReads(state, sv, sequenceNumber_, stepIdx);
  }
  if (doTiming) {
    countD2hDependencies(state, sv);
  }
  recordD2hProducers(state, sv);
  if (WaveConfig::get().trace & WaveConfig::kFrame) {
    checkNoReleasedReads(state, sv, sequenceNumber_, stepIdx);
  }
  sv.executedStep = state.executedSteps;
  ++state.executedSteps;
  if (stepLastUse) {
    releaseLastUseAtStep(state, stepIdx, sv, lastUseReleaseStep);
  }

  if (sv.kernels.empty()) {
    if (WaveConfig::get().trace &
        (WaveConfig::kNodes | WaveConfig::kLaunches)) {
      traceStep(stepIdx, sv, currentGridChoices);
    }
    runShortcutStandalones(
        sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
    auto& stepEvents = newStepEvents(state, sequenceNumber_, stepIdx);
    if (sv.hasGpuStandalones) {
      if (state.lastWaveDone != nullptr) {
        waitOnce(state.lastWaveDone, torchStream(), state.torchWaitedFor);
      }
      if (doEventTiming) {
        stepEvents.standaloneBegin->record(torchStream());
      }
    }
    auto tStandalone = doTiming ? Clock::now() : Clock::time_point{};
    runStandalones(
        sv.standalones,
        state,
        *state.kernelMap,
        *state.standaloneIndices,
        *state.standaloneStats,
        doTiming);
    if (sv.hasGpuStandalones) {
      recordStepDone(stepEvents, torchStream(), /*wave=*/false);
      state.lastStandaloneDone = stepEvents.standaloneDone.get();
    }
    if (doTiming) {
      sv.standaloneUs = elapsed(tStandalone);
      sv.currentBytes = currentAllocatedBytes();
    }
    if (countElidedClones) {
      addElidedCloneBytes(sv);
    }
    ranStandalones = true;
    state.launchDebugInfos.push_back(
        {nullptr, nullptr, 0, sequenceNumber_, stepIdx});
    {
      bool timeRefCheck = doTiming && WaveConfig::get().referenceFrame;
      if (timeRefCheck) {
        syncWaveStream(state);
        syncTorchDefaultStream();
      }
      if (WaveConfig::get().referenceFrame != nullptr) {
        resolveAllPendingReturns(state);
      }
      auto tRefCheck = timeRefCheck ? Clock::now() : Clock::time_point{};
      verifyAgainstReference(sv.shortcutStandalones, frame, state);
      verifyAgainstReference(sv.standalones, frame, state);
      if (timeRefCheck) {
        sv.refCheckUs += elapsed(tRefCheck);
      }
    }
    sv.executionStage = ExecutionStage::kAllocated;
    return;
  }

  if (!state.traceState.empty()) {
    for (const auto& launch : sv.kernels) {
      traceFrameValues("input", launch.actualInputs, frame, state.traceState);
    }
  }

  auto* deviceDebugBase =
      reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
  if (doTiming) {
    sv.inputBytes = 0;
    sv.outputBytes = 0;
    for (size_t i = 0; i < sv.kernels.size(); ++i) {
      for (size_t j = 0; j < sv.kernels[i].tensorsInFrame.size(); ++j) {
        auto off = sv.kernels[i].tensorOffsets[j];
        auto* t =
            reinterpret_cast<Tensor*>(pinnedBase + sv.paramOffsets[i] + off);
        auto bytes = static_cast<int64_t>(t->numEl) * t->elementSize;
        if (sv.kernels[i].shapeOnlyTensorIndices.count(j)) {
          continue;
        }
        if (j < static_cast<size_t>(sv.kernels[i].launch->op->numInputs())) {
          sv.inputBytes += bytes;
        } else {
          sv.outputBytes += bytes;
        }
      }
    }
  }

  {
    auto t0 = Clock::now();
    if (gridSizesMatch(sv.kernels, sv)) {
      blockSize = sv.cachedBlockSize;
    } else {
      blockSize = makeGrid(
          sv.kernels,
          sv,
          sv.blockSize,
          kernel_->maxOccupancy(sv.entry),
          /*staticSharedPerBlock=*/0,
          occupancyQuery(kernel_.get(), sv.entry));
      TORCH_CHECK(
          (blockSize & (blockSize - 1)) == 0,
          "Block size must be a power of two, got ",
          blockSize);
      sv.cachedBlockSize = blockSize;
      ++sv.gridVersion;
      updateGridSizeBounds(sv.kernels, sv);
    }
    if (doTiming) {
      sv.gridUs = elapsed(t0);
    }
    if (WaveConfig::get().trace & WaveConfig::kGrid) {
      traceGrid(sequenceNumber_, stepIdx, sv);
    }
  }

  auto numBlocks = sv.blocks.size();
  TORCH_CHECK(
      numBlocks <= static_cast<size_t>(sv.blockCapacity),
      "makeGrid produced ",
      numBlocks,
      " blocks for node ",
      sequenceNumber_,
      " step ",
      stepIdx,
      " but the buffer was reserved for ",
      sv.blockCapacity);
  {
    auto t0 = Clock::now();
    writeBlockInfos(sv, pinnedBase, deviceBase, deviceDebugBase);
    if (doTiming) {
      sv.blockInfoUs = elapsed(t0);
      sv.fillUs += sv.blockInfoUs;
    }
  }

  if (WaveConfig::get().trace & (WaveConfig::kNodes | WaveConfig::kLaunches)) {
    traceStep(stepIdx, sv, currentGridChoices);
  }

  state.launchDebugInfos.push_back(
      {reinterpret_cast<DebugInfo*>(pinnedBase + totalPinnedBytes),
       deviceDebugBase,
       static_cast<int32_t>(numBlocks),
       sequenceNumber_,
       stepIdx});

  auto& stepEvents = newStepEvents(state, sequenceNumber_, stepIdx);
  int64_t standaloneElapsed = 0;
  auto runStepStandalones = [&]() {
    if (!sv.shortcutStandalones.empty()) {
      runShortcutStandalones(
          sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
    }
    if (!sv.standalones.empty()) {
      if (state.lastWaveDone != nullptr) {
        waitOnce(state.lastWaveDone, torchStream(), state.torchWaitedFor);
      }
      if (doEventTiming) {
        stepEvents.standaloneBegin->record(torchStream());
      }
      auto tStandalone = doTiming ? Clock::now() : Clock::time_point{};
      runStandalones(
          sv.standalones,
          state,
          *state.kernelMap,
          *state.standaloneIndices,
          *state.standaloneStats,
          doTiming);
      recordStepDone(stepEvents, torchStream(), /*wave=*/false);
      state.lastStandaloneDone = stepEvents.standaloneDone.get();
      if (doTiming) {
        standaloneElapsed = elapsed(tStandalone);
      }
      ranStandalones = true;
    }
  };

  if (state.lastStandaloneDone != nullptr) {
    waitOnce(state.lastStandaloneDone, *state.stream, state.waveWaitedFor);
  }
  if (doEventTiming) {
    stepEvents.waveBegin->record(*state.stream);
  }

  prepareHostCopies(state, sv);
  const bool hasTransfer = returnBegin >= 0 || !sv.hostCopies.empty();
  const bool deferReturn = WaveConfig::get().deferD2h && hasTransfer &&
      !WaveConfig::get().debugSingleOps;
  {
    auto tLaunch = Clock::now();
    sv.setup = threadSetupCounters() - setupBefore;
    launch(
        static_cast<int32_t>(numBlocks),
        blockSize,
        pinnedBase,
        deviceBase,
        sv.blockInfoOffset +
            static_cast<int64_t>(numBlocks) *
                static_cast<int64_t>(sizeof(BlockInfo)),
        returnBegin,
        returnEnd,
        deviceDebugBase,
        state.stream.get(),
        sv,
        stepIdx,
        deferReturn,
        runStepStandalones,
        &stepEvents);
    state.lastWaveDone = stepEvents.waveDone.get();
    advanceCompletedStages(state);

    if (deferReturn) {
      state.pendingReturns.push_back(
          {sequenceNumber_,
           stepIdx,
           sv.executedStep,
           stepEvents.waveDone.get()});
    } else if (hasTransfer) {
      processReturnData(sv, frame, pinnedBase);
    }
    if (doTiming) {
      sv.kernelUs = elapsed(tLaunch);
      sv.standaloneUs = standaloneElapsed;
      sv.standaloneBound = standaloneElapsed > sv.kernelUs;
      sv.noDtoH = (returnBegin < 0);
      sv.currentBytes = currentAllocatedBytes();
    }
    if (countElidedClones) {
      addElidedCloneBytes(sv);
    }
  }

  if (!state.traceState.empty()) {
    syncWaveStream(state);
    resolveAllPendingReturns(state);
    for (const auto& launch : sv.kernels) {
      traceFrameValues("output", launch.actualOutputs, frame, state.traceState);
    }
  }

  {
    bool timeRefCheck = doTiming && WaveConfig::get().referenceFrame;
    if (timeRefCheck) {
      syncWaveStream(state);
      syncTorchDefaultStream();
    }
    if (WaveConfig::get().referenceFrame != nullptr) {
      resolveAllPendingReturns(state);
    }
    auto tRefCheck = timeRefCheck ? Clock::now() : Clock::time_point{};
    verifyAgainstReference(sv.shortcutStandalones, frame, state);
    verifyAgainstReference(sv.standalones, frame, state);
    verifyAgainstReference(sv.kernels, frame, state);
    if (timeRefCheck) {
      sv.refCheckUs += elapsed(tRefCheck);
    }
  }
  sv.executionStage = ExecutionStage::kAllocated;
}

void CompositeInvocation::finishNode(NodeRun& run) {
  auto& state = run.state;
  const bool ranStandalones = run.ranStandalones;
  const int32_t lastExecStep = run.lastExecStep;
  const bool stepLastUse = run.stepLastUse;
  const auto& lastUseReleaseStep = run.lastUseReleaseStep;
  const int64_t reuseCount0 = run.reuseCount0;
  const int64_t reuseBytes0 = run.reuseBytes0;
  if (ranStandalones && !WaveConfig::get().runAhead) {
    syncTorchDefaultStream();
  }

  if (WaveConfig::get().freeIntermediates && lastExecStep >= 0 &&
      !lastUseIds_.empty()) {
    auto& lastSv =
        getStepVectors(state.stepVectors, sequenceNumber_, lastExecStep);
    std::vector<nativert::ValueId> atNodeEnd;
    for (size_t i = 0; i < lastUseIds_.size(); ++i) {
      const int32_t releasedAt = stepLastUse ? lastUseReleaseStep[i] : -1;
      if (releasedAt < 0) {
        atNodeEnd.push_back(lastUseIds_[i]);
      }
      if (releasedAt < 0 || releasedAt == lastExecStep) {
        ++state.numLastUseAtNodeEnd;
      } else {
        ++state.numLastUseEarly;
      }
    }
    if (lastSv.executionStage == ExecutionStage::kSynced) {
      resolvePendingReturns(state, lastSv.executedStep);
      freeLastUseNow(state, atNodeEnd);
    } else {
      for (auto id : atNodeEnd) {
        addLastUseId(state, lastSv, id);
      }
    }
  }

  if ((WaveConfig::get().trace & WaveConfig::kTiming) &&
      gElementwiseReuseCount > reuseCount0) {
    std::cout << "  node " << sequenceNumber_ << " elementwise input reuse: "
              << (gElementwiseReuseCount - reuseCount0) << " tensors, "
              << (gElementwiseReuseBytes - reuseBytes0) / 1024
              << " KB written in place" << std::endl;
  }
}

void CompositeInvocation::launch(
    int32_t numBlocks,
    int32_t blockSize,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t h2dBytes,
    int32_t returnBegin,
    int32_t returnEnd,
    DebugInfo* deviceDebugBase,
    facebook::velox::wave::Stream* stream,
    const StepVectors& sv,
    int32_t stepIdx,
    bool deferReturn,
    const std::function<void()>& betweenLaunchAndSync,
    StepEvents* events) {
  TorchWaveParams params{};
  params.info = reinterpret_cast<BlockInfo*>(deviceBase + sv.blockInfoOffset);
  params.debugInfo = deviceDebugBase;
  void* args[] = {&params};

  // Ops declare their extern __shared__ needs through
  // Metadata::dynamicSharedMemory; the ops of a step share one launch, so the
  // launch takes the max. Steps with no such op launch with zero and keep the
  // occupancy they would have had.
  const int32_t dynShared = dynamicSharedBytes(sv.kernels, blockSize);

  // opBarrier (Core.cuh) is a counter spin-wait that blocks until numBlocksInOp
  // blocks have arrived, so it needs those blocks co-resident -- which only a
  // cooperative launch guarantees. A barrier op assigned a single block passes
  // its barrier immediately (the count reaches 1 as soon as that block runs),
  // so it needs no co-residency. sv.isCgGrid merely marks "this step has a
  // barrier op"; a cooperative launch is only actually required when some
  // barrier op spans more than one block. Refining the decision here lets a
  // wide fan-out of single-block ops (whose total block count can exceed the
  // device co-residency limit) launch normally instead of failing the
  // cooperative launch's block cap ("too many blocks in cooperative launch").
  bool cooperative = false;
  if (sv.isCgGrid) {
    for (size_t ki = 0; ki < sv.kernels.size(); ++ki) {
      const auto& kd = sv.kernels.at(ki);
      if (kd.launch && kd.launch->op &&
          !kd.launch->op->barrierCounters().empty() &&
          ki < sv.numBlocksPerLaunch.size() &&
          sv.numBlocksPerLaunch.at(ki) > 1) {
        cooperative = true;
        break;
      }
    }
  }

  auto* pinnedBlocks =
      reinterpret_cast<BlockInfo*>(pinnedBase + sv.blockInfoOffset);

  // Ordinarily one launch over every block of the step. A step the packer
  // split runs its launches back to back on the same stream, each over its own
  // slice of the block array and with only the shared memory its own ops need
  // -- which is where the occupancy a split buys actually lands.
  std::vector<LaunchSegment> wholeStep;
  if (sv.segments.empty()) {
    wholeStep.push_back(
        {.firstBlock = 0,
         .numBlocks = numBlocks,
         .dynamicShared = dynShared,
         .cooperative = cooperative});
  }
  const auto& segments = sv.segments.empty() ? wholeStep : sv.segments;
  auto* deviceBlocks =
      reinterpret_cast<BlockInfo*>(deviceBase + sv.blockInfoOffset);
  auto launchSegment = [&](const LaunchSegment& segment) {
    params.info = deviceBlocks + segment.firstBlock;
    // ENTRY reads the block's DebugInfo at blockIdx.x of whatever base it is
    // handed, so the base has to move with the segment; otherwise every launch
    // would overwrite the first one's per-block timings.
    params.debugInfo = deviceDebugBase != nullptr
        ? deviceDebugBase + segment.firstBlock
        : nullptr;
    // A step that was not split keeps deciding this here, from the block
    // counts as they stand at launch time rather than as makeGrid left them.
    // The two agree, but this one cannot go stale behind a reused grid.
    const bool launchCooperative =
        segments.size() > 1 ? segment.cooperative : cooperative;
    if (launchCooperative) {
      kernel_->launchCooperative(
          sv.entry, segment.numBlocks, segment.dynamicShared, stream, args);
    } else {
      kernel_->launch(
          sv.entry, segment.numBlocks, segment.dynamicShared, stream, args);
    }
  };

  if (WaveConfig::get().debugSingleOps) {
    std::vector<int32_t> originalOps(numBlocks);
    for (int32_t b = 0; b < numBlocks; ++b) {
      originalOps[b] = pinnedBlocks[b].op;
    }

    // Transfer pinned buffer to device once.
    stream->hostToDeviceAsync(deviceBase, pinnedBase, h2dBytes);
    stream->wait();

    // One op at a time, inside one launch at a time. maybePartition declines to
    // partition under this flag, but a step whose grid was laid out by the
    // normal pass keeps that pass's segments, and a step split because it
    // exceeded the cooperative capacity must be walked segment by segment:
    // launching its whole block array as one cooperative grid is wider than the
    // device holds co-resident, and fails here alone.
    for (const auto& segment : segments) {
      // Per launch, not per step: an op with blocks in two launches has to run
      // in both. Hoisting this out of the loop would silently drop its second
      // half and leave the output partly written.
      folly::F14FastSet<uintptr_t> launched;
      const int32_t segFirst = segment.firstBlock;
      const int32_t segEnd = segment.firstBlock + segment.numBlocks;
      params.info = deviceBlocks + segFirst;
      params.debugInfo =
          deviceDebugBase != nullptr ? deviceDebugBase + segFirst : nullptr;
      for (int32_t active = segFirst; active < segEnd; ++active) {
        // launchIndices has one entry per block; at() so a short vector fails
        // loudly instead of reading out of bounds.
        auto launchIdx = sv.launchIndices.at(active);
        // A barrier op needs cooperative grouping only when it spans more than
        // one block (see 'cooperative' above): opBarrier waits for
        // numBlocksInOp arrivals, which is immediate for a single-block op.
        bool hasBarriers =
            launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
            sv.kernels.at(launchIdx).launch &&
            sv.kernels.at(launchIdx).launch->op &&
            !sv.kernels.at(launchIdx).launch->op->barrierCounters().empty() &&
            launchIdx < static_cast<int32_t>(sv.numBlocksPerLaunch.size()) &&
            sv.numBlocksPerLaunch.at(launchIdx) > 1;

        // Only a barrier op has to move as a unit. Everything else advances one
        // block per launch, which is what makes a failure land on a block
        // rather than on an op. A cg step still launches cooperatively -- the
        // kernel was compiled that way -- it just runs with one live block.
        bool groupAll = hasBarriers;

        // Quiet this launch's blocks; the rest are not in the grid.
        setOpCodes(
            deviceBlocks, segFirst, segment.numBlocks, kDebugNoOp, stream);

        if (groupAll) {
          auto* inv = sv.kernels.at(launchIdx).invocation;
          if (!launched.insert(reinterpret_cast<intptr_t>(inv)).second) {
            continue;
          }
          // Activate the op's blocks inside this launch. A barrier op is never
          // split across launches, so this reaches all of them.
          for (int32_t b = segFirst; b < segEnd; ++b) {
            auto bIdx = sv.launchIndices.at(b);
            bool sameOp = bIdx < static_cast<int32_t>(sv.kernels.size()) &&
                sv.kernels.at(bIdx).invocation == inv;
            if (sameOp) {
              setOpCodes(deviceBlocks, b, 1, originalOps[b], stream);
            }
          }
        } else {
          setOpCodes(deviceBlocks, active, 1, originalOps[active], stream);
        }

        // Reset barrier counters on device for the active op. Ops without
        // barriers have an empty barrierCounters(), so this loop is a no-op for
        // them even when it runs under a cooperative grid.
        if (groupAll) {
          for (size_t li = 0; li < sv.kernels.size(); ++li) {
            if (sv.kernels[li].invocation == sv.kernels[launchIdx].invocation) {
              auto* kernelOp = sv.kernels[li].launch->op;
              for (auto offset : kernelOp->barrierCounters()) {
                int32_t zero = 0;
                auto* dest = deviceBase + sv.paramOffsets.at(li) + offset;
                stream->hostToDeviceAsync(dest, &zero, sizeof(zero));
              }
            }
          }
        }

        try {
          if (groupAll || segment.cooperative) {
            kernel_->launchCooperative(
                kernel_->entryFor(blockSize, stepIdx),
                segment.numBlocks,
                segment.dynamicShared,
                stream,
                args);
          } else {
            kernel_->launch(
                kernel_->entryFor(blockSize, stepIdx),
                segment.numBlocks,
                segment.dynamicShared,
                stream,
                args);
          }
          stream->wait();
        } catch (const std::exception& e) {
          auto opCode = originalOps[active];
          std::string opText;
          std::string paramText;
          if (launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
              sv.kernels.at(launchIdx).launch &&
              sv.kernels.at(launchIdx).launch->op) {
            auto* kernelOp = sv.kernels.at(launchIdx).launch->op;
            opText = kernelOp->toString(sv.kernels.at(launchIdx).invocation);
            auto* opParams = pinnedBase + sv.paramOffsets.at(launchIdx);
            paramText = dumpOpParams(
                *kernelOp, opParams, sv.kernels.at(launchIdx).invocation);
          }
          LOG(ERROR) << "debug_single_ops: block " << active << " opCode "
                     << opCode << " blockInOp "
                     << pinnedBlocks[active].blockInOp << " stepIdx " << stepIdx
                     << " op: " << opText << "\nparams:\n"
                     << paramText << "error: " << e.what();
          throw;
        }
      }
    }

    // D2H transfer after all blocks complete.
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
    }
    if (returnBegin >= 0 || !sv.hostCopies.empty()) {
      enqueueHostCopies(sv, *stream);
      stream->wait();
    }
    // The debug path already syncs after every block; just close the event
    // chain so the ordering edges stay consistent with the normal path.
    if (events != nullptr) {
      recordStepDone(*events, *stream, /*wave=*/true);
    }

    if (betweenLaunchAndSync) {
      betweenLaunchAndSync();
    }
  } else {
    stream->hostToDeviceAsync(deviceBase, pinnedBase, h2dBytes);
    for (const auto& segment : segments) {
      if (!WaveConfig::get().launchPerOp) {
        launchSegment(segment);
        continue;
      }
      // An op's blocks are contiguous in the step's block array, so each run
      // of one launch index is one op.  A barrier op's blocks stay together
      // in their run, which keeps them in one launch.
      const int32_t segmentEnd = segment.firstBlock + segment.numBlocks;
      for (int32_t runStart = segment.firstBlock; runStart < segmentEnd;) {
        int32_t runEnd = runStart + 1;
        while (runEnd < segmentEnd &&
               sv.launchIndices.at(runEnd) == sv.launchIndices.at(runStart)) {
          ++runEnd;
        }
        LaunchSegment opSegment = segment;
        opSegment.firstBlock = runStart;
        opSegment.numBlocks = runEnd - runStart;
        launchSegment(opSegment);
        runStart = runEnd;
      }
    }
    if (returnBegin >= 0 || !sv.hostCopies.empty()) {
      if (returnBegin >= 0) {
        stream->deviceToHostAsync(
            pinnedBase + returnBegin,
            deviceBase + returnBegin,
            returnEnd - returnBegin);
      }
      enqueueHostCopies(sv, *stream);
      // Record before the host wait, so the timestamp is the kernel/D2H
      // completion rather than the host's observation of it, and so the
      // ordering edge is established as early as possible.
      if (events != nullptr) {
        recordStepDone(*events, *stream, /*wave=*/true);
      }
      if (betweenLaunchAndSync) {
        betweenLaunchAndSync();
      }
      if (!deferReturn) {
        // A real host wait: the caller reads the pinned buffer as soon as this
        // returns. With deferReturn the caller records waveDone instead and
        // the wait moves to the first later step that reads the data.
        stream->wait();
      }
    } else {
      if (events != nullptr) {
        recordStepDone(*events, *stream, /*wave=*/true);
      }
      if (betweenLaunchAndSync) {
        betweenLaunchAndSync();
      }
      // No wait under kTiming any more. Its only purpose was to make the
      // host-measured kernelUs approximate GPU time; the event pair does that
      // properly, and the wait destroyed the overlap this is here to measure.
    }
  }
}

void CompositeInvocation::traceStep(
    int32_t stepIdx,
    const StepVectors& sv,
    const std::vector<GridChoice>& gridChoices) {
  std::cout << "== " << sequenceNumber_ << " step " << stepIdx << std::endl;

  // Build map from OpInvocation* to its index in ops.
  std::unordered_map<const OpInvocation*, int32_t> opInvocationIndex;
  for (size_t i = 0; i < ops_.size(); ++i) {
    opInvocationIndex[&ops_[i]] = static_cast<int32_t>(i);
  }

  // Build map from ProjectOperation* to its distinct ordinal.
  std::unordered_map<ProjectOperation*, int32_t> projectOpIndex;
  for (const auto& op : ops_) {
    auto* projectOp = op.projectOp();
    if (projectOpIndex.find(projectOp) == projectOpIndex.end()) {
      projectOpIndex[projectOp] = static_cast<int32_t>(projectOpIndex.size());
    }
  }

  for (const auto& launch : sv.standalones) {
    auto opIdx = opInvocationIndex[launch.invocation];
    std::cout << sequenceNumber_ << "." << opIdx << " standalone "
              << standaloneToString(launch.standalone);
  }

  for (const auto& launch : sv.shortcutStandalones) {
    auto opIdx = opInvocationIndex[launch.invocation];
    std::cout << sequenceNumber_ << "." << opIdx << " shortcut "
              << standaloneToString(launch.standalone);
  }

  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    const auto& launch = sv.kernels[i];
    auto opIdx = opInvocationIndex[launch.invocation];
    auto distinctOpIdx = projectOpIndex[launch.invocation->projectOp()];
    auto* projectOp = launch.invocation->projectOp();
    const char* gridLabel = "M";
    if (gridChoices[opIdx].singleBlock) {
      gridLabel = "S";
    } else if (gridChoices[opIdx].grid == &projectOp->cgGrid()) {
      gridLabel = "CG";
    }
    std::string opStr;
    if (launch.standalone) {
      opStr = "standalone";
    } else {
      opStr = launch.launch->op->toString(launch.invocation);
    }
    std::cout << sequenceNumber_ << "." << opIdx << " " << gridLabel << " op "
              << distinctOpIdx << " " << launch.numElements
              << " blocks=" << sv.numBlocksPerLaunch.at(i)
              << " opcode=" << launch.launch->op->opCode() << " " << opStr
              << std::endl;
  }
}

std::string CompositeInvocation::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;

  // Collect distinct ProjectOperations.
  std::vector<ProjectOperation*> projectOps;
  std::unordered_map<ProjectOperation*, int32_t> projectOpIndex;
  for (const auto& op : ops_) {
    auto* po = op.projectOp();
    if (projectOpIndex.find(po) == projectOpIndex.end()) {
      projectOpIndex[po] = static_cast<int32_t>(projectOps.size());
      projectOps.push_back(po);
    }
  }

  // Print OpInvocations with their ProjectOperation ordinal and bindings.
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto it = projectOpIndex.find(ops_[i].projectOp());
    ss << ordinal << "." << i << ": ProjectOp " << it->second;
    const auto& bindings = ops_[i].bindings();
    for (const auto& [formalId, actualId] : bindings) {
      ss << " %" << formalId << " = %" << actualId;
    }
    ss << "\n";
  }

  // Print distinct ops with their grids.
  ss << "\nDistinct Ops\n";
  for (size_t i = 0; i < projectOps.size(); ++i) {
    ss << "Op " << ordinal << "." << i << "\n";
    ss << projectOps[i]->toString(mode);
  }

  return ss.str();
}

// --- CompiledNode ---

void CompiledNode::execute(ExecutionState& state) {
  kernels_->execute(state);
}

std::string CompiledNode::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;
  if (mode == kExprs) {
    // Collect distinct ProjectOperations and count invocations.
    std::vector<ProjectOperation*> projectOps;
    std::unordered_map<ProjectOperation*, int32_t> invocationCount;
    for (const auto& op : kernels_->ops()) {
      auto* po = op.projectOp();
      if (invocationCount[po]++ == 0) {
        projectOps.push_back(po);
      }
    }
    for (auto* po : projectOps) {
      ss << invocationCount[po] << "x " << po->subgraph().toString(mode)
         << "\n";
    }
  } else {
    ss << kernels_->toString(mode, ordinal);
  }
  return ss.str();
}

} // namespace torch::wave
