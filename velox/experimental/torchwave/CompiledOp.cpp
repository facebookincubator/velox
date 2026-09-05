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
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/wave/common/KernelFsCache.h"

#include <ATen/ATen.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
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

void fillEmptyTensorParam(void* dest) {
  memset(dest, 0, sizeof(Tensor));
}

void fillShapeOnlyTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= kMaxDims,
      "Tensors with more than ",
      kMaxDims,
      " dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = nullptr;
  t->rank = static_cast<int8_t>(tensor.dim());
  for (int i = 0; i < kMaxDims; ++i) {
    t->dims[i] = i < tensor.dim() ? static_cast<int32_t>(tensor.size(i)) : 0;
    t->strides[i] = 0;
  }
  t->numEl = static_cast<uint32_t>(tensor.numel());
  t->status = Tensor::kUninited;
}

void fillTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= kMaxDims,
      "Tensors with more than ",
      kMaxDims,
      " dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = tensor.data_ptr();
  t->rank = static_cast<int8_t>(tensor.dim());
  t->elementSize = tensor.element_size();
  t->elementType = static_cast<uint8_t>(tensor.scalar_type());
  for (int i = 0; i < kMaxDims; ++i) {
    t->dims[i] = i < tensor.dim() ? static_cast<int32_t>(tensor.size(i)) : 0;
    t->strides[i] = i < tensor.dim()
        ? (tensor.size(i) == 1 ? 0 : static_cast<int32_t>(tensor.stride(i)))
        : 0;
  }
  t->numEl = static_cast<uint32_t>(tensor.numel());
  t->status = Tensor::kUninited;
}

std::string tensorToString(const Tensor& t) {
  std::stringstream ss;
  ss << "Tensor{storage=" << t.storage << " rank=" << static_cast<int>(t.rank)
     << " dims=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.dims[i];
  }
  ss << "] strides=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.strides[i];
  }
  ss << "] numEl=" << t.numEl << " status=" << t.status << "}";
  return ss.str();
}

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

} // namespace

at::Tensor paramTensor(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %",
      value->id(),
      " not found in FormalToActual map");
  const auto& iv = frame.getIValue(it->second);
  TORCH_CHECK(
      iv.isTensor(),
      "paramTensor: actual value %",
      it->second,
      " (formal %",
      value->id(),
      ") is not a tensor (tag=",
      iv.tagKind(),
      ") -- freed while still needed?");
  return frame.getTensor(it->second);
}

int64_t paramSymInt(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %",
      value->id(),
      " not found in FormalToActual map");
  return frame.getSymInt(it->second);
}

namespace {

NodeCP actualNode(NodeCP formalNode, const NodeMap& nodeMap) {
  auto it = nodeMap.find(formalNode);
  return it != nodeMap.end() ? it->second : formalNode;
}

} // namespace

int64_t paramIntByName(
    NodeCP node,
    std::string_view name,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    const NodeMap& nodeMap) {
  auto* input = node->tryGetInput(name);
  if (input) {
    return paramSymInt(input->value, frame, map);
  }
  auto* actual = actualNode(node, nodeMap);
  const auto* attr = actual->tryGetAttribute(name);
  TORCH_CHECK(
      attr, actual->target(), ": missing input or attribute '", name, "'");
  return std::get<int64_t>(attr->value);
}

std::vector<int64_t> paramIntListByName(
    NodeCP node,
    std::string_view name,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    const NodeMap& nodeMap) {
  auto* input = node->tryGetInput(name);
  if (input) {
    auto it = map.find(input->value->id());
    TORCH_CHECK(it != map.end(), node->target(), ": '", name, "' not in map");
    auto& ivalue = frame.getIValue(it->second);
    if (!ivalue.isNone()) {
      return ivalue.toIntVector();
    }
    auto& idToValue = waveGraph()->idToValue();
    auto valueIt = idToValue.find(it->second);
    TORCH_CHECK(
        valueIt != idToValue.end(),
        node->target(),
        ": '",
        name,
        "' value id not in idToValue");
    auto* producer = valueIt->second->producer();
    TORCH_CHECK(
        producer && producer->target() == "prim.ListPack",
        node->target(),
        ": expected prim.ListPack producer for '",
        name,
        "'");
    // The pack was resolved through idToValue, which is keyed by actual ids, so
    // its operands are already actual Values. 'map' is keyed by formal ids and
    // normally has no entry for them; fall back to the id as-is, the same way
    // repeatReserve resolves its self tensor.
    std::vector<int64_t> result;
    for (const auto& elem : producer->inputs()) {
      auto elemId = elem.value->id();
      if (auto elemIt = map.find(elemId); elemIt != map.end()) {
        elemId = elemIt->second;
      }
      result.push_back(frame.getIValue(elemId).toInt());
    }
    return result;
  }
  auto* actual = actualNode(node, nodeMap);
  const auto* attr = actual->tryGetAttribute(name);
  TORCH_CHECK(
      attr, actual->target(), ": missing input or attribute '", name, "'");
  return std::get<std::vector<int64_t>>(attr->value);
}

std::vector<std::vector<Dim>> elementwiseInputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    int32_t ordinal) {
  std::vector<int64_t> bestShape;
  int64_t bestNumel = -1;

  std::function<void(ValueCP)> trace = [&](ValueCP v) {
    auto it = map.find(v->id());
    if (it != map.end()) {
      auto& ivalue = frame.getIValue(it->second);
      if (ivalue.isTensor()) {
        auto numel = ivalue.toTensor().numel();
        if (numel > bestNumel) {
          bestNumel = numel;
          auto sizes = ivalue.toTensor().sizes();
          bestShape.assign(sizes.begin(), sizes.end());
        }
        return;
      }
    }
    auto* producer = v->producer();
    if (!producer) {
      return;
    }
    auto* meta = Registry::metadata(producer->target());
    if (meta && meta->elementwise) {
      for (const auto& input : producer->inputs()) {
        trace(input.value);
      }
    }
  };

  trace(node->inputs()[ordinal].value);

  TORCH_CHECK(
      bestNumel >= 0, "elementwiseInputShape: no tensor found in frame");
  return {std::vector<Dim>(bestShape.begin(), bestShape.end())};
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
// Dynamic shared memory a launch of 'launches' needs: the max over the ops,
// since the ops share one kernel launch.
int32_t dynamicSharedBytes(const std::vector<LaunchData>& launches) {
  int64_t bytes = 0;
  for (const auto& launch : launches) {
    if (launch.launch && launch.launch->op) {
      bytes = std::max(bytes, launch.launch->op->dynamicSharedBytes());
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
int32_t targetBlockCount(
    int32_t maxBlocksPerSM,
    int32_t dynSharedPerBlock,
    int32_t staticSharedPerBlock) {
  auto* device = facebook::velox::wave::currentDevice();
  int32_t numSMs = WaveConfig::get().numSms;
  if (numSMs == 0) {
    numSMs = device ? device->numSM : kDefaultNumSMs;
  }
  int32_t blocksPerSM =
      maxBlocksPerSM > 0 ? maxBlocksPerSM : kDefaultBlocksPerSM;
  if (dynSharedPerBlock > 0) {
    const int32_t sharedPerSM =
        device ? device->sharedMemPerSM : kDefaultSharedMemPerSM;
    blocksPerSM = std::max(
        1,
        std::min(
            blocksPerSM,
            sharedPerSM / (dynSharedPerBlock + staticSharedPerBlock)));
  }
  return numSMs * blocksPerSM;
}

int32_t makeGrid(
    std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM,
    int32_t staticSharedPerBlock) {
  const int32_t blockSize = WaveConfig::get().blockSize;

  // Compute cost per launch: numElements * unitCost * costAdjustFactor.
  sv.costs.resize(launches.size());
  float totalCost = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    float adjust =
        launches[i].costAdjustFactor > 0 ? launches[i].costAdjustFactor : 1.0f;
    sv.costs[i] = static_cast<float>(launches[i].numElements) *
        launches[i].launch->op->unitCost() * adjust;
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
    float costPerInput = launches[i].launch->op->unitCost() * adjust;
    int32_t elementsPerBlock =
        blockSize * elementsPerThreadForCost(costPerInput);
    sv.maxBlocks[i] = static_cast<int32_t>(
        (launches[i].numElements + elementsPerBlock - 1) / elementsPerBlock);
    if (sv.maxBlocks[i] < 1) {
      sv.maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count and kernel occupancy.
  int32_t maxBlocks = targetBlockCount(
      maxBlocksPerSM, dynamicSharedBytes(launches), staticSharedPerBlock);
  int32_t targetBlocks = maxBlocks;

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

  // For cooperative grids, cap total blocks at what the GPU can run
  // concurrently.
  if (sv.isCgGrid && totalAssigned > targetBlocks) {
    // Trim excess blocks from launches with the most blocks first,
    // preserving the proportional allocation for small launches.
    while (totalAssigned > targetBlocks) {
      int32_t before = totalAssigned;
      // Find the current max block count.
      int32_t maxVal = 1;
      for (size_t i = 0; i < launches.size(); ++i) {
        maxVal = std::max(maxVal, sv.numBlocksPerLaunch[i]);
      }
      if (maxVal <= 1) {
        break;
      }
      // Remove one block from all launches at the max level.
      for (size_t i = 0; i < launches.size() && totalAssigned > targetBlocks;
           ++i) {
        if (sv.numBlocksPerLaunch[i] == maxVal) {
          --sv.numBlocksPerLaunch[i];
          --totalAssigned;
        }
      }
      if (totalAssigned == before) {
        break;
      }
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

  // Fill blocks and launchIndices.
  sv.blocks.resize(totalAssigned);
  sv.launchIndices.resize(totalAssigned);
  int32_t blockIdx = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    auto opCode = launches[i].launch->op->opCode();
    auto nBlocks = sv.numBlocksPerLaunch[i];
    for (int32_t b = 0; b < nBlocks; ++b) {
      auto& info = sv.blocks[blockIdx];
      info.op = opCode;
      info.blockInOp = b;
      info.numBlocksInOp = nBlocks;
      info.params = nullptr;
      sv.launchIndices.at(blockIdx) = static_cast<int32_t>(i);
      ++blockIdx;
    }
  }
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
  if (state.returnedAtStep.empty()) {
    return;
  }
  // Producing step of the nearest returned value an op reads, or -1.
  auto producerOf = [&](const std::vector<nativert::ValueId>& ids) {
    int32_t best = -1;
    for (auto id : ids) {
      auto it = state.returnedAtStep.find(id);
      if (it != state.returnedAtStep.end() && it->second > best) {
        best = it->second;
      }
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
    // An output produced by a host-side view (a viewNode output desc, e.g. a
    // leaf-input slice) takes its shape and offset operands from that view
    // node, not from this op's inputs -- a slice end is typically an item() of
    // a device result. Those ids appear nowhere in actualInputs, which is why
    // KernelOperation adds them to orderingInputs_ separately. Skip the base
    // operand, which is the tensor being viewed rather than a bound.
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
        if (static_cast<int32_t>(k) == baseOrdinal ||
            viewInputs[k].value == nullptr) {
          continue;
        }
        auto id = viewInputs[k].value->id();
        auto it = state.returnedAtStep.find(id);
        if (it != state.returnedAtStep.end() && it->second > producer) {
          producer = it->second;
        }
        auto tt = tainted.find(id);
        if (tt != tainted.end() && tt->second > producer) {
          producer = tt->second;
        }
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
  for (const auto& data : sv.kernels) {
    for (auto id : data.returnValues) {
      state.returnedAtStep[id] = state.executedSteps;
    }
  }
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
      auto pushValue = [&](ValueCP v) {
        args.push_back(v);
        intArgs.push_back(0);
      };
      auto pushInt = [&](int64_t c) {
        args.push_back(nullptr);
        intArgs.push_back(c);
      };
      const auto* meta = Registry::metadata(standalone->target());
      if (meta != nullptr && meta->functionSchema != nullptr) {
        for (const auto& arg : meta->functionSchema->arguments()) {
          if (const auto* in = standalone->tryGetInput(arg.name())) {
            pushValue(in->value);
          } else if (
              const auto* attr = standalone->tryGetAttribute(arg.name())) {
            if (std::holds_alternative<int64_t>(attr->value)) {
              pushInt(std::get<int64_t>(attr->value));
            } else if (std::holds_alternative<std::vector<int64_t>>(
                           attr->value)) {
              const auto& vec = std::get<std::vector<int64_t>>(attr->value);
              intList.assign(vec.begin(), vec.end());
              pushValue(nullptr);
            } else {
              pushValue(nullptr);
            }
          } else {
            pushValue(nullptr);
          }
        }
      } else {
        // prim.ListPack and other schemaless ops: every input is a value.
        for (auto& input : standalone->inputs()) {
          pushValue(input.value);
        }
      }
    }
  } else {
    // Kernel op: translate sizeExpr, inputs, outputs, and output descs.
    auto* kernelOp = launch.op;
    sizeExpr = kernelOp->sizeExpr().toActual(bindings, idToValue);

    const auto& orderedInputs = kernelOp->orderedInputs();
    auto nInputs = kernelOp->numInputs();
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
      actualDesc.sizeExpr = desc.sizeExpr.toActual(bindings, idToValue);
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
  auto entryPoint = "torch::wave::" + kernelName;

  std::stringstream ss;
  ss << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
  for (const auto& inc : includes) {
    ss << "#include \"" << inc << "\"\n";
  }
  ss << "\nnamespace torch::wave {\n\n";
  for (const auto& kop : kernelOpStorage_) {
    if (!kop->helperCode().empty()) {
      ss << kop->helperCode();
    }
  }
  // An op whose device function is register-hungry would otherwise cost every
  // other op in this kernel its occupancy, so honor the largest blocks-per-SM
  // any of them asks for.
  int32_t minBlocksPerSm = 0;
  for (const auto& kop : kernelOpStorage_) {
    minBlocksPerSm = std::max(minBlocksPerSm, kop->minBlocksPerSm());
  }
  ss << "__global__ ";
  if (minBlocksPerSm > 0) {
    ss << "__launch_bounds__(" << WaveConfig::get().blockSize << ", "
       << minBlocksPerSm << ") ";
  }
  ss << "void " << kernelName << "(TorchWaveParams params) {\n"
     << "  ENTRY;\n";
  eltTrace(
      ss,
      "\"entry blockIdx %d op %d blockInOp %d\\n\", blockIdx.x, blockInfo.op, blockInfo.blockInOp");
  std::unordered_set<std::string> emittedDecls;
  for (const auto& kop : kernelOpStorage_) {
    for (const auto& decl : kop->sharedDeclarations()) {
      if (emittedDecls.insert(decl).second) {
        ss << decl;
      }
    }
  }
  ss << "  switch (blockInfo.op) {\n";
  {
    WithPrintOptions printGuard("D3,L4,S");
    for (const auto& kop : kernelOpStorage_) {
      ss << "    // " << kop->label() << "\n";
      auto opStr = kop->toString();
      std::istringstream lines(opStr);
      std::string line;
      while (std::getline(lines, line)) {
        ss << "    // " << line << "\n";
      }
      ss << "    case " << kop->opCode() << ": {\n";

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
          paramOffs.push_back(kop->paramOffset(v));
          outputOffs.push_back(outputOff);
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
          ss << "    static int32_t " << name << "[] = {";
          for (size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) {
              ss << ", ";
            }
            ss << arr[i];
          }
          ss << "};\n";
        };
        if (!normalParam.empty()) {
          ss << "  {\n";
          emitArray("paramOffsets", normalParam);
          emitArray("outputOffsets", normalOutput);
          emitArray("altOffsets", normalAlt);
          ss << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
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
          ss << "  {\n";
          emitArray("gatherOffsets", gatherParam);
          ss << "    for (auto i = threadIdx.x; i < sizeof(gatherOffsets) / sizeof(gatherOffsets[0]); i += blockDim.x) {\n"
             << "      param<Tensor>(blockInfo, gatherOffsets[i])->ensureIndexCalculator();\n"
             << "    }\n"
             << "  }\n";
        }
        ss << "  __syncthreads();\n";
      }
      ss << kop->code() << "      break;\n"
         << "    }\n";
    }
  } // WithPrintOptions scope
  ss << "  }\n"
     << "  LEAVE();\n"
     << "}\n\n"
     << "} // namespace torch::wave\n";

  auto code = ss.str();

  // Save to /tmp for debugging. Kernel ids are per-construction (not globally
  // unique), so append a process-wide ordinal to keep filenames distinct when
  // several graphs compile concurrently.
  static std::atomic<int64_t> debugDumpOrdinal{0};
  auto filePath = "/tmp/kernel" + std::to_string(kernelId) + "_" +
      std::to_string(debugDumpOrdinal++) + ".cu";
  {
    std::ofstream out(filePath);
    out << code;
  }

  if (!FLAGS_debug_kernel_dir.empty()) {
    auto debugFile = "kernel" + std::to_string(kernelId) + ".cu";
    auto debugPath = FLAGS_debug_kernel_dir + "/" + debugFile;
    std::ifstream in(debugPath);
    if (in.good()) {
      code = std::string(
          std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
      LOG(INFO) << "Reading code of kernel" << kernelId << " from "
                << debugPath;
    } else {
      LOG(INFO) << "No debug version for " << debugFile;
    }
  }

  entryPoint_ = entryPoint;
  text_ = code;

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

  // Only compile the kernel if a GPU is available. The one-time
  // NVRTC/system-header initialization (CompiledKernel::initialize()) is run
  // eagerly on the main thread by torch::wave::initialize() before any kernel
  // is compiled, so the async compile enqueued below never triggers it lazily
  // on a Wave compile-pool thread (which deadlocks warmup() in heavyweight
  // NCCL/Thrift/folly hosts -- T275179010).
  if (facebook::velox::wave::currentDevice()) {
    auto genFunc = [code = std::move(code),
                    entryPoint,
                    filePath]() -> facebook::velox::wave::KernelSpec {
      facebook::velox::wave::KernelSpec spec;
      spec.code = code;
      spec.entryPoints = {entryPoint};
      spec.filePath = filePath;
      spec.numHeaders = 0;
      spec.headers = nullptr;
      return spec;
    };
    if (!WaveConfig::get().kernelCacheDir.empty()) {
      static facebook::velox::wave::KernelFsCache cache(
          WaveConfig::get().kernelCacheDir);
      kernel_ = cache.getKernel(text_, std::move(genFunc));
    } else {
      kernel_ = facebook::velox::wave::CompiledKernel::getKernel(
          text_, std::move(genFunc));
    }
  }
}

void CompositeKernel::warmup() {
  if (!kernel_) {
    return;
  }
  TorchWaveParams params{};
  memset(&params, 0, sizeof(params));
  params.info = nullptr;
  params.debugInfo = nullptr;
  params.inlineInfo[0].op = kDebugNoOp;
  void* args[] = {&params};
  facebook::velox::wave::Stream stream;
  kernel_->launch(0, 1, 1, 0, &stream, args);
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

facebook::velox::wave::KernelInfo CompositeKernel::kernelInfo() const {
  if (kernel_) {
    return kernel_->info(0);
  }
  return {};
}

void CompositeKernel::launch(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launch(0, numBlocks, numThreads, sharedMemory, stream, args);
}

void CompositeKernel::launchCooperative(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launchCooperative(
      0, numBlocks, numThreads, sharedMemory, stream, args);
}

std::string CompositeKernel::toString(Listing /*mode*/) const {
  std::stringstream ss;
  for (const auto& kop : kernelOpStorage_) {
    ss << kop->toString();
  }
  auto info = kernelInfo();
  if (info.numRegs > 0) {
    ss << "entry=" << entryPoint_ << " " << info.toString() << "\n";
  }
  return ss.str();
}

namespace {

void traceParamOffsets(const LaunchData& launch) {
  if (!(WaveConfig::get().trace & WaveConfig::kFrame)) {
    return;
  }
  for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
    std::cout << "  %" << launch.tensorsInFrame[i]
              << " offset = " << launch.tensorOffsets[i] << std::endl;
  }
  for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
    std::cout << "  %" << launch.scalarsInFrame[i] << " = "
              << launch.scalarOffsets[i] << std::endl;
  }
}

void trackReturnValue(
    nativert::ValueId actualId,
    int32_t offset,
    int32_t size,
    LaunchData& launch,
    int32_t& returnCounter,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
      actualId == launch.returnValues[returnCounter]) {
    launch.returnOffsets.push_back(offset);
    if (returnBegin == -1) {
      returnBegin = offset;
    }
    returnEnd = offset + size;
    ++returnCounter;
  }
}

void fillTensorListParam(
    LaunchData& launch,
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase,
    const KernelOperation& kernelOp,
    ValueCP listValue,
    const FormalToActual& bindings,
    std::unordered_set<int32_t>& filledOffsets) {
  auto elements = listValue->getListElements();
  TensorListParam tlp;
  tlp.listOffset = kernelOp.paramOffset(listValue);
  for (auto* elem : elements) {
    auto elemId = elem->id();
    auto it = bindings.find(elemId);
    auto actualId = it != bindings.end() ? it->second : elemId;
    auto elemOffset = kernelOp.paramOffset(elem);
    tlp.elementOffsets.push_back(elemOffset);
    tlp.elementIds.push_back(actualId);
    if (filledOffsets.insert(elemOffset).second) {
      const auto& elemIv = frame.getIValue(actualId);
      TORCH_CHECK(
          elemIv.isTensor(),
          "fillTensorListParam: list %",
          listValue->id(),
          " element actual %",
          actualId,
          " (formal %",
          elemId,
          ") is not a tensor (tag=",
          elemIv.tagKind(),
          ") -- freed while still needed?");
      fillTensorParam(elemIv.toTensor(), paramBase + elemOffset);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(elemOffset);
    }
  }
  // Write TensorList struct header. The tensors pointer array follows
  // the struct and is patched with device-side addresses later.
  auto* tl = reinterpret_cast<TensorList*>(paramBase + tlp.listOffset);
  tl->size = static_cast<int64_t>(elements.size());
  tl->tensors = nullptr;
  launch.tensorLists.push_back(std::move(tlp));
}

void patchTensorListPointers(
    const LaunchData& launch,
    uint8_t* paramBase,
    uint8_t* deviceBase) {
  for (const auto& tlp : launch.tensorLists) {
    auto* tl = reinterpret_cast<TensorList*>(paramBase + tlp.listOffset);
    auto* ptrArray = reinterpret_cast<Tensor**>(
        paramBase + tlp.listOffset + sizeof(TensorList));
    tl->tensors = reinterpret_cast<Tensor**>(
        deviceBase + tlp.listOffset + sizeof(TensorList));
    for (size_t j = 0; j < tlp.elementOffsets.size(); ++j) {
      ptrArray[j] =
          reinterpret_cast<Tensor*>(deviceBase + tlp.elementOffsets[j]);
    }
  }
}

void fillLaunchParams(
    LaunchData& launch,
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  if (!launch.tensorsInFrame.empty() || !launch.scalarsInFrame.empty()) {
    // Cached path: fill only variable tensors and scalars, skip constants.
    TORCH_CHECK(
        launch.tensorsInFrame.size() == launch.tensorOffsets.size(),
        "tensorsInFrame/tensorOffsets size mismatch");
    for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
      if (launch.shapeOnlyTensorIndices.count(i)) {
        fillShapeOnlyTensorParam(
            frame.getIValue(launch.tensorsInFrame[i]).toTensor(),
            paramBase + launch.tensorOffsets[i]);
      } else {
        fillTensorParam(
            frame.getIValue(launch.tensorsInFrame[i]).toTensor(),
            paramBase + launch.tensorOffsets[i]);
      }
    }
    TORCH_CHECK(
        launch.scalarsInFrame.size() == launch.scalarOffsets.size(),
        "scalarsInFrame/scalarOffsets size mismatch");
    for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
      fillScalarParam(
          frame.getIValue(launch.scalarsInFrame[i]),
          paramBase + launch.scalarOffsets[i],
          launch.scalarsInFrame[i]);
    }
    for (auto offset : launch.scalarOutputOffsets) {
      *reinterpret_cast<int64_t*>(paramBase + offset) = 0;
    }
    for (auto offset : launch.launch->op->barrierCounters()) {
      *reinterpret_cast<int32_t*>(paramBase + offset) = 0;
    }
    if (!launch.returnValues.empty()) {
      if (returnBegin == -1) {
        returnBegin = launch.returnOffsets.front();
      }
      auto lastType = launch.returnTypes.back();
      int32_t lastSize =
          lastType == nativert::Type::Kind::Tensor ? sizeof(Tensor) : 8;
      returnEnd = launch.returnOffsets.back() + lastSize;
    }
    traceParamOffsets(launch);
    return;
  }

  auto* kernelOp = launch.launch->op;
  const auto& orderedInputs = kernelOp->orderedInputs();
  auto numInputs = kernelOp->numInputs();
  const auto& bindings = launch.invocation->bindings();

  // Track which tensor offsets have been filled to avoid duplicates
  // when multiple TensorLists share elements.
  std::unordered_set<int32_t> filledOffsets;

  // Fill input params, recording tensor/scalar values and their offsets.
  int32_t returnCounter = 0;
  for (int32_t i = 0; i < numInputs; ++i) {
    auto* formalValue = orderedInputs[i];
    if (formalValue->type().kind() == nativert::Type::Kind::TensorList) {
      fillTensorListParam(
          launch,
          frame,
          paramBase,
          *kernelOp,
          formalValue,
          bindings,
          filledOffsets);
      continue;
    }
    auto offset = kernelOp->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualInputs.at(i);
    auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      fillTensorParam(ivalue.toTensor(), dest);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else if (
        ivalue.isNone() &&
        formalValue->type().kind() == nativert::Type::Kind::Tensor) {
      fillEmptyTensorParam(dest);
      launch.numElements = 0;
    } else {
      fillScalarParam(ivalue, dest, actualId);
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
    }
    trackReturnValue(
        actualId,
        offset,
        ivalue.isTensor() ? static_cast<int32_t>(sizeof(Tensor)) : 8,
        launch,
        returnCounter,
        returnBegin,
        returnEnd);
  }

  // Fill output params, recording values and offsets.
  //
  // Every index into a 'launch' vector below is bounds-checked where it is
  // used -- by this loop's own bound, or by an explicit size test in the same
  // condition. ParameterUncheckedArrayBounds credits neither, only an
  // emptiness assert on the parameter, which would state a false invariant:
  // an op with no tensor-typed outputs legitimately has these empty.
  // NOLINTBEGIN(facebook-hte-ParameterUncheckedArrayBounds)
  for (size_t i = 0; i < launch.actualOutputs.size(); ++i) {
    auto* formalValue = orderedInputs[numInputs + i];
    if (formalValue->type().kind() == nativert::Type::Kind::TensorList) {
      auto listOffset = kernelOp->paramOffset(formalValue);
      fillTensorListParam(
          launch,
          frame,
          paramBase,
          *kernelOp,
          formalValue,
          bindings,
          filledOffsets);
      auto actualId = launch.actualOutputs[i];
      if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
          actualId == launch.returnValues[returnCounter]) {
        launch.returnOffsets.push_back(listOffset);
        // fillTensorListParam above appended the entry, so back() is safe.
        TORCH_CHECK(!launch.tensorLists.empty());
        const auto& tlp = launch.tensorLists.back();
        for (auto elemOff : tlp.elementOffsets) {
          if (returnBegin == -1) {
            returnBegin = elemOff;
          } else {
            returnBegin = std::min(returnBegin, elemOff);
          }
          returnEnd = std::max(
              returnEnd, elemOff + static_cast<int32_t>(sizeof(Tensor)));
        }
        ++returnCounter;
      }
      continue;
    }
    auto offset = kernelOp->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualOutputs[i];
    bool isTensorOutput = i < launch.actualOutputTypes.size() &&
        launch.actualOutputTypes[i] == nativert::Type::Kind::Tensor;
    if (isTensorOutput) {
      const auto& ivalue = frame.getIValue(actualId);
      if (ivalue.isNone()) {
        launch.numElements = 0;
        fillEmptyTensorParam(dest);
        continue;
      }
      TORCH_CHECK(
          ivalue.isTensor(),
          "Expected tensor for output param: value %",
          actualId,
          " opCode ",
          kernelOp->opCode(),
          " output index ",
          i,
          " isNone ",
          ivalue.isNone());
      bool isShapeOnly = false;
      if (i < launch.actualOutputDescs.size()) {
        isShapeOnly = launch.actualOutputDescs[i].shapeOnly;
      }
      if (isShapeOnly) {
        fillShapeOnlyTensorParam(ivalue.toTensor(), dest);
        launch.shapeOnlyTensorIndices.insert(launch.tensorsInFrame.size());
      } else {
        fillTensorParam(ivalue.toTensor(), dest);
      }
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      // Non-tensor output: write a 64-bit zero placeholder. The kernel writes
      // the real value; record the offset so the cached path re-zeroes it
      // rather than (incorrectly) filling it from the frame as if an input.
      *reinterpret_cast<int64_t*>(dest) = 0;
      launch.scalarOutputOffsets.push_back(offset);
    }
    trackReturnValue(
        actualId,
        offset,
        isTensorOutput ? static_cast<int32_t>(sizeof(Tensor)) : 8,
        launch,
        returnCounter,
        returnBegin,
        returnEnd);
  }
  // NOLINTEND(facebook-hte-ParameterUncheckedArrayBounds)

  // Fill constant params (first time only, constants don't change).
  auto constantOffset = kernelOp->constantAreaOffset();
  const auto& opConstants = launch.invocation->constants();
  for (auto idx : launch.launch->constantIndices) {
    auto* dest = paramBase + constantOffset;
    // Constants carry no Value id; pass -1.
    fillScalarParam(*opConstants[idx], dest, -1);
    constantOffset += 8;
  }

  for (auto offset : kernelOp->altTensorOffsets()) {
    auto* t = reinterpret_cast<Tensor*>(paramBase + offset);
    t->status = Tensor::kUninited;
  }

  for (auto offset : kernelOp->barrierCounters()) {
    *reinterpret_cast<int32_t*>(paramBase + offset) = 0;
  }

  traceParamOffsets(launch);
}

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

void ensureCudaTensor(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId actualId,
    c10::IntArrayRef dims,
    const std::string& sizeKey = std::string("dyn"),
    ExecutionState* state = nullptr) {
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
      frame.setIValue(
          actualId,
          at::empty(
              dims,
              at::TensorOptions()
                  .dtype(
                      typeMeta != nullptr ? typeMeta->dtype()
                                          : c10::ScalarType::Float)
                  .device(at::kMeta)));
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
    // A pooled buffer of exactly this size costs a pop instead of an allocator
    // call, which is the whole cost at any size.
    const auto bytes = requestedBytes(meta->dtype());
    if (state != nullptr) {
      auto donated = takeDonatedTensor(*state, bytes, meta->dtype(), dims);
      if (donated.defined()) {
        frame.setIValue(actualId, std::move(donated));
        return;
      }
      // Missed, so this allocation grows the footprint. Shed pooled buffers if
      // holding them would push past the delayed-free ceiling.
      const auto limit = WaveConfig::get().maxDelayedFree;
      if (limit > 0) {
        evictDonatable(*state, limit);
      }
    }
    at::Tensor tensor;
    {
      ScopedAllocCall timed;
      tensor = at::empty(
          dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
    }
    frame.setIValue(actualId, std::move(tensor));
  }
}

// Serializes a SizeExpr into a canonical string. The expression is fixed at
// compile time and its value depends only on the numel of the values it names,
// so two outputs whose keys match are provably the same element count on every
// run. kMax / kSum are commutative, so the value list is sorted to keep the key
// canonical.
void appendSizeExprKey(const SizeExpr& expr, std::string& out) {
  out += std::to_string(static_cast<int>(expr.op));
  if (expr.broadcast) {
    out += 'B';
  }
  out += '(';
  auto values = expr.values;
  std::sort(values.begin(), values.end());
  for (auto value : values) {
    out += std::to_string(value);
    out += ',';
  }
  for (const auto& shape : expr.constShapes) {
    out += '[';
    for (auto dim : shape) {
      out += std::to_string(dim);
      out += ' ';
    }
    out += ']';
  }
  for (const auto& arg : expr.args) {
    appendSizeExprKey(arg, out);
  }
  out += ')';
}

// The static size key for an output: the size expression plus the element
// width, since equal element counts only mean equal bytes at equal width.
// "dyn" when the shape comes from a reserveShape lambda, which reads the frame
// and cannot be compared before running.
std::string staticSizeKey(
    const OutputDesc& desc,
    const ValueTypes& types,
    nativert::ValueId actualId) {
  if (desc.reserveShape || desc.sizeExpr.op == SizeShortcut::kNone) {
    return "dyn";
  }
  std::string key;
  appendSizeExprKey(desc.sizeExpr, key);
  const auto* meta = static_cast<size_t>(actualId) < types.types.size()
      ? types.types[actualId]
      : nullptr;
  key += '#';
  key += std::to_string(
      meta ? static_cast<int>(c10::elementSize(meta->dtype())) : 0);
  return key;
}

// True for the list kinds whose elements are tensors, i.e. the cases where a
// pack/unpack only moves handles.
bool isTensorListKind(nativert::Type::Kind kind) {
  return kind == nativert::Type::Kind::TensorList ||
      kind == nativert::Type::Kind::NestedTensorList ||
      kind == nativert::Type::Kind::OptionalTensorList;
}

void allocateLaunchOutputs(
    const LaunchData& launch,
    ExecutionState* state,
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId largestId,
    const folly::F14FastMap<NodeCP, nativert::OpKernel*>* kernelMap,
    const IdToValueMap& idToValue,
    const folly::F14FastSet<nativert::ValueId>& reusableIds) {
  const auto& descs = launch.actualOutputDescs;
  const auto& actualOutputs = launch.actualOutputs;
  const auto& outputTypes = launch.actualOutputTypes;

  // Outputs that another output aliases in place (its aliasSelfId). These are
  // materialized whole-tensor bases -- e.g. the elementwise 'self' of a fused
  // tw.select_scatter -- and must keep their own shape, not be resized to the
  // op's largest elementwise input (which is the scatter 'src', smaller than
  // 'self') by the largestId shortcut below.
  folly::F14FastSet<nativert::ValueId> aliasSelfTargets;
  for (const auto& desc : descs) {
    if (desc.aliasSelfId) {
      aliasSelfTargets.insert(*desc.aliasSelfId);
    }
  }

  // Shortcut: if largestId is set, resize tensor outputs to match it.
  if (largestId >= 0) {
    auto& largestIv = frame.getIValue(largestId);
    auto dims = largestIv.toTensor().sizes();

    // Elementwise in-place reuse: the largest input determines the output
    // shape, so when it is flagged reusable/overwritable, contiguous, and
    // CUDA-resident its buffer can back the output (write the result in place)
    // instead of allocating a fresh tensor. Only apply when there is exactly
    // one real tensor output, so overwriting largestId cannot clobber an input
    // that another output still reads. Gated by WaveConfig::enableReuse.
    int32_t tensorOutputs = 0;
    for (size_t i = 0; i < descs.size(); ++i) {
      if (i < outputTypes.size() &&
          outputTypes[i] != nativert::Type::Kind::Tensor) {
        continue;
      }
      if (descs[i].viewNode || descs[i].delegated || descs[i].aliasSelfId) {
        continue;
      }
      ++tensorOutputs;
    }
    bool tryReuse = WaveConfig::get().enableReuse && tensorOutputs == 1 &&
        largestIv.isTensor() && largestIv.toTensor().is_cuda() &&
        largestIv.toTensor().is_contiguous() &&
        reusableIds.count(largestId) > 0;
    // Not over a concat group's band. Reuse trades an output allocation for an
    // in-place write, but a placed output has no allocation to trade: its
    // buffer is a region of the concat result, which the concat no longer
    // copies into. Taking the input's buffer instead would leave that region
    // unwritten. Nor may the input be a band: its region belongs to a result
    // that outlives it, and this op's output would overwrite it.
    if (tryReuse && state != nullptr && state->waveGraph != nullptr &&
        state->waveGraph->isConcatPlaced(largestId)) {
      tryReuse = false;
    }
    // Do not reuse largestId's buffer if any OTHER kernel input aliases its
    // storage (e.g. a fused view/slice of it): writing the output in place
    // would clobber that operand's reads and corrupt the result.
    if (tryReuse) {
      for (auto inId : launch.actualInputs) {
        if (inId == largestId) {
          continue;
        }
        auto& iv = frame.getIValue(inId);
        if (iv.isTensor() && iv.toTensor().is_alias_of(largestIv.toTensor())) {
          tryReuse = false;
          break;
        }
      }
    }

    for (size_t i = 0; i < descs.size(); ++i) {
      if (i < outputTypes.size() &&
          outputTypes[i] != nativert::Type::Kind::Tensor) {
        continue;
      }
      if (descs[i].viewNode && kernelMap) {
        auto it = kernelMap->find(descs[i].viewNode);
        TORCH_CHECK(
            it != kernelMap->end(),
            "No kernel for view node ",
            descs[i].viewNode->target());
        executeNode(descs[i].viewNode, it->second, frame);
        continue;
      }
      if (descs[i].delegated) {
        continue;
      }
      auto actualId = actualOutputs[i];
      // A materialized whole-tensor base that a later output aliases in place
      // (e.g. the elementwise 'self' of a fused tw.select_scatter) is sized by
      // its own shape, not the op's largest elementwise input. Reserved here,
      // ahead of the aliasing output, so that output inherits self's full shape
      // via the alias() below.
      if (aliasSelfTargets.count(actualId) &&
          descs[i].sizeExpr.op != SizeShortcut::kNone) {
        auto exprDims = descs[i].sizeExpr.dims(&frame);
        std::vector<int64_t> ownDims(exprDims.begin(), exprDims.end());
        ensureCudaTensor(frame, types, actualId, ownDims);
        continue;
      }
      if (descs[i].aliasSelfId) {
        auto& selfIv = frame.getIValue(*descs[i].aliasSelfId);
        if (selfIv.isTensor()) {
          // In-place op output: a view sharing self's storage (see general
          // path below).
          frame.setIValue(actualId, selfIv.toTensor().alias());
          continue;
        }
      }
      if (tryReuse && actualId != largestId &&
          !(state != nullptr && state->waveGraph != nullptr &&
            state->waveGraph->isConcatPlaced(actualId))) {
        auto* meta = types.types.at(actualId);
        if (meta && meta->dtype() == largestIv.toTensor().scalar_type()) {
          traceTensor(actualId, dims, "reuse");
          if (WaveConfig::get().trace & WaveConfig::kTiming) {
            std::cout << "  REUSE out=%" << actualId << " in=%" << largestId
                      << std::endl;
          }
          gElementwiseReuseCount += 1;
          gElementwiseReuseBytes +=
              static_cast<int64_t>(largestIv.toTensor().nbytes());
          // Share largestId's storage: the elementwise loop reads each element
          // before writing the same index, so the in-place write is safe.
          frame.setIValue(actualId, largestIv.toTensor());
          continue;
        }
      }
      ensureCudaTensor(frame, types, actualId, dims);
    }
    return;
  }

  const auto& bindings = launch.invocation->bindings();
  const auto& nodeMap = launch.invocation->nodeMap();
  for (size_t i = 0; i < descs.size(); ++i) {
    // Skip non-tensor and non-tensor-list outputs.
    if (i < outputTypes.size() &&
        outputTypes[i] != nativert::Type::Kind::Tensor &&
        outputTypes[i] != nativert::Type::Kind::TensorList) {
      continue;
    }
    if (descs[i].viewNode && kernelMap) {
      auto it = kernelMap->find(descs[i].viewNode);
      TORCH_CHECK(
          it != kernelMap->end(),
          "No kernel for view node ",
          descs[i].viewNode->target());
      try {
        executeNode(descs[i].viewNode, it->second, frame);
      } catch (...) {
      }
      continue;
    }
    if (descs[i].delegated) {
      continue;
    }
    auto actualId = actualOutputs[i];

    // In-place op output (Tensor(a!)): reserve as a view sharing the mutated
    // self's storage, so the returned tensor aliases self and reflects later
    // in-place mutations, rather than being a fresh copy.
    if (descs[i].aliasSelfId) {
      auto& selfIv = frame.getIValue(*descs[i].aliasSelfId);
      if (selfIv.isTensor()) {
        frame.setIValue(actualId, selfIv.toTensor().alias());
        continue;
      }
    }

    // TensorList output: expand to component Values and allocate each.
    if (i < outputTypes.size() &&
        outputTypes[i] == nativert::Type::Kind::TensorList) {
      if (!descs[i].reserveShape) {
        continue;
      }
      auto shapes = descs[i].reserveShape(frame, bindings, nodeMap);
      auto valueIt = idToValue.find(actualId);
      TORCH_CHECK(
          valueIt != idToValue.end(),
          "TensorList output value not found: ",
          actualId);
      auto elements = valueIt->second->getListElements();
      TORCH_CHECK(
          shapes.size() == elements.size(),
          "reserveShape returned ",
          shapes.size(),
          " shapes but TensorList has ",
          elements.size(),
          " elements");
      for (size_t j = 0; j < elements.size(); ++j) {
        auto elemId = elements[j]->id();
        auto elemActualIt = bindings.find(elemId);
        auto elemActualId =
            elemActualIt != bindings.end() ? elemActualIt->second : elemId;
        std::vector<int64_t> dims(shapes[j].begin(), shapes[j].end());
        ensureCudaTensor(
            frame, types, elemActualId, dims, std::string("dyn"), state);
      }
      continue;
    }

    std::vector<int64_t> dims;
    if (descs[i].reserveShape) {
      auto shapes = descs[i].reserveShape(frame, bindings, nodeMap);
      TORCH_CHECK(
          !shapes.empty(),
          "OutputReserveFunc returned empty shapes for output ",
          i);
      dims.assign(shapes[0].begin(), shapes[0].end());
    } else if (descs[i].sizeExpr.op != SizeShortcut::kNone) {
      auto exprDims = descs[i].sizeExpr.dims(&frame);
      dims.assign(exprDims.begin(), exprDims.end());
    } else {
      continue;
    }
    if (descs[i].shapeOnly) {
      auto& existing = frame.getIValue(actualId);
      if (existing.isTensor()) {
        auto& tensor = existing.toTensor();
        if (tensor.sizes() != dims) {
          tensor.resize_(dims);
        }
      } else {
        auto tensor = at::empty(
            dims, at::TensorOptions().dtype(at::kFloat).device(at::kMeta));
        frame.setIValue(actualId, std::move(tensor));
      }
      continue;
    }
    ensureCudaTensor(
        frame,
        types,
        actualId,
        dims,
        staticSizeKey(descs[i], types, actualId),
        state);
  }
}

[[maybe_unused]] int32_t launchParamSize(const LaunchData& launch) {
  return launch.launch->op->altParamOffset();
}

facebook::velox::wave::WaveBufferPtr& getOrAllocateBuffer(
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>& buffers,
    int32_t sequenceNumber,
    int32_t stepIdx,
    int64_t requiredBytes,
    facebook::velox::wave::GpuArena* arena,
    const std::function<void(void*, int64_t)>& initFunc = nullptr) {
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

// When sizeExpr is kNone (0-arg elementwise), gets numElements from the
// single output desc's reserveShape.
int64_t numElementsFromReserve(
    LaunchData& data,
    nativert::ExecutionFrame& frame) {
  const auto& descs = data.actualOutputDescs;
  TORCH_CHECK(
      descs.size() == 1,
      "sizeExpr is kNone but kernel op has ",
      descs.size(),
      " output descs, expected 1");
  TORCH_CHECK(
      descs[0].reserveShape,
      "sizeExpr is kNone but output desc has no reserveShape");
  const auto& bindings = data.invocation->bindings();
  const auto& nodeMap = data.invocation->nodeMap();
  auto shapes = descs[0].reserveShape(frame, bindings, nodeMap);
  TORCH_CHECK(!shapes.empty(), "reserveShape returned empty shapes");
  int64_t numElements = 1;
  for (auto dim : shapes[0]) {
    numElements *= dim;
  }
  return numElements;
}

} // namespace

// --- CompositeInvocation ---

namespace {
// Defined further down, next to the other pending-return helpers.
int32_t neededPendingStep(
    const ExecutionState& state,
    const folly::F14FastSet<nativert::ValueId>& readIds);
} // namespace

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
  // that reach the host only through reserveShape / sizeExpr at gather time and
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
  const auto numSlots = static_cast<int32_t>(sv.slotOffsets.size());
  sv.blockCapacity =
      2 * numSlots +
      targetBlockCount(
          kernel_->kernelInfo().maxOccupancy0,
          // A bound, so skip the shared-memory reduction: it only ever lowers
          // the count makeGrid will actually use, and a step's dynamic
          // shared-memory need is not known until its launches are gathered.
          /*dynSharedPerBlock=*/0,
          /*staticSharedPerBlock=*/0);
}

bool CompositeInvocation::chooseGridVariant(
    ExecutionState& state,
    GridChoice& gridChoice,
    OpInvocation& op,
    int32_t stepIdx,
    size_t launchIndex,
    bool hasByLargestInput,
    LaunchData& data,
    nativert::ValueId& largestId) {
  auto* projectOp = op.projectOp();
  bool wantSingleBlock;
  if (WaveConfig::get().useSingleBlock.has_value()) {
    wantSingleBlock = *WaveConfig::get().useSingleBlock;
  } else {
    wantSingleBlock = data.numElements <= projectOp->singleBlockMaxSize();
  }
  LaunchGrid* newGrid = nullptr;
  if (wantSingleBlock != gridChoice.singleBlock) {
    if (wantSingleBlock) {
      newGrid = &projectOp->singleBlockGrid();
    } else {
      newGrid = &projectOp->grid();
    }
  }
  if (!wantSingleBlock && !projectOp->cgGrid().empty() &&
      WaveConfig::get().isCg.has_value() && *WaveConfig::get().isCg) {
    newGrid = &projectOp->cgGrid();
  }
  // A scanOutputReturnBarrier op takes a launch break only in the multi-block
  // grid, so its multi-block grid has more steps than its single-block
  // variant. The grid-choice kernel can therefore sit at a stepIdx that exists
  // only in the current (longer) grid; switching to a shorter variant here
  // would index it out of bounds (the initial access in the caller is guarded,
  // but these post-swap accesses are not). Only switch when the target grid
  // actually has this step. Otherwise keep the current grid -- it is a
  // complete, correct plan for this op -- so the launch still runs, just under
  // the already-selected variant. The op's earlier steps already ran under that
  // variant, so this also keeps the whole op on one consistent grid.
  if (newGrid == nullptr || newGrid == gridChoice.grid ||
      stepIdx >= static_cast<int32_t>(newGrid->size())) {
    return false;
  }
  gridChoice.singleBlock = wantSingleBlock;
  gridChoice.grid = newGrid;
  const auto& newLaunch = (*gridChoice.grid)[stepIdx][launchIndex];
  data = LaunchData(newLaunch, op, state.waveGraph->idToValue());
  largestId = -1;
  if (data.sizeExpr.op == SizeShortcut::kNone) {
    data.numElements = numElementsFromReserve(data, *state.frame);
  } else {
    data.numElements = data.sizeExpr.numElements(
        state.frame, hasByLargestInput ? &largestId : nullptr);
  }
  return true;
}

void CompositeInvocation::sizeForUnreadyOperands(
    LaunchData& data,
    const Launch& launch,
    nativert::ExecutionFrame& frame) {
  // If a tensor input this kernel does NOT produce itself is None, skip the
  // kernel (set numElements=0 so makeGrid assigns 0 blocks). Two kinds of input
  // can be None without meaning "not ready":
  //
  //  - A value the op computes itself. Its producer is one of the op's own
  //    nodes, so the launch writes it before the code that reads it (ordered by
  //    the op's internal barriers); it is simply not materialized yet at host
  //    sizing time.
  //  - A non-tensor. An absent optional argument -- clamp's `min`, say -- is
  //    legitimately None and says nothing about readiness.
  //
  // Counting either one starved the launch to a single block: it zeroed
  // numElements and left the cg path to rebuild a size from static shapes
  // alone.
  const auto& opInputs = launch.op->orderedInputs();
  const auto& opNodes = launch.op->allNodes();
  auto numOpInputs = static_cast<size_t>(launch.op->numInputs());
  // orderedInputs is the inputs followed by the outputs, and actualInputs is
  // translated from its first numInputs entries, so these two sizes always
  // agree. Scanning fewer than all the actual inputs would let a None slip past
  // and starve the launch to a single block, so this is checked rather than
  // clamped.
  TORCH_CHECK(
      data.actualInputs.size() == numOpInputs && numOpInputs <= opInputs.size(),
      "Kernel op input count mismatch: ",
      data.actualInputs.size(),
      " actual vs ",
      numOpInputs,
      " formal of ",
      opInputs.size(),
      " ordered");
  for (size_t k = 0; k < numOpInputs; ++k) {
    const auto* formal = opInputs[k];
    auto kind = formal->type().kind();
    if (kind != nativert::Type::Kind::Tensor &&
        kind != nativert::Type::Kind::TensorList) {
      continue;
    }
    const auto* producer = formal->producer();
    if (producer != nullptr && opNodes.count(producer) > 0) {
      continue;
    }
    auto& iv = frame.getIValue(data.actualInputs[k]);
    if (iv.isNone()) {
      data.numElements = 0;
      break;
    }
  }
  if (data.numElements > 0) {
    auto* collector = currentAllocCollector();
    for (size_t oi = 0; oi < data.actualOutputs.size(); ++oi) {
      if (oi < data.actualOutputTypes.size() &&
          data.actualOutputTypes[oi] == nativert::Type::Kind::Tensor) {
        const auto outputId = data.actualOutputs[oi];
        // A member of one of this step's allocation groups has no tensor until
        // the group is carved, which is after this pass. That None is not the
        // one below: the value is produced right here, and zeroing the launch
        // over it would starve the op to a block.
        if (collector != nullptr && collector->owns(outputId)) {
          continue;
        }
        const auto& oiv = frame.getIValue(outputId);
        // A None output comes from a later PN -- its tensor is not
        // materialized, so the kernel must not launch yet. An empty (0-element)
        // output is handled in device code (the elementwise size head sets
        // size=0 -> 0 iterations), so it does not zero the whole launch here --
        // that would wrongly skip the non-empty lanes of a multi-output kernel.
        if (oiv.isNone()) {
          data.numElements = 0;
          break;
        }
      }
    }
  }
  // Under a cooperative grid the whole step launches as ONE kernel, so an op
  // cannot be skipped -- numElements only sets its block share. A view-rooted
  // op (e.g. slice->clamp) fused into the step reads an input that is a
  // step-internal intermediate: None/unallocated at host sizing time, so the
  // guards above zero its numElements and it is starved to ~1 block even though
  // it runs correctly once the cooperative kernel materializes that input
  // mid-launch (op 138: 6 of 480 blocks, ~85ms). Recover a grid size from the
  // kernel's concrete static input shapes (TensorMeta is available without
  // materialization). numElements only drives the grid; the kernel loops to the
  // true size on device, so an over-estimate is safe (surplus blocks
  // early-out).
  if (data.numElements == 0 && WaveConfig::get().isCg.value_or(false)) {
    int64_t staticNumElements = 0;
    for (const auto* tensorMeta : launch.op->inputTypes()) {
      if (tensorMeta != nullptr && !tensorMeta->hasSymbolicShape()) {
        int64_t numElements = 1;
        for (auto extent : tensorMeta->sizes()) {
          numElements *= extent;
        }
        if (numElements > staticNumElements) {
          staticNumElements = numElements;
        }
      }
    }
    if (staticNumElements > 0) {
      data.numElements = staticNumElements;
    }
  }
}

void CompositeInvocation::fillLaunchParamBlock(
    LaunchData& data,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t paramOffset,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  int32_t launchReturnBegin = -1;
  int32_t launchReturnEnd = -1;
  fillLaunchParams(
      data,
      frame,
      pinnedBase + paramOffset,
      launchReturnBegin,
      launchReturnEnd);
  patchTensorListPointers(
      data, pinnedBase + paramOffset, deviceBase + paramOffset);
  if (launchReturnBegin < 0) {
    return;
  }
  // Min/max rather than first/last: the two passes fill in dependency order,
  // not in ascending offset order.
  const auto begin = static_cast<int32_t>(paramOffset + launchReturnBegin);
  const auto end = static_cast<int32_t>(paramOffset + launchReturnEnd);
  returnBegin = returnBegin < 0 ? begin : std::min(returnBegin, begin);
  returnEnd = std::max(returnEnd, end);
}

void CompositeInvocation::gatherLaunches(
    ExecutionState& state,
    std::vector<GridChoice>& grids,
    int32_t stepIdx,
    StepVectors& sv,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    bool deferredOnly,
    std::vector<std::pair<size_t, DeferReason>>& deferred,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  const auto& idToValue = state.waveGraph->idToValue();
  LaunchCursor cursor;
  if (!deferredOnly) {
    sv.gridChanged = false;
    sv.isCgGrid = false;
    sv.hasGpuStandalones = false;
    layoutParamSlots(stepIdx, sv);
  }
  const bool doTiming = WaveConfig::get().printTiming ||
      (WaveConfig::get().trace & WaveConfig::kTiming);
  const int64_t maxDelayedFree = WaveConfig::get().maxDelayedFree;

  // Everything the pending transfers bring back, folded into one signature and
  // one id set -- built once for the whole step, not per op. Actual ids: the
  // formal-to-actual translation already happened when opReadIds was built.
  uint64_t pendingSignature = 0;
  if (!deferredOnly && !state.pendingReturns.empty()) {
    for (const auto& pending : state.pendingReturns) {
      const auto& producer =
          state.stepVectors.at(pending.sequenceNumber).at(pending.stepIdx);
      for (const auto& data : producer.kernels) {
        for (auto id : data.returnValues) {
          pendingSignature |= uint64_t{1} << (static_cast<uint64_t>(id) & 63);
        }
      }
    }
  }
  // The memory cap is fixed for the step, so the per-op test is one compare.
  const bool overMemoryCap =
      maxDelayedFree > 0 && state.delayedFreeBytes > maxDelayedFree;

  // Ops still waiting on a transfer this step's first pass could not resolve,
  // or on memory the device has not released yet. The whole op is deferred:
  // sizing, allocation and fill all read the frame. The common case costs one
  // AND and one branch; only an op whose signature overlaps pays for the exact
  // set lookup.
  auto deferReasonFor = [&](size_t opIndex) -> std::optional<DeferReason> {
    if (pendingSignature != 0 && opIndex < sv.opReadSignature.size() &&
        (sv.opReadSignature[opIndex] & pendingSignature) != 0 &&
        neededPendingStep(state, sv.opReadIds[opIndex]) >= 0) {
      return DeferReason::kTransfer;
    }
    if (overMemoryCap) {
      return DeferReason::kMemory;
    }
    return std::nullopt;
  };

  // Set of ops the first pass skipped, so the second pass can pick them out
  // without re-testing (their dependency is resolved by then).
  folly::F14FastSet<size_t> deferredOps;
  if (deferredOnly) {
    for (const auto& [opIndex, reason] : deferred) {
      deferredOps.insert(opIndex);
    }
  }

  for (size_t i = 0; i < ops_.size(); ++i) {
    auto* grid = grids.at(i).grid;
    if (stepIdx >= static_cast<int32_t>(grid->size())) {
      continue;
    }
    // The index and offset bookkeeping below has to run for every op on the
    // first pass whatever its dependencies, or a deferred op would shift the
    // parameter offsets and launch indices of the ops after it.
    bool skipSetup = false;
    if (deferredOnly) {
      skipSetup = deferredOps.count(i) == 0;
    } else if (auto reason = deferReasonFor(i)) {
      skipSetup = true;
      deferred.emplace_back(i, *reason);
    }
    if (deferredOnly && skipSetup) {
      // Nothing to set up here, and the first pass already recorded how many
      // launches this op owns, so step the cursor over it instead of walking
      // its launches again.
      cursor.kernel += sv.opKernelCount.at(i);
      cursor.standalone += sv.opStandaloneCount.at(i);
      cursor.shortcut += sv.opShortcutCount.at(i);
      continue;
    }
    const LaunchCursor opBegin = cursor;
    auto* step = &(*grid)[stepIdx];
    // Slot of the next kernel launch of this op, indexing the reservation
    // layoutParamSlots made for the op's widest variant.
    int32_t opKernelSlot = 0;
    for (size_t j = 0; j < step->size(); ++j) {
      auto& launch = (*step)[j];
      if (launch.op != nullptr) {
        auto slot = sv.opSlotBegin.at(i) + opKernelSlot;
        TORCH_CHECK(
            slot < sv.opSlotBegin.at(i + 1),
            "Kernel launch slot beyond the reservation for op ",
            i,
            " at step ",
            stepIdx);
        if (cursor.kernel >= static_cast<int32_t>(sv.paramOffsets.size())) {
          sv.paramOffsets.resize(cursor.kernel + 1);
        }
        sv.paramOffsets.at(cursor.kernel) = sv.slotOffsets.at(slot);
        ++opKernelSlot;
        bool isNew = cursor.kernel >= static_cast<int32_t>(sv.kernels.size());
        if (isNew) {
          sv.kernels.emplace_back(launch, ops_[i], idToValue);
        }
        auto& data = sv.kernels.at(cursor.kernel);
        if (skipSetup) {
          // Sizing reads the frame, so it waits for the second pass too. The
          // launch keeps its slot and its index; only its contents are unset.
          ++cursor.kernel;
          continue;
        }
        // Sizing this launch is a sequence of choices, each able to overturn
        // the one before, so the order below is the meaning:
        //   1. an element count, from the reserve or the size expression;
        //   2. the grid variant that count implies, which if it moves rebuilds
        //      the launch data and so the count with it;
        //   3. allocation of the outputs at that size;
        //   4. zero, if an operand the launch does not produce is still None --
        //      unless the cooperative grid is on, which recovers a size from
        //      static shapes because a cooperative step cannot skip an op;
        //   5. the parameter block, filled from whatever survived.
        bool hasByLargestInput = !data.launch->op->outputDescs().empty() &&
            data.launch->op->outputDescs()[0].byLargestInput;
        nativert::ValueId largestId = -1;
        if (data.sizeExpr.op == SizeShortcut::kNone) {
          data.numElements = numElementsFromReserve(data, *state.frame);
        } else {
          data.numElements = data.sizeExpr.numElements(
              state.frame, hasByLargestInput ? &largestId : nullptr);
        }

        if (launch.op->isGridChoice() &&
            chooseGridVariant(
                state,
                grids[i],
                ops_[i],
                stepIdx,
                j,
                hasByLargestInput,
                data,
                largestId)) {
          grid = grids[i].grid;
          step = &(*grid)[stepIdx];
          if (!isNew) {
            sv.gridChanged = true;
          }
        }

        // Check if any viewNode output descs have unavailable inputs.
        // If so, skip allocateLaunchOutputs — the viewNode would
        // crash on None inputs from a later PN.
        //
        // Timed on its own into allocUs, which is what makes the report say
        // how much of the interpretation is allocation -- the part no amount of
        // compiling the host path can remove. The pinned/device buffers are
        // deliberately not counted: those come from a preallocated arena and
        // cost nothing.
        const uint64_t tAlloc = doTiming ? folly::hardware_timestamp() : 0;
        const int64_t allocCallBefore = threadAllocCallUs();
        allocateLaunchOutputs(
            data,
            &state,
            *state.frame,
            *state.valueTypes,
            largestId,
            state.kernelMap,
            idToValue,
            reusableIds_);
        if (doTiming) {
          sv.allocUs += static_cast<int64_t>(
              static_cast<double>(folly::hardware_timestamp() - tAlloc) /
              tscTicksPerMicro());
          sv.allocCallUs += threadAllocCallUs() - allocCallBefore;
        }
        sizeForUnreadyOperands(data, launch, *state.frame);
        if (!launch.op->barrierCounters().empty()) {
          sv.isCgGrid = true;
        }
        // Fill this launch's parameter block now, while its sizes and output
        // tensors are still in cache, instead of in a second sweep over
        // sv.kernels. The block's offset comes from the slot reservation, so it
        // does not move when a later op switches grid variant or is deferred.
        //
        // Skipped under an installed collector: there, a launch's outputs are
        // members of a group that does not exist until every one of its members
        // has been sized, so there is no tensor to read yet and fillStepParams
        // sweeps the step once the groups are materialized. Keyed on the
        // collector rather than the config so a step that groups nothing still
        // fills inline.
        if (currentAllocCollector() == nullptr) {
          const uint64_t tFill = doTiming ? folly::hardware_timestamp() : 0;
          fillLaunchParamBlock(
              data,
              *state.frame,
              pinnedBase,
              deviceBase,
              sv.paramOffsets.at(cursor.kernel),
              returnBegin,
              returnEnd);
          if (doTiming) {
            sv.fillUs += static_cast<int64_t>(
                static_cast<double>(folly::hardware_timestamp() - tFill) /
                tscTicksPerMicro());
          }
        }
        ++cursor.kernel;
      } else if (launch.standaloneShortcut != StandaloneShortcut::kNone) {
        // Metadata-only shortcut op: separate list, tight switch loop, no sync.
        if (cursor.shortcut >=
            static_cast<int32_t>(sv.shortcutStandalones.size())) {
          sv.shortcutStandalones.emplace_back(launch, ops_[i], idToValue);
        }
        ++cursor.shortcut;
      } else {
        if (cursor.standalone >= static_cast<int32_t>(sv.standalones.size())) {
          sv.standalones.emplace_back(launch, ops_[i], idToValue);
        }
        // A standalone that does real device work needs the wave stream synced
        // before it (and before this step's fused kernel). Metadata-only ops
        // (host-only, e.g. a SymInt-list prim.ListPack) need no sync.
        if (!launch.metadataOnly) {
          sv.hasGpuStandalones = true;
        }
        ++cursor.standalone;
      }
    }
    if (!deferredOnly) {
      sv.opKernelCount.at(i) = cursor.kernel - opBegin.kernel;
      sv.opStandaloneCount.at(i) = cursor.standalone - opBegin.standalone;
      sv.opShortcutCount.at(i) = cursor.shortcut - opBegin.shortcut;
    }
  }
  // Everything downstream -- makeGrid, the param fill, the BlockInfo wiring --
  // walks all of sv.kernels, so a gather that produced fewer launches than a
  // previous one for this step would run a stale launch with a stale param
  // offset. The vectors only ever grow, so say so loudly rather than write
  // params outside the slot reservation.
  TORCH_CHECK(
      cursor.kernel == static_cast<int32_t>(sv.kernels.size()),
      "Gathered ",
      cursor.kernel,
      " kernel launches for node ",
      sequenceNumber_,
      " step ",
      stepIdx,
      " but the step holds ",
      sv.kernels.size(),
      " from an earlier gather");
}

// Invalidates cached grid, launch, and param state for the given step and all
// subsequent steps. Called when the grid choice changes at runtime (e.g.
// switching between single-block and multi-block). Frees pinned buffers so
// that fillLaunchParams writes a complete fresh copy on the next execution
// instead of incrementally updating stale data from the previous grid layout.
void invalidateReusedState(
    std::vector<StepVectors>& steps,
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>&
        pinnedBuffers,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  for (auto i = stepIdx; i < static_cast<int32_t>(steps.size()); ++i) {
    auto& sv = steps[i];
    sv.hasGridCache = false;
    sv.hasLaunchCache = false;
    for (auto& data : sv.kernels) {
      data.tensorsInFrame.clear();
      data.tensorOffsets.clear();
      data.scalarsInFrame.clear();
      data.scalarOffsets.clear();
    }
  }
  if (sequenceNumber < static_cast<int32_t>(pinnedBuffers.size())) {
    auto& buffers = pinnedBuffers[sequenceNumber];
    for (auto i = stepIdx; i < static_cast<int32_t>(buffers.size()); ++i) {
      buffers[i].reset();
    }
  }
}

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
      auto actualOpt = asTensor(frame.getIValue(actualId));
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
          auto actualOpt = asTensor(frame.getIValue(actualId));
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
        auto actualOpt = asTensor(frame.getIValue(actualId));
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
        "{} reference mismatches, {} corrupted, {} passed ({})",
        numMismatches,
        numCorrupted,
        numPassed,
        passedIds);
    if (WaveConfig::get().continueAfterMismatch) {
      LOG(ERROR) << msg;
    } else {
      TORCH_CHECK(false, msg);
    }
  }
}

} // namespace

// Resizes 'tensor' in 'frame' to match the dims returned by the device kernel.
void resizeTensorFromDevice(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    const uint8_t* pinnedBase,
    int64_t absOffset,
    bool trace) {
  auto* t = reinterpret_cast<const Tensor*>(pinnedBase + absOffset);
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
  c10::IValue ival;
  if constexpr (std::is_same_v<T, double>) {
    ival = c10::IValue(val);
  } else if constexpr (std::is_same_v<T, bool>) {
    ival = c10::IValue(val);
  } else {
    ival = c10::IValue(static_cast<int64_t>(val));
  }
  frame.setIValue(id, ival);
  if (trace) {
    if constexpr (std::is_same_v<T, bool>) {
      std::cout << "  D2H: %" << id << " = " << (val ? "true" : "false")
                << std::endl;
    } else {
      std::cout << "  D2H: %" << id << " = " << val << std::endl;
    }
  }
}

void processReturnData(
    StepVectors& sv,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
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
        frame.setIValue(actualId, c10::IValue(val));
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

namespace {

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

// Latest executed step among the pending transfers whose returned values this
// step can read, or -1 when it reads none of them -- in which case the step
// runs start to finish while every one of them is still in flight. sv.readIds
// is a union over the step's grid variants, so this does not depend on a
// variant choice gatherLaunches has not made yet.
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
  }
  return -1;
}

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

void CompositeInvocation::fillStepParams(
    ExecutionState& state,
    StepVectors& sv,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  const bool doTiming = WaveConfig::get().printTiming ||
      (WaveConfig::get().trace & WaveConfig::kTiming);
  const uint64_t tFill = doTiming ? folly::hardware_timestamp() : 0;
  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    auto& data = sv.kernels[i];
    const auto paramOffset = sv.paramOffsets[i];
    int32_t launchReturnBegin = -1;
    int32_t launchReturnEnd = -1;
    fillLaunchParams(
        data,
        *state.frame,
        pinnedBase + paramOffset,
        launchReturnBegin,
        launchReturnEnd);
    patchTensorListPointers(
        data, pinnedBase + paramOffset, deviceBase + paramOffset);
    if (launchReturnBegin >= 0) {
      const auto begin = static_cast<int32_t>(paramOffset + launchReturnBegin);
      const auto end = static_cast<int32_t>(paramOffset + launchReturnEnd);
      returnBegin = returnBegin < 0 ? begin : std::min(returnBegin, begin);
      returnEnd = std::max(returnEnd, end);
    }
  }
  if (doTiming) {
    sv.fillUs += static_cast<int64_t>(
        (folly::hardware_timestamp() - tFill) / tscTicksPerMicro());
  }
}

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

void CompositeInvocation::releaseLastUseAtStep(
    ExecutionState& state,
    const std::vector<GridChoice>& grids,
    int32_t stepIdx,
    StepVectors& sv,
    std::vector<int32_t>& releaseStep) {
  // An op with a grid of L steps does its last work at step L-1, so once every
  // reader of a value is past that, no later step of this node can read it.
  // The grid length is read after gatherLaunches, which is the only place a
  // variant is switched and only ever at a step the op still has -- so an op
  // already past its end cannot grow one later.
  auto readersDone = [&](const std::vector<int32_t>& readers) {
    for (auto op : readers) {
      if (static_cast<int32_t>(grids.at(op).grid->size()) > stepIdx + 1) {
        return false;
      }
    }
    return true;
  };
  const bool traceFrame = (WaveConfig::get().trace & WaveConfig::kFrame) != 0;
  for (size_t i = 0; i < lastUseIds_.size(); ++i) {
    const auto& readers = lastUseReaderOps_[i];
    // releaseStep is sized to lastUseIds_.size() by the only caller, which
    // also gates the call on stepLastUse; i runs over the same bound.
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    if (releaseStep[i] >= 0 || readers.empty() || !readersDone(readers)) {
      continue;
    }
    releaseStep[i] = stepIdx;
    addLastUseId(state, sv, lastUseIds_[i]);
    if (traceFrame) {
      state.releasedAtStep[lastUseIds_[i]] = state.executedSteps;
    }
  }
}

// Central execution loop: for each step, gathers launches from the
// subgraph, builds the thread-block grid, fills kernel parameters
// (tensor pointers, scalars, shapes), transfers params H2D, launches
// the CUDA kernel, transfers return values D2H, and verifies against
// the reference frame if set.
void CompositeInvocation::execute(ExecutionState& state) {
  if (allocGroupEnabled()) {
    executeAllocGroups(state);
    return;
  }
  Timer ex("comp inv execute", WaveConfig::get().printTiming);
  auto& frame = *state.frame;
  const int64_t reuseCount0 = gElementwiseReuseCount;
  const int64_t reuseBytes0 = gElementwiseReuseBytes;

  if (WaveConfig::get().trace & (WaveConfig::kNodes | WaveConfig::kLaunches)) {
    std::cout << "==== Node " << sequenceNumber_ << std::endl;
  }

  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber_, 0);
  auto& gridChoices = sv0.gridChoices;
  // Reset each op's grid-variant choice to the multi-block default on every
  // execution. gridChoices lives in the pooled ExecutionState, so it would
  // otherwise carry a prior run's evolved choice into the next frame reuse.
  // The single-block variant of a scanOutputReturnBarrier op has fewer steps
  // than its multi-block variant, so starting a reused frame from a persisted
  // single-block choice drops that op's multi-block-only steps, leaving their
  // outputs unproduced (None) and crashing a later consumer. Re-deriving from
  // the default each run makes every execution schedule identically to the
  // first; gatherLaunches re-applies the single-block switch as needed.
  gridChoices.clear();
  for (auto& op : ops_) {
    gridChoices.push_back({0, false, &op.projectOp()->grid()});
  }

  using Clock = std::chrono::high_resolution_clock;
  bool doTiming = WaveConfig::get().printTiming ||
      (WaveConfig::get().trace & WaveConfig::kTiming);
  // Event timing is keyed strictly on kTiming, so printTiming stays the cheap
  // host-only path and does not allocate timing events.
  bool doEventTiming = (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - start)
        .count();
  };

  // Track eager standalone execution so the default CUDA stream can be
  // synchronized before this invocation returns. Eager standalone ops run on
  // the default stream while wave kernels run on the wave stream, and the two
  // are otherwise unordered, so a final default-stream sync is needed.
  bool ranStandalones = false;

  // Copying saved by this node's elided clones. Each input is charged to the
  // first step where it has a tensor; 'elidedCounted' keeps the later steps of
  // this invocation from counting it again.
  const bool countElidedClones =
      (WaveConfig::get().trace & WaveConfig::kTiming) &&
      !elidedCloneInputs_.empty();
  std::vector<bool> elidedCounted(
      countElidedClones ? elidedCloneInputs_.size() : 0, false);
  auto addElidedCloneBytes = [&](StepVectors& sv) {
    for (size_t i = 0; i < elidedCloneInputs_.size(); ++i) {
      if (elidedCounted[i]) {
        continue;
      }
      const auto& [valueId, numClones] = elidedCloneInputs_[i];
      const auto& ivalue = frame.getIValue(valueId);
      if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
        continue;
      }
      const auto& tensor = ivalue.toTensor();
      sv.elidedCloneBytes += tensor.numel() * tensor.element_size() * numClones;
      elidedCounted[i] = true;
    }
  };

  const bool stepLastUse = stepLevelRelease();
  std::vector<int32_t> lastUseReleaseStep(
      stepLastUse ? lastUseIds_.size() : 0, -1);

  int32_t blockSize;
  int32_t lastExecStep = -1;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    // No step-level wait for pending transfers here. Waiting for everything the
    // step can read would resolve every dependency before gatherLaunches got to
    // test any op, so nothing would ever be deferred and the step would stall
    // whole for one op's transfer. The per-op test inside gatherLaunches drives
    // it instead: independent ops are set up first, then the wait, then the
    // rest.
    drainBeforeStepIfRequested(state);
    // Running ahead delays every free until the device catches up, so give the
    // memory back before this step allocates if too much has piled up.
    enforceDelayedFreeLimit(state);
    // Re-fetch since the resize above may have invalidated the reference.
    auto& currentGridChoices =
        state.stepVectors.at(sequenceNumber_).at(0).gridChoices;

    // The parameter buffer needs only the step's slot reservation and block
    // capacity, both fixed by layoutParamSlots from the compiled grids, so it
    // can be acquired before anything is gathered. That is what lets each op be
    // filled in the same pass that sizes and allocates it, instead of in a
    // second sweep over sv.kernels.
    layoutParamSlots(stepIdx, sv);
    sv.blockInfoOffset = roundUp(sv.paramRegionBytes, int64_t{16});
    const int64_t totalPinnedBytes = sv.blockInfoOffset +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(BlockInfo));
    const int64_t totalAllocBytes = totalPinnedBytes +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(DebugInfo));
    uint8_t* pinnedBase = nullptr;
    uint8_t* deviceBase = nullptr;
    // Not timed into allocUs: these come from a preallocated arena and cost
    // nothing, so counting them would hide the allocation that does cost.
    auto acquireBuffers = [&]() {
      pinnedBase = getOrAllocateBuffer(
                       state.pinnedBuffers,
                       sequenceNumber_,
                       stepIdx,
                       totalAllocBytes,
                       state.pinnedArena,
                       WaveConfig::get().debugSingleOps
                           ? std::function<void(void*, int64_t)>(
                                 [](void* ptr, int64_t bytes) {
                                   memset(ptr, 0xaa, bytes);
                                 })
                           : nullptr)
                       ->as<uint8_t>();
      deviceBase = getOrAllocateBuffer(
                       state.deviceBuffers,
                       sequenceNumber_,
                       stepIdx,
                       totalAllocBytes,
                       state.deviceArena)
                       ->as<uint8_t>();
      // The arena always hands back a buffer, and every use below indexes off
      // these; say so once rather than at each use.
      TORCH_CHECK(pinnedBase != nullptr && deviceBase != nullptr);
    };
    acquireBuffers();
    TORCH_CHECK(pinnedBase != nullptr && deviceBase != nullptr);

    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    std::vector<std::pair<size_t, DeferReason>> deferred;
    {
      auto t0 = Clock::now();
      sv.allocUs = 0;
      sv.allocCallUs = 0;
      sv.fillUs = 0;
      // Pooled across executions, so last run's groups would otherwise add to
      // this one's.
      sv.allocGroups = 0;
      sv.allocGroupTensors = 0;
      gatherLaunches(
          state,
          currentGridChoices,
          stepIdx,
          sv,
          pinnedBase,
          deviceBase,
          /*deferredOnly=*/false,
          deferred,
          returnBegin,
          returnEnd);
      if (sv.gridChanged) {
        ++state.numGridRedos;
        // A variant switch changed this step's launch layout, so the caches the
        // fill just used and the parameter blocks it wrote belong to the old
        // one. invalidateReusedState also drops the pinned buffer, so the bases
        // have to be taken again before the redo.
        // stepVectors has one entry per sequence number, created before the
        // invocation runs.
        // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
        auto& seqStepVectors = state.stepVectors[sequenceNumber_];
        invalidateReusedState(
            seqStepVectors, state.pinnedBuffers, sequenceNumber_, stepIdx);
        acquireBuffers();
        TORCH_CHECK(pinnedBase != nullptr && deviceBase != nullptr);
        deferred.clear();
        returnBegin = -1;
        returnEnd = -1;
        sv.allocUs = 0;
        sv.allocCallUs = 0;
        sv.fillUs = 0;
        gatherLaunches(
            state,
            currentGridChoices,
            stepIdx,
            sv,
            pinnedBase,
            deviceBase,
            /*deferredOnly=*/false,
            deferred,
            returnBegin,
            returnEnd);
      }
      if (doTiming) {
        sv.gatherUs = elapsed(t0) - sv.allocUs - sv.fillUs;
      }
    }
    // StepVectors are pooled and reused across executions; reset the
    // accumulated ref-check time and elided-clone bytes so they reflect only
    // this run (other timing fields are overwritten with '=' at their
    // measurement point).
    sv.refCheckUs = 0;
    sv.elidedCloneBytes = 0;
    if (sv.kernels.empty() && sv.standalones.empty() &&
        sv.shortcutStandalones.empty()) {
      break;
    }
    lastExecStep = stepIdx;
    // Past the break above, so only steps that actually do work are sampled,
    // and after every wait this step had to take, so the sample is the device
    // state its interpretation gets to overlap with. The gather that precedes
    // it is charged to the previous sample's window, which is the conservative
    // direction: it can only make the measured depth look larger.
    sampleRunAhead(state);

    // Second pass: everything that could be set up without waiting is done, so
    // now wait for exactly what the rest need and finish them. Only makeGrid
    // and the BlockInfo array are left after this.
    if (!deferred.empty()) {
      bool forMemory = false;
      int32_t throughStep = -1;
      for (const auto& [opIndex, reason] : deferred) {
        if (reason == DeferReason::kMemory) {
          forMemory = true;
        } else {
          throughStep = std::max(
              throughStep, neededPendingStep(state, sv.opReadIds[opIndex]));
        }
      }
      // Deliberately outside the timed region below: this blocks the host on
      // the device, so it is CPU idle waiting for the GPU, not interpretation.
      // Counting it as interpretation is what made the second pass look like
      // it had made the setup loop slower.
      if (forMemory) {
        // Memory only comes back as the device catches up, so this waits for
        // everything rather than for one transfer.
        enforceDelayedFreeLimit(state);
        resolveAllPendingReturns(state);
      } else {
        resolvePendingReturns(state, throughStep);
      }
      auto t0 = Clock::now();
      const auto allocBefore = sv.allocUs;
      const auto fillBefore = sv.fillUs;
      gatherLaunches(
          state,
          currentGridChoices,
          stepIdx,
          sv,
          pinnedBase,
          deviceBase,
          /*deferredOnly=*/true,
          deferred,
          returnBegin,
          returnEnd);
      if (doTiming) {
        sv.gatherUs +=
            elapsed(t0) - (sv.allocUs - allocBefore) - (sv.fillUs - fillBefore);
        state.numDeferredOps += static_cast<int32_t>(deferred.size());
        ++state.numDeferredSteps;
      }
    }

    // Which of this step's ops are blocked on an earlier step's D2H, and which
    // values this step will itself send back. Marked before the step runs, so
    // the producer map holds only genuinely earlier steps.
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
    // Release what no later step of this node reads. Done here rather than
    // after the loop because a step that has already been swept to kSynced is
    // never revisited, so a value stamped onto it later would never be freed.
    if (stepLastUse) {
      releaseLastUseAtStep(
          state, currentGridChoices, stepIdx, sv, lastUseReleaseStep);
    }

    if (sv.kernels.empty()) {
      if (WaveConfig::get().trace &
          (WaveConfig::kNodes | WaveConfig::kLaunches)) {
        traceStep(stepIdx, sv, currentGridChoices);
      }
      // Metadata-only shortcut standalones are host-only and need no
      // wave-stream sync; run them first in their tight, batch-timed loop.
      runShortcutStandalones(
          sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
      // Order the eager standalones after the last wave kernel, on the device
      // rather than by blocking the host. They run on the torch stream and read
      // inputs produced by wave kernels; without the edge an eager op can read
      // a wave-stream buffer whose producing kernel has not completed, since
      // the wave streams are non-blocking and the two are otherwise unordered.
      // Shortcut ops only touch host-side tensor metadata, so they need none.
      //
      // Enqueue the wait BEFORE recording standaloneBegin, so that event's
      // timestamp is when this step's GPU work actually started with its
      // dependency already satisfied. Recorded first it would fire as soon as
      // the stream drained and silently understate idle.
      auto& stepEvents = newStepEvents(state, sequenceNumber_, stepIdx);
      if (sv.hasGpuStandalones) {
        if (state.lastWaveDone != nullptr) {
          state.lastWaveDone->wait(torchStream());
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
        stepEvents.standaloneDone->record(torchStream());
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
        // Drain streams outside the timed region (real standalone/GPU work
        // belongs in e2e); time only the device-to-host copy and comparison.
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
      // Standalone-only step issued; the next wave-stream wait advances it to
      // kSynced (and frees its lastUseIds if freeIntermediates is on).
      sv.executionStage = ExecutionStage::kAllocated;
      continue;
    }

    // Trace inputs of kernel launches before execution.
    if (!state.traceState.empty()) {
      for (const auto& launch : sv.kernels) {
        traceFrameValues("input", launch.actualInputs, frame, state.traceState);
      }
    }

    auto* deviceDebugBase =
        reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
    {
      // Sizing, output allocation and the parameter fill all happened per op in
      // gatherLaunches above; only the byte accounting is left here.
      if (doTiming) {
        sv.inputBytes = 0;
        sv.outputBytes = 0;
        for (size_t i = 0; i < sv.kernels.size(); ++i) {
          for (size_t j = 0; j < sv.kernels[i].tensorsInFrame.size(); ++j) {
            auto off = sv.kernels[i].tensorOffsets[j];
            auto* t = reinterpret_cast<Tensor*>(
                pinnedBase + sv.paramOffsets[i] + off);
            auto bytes = static_cast<int64_t>(t->numEl) * t->elementSize;
            if (sv.kernels[i].shapeOnlyTensorIndices.count(j)) {
              continue;
            }
            if (j <
                static_cast<size_t>(sv.kernels[i].launch->op->numInputs())) {
              sv.inputBytes += bytes;
            } else {
              sv.outputBytes += bytes;
            }
          }
        }
      }
    }

    // Grid last: the params are already in the pinned buffer, so the only work
    // left after this point is the BlockInfo array, which is what the block
    // count actually determines.
    {
      auto t0 = Clock::now();
      if (gridSizesMatch(sv.kernels, sv)) {
        blockSize = sv.cachedBlockSize;
      } else {
        const auto kernelInfo = kernel_->kernelInfo();
        blockSize = makeGrid(
            sv.kernels, sv, kernelInfo.maxOccupancy0, kernelInfo.sharedMemory);
        TORCH_CHECK(
            (blockSize & (blockSize - 1)) == 0,
            "Block size must be a power of two, got ",
            blockSize);
        sv.cachedBlockSize = blockSize;
        updateGridSizeBounds(sv.kernels, sv);
      }
      if (doTiming) {
        sv.gridUs = elapsed(t0);
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
      auto* pinnedBlocks =
          reinterpret_cast<BlockInfo*>(pinnedBase + sv.blockInfoOffset);
      if (!sv.blocks.empty()) {
        memcpy(pinnedBlocks, sv.blocks.data(), numBlocks * sizeof(BlockInfo));
      }
      for (size_t b = 0; b < numBlocks; ++b) {
        auto idx = sv.launchIndices[b];
        pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
        pinnedBlocks[b].debugInfo = deviceDebugBase + b;
      }
      if (doTiming) {
        sv.fillUs += elapsed(t0);
      }
    }

    if (WaveConfig::get().trace &
        (WaveConfig::kNodes | WaveConfig::kLaunches)) {
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
      // Metadata-only shortcut ops: host-only, tight batch-timed loop, no sync.
      if (!sv.shortcutStandalones.empty()) {
        runShortcutStandalones(
            sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
      }
      if (!sv.standalones.empty()) {
        // This lambda runs inside launch(), i.e. after the kernel has been
        // enqueued on the wave stream. The standalones wait on lastWaveDone --
        // step N-1's kernel -- not on this step's, which is exactly what lets
        // the two overlap.
        if (state.lastWaveDone != nullptr) {
          state.lastWaveDone->wait(torchStream());
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
        stepEvents.standaloneDone->record(torchStream());
        state.lastStandaloneDone = stepEvents.standaloneDone.get();
        if (doTiming) {
          standaloneElapsed = elapsed(tStandalone);
        }
        ranStandalones = true;
      }
    };

    // Order this step's kernel after the previous step's eager standalones,
    // which may have produced buffers it reads. Device-side, so the host does
    // not block; the wait is enqueued before waveBegin is recorded so that
    // event marks the real start of GPU work (see the kernel-less path above).
    if (state.lastStandaloneDone != nullptr) {
      state.lastStandaloneDone->wait(*state.stream);
    }
    if (doEventTiming) {
      stepEvents.waveBegin->record(*state.stream);
    }

    // Under debug_single_ops launch() waits after every block anyway, so there
    // is nothing to defer there.
    const bool deferReturn = WaveConfig::get().deferD2h && returnBegin >= 0 &&
        !WaveConfig::get().debugSingleOps;
    {
      auto tLaunch = Clock::now();
      launch(
          static_cast<int32_t>(numBlocks),
          blockSize,
          pinnedBase,
          deviceBase,
          // Only the params and the BlockInfos actually in use go to the
          // device; the rest of the reservation is capacity, not content.
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
      // A device-side event wait orders the GPU but tells the host nothing, so
      // steps cannot be declared synced by it. Sweep with the non-blocking
      // Event::query() instead to free what has actually completed.
      advanceCompletedStages(state);

      if (deferReturn) {
        // The transfer is in flight and the pinned buffer is not readable yet.
        // A later step that needs it waits then; one that does not runs right
        // past it.
        state.pendingReturns.push_back(
            {sequenceNumber_,
             stepIdx,
             sv.executedStep,
             stepEvents.waveDone.get()});
      } else if (returnBegin >= 0) {
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

    // Trace outputs of kernel launches after execution.
    if (!state.traceState.empty()) {
      syncWaveStream(state);
      // The traced values include shapes that only the return data settles.
      resolveAllPendingReturns(state);
      for (const auto& launch : sv.kernels) {
        traceFrameValues(
            "output", launch.actualOutputs, frame, state.traceState);
      }
    }

    {
      // Reference-frame checking does an extra device-to-host copy and a
      // host-side comparison. This is debug-only overhead that inflates the
      // measured wall time, so time it separately when it is on so the report
      // can subtract it from the e2e time. Drain the wave and default streams
      // first, OUTSIDE the timed region: that wait is for real GPU/standalone
      // work that belongs in the e2e time, not the checking overhead. The waits
      // inside verifyAgainstReference are then no-ops, so the timed span covers
      // only the device-to-host copy and comparison.
      bool timeRefCheck = doTiming && WaveConfig::get().referenceFrame;
      if (timeRefCheck) {
        syncWaveStream(state);
        syncTorchDefaultStream();
      }
      if (WaveConfig::get().referenceFrame != nullptr) {
        // The check walks this step's outputs, whose shapes the return data
        // settles, so it has to see a complete frame.
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
    // Kernel launched and its outputs allocated; the next wave-stream wait
    // advances this step to kSynced.
    sv.executionStage = ExecutionStage::kAllocated;
  }

  // If any eager standalone op ran, synchronize the default CUDA stream before
  // returning. The eager ops run on the default stream and are otherwise
  // unordered against wave-stream kernels of later invocations, which can
  // recycle arena buffers an eager op still reads. This sync follows any
  // wave-stream sync already done above (e.g. a device-to-host transfer).
  //
  // Under WaveConfig::runAhead it is skipped. What it guards is now carried
  // device-side: every later wave kernel waits on lastStandaloneDone before it
  // launches, and a step's frame values are only freed once its own
  // standaloneDone has been observed complete, so nothing can recycle a buffer
  // an in-flight eager op still reads. executeWave still drains both streams
  // before it returns. Draining here instead costs one host stall per node,
  // which is what keeps the host from ever getting ahead of the device.
  if (ranStandalones && !WaveConfig::get().runAhead) {
    // Each step's standaloneUs already covers its eager ops (the per-op
    // default- stream sync in runStandalones drains them at their own step), so
    // this final sync is only for correctness -- it must not be folded back
    // into any step's standalone time, which previously charged the whole-graph
    // async tail to the first standalone step and over-reported standalone time
    // past e2e.
    syncTorchDefaultStream();
  }

  // Stamp the last-use values no step claimed onto this node's last executed
  // step, so they are released by the wave-stream sync that advances that step
  // to kSynced, with no dedicated sync of their own. advanceSyncedStages
  // performs the release -- unless that step has already been swept, which
  // syncEachStep does at the top of the iteration that ends the loop. A kSynced
  // step is never revisited, so free those here instead of leaving them for a
  // sweep that will not come.
  if (WaveConfig::get().freeIntermediates && lastExecStep >= 0 &&
      !lastUseIds_.empty()) {
    auto& lastSv =
        getStepVectors(state.stepVectors, sequenceNumber_, lastExecStep);
    std::vector<nativert::ValueId> atNodeEnd;
    for (size_t i = 0; i < lastUseIds_.size(); ++i) {
      // Sized to lastUseIds_.size() when stepLastUse, and only read then.
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
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
      // Same hazard as in the sweeps: a value freed here can be one this step
      // sent back, and parsing into a cleared frame slot afterwards would read
      // a None. kSynced implies the transfer has landed, so this only copies.
      resolvePendingReturns(state, lastSv.executedStep);
      freeLastUseNow(state, atNodeEnd);
    } else {
      for (auto id : atNodeEnd) {
        addLastUseId(state, lastSv, id);
      }
    }
  }

  // Per-node elementwise input-reuse summary (alongside the alloc traces
  // above).
  if ((WaveConfig::get().trace & WaveConfig::kTiming) &&
      gElementwiseReuseCount > reuseCount0) {
    std::cout << "  node " << sequenceNumber_ << " elementwise input reuse: "
              << (gElementwiseReuseCount - reuseCount0) << " tensors, "
              << (gElementwiseReuseBytes - reuseBytes0) / 1024
              << " KB written in place" << std::endl;
  }
}

// The allocation-group execute path.
//
// Defined here rather than in AllocGroup.cpp, where the rest of the mode lives,
// because the step loop is built almost entirely out of this file's internal
// helpers -- getStepVectors, the buffer arena, the pending-return machinery,
// the event bookkeeping -- none of which is declared in a header. Moving them
// out to move this out would be a much larger change than the mode. Everything
// that does not need them (the plan, the grouping, the collector, the buffer
// carving) is in AllocGroup.cpp.
//
// Differs from execute() only in how a step's outputs are allocated:
//
//   1. The grid is taken as fixed. execute() starts every op on the multi-block
//      default and lets gatherLaunches switch it; here the cooperative grid is
//      chosen up front, because the plan's step indices are indices into it and
//      a switch would silently renumber them.
//   2. A collector is installed for the step's groups, which diverts their
//      members' allocations into shape records and suppresses the inline
//      parameter fill.
//   3. Groups are carved as they complete: the sync-free ones after the first
//      pass, before the host waits on anything, and the rest after the deferred
//      pass has run.
//   4. The parameters are filled in one sweep at the end, once every output
//      tensor exists.
void CompositeInvocation::executeAllocGroups(ExecutionState& state) {
  Timer ex("comp inv execute allocgroup", WaveConfig::get().printTiming);
  auto& frame = *state.frame;
  const int64_t reuseCount0 = gElementwiseReuseCount;
  const int64_t reuseBytes0 = gElementwiseReuseBytes;

  if (WaveConfig::get().trace & (WaveConfig::kNodes | WaveConfig::kLaunches)) {
    std::cout << "==== Node " << sequenceNumber_ << " (alloc groups)"
              << std::endl;
  }

  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber_, 0);
  auto& gridChoices = sv0.gridChoices;
  gridChoices.clear();
  for (auto& op : ops_) {
    // Fixed at the cooperative grid, not the multi-block default that
    // gatherLaunches would switch away from. An op with no cooperative variant
    // keeps its own grid; the plan was built from the same choice, so the two
    // agree.
    auto& cg = op.projectOp()->cgGrid();
    gridChoices.push_back(
        {0, false, cg.empty() ? &op.projectOp()->grid() : &cg});
  }

  // A lifetime crosses nodes: the node that allocates a value is rarely the one
  // whose last use releases it, so the grouping is decided for the graph as a
  // whole and each node is handed the groups it allocates. Built on the first
  // execution that reaches this, since the mode is a runtime choice, and shared
  // by every later one -- the compiled grids it reads do not change.
  if (allocGroupPlan_ == nullptr) {
    state.waveGraph->ensureAllocGroupPlans([&] {
      installGraphAllocGroupPlans(*state.waveGraph, *state.valueTypes);
    });
    TORCH_CHECK(
        allocGroupPlan_ != nullptr,
        "Node ",
        sequenceNumber_,
        " got no allocation-group plan; it is not one of the compiled nodes the "
        "graph-wide plan was built from");
  }
  const auto& plan = *allocGroupPlan_;

  using Clock = std::chrono::high_resolution_clock;
  const bool doTiming = WaveConfig::get().printTiming ||
      (WaveConfig::get().trace & WaveConfig::kTiming);
  const bool doEventTiming =
      (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - start)
        .count();
  };

  bool ranStandalones = false;

  const bool countElidedClones =
      (WaveConfig::get().trace & WaveConfig::kTiming) &&
      !elidedCloneInputs_.empty();
  std::vector<bool> elidedCounted(
      countElidedClones ? elidedCloneInputs_.size() : 0, false);
  auto addElidedCloneBytes = [&](StepVectors& sv) {
    for (size_t i = 0; i < elidedCloneInputs_.size(); ++i) {
      if (elidedCounted[i]) {
        continue;
      }
      const auto& [valueId, numClones] = elidedCloneInputs_[i];
      const auto& ivalue = frame.getIValue(valueId);
      if (!ivalue.isTensor() || !ivalue.toTensor().defined()) {
        continue;
      }
      const auto& tensor = ivalue.toTensor();
      sv.elidedCloneBytes += tensor.numel() * tensor.element_size() * numClones;
      elidedCounted[i] = true;
    }
  };

  const bool stepLastUse = stepLevelRelease();
  std::vector<int32_t> lastUseReleaseStep(
      stepLastUse ? lastUseIds_.size() : 0, -1);

  int32_t blockSize;
  int32_t lastExecStep = -1;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    drainBeforeStepIfRequested(state);
    enforceDelayedFreeLimit(state);
    auto& currentGridChoices =
        state.stepVectors.at(sequenceNumber_).at(0).gridChoices;

    layoutParamSlots(stepIdx, sv);
    sv.blockInfoOffset = roundUp(sv.paramRegionBytes, int64_t{16});
    const int64_t totalPinnedBytes = sv.blockInfoOffset +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(BlockInfo));
    const int64_t totalAllocBytes = totalPinnedBytes +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(DebugInfo));
    uint8_t* pinnedBase = getOrAllocateBuffer(
                              state.pinnedBuffers,
                              sequenceNumber_,
                              stepIdx,
                              totalAllocBytes,
                              state.pinnedArena,
                              WaveConfig::get().debugSingleOps
                                  ? std::function<void(void*, int64_t)>(
                                        [](void* ptr, int64_t bytes) {
                                          memset(ptr, 0xaa, bytes);
                                        })
                                  : nullptr)
                              ->as<uint8_t>();
    uint8_t* deviceBase = getOrAllocateBuffer(
                              state.deviceBuffers,
                              sequenceNumber_,
                              stepIdx,
                              totalAllocBytes,
                              state.deviceArena)
                              ->as<uint8_t>();

    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    std::vector<std::pair<size_t, DeferReason>> deferred;

    // The step's groups. Empty is the common case for a step whose outputs all
    // escape, and installing an empty collector anyway is what keeps the fill
    // suppression uniform across every step of the mode.
    std::vector<const AllocGroup*> stepGroups;
    if (stepIdx < static_cast<int32_t>(plan.groupsByStep.size())) {
      for (auto g : plan.groupsByStep[stepIdx]) {
        stepGroups.push_back(&plan.groups[g]);
      }
    }
    AllocGroupCollector collector(stepGroups);

    // Carves the step's groups. Called once before the host waits -- which is
    // what picks up the sync-free ones -- and again after the deferred pass;
    // groups already carved are skipped the second time.
    //
    // Before the wait only a group all of whose members are sized can be
    // carved: an unsized member may simply be one a deferred op has not reached
    // yet. After the deferred pass the step's sizing is over, so what is still
    // unsized never will be -- the plan proposed a member the sizing path
    // allocates some other way -- and the group is carved without it.
    auto materializeReady = [&](bool afterWait) {
      for (size_t g = 0; g < collector.numGroups(); ++g) {
        if (collector.materialized(g) ||
            (!afterWait && !collector.complete(g))) {
          continue;
        }
        // A concat group is laid out by the shape of the result rather than by
        // packing slots, and its members are the regions of that result. A
        // member the sizing pass never reached is not dropped the way an
        // ordinary group's is: the layout still has to account for its extent,
        // so it is measured from the frame and simply not carved.
        const auto* concat = collector.concatLayout(g);
        int32_t numCarved = 0;
        AllocGroupBuffer buffer;
        const uint64_t tAlloc = doTiming ? folly::hardware_timestamp() : 0;
        const int64_t allocCallBefore = threadAllocCallUs();
        if (concat != nullptr) {
          buffer = materializeConcatGroup(
              *concat, collector.requests(g), collector.sizedMask(g), frame);
          for (const auto& slot : buffer.slots) {
            numCarved += slot.defined() ? 1 : 0;
          }
        } else {
          auto requests = collector.sizedRequests(g);
          if (requests.empty()) {
            collector.markMaterialized(g);
            continue;
          }
          buffer = materializeAllocGroup(requests, frame);
          numCarved = static_cast<int32_t>(requests.size());
        }
        collector.markMaterialized(g);
        ++sv.allocGroups;
        sv.allocGroupTensors += numCarved;
        if (doTiming) {
          sv.allocUs += static_cast<int64_t>(
              (folly::hardware_timestamp() - tAlloc) / tscTicksPerMicro());
          sv.allocCallUs += threadAllocCallUs() - allocCallBefore;
        }
        if (allocTraceEnabled()) {
          logAllocEvent("allocgroup", -1, buffer.totalBytes);
        }
      }
    };

    {
      auto t0 = Clock::now();
      sv.allocUs = 0;
      sv.allocCallUs = 0;
      sv.fillUs = 0;
      // Pooled across executions, so last run's groups would otherwise add to
      // this one's.
      sv.allocGroups = 0;
      sv.allocGroupTensors = 0;
      gatherLaunches(
          state,
          currentGridChoices,
          stepIdx,
          sv,
          pinnedBase,
          deviceBase,
          /*deferredOnly=*/false,
          deferred,
          returnBegin,
          returnEnd);
      // The grid was fixed before the loop, so nothing can have switched it.
      // If it ever did, the plan's step indices would no longer name the steps
      // the groups were built from and a group could outlive its buffer.
      TORCH_CHECK(
          !sv.gridChanged,
          "Grid variant switched at node ",
          sequenceNumber_,
          " step ",
          stepIdx,
          " in allocation-group mode, where the grid is fixed before execution");
      if (doTiming) {
        sv.gatherUs = elapsed(t0) - sv.allocUs - sv.fillUs;
      }
    }
    // Everything sizeable without blocking is sized, so the groups that need no
    // transfer are complete: carve them before the host waits on anything.
    materializeReady(/*afterWait=*/false);

    sv.refCheckUs = 0;
    sv.elidedCloneBytes = 0;
    if (sv.kernels.empty() && sv.standalones.empty() &&
        sv.shortcutStandalones.empty()) {
      break;
    }
    lastExecStep = stepIdx;
    sampleRunAhead(state);

    if (!deferred.empty()) {
      bool forMemory = false;
      int32_t throughStep = -1;
      for (const auto& [opIndex, reason] : deferred) {
        if (reason == DeferReason::kMemory) {
          forMemory = true;
        } else {
          throughStep = std::max(
              throughStep, neededPendingStep(state, sv.opReadIds[opIndex]));
        }
      }
      if (forMemory) {
        enforceDelayedFreeLimit(state);
        resolveAllPendingReturns(state);
      } else {
        resolvePendingReturns(state, throughStep);
      }
      auto t0 = Clock::now();
      const auto allocBefore = sv.allocUs;
      const auto fillBefore = sv.fillUs;
      gatherLaunches(
          state,
          currentGridChoices,
          stepIdx,
          sv,
          pinnedBase,
          deviceBase,
          /*deferredOnly=*/true,
          deferred,
          returnBegin,
          returnEnd);
      if (doTiming) {
        sv.gatherUs +=
            elapsed(t0) - (sv.allocUs - allocBefore) - (sv.fillUs - fillBefore);
        state.numDeferredOps += static_cast<int32_t>(deferred.size());
        ++state.numDeferredSteps;
      }
    }
    // The transfers have landed, so the groups that were waiting on a returned
    // count are sized now.
    materializeReady(/*afterWait=*/true);

    // What the plan claimed and the sizing pass did not deliver. Not an error
    // -- those values were allocated the ordinary way -- but it is the gap
    // between the grouping and what the mode can actually fold, so it is worth
    // seeing on a measured run.
    if (doTiming) {
      for (size_t g = 0; g < collector.numGroups(); ++g) {
        const auto notSized = collector.missing(g);
        if (!notSized.empty()) {
          std::cout << "  node " << sequenceNumber_ << " step " << stepIdx
                    << " group " << g << ": " << notSized.size() << " of "
                    << collector.requests(g).size()
                    << " values were not sized by any launch and were left to "
                       "the ordinary path"
                    << std::endl;
        }
      }
    }

    // Every output tensor exists now, so the parameter blocks can be filled --
    // the one sweep that replaces the per-launch fill the ordinary path does.
    fillStepParams(state, sv, pinnedBase, deviceBase, returnBegin, returnEnd);

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
      releaseLastUseAtStep(
          state, currentGridChoices, stepIdx, sv, lastUseReleaseStep);
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
          state.lastWaveDone->wait(torchStream());
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
        stepEvents.standaloneDone->record(torchStream());
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
      continue;
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
        blockSize =
            makeGrid(sv.kernels, sv, kernel_->kernelInfo().maxOccupancy0);
        TORCH_CHECK(
            (blockSize & (blockSize - 1)) == 0,
            "Block size must be a power of two, got ",
            blockSize);
        sv.cachedBlockSize = blockSize;
        updateGridSizeBounds(sv.kernels, sv);
      }
      if (doTiming) {
        sv.gridUs = elapsed(t0);
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
      auto* pinnedBlocks =
          reinterpret_cast<BlockInfo*>(pinnedBase + sv.blockInfoOffset);
      if (!sv.blocks.empty()) {
        memcpy(pinnedBlocks, sv.blocks.data(), numBlocks * sizeof(BlockInfo));
      }
      for (size_t b = 0; b < numBlocks; ++b) {
        auto idx = sv.launchIndices[b];
        pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
        pinnedBlocks[b].debugInfo = deviceDebugBase + b;
      }
      if (doTiming) {
        sv.fillUs += elapsed(t0);
      }
    }

    if (WaveConfig::get().trace &
        (WaveConfig::kNodes | WaveConfig::kLaunches)) {
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
          state.lastWaveDone->wait(torchStream());
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
        stepEvents.standaloneDone->record(torchStream());
        state.lastStandaloneDone = stepEvents.standaloneDone.get();
        if (doTiming) {
          standaloneElapsed = elapsed(tStandalone);
        }
        ranStandalones = true;
      }
    };

    if (state.lastStandaloneDone != nullptr) {
      state.lastStandaloneDone->wait(*state.stream);
    }
    if (doEventTiming) {
      stepEvents.waveBegin->record(*state.stream);
    }

    const bool deferReturn = WaveConfig::get().deferD2h && returnBegin >= 0 &&
        !WaveConfig::get().debugSingleOps;
    {
      auto tLaunch = Clock::now();
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
      } else if (returnBegin >= 0) {
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
        traceFrameValues(
            "output", launch.actualOutputs, frame, state.traceState);
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
  const int32_t dynShared = dynamicSharedBytes(sv.kernels);

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

  if (WaveConfig::get().debugSingleOps) {
    std::vector<int32_t> originalOps(numBlocks);
    for (int32_t b = 0; b < numBlocks; ++b) {
      originalOps[b] = pinnedBlocks[b].op;
    }

    // Transfer pinned buffer to device once.
    stream->hostToDeviceAsync(deviceBase, pinnedBase, h2dBytes);
    stream->wait();

    auto* deviceBlocks =
        reinterpret_cast<BlockInfo*>(deviceBase + sv.blockInfoOffset);

    // Run blocks individually or grouped. Ops with barrierCounters need all
    // blocks of the same project op launched together with cooperative launch.
    folly::F14FastSet<uintptr_t> launched;
    for (int32_t active = 0; active < numBlocks; ++active) {
      // launchIndices has one entry per block; guard the access so a short
      // vector fails loudly instead of reading out of bounds.
      TORCH_CHECK(active < static_cast<int32_t>(sv.launchIndices.size()));
      auto launchIdx = sv.launchIndices[active];
      // A barrier op needs cooperative grouping only when it spans more than
      // one block (see 'cooperative' above): opBarrier waits for numBlocksInOp
      // arrivals, which is immediate for a single-block op.
      bool hasBarriers = launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
          sv.kernels[launchIdx].launch && sv.kernels[launchIdx].launch->op &&
          !sv.kernels[launchIdx].launch->op->barrierCounters().empty() &&
          launchIdx < static_cast<int32_t>(sv.numBlocksPerLaunch.size()) &&
          sv.numBlocksPerLaunch[launchIdx] > 1;

      // Under a cooperative grid the whole step is compiled as one cooperative
      // kernel whose cross-block barriers require every block of an op to be
      // co-resident and launched cooperatively. Single-stepping a subset of an
      // op's blocks, or launching that kernel via the regular (non-cooperative)
      // path, faults with an illegal memory access. So when the step needs a
      // cooperative launch, treat every op like a barrier op: activate all of
      // its blocks and launch cooperatively, mirroring the non-debug path
      // below.
      bool groupAndCooperative = hasBarriers || cooperative;

      // Set all opcodes to kDebugNoOp on device.
      setOpCodes(deviceBlocks, 0, numBlocks, kDebugNoOp, stream);

      if (groupAndCooperative) {
        auto* inv = sv.kernels.at(launchIdx).invocation;
        if (!launched.insert(reinterpret_cast<intptr_t>(inv)).second) {
          continue;
        }
        // Activate all blocks belonging to the same project op.
        for (int32_t b = 0; b < numBlocks; ++b) {
          auto bIdx = sv.launchIndices[b];
          bool sameOp = bIdx < static_cast<int32_t>(sv.kernels.size()) &&
              sv.kernels[bIdx].invocation == inv;
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
      if (groupAndCooperative) {
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
        if (groupAndCooperative) {
          kernel_->launchCooperative(
              numBlocks, blockSize, dynShared, stream, args);
        } else {
          kernel_->launch(numBlocks, blockSize, dynShared, stream, args);
        }
        stream->wait();
      } catch (const std::exception& e) {
        auto opCode = originalOps[active];
        std::string opText;
        std::string paramText;
        if (launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
            sv.kernels[launchIdx].launch && sv.kernels[launchIdx].launch->op) {
          auto* kernelOp = sv.kernels[launchIdx].launch->op;
          opText = kernelOp->toString(sv.kernels[launchIdx].invocation);
          auto* opParams = pinnedBase + sv.paramOffsets.at(launchIdx);
          paramText = dumpOpParams(
              *kernelOp, opParams, sv.kernels[launchIdx].invocation);
        }
        LOG(ERROR) << "debug_single_ops: block " << active << " opCode "
                   << opCode << " blockInOp " << pinnedBlocks[active].blockInOp
                   << " stepIdx " << stepIdx << " op: " << opText
                   << "\nparams:\n"
                   << paramText << "error: " << e.what();
        throw;
      }
    }

    // D2H transfer after all blocks complete.
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
      stream->wait();
    }
    // The debug path already syncs after every block; just close the event
    // chain so the ordering edges stay consistent with the normal path.
    if (events != nullptr) {
      events->waveDone->record(*stream);
    }

    if (betweenLaunchAndSync) {
      betweenLaunchAndSync();
    }
  } else {
    stream->hostToDeviceAsync(deviceBase, pinnedBase, h2dBytes);
    if (cooperative) {
      kernel_->launchCooperative(numBlocks, blockSize, dynShared, stream, args);
    } else {
      kernel_->launch(numBlocks, blockSize, dynShared, stream, args);
    }
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
      // Record before the host wait, so the timestamp is the kernel/D2H
      // completion rather than the host's observation of it, and so the
      // ordering edge is established as early as possible.
      if (events != nullptr) {
        events->waveDone->record(*stream);
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
        events->waveDone->record(*stream);
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
