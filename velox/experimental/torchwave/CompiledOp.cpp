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
#include <gflags/gflags.h>
#include <algorithm>
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

std::string dumpOpParams(const KernelOperation& op, uint8_t* paramBase) {
  std::stringstream ss;
  const auto& inputs = op.orderedInputs();
  auto numInputs = op.numInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto offset = op.paramOffset(inputs[i]);
    bool isOutput = static_cast<int32_t>(i) >= numInputs;
    ss << "  " << (isOutput ? "output" : "input") << "[" << i
       << "] offset=" << offset;
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
    std::vector<int64_t> result;
    for (const auto& elem : producer->inputs()) {
      auto elemIt = map.find(elem.value->id());
      TORCH_CHECK(
          elemIt != map.end(), node->target(), ": ListPack element not in map");
      result.push_back(frame.getIValue(elemIt->second).toInt());
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

int32_t makeGrid(
    std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM) {
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
  int32_t numSMs = WaveConfig::get().numSms;
  if (numSMs == 0) {
    numSMs = kDefaultNumSMs;
    auto* device = facebook::velox::wave::currentDevice();
    if (device) {
      numSMs = device->numSM;
    }
  }
  int32_t blocksPerSM =
      maxBlocksPerSM > 0 ? maxBlocksPerSM : kDefaultBlocksPerSM;
  int32_t maxBlocks = numSMs * blocksPerSM;
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
  // leave those on the generic path.
  if (standaloneShortcut == StandaloneShortcut::kListPack &&
      (standaloneNode->outputs().empty() ||
       standaloneNode->outputs()[0]->type().kind() !=
           nativert::Type::Kind::TensorList)) {
    standaloneShortcut = StandaloneShortcut::kNone;
  }
  auto* meta = Registry::metadata(standaloneNode->target());
  // prim.ListPack has no registry entry but is metadata-only by definition;
  // every other op's metadata-only status comes from its Metadata.
  metadataOnly =
      meta ? meta->metadataOnly : (standaloneNode->target() == "prim.ListPack");
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
    std::vector<Launch> prePassStandalones)
    : kernel_(std::move(kernel)),
      ops_(std::move(ops)),
      ivalueStorage_(std::move(ivalueStorage)),
      sequenceNumber_(sequenceNumber),
      lastUseIds_(std::move(lastUseIds)),
      prePassStandalones_(std::move(prePassStandalones)) {}

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
    const std::unordered_set<std::string>& includes)
    : ops_(std::move(ops)), kernelOpStorage_(std::move(kernelOps)) {
  auto kernelId = CompileCtx::nextKernelId();
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
  ss << "__global__ void " << kernelName << "(TorchWaveParams params) {\n"
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

      if (!paramOffs.empty()) {
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
        ss << "  {\n";
        emitArray("paramOffsets", paramOffs);
        emitArray("outputOffsets", outputOffs);
        emitArray("altOffsets", altOffs);
        ss << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
           << "      if (altOffsets[i] != -1) {\n"
           << "        copyTensorHead(param<Tensor>(blockInfo, paramOffsets[i]), param<Tensor>(blockInfo, altOffsets[i]));\n"
           << "        param<Tensor>(blockInfo, altOffsets[i])->init<true>(param<Tensor>(blockInfo, outputOffsets[i]));\n"
           << "      } else {\n"
           << "        param<Tensor>(blockInfo, paramOffsets[i])->init<true>(outputOffsets[i] != paramOffsets[i] ? param<Tensor>(blockInfo, outputOffsets[i]) : nullptr);\n"
           << "      }\n"
           << "    }\n"
           << "  }\n"
           << "  __syncthreads();\n";
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

  // Save to /tmp for debugging.
  auto filePath = "/tmp/kernel" + std::to_string(kernelId) + ".cu";
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
      bool isShapeOnly = i < launch.actualOutputDescs.size() &&
          launch.actualOutputDescs[i].shapeOnly;
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

void ensureCudaTensor(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId actualId,
    c10::IntArrayRef dims) {
  auto& existing = frame.getIValue(actualId);
  if (existing.isTensor() && existing.toTensor().is_cuda()) {
    auto& tensor = existing.toTensor();
    if (tensor.sizes() != dims) {
      traceTensor(actualId, dims, "resize");
      tensor.resize_(dims);
    } else {
      traceTensor(actualId, dims, "keep");
    }
  } else {
    auto* meta = types.types.at(actualId);
    if (!meta) {
      return;
    }
    traceTensor(actualId, dims, "alloc");
    auto tensor = at::empty(
        dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
    frame.setIValue(actualId, std::move(tensor));
  }
}

void allocateLaunchOutputs(
    const LaunchData& launch,
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId largestId,
    const folly::F14FastMap<NodeCP, nativert::OpKernel*>* kernelMap,
    const IdToValueMap& idToValue) {
  const auto& descs = launch.actualOutputDescs;
  const auto& actualOutputs = launch.actualOutputs;
  const auto& outputTypes = launch.actualOutputTypes;

  // Shortcut: if largestId is set, resize tensor outputs to match it.
  if (largestId >= 0) {
    auto dims = frame.getIValue(largestId).toTensor().sizes();
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
      if (descs[i].aliasSelfId) {
        auto& selfIv = frame.getIValue(*descs[i].aliasSelfId);
        if (selfIv.isTensor()) {
          // In-place op output: a view sharing self's storage (see general
          // path below).
          frame.setIValue(actualId, selfIv.toTensor().alias());
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
        ensureCudaTensor(frame, types, elemActualId, dims);
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
    ensureCudaTensor(frame, types, actualId, dims);
  }
}

int32_t launchParamSize(const LaunchData& launch) {
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

void CompositeInvocation::gatherLaunches(
    const ExecutionState& state,
    std::vector<GridChoice>& grids,
    int32_t stepIdx,
    StepVectors& sv) {
  const auto& idToValue = state.waveGraph->idToValue();
  int32_t kernelIdx = 0;
  int32_t standaloneIdx = 0;
  int32_t shortcutIdx = 0;
  sv.gridChanged = false;
  sv.isCgGrid = false;
  sv.hasGpuStandalones = false;
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto* grid = grids.at(i).grid;
    if (stepIdx >= static_cast<int32_t>(grid->size())) {
      continue;
    }
    auto* step = &(*grid)[stepIdx];
    for (size_t j = 0; j < step->size(); ++j) {
      auto& launch = (*step)[j];
      if (launch.op != nullptr) {
        bool isNew = kernelIdx >= static_cast<int32_t>(sv.kernels.size());
        if (isNew) {
          sv.kernels.emplace_back(launch, ops_[i], idToValue);
        }
        auto& data = sv.kernels.at(kernelIdx);
        bool hasByLargestInput = !data.launch->op->outputDescs().empty() &&
            data.launch->op->outputDescs()[0].byLargestInput;
        nativert::ValueId largestId = -1;
        if (data.sizeExpr.op == SizeShortcut::kNone) {
          data.numElements = numElementsFromReserve(data, *state.frame);
        } else {
          data.numElements = data.sizeExpr.numElements(
              state.frame, hasByLargestInput ? &largestId : nullptr);
        }

        if (launch.op->isGridChoice()) {
          auto* projectOp = ops_[i].projectOp();
          bool wantSingleBlock;
          if (WaveConfig::get().useSingleBlock.has_value()) {
            wantSingleBlock = *WaveConfig::get().useSingleBlock;
          } else {
            wantSingleBlock =
                data.numElements <= projectOp->singleBlockMaxSize();
          }
          LaunchGrid* newGrid = nullptr;
          if (wantSingleBlock != grids[i].singleBlock) {
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
          // A scanOutputReturnBarrier op takes a launch break only in the
          // multi-block grid, so its multi-block grid has more steps than its
          // single-block variant. The grid-choice kernel can therefore sit at a
          // stepIdx that exists only in the current (longer) grid; switching to
          // a shorter variant here would index it out of bounds (the initial
          // access above is guarded, but these post-swap accesses are not).
          // Only switch when the target grid actually has this step. Otherwise
          // keep the current grid -- it is a complete, correct plan for this op
          // -- so the launch still runs, just under the already-selected
          // variant. The op's earlier steps already ran under that variant, so
          // this also keeps the whole op on one consistent grid.
          if (newGrid && newGrid != grids[i].grid &&
              stepIdx < static_cast<int32_t>(newGrid->size())) {
            grids[i].singleBlock = wantSingleBlock;
            grids[i].grid = newGrid;
            grid = newGrid;
            step = &(*grid)[stepIdx];
            if (!isNew) {
              sv.gridChanged = true;
            }
            auto& newLaunch = (*grids[i].grid)[stepIdx][j];
            data = LaunchData(newLaunch, ops_[i], idToValue);
            largestId = -1;
            if (data.sizeExpr.op == SizeShortcut::kNone) {
              data.numElements = numElementsFromReserve(data, *state.frame);
            } else {
              data.numElements = data.sizeExpr.numElements(
                  state.frame, hasByLargestInput ? &largestId : nullptr);
            }
          }
        }

        // Check if any viewNode output descs have unavailable inputs.
        // If so, skip allocateLaunchOutputs — the viewNode would
        // crash on None inputs from a later PN.
        allocateLaunchOutputs(
            data,
            *state.frame,
            *state.valueTypes,
            largestId,
            state.kernelMap,
            idToValue);
        // If any tensor input or output is None, skip this kernel
        // (set numElements=0 so makeGrid assigns 0 blocks).
        for (auto inputId : data.actualInputs) {
          auto& iv = state.frame->getIValue(inputId);
          if (iv.isNone()) {
            data.numElements = 0;
            break;
          }
        }
        if (data.numElements > 0) {
          for (size_t oi = 0; oi < data.actualOutputs.size(); ++oi) {
            if (oi < data.actualOutputTypes.size() &&
                data.actualOutputTypes[oi] == nativert::Type::Kind::Tensor) {
              const auto& oiv = state.frame->getIValue(data.actualOutputs[oi]);
              // A None output comes from a later PN -- its tensor is not
              // materialized, so the kernel must not launch yet. An empty
              // (0-element) output is handled in device code (the elementwise
              // size head sets size=0 -> 0 iterations), so it does not zero the
              // whole launch here -- that would wrongly skip the non-empty
              // lanes of a multi-output kernel.
              if (oiv.isNone()) {
                data.numElements = 0;
                break;
              }
            }
          }
        }
        // Under a cooperative grid the whole step launches as ONE kernel, so an
        // op cannot be skipped -- numElements only sets its block share. A
        // view-rooted op (e.g. slice->clamp) fused into the step reads an input
        // that is a step-internal intermediate: None/unallocated at host sizing
        // time, so the guards above zero its numElements and it is starved to
        // ~1 block even though it runs correctly once the cooperative kernel
        // materializes that input mid-launch (op 138: 6 of 480 blocks, ~85ms).
        // Recover a grid size from the kernel's concrete static input shapes
        // (TensorMeta is available without materialization). numElements only
        // drives the grid; the kernel loops to the true size on device, so an
        // over-estimate is safe (surplus blocks early-out).
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
        if (!launch.op->barrierCounters().empty()) {
          sv.isCgGrid = true;
        }
        ++kernelIdx;
      } else if (launch.standaloneShortcut != StandaloneShortcut::kNone) {
        // Metadata-only shortcut op: separate list, tight switch loop, no sync.
        if (shortcutIdx >=
            static_cast<int32_t>(sv.shortcutStandalones.size())) {
          sv.shortcutStandalones.emplace_back(launch, ops_[i], idToValue);
        }
        ++shortcutIdx;
      } else {
        if (standaloneIdx >= static_cast<int32_t>(sv.standalones.size())) {
          sv.standalones.emplace_back(launch, ops_[i], idToValue);
        }
        // A standalone that does real device work needs the wave stream synced
        // before it (and before this step's fused kernel). Metadata-only ops
        // (host-only, e.g. a SymInt-list prim.ListPack) need no sync.
        if (!launch.metadataOnly) {
          sv.hasGpuStandalones = true;
        }
        ++standaloneIdx;
      }
    }
  }
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
  int32_t numMismatches = 0;
  std::string passedIds;
  int32_t numPassed = 0;
  for (const auto& data : launches) {
    bool nodeChecked = false;
    for (size_t oi = 0; oi < data.actualOutputs.size(); ++oi) {
      auto actualId = data.actualOutputs[oi];
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
  for (int d = 0; d < t->rank; ++d) {
    newDims[d] = t->dims[d];
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

void CompositeInvocation::processReturnData(
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

// Central execution loop: for each step, gathers launches from the
// subgraph, builds the thread-block grid, fills kernel parameters
// (tensor pointers, scalars, shapes), transfers params H2D, launches
// the CUDA kernel, transfers return values D2H, and verifies against
// the reference frame if set.
void CompositeInvocation::execute(ExecutionState& state) {
  Timer ex("comp inv execute", WaveConfig::get().printTiming);
  auto& frame = *state.frame;

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

  int32_t blockSize;
  int32_t lastExecStep = -1;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    // Re-fetch since the resize above may have invalidated the reference.
    auto& currentGridChoices =
        state.stepVectors.at(sequenceNumber_).at(0).gridChoices;

    {
      auto t0 = Clock::now();
      gatherLaunches(state, currentGridChoices, stepIdx, sv);
      if (doTiming) {
        sv.gatherUs = elapsed(t0);
      }
    }
    // StepVectors are pooled and reused across executions; reset the
    // accumulated ref-check time so it reflects only this run (other timing
    // fields are overwritten with '=' at their measurement point).
    sv.refCheckUs = 0;
    if (sv.gridChanged) {
      invalidateReusedState(
          state.stepVectors[sequenceNumber_],
          state.pinnedBuffers,
          sequenceNumber_,
          stepIdx);
    }
    if (sv.kernels.empty() && sv.standalones.empty() &&
        sv.shortcutStandalones.empty()) {
      break;
    }
    lastExecStep = stepIdx;

    if (sv.kernels.empty()) {
      if (WaveConfig::get().trace &
          (WaveConfig::kNodes | WaveConfig::kLaunches)) {
        traceStep(stepIdx, sv, currentGridChoices);
      }
      // Metadata-only shortcut standalones are host-only and need no
      // wave-stream sync; run them first in their tight, batch-timed loop.
      runShortcutStandalones(
          sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
      // Wait for the wave stream before running eager standalone ops only when
      // a standalone does device-side work. Such ops run on the default stream
      // and read inputs produced by wave kernels; without this wait the eager
      // op can read a wave-stream buffer whose producing kernel (or a pending
      // arena recycle) has not completed, since the two streams are otherwise
      // unordered. Shortcut ops only touch host-side tensor metadata, so they
      // need no sync.
      if (sv.hasGpuStandalones) {
        syncWaveStream(state);
      }
      auto tStandalone = doTiming ? Clock::now() : Clock::time_point{};
      runStandalones(
          sv.standalones,
          state,
          *state.kernelMap,
          *state.standaloneIndices,
          *state.standaloneStats,
          doTiming);
      if (doTiming) {
        sv.standaloneUs = elapsed(tStandalone);
        sv.currentBytes = currentAllocatedBytes();
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
    auto blockInfoBytes = static_cast<int64_t>(numBlocks) *
        static_cast<int64_t>(sizeof(BlockInfo));

    int64_t totalPinnedBytes;
    int64_t totalAllocBytes;
    uint8_t* pinnedBase;
    uint8_t* deviceBase;
    {
      auto t0 = Clock::now();
      sv.paramOffsets.resize(sv.kernels.size());
      int64_t paramCursor = blockInfoBytes;
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        sv.paramOffsets[i] = paramCursor;
        paramCursor += launchParamSize(sv.kernels[i]);
      }
      totalPinnedBytes = paramCursor;

      auto debugInfoBytes = static_cast<int64_t>(numBlocks) *
          static_cast<int64_t>(sizeof(DebugInfo));
      totalAllocBytes = totalPinnedBytes + debugInfoBytes;

      auto& pinnedBuffer = getOrAllocateBuffer(
          state.pinnedBuffers,
          sequenceNumber_,
          stepIdx,
          totalAllocBytes,
          state.pinnedArena,
          WaveConfig::get().debugSingleOps
              ? std::function<void(void*, int64_t)>(
                    [](void* ptr, int64_t bytes) { memset(ptr, 0xaa, bytes); })
              : nullptr);
      auto& deviceBuffer = getOrAllocateBuffer(
          state.deviceBuffers,
          sequenceNumber_,
          stepIdx,
          totalAllocBytes,
          state.deviceArena);
      pinnedBase = pinnedBuffer->as<uint8_t>();
      deviceBase = deviceBuffer->as<uint8_t>();
      if (doTiming) {
        sv.allocUs = elapsed(t0);
      }
    }

    auto* deviceDebugBase =
        reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    {
      auto t0 = Clock::now();
      if (!sv.blocks.empty()) {
        memcpy(pinnedBase, sv.blocks.data(), blockInfoBytes);
      }

      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        int32_t rb = -1;
        int32_t re = -1;
        fillLaunchParams(
            sv.kernels[i], frame, pinnedBase + sv.paramOffsets[i], rb, re);
        if (rb >= 0) {
          if (returnBegin == -1) {
            returnBegin = static_cast<int32_t>(sv.paramOffsets[i] + rb);
          }
          returnEnd = static_cast<int32_t>(sv.paramOffsets[i] + re);
        }
      }

      auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);
      for (size_t b = 0; b < numBlocks; ++b) {
        auto idx = sv.launchIndices[b];
        pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
        pinnedBlocks[b].debugInfo = deviceDebugBase + b;
      }

      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        patchTensorListPointers(
            sv.kernels[i],
            pinnedBase + sv.paramOffsets[i],
            deviceBase + sv.paramOffsets[i]);
      }
      if (doTiming) {
        sv.fillUs = elapsed(t0);
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

    int64_t standaloneElapsed = 0;
    auto runStepStandalones = [&]() {
      // Metadata-only shortcut ops: host-only, tight batch-timed loop, no sync.
      if (!sv.shortcutStandalones.empty()) {
        runShortcutStandalones(
            sv.shortcutStandalones, state, doTiming, sv.shortcutUs);
      }
      if (!sv.standalones.empty()) {
        auto tStandalone = doTiming ? Clock::now() : Clock::time_point{};
        runStandalones(
            sv.standalones,
            state,
            *state.kernelMap,
            *state.standaloneIndices,
            *state.standaloneStats,
            doTiming);
        if (doTiming) {
          standaloneElapsed = elapsed(tStandalone);
        }
        ranStandalones = true;
      }
    };

    // If this step has device-side standalones, wait for the wave stream before
    // the fused kernel launch. Such standalones run on the default stream and
    // may read results of prior wave fused kernels; without this wait those
    // results may not be complete, since the two streams are otherwise
    // unordered. Shortcut standalones only touch host-side tensor metadata, so
    // they need no sync.
    if (sv.hasGpuStandalones) {
      syncWaveStream(state);
    }

    {
      auto tLaunch = Clock::now();
      launch(
          static_cast<int32_t>(numBlocks),
          blockSize,
          pinnedBase,
          deviceBase,
          totalPinnedBytes,
          returnBegin,
          returnEnd,
          deviceDebugBase,
          state.stream.get(),
          sv,
          stepIdx,
          runStepStandalones);

      if (returnBegin >= 0) {
        processReturnData(sv, frame, pinnedBase);
      }
      if (doTiming) {
        sv.kernelUs = elapsed(tLaunch);
        sv.standaloneUs = standaloneElapsed;
        sv.standaloneBound = standaloneElapsed > sv.kernelUs;
        sv.noDtoH = (returnBegin < 0);
        sv.currentBytes = currentAllocatedBytes();
      }
    }

    // Trace outputs of kernel launches after execution.
    if (!state.traceState.empty()) {
      syncWaveStream(state);
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
  if (ranStandalones) {
    // Each step's standaloneUs already covers its eager ops (the per-op
    // default- stream sync in runStandalones drains them at their own step), so
    // this final sync is only for correctness -- it must not be folded back
    // into any step's standalone time, which previously charged the whole-graph
    // async tail to the first standalone step and over-reported standalone time
    // past e2e.
    syncTorchDefaultStream();
  }

  // Stamp this node's last-use values onto its last executed step so they are
  // released by the wave-stream sync that advances that step to kSynced, with
  // no dedicated sync of their own. advanceSyncedStages performs the release.
  if (WaveConfig::get().freeIntermediates && lastExecStep >= 0 &&
      !lastUseIds_.empty()) {
    getStepVectors(state.stepVectors, sequenceNumber_, lastExecStep)
        .lastUseIds = lastUseIds_;
  }
}

void CompositeInvocation::launch(
    int32_t numBlocks,
    int32_t blockSize,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t totalPinnedBytes,
    int32_t returnBegin,
    int32_t returnEnd,
    DebugInfo* deviceDebugBase,
    facebook::velox::wave::Stream* stream,
    const StepVectors& sv,
    int32_t stepIdx,
    std::function<void()> betweenLaunchAndSync) {
  TorchWaveParams params{};
  params.info = reinterpret_cast<BlockInfo*>(deviceBase);
  params.debugInfo = deviceDebugBase;
  void* args[] = {&params};

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
      const auto& kd = sv.kernels[ki];
      if (kd.launch && kd.launch->op &&
          !kd.launch->op->barrierCounters().empty() &&
          ki < sv.numBlocksPerLaunch.size() && sv.numBlocksPerLaunch[ki] > 1) {
        cooperative = true;
        break;
      }
    }
  }

  auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);

  if (WaveConfig::get().debugSingleOps) {
    std::vector<int32_t> originalOps(numBlocks);
    for (int32_t b = 0; b < numBlocks; ++b) {
      originalOps[b] = pinnedBlocks[b].op;
    }

    // Transfer pinned buffer to device once.
    stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
    stream->wait();

    auto* deviceBlocks = reinterpret_cast<BlockInfo*>(deviceBase);

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
        auto* inv = sv.kernels[launchIdx].invocation;
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
              auto* dest = deviceBase + sv.paramOffsets[li] + offset;
              stream->hostToDeviceAsync(dest, &zero, sizeof(zero));
            }
          }
        }
      }

      try {
        if (groupAndCooperative) {
          kernel_->launchCooperative(numBlocks, blockSize, 0, stream, args);
        } else {
          kernel_->launch(numBlocks, blockSize, 0, stream, args);
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
          paramText = dumpOpParams(*kernelOp, opParams);
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

    if (betweenLaunchAndSync) {
      betweenLaunchAndSync();
    }
  } else {
    stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
    if (cooperative) {
      kernel_->launchCooperative(numBlocks, blockSize, 0, stream, args);
    } else {
      kernel_->launch(numBlocks, blockSize, 0, stream, args);
    }
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
      if (betweenLaunchAndSync) {
        betweenLaunchAndSync();
      }
      stream->wait();
    } else {
      if (betweenLaunchAndSync) {
        betweenLaunchAndSync();
      }
      if (WaveConfig::get().trace & WaveConfig::kTiming) {
        stream->wait();
      }
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
