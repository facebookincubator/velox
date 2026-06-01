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
#include "velox/experimental/torchwave/GraphOptimizer.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/wave/common/KernelFsCache.h"

#include <ATen/ATen.h>
#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <type_traits>

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
namespace torch::wave {

namespace {

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
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? static_cast<int32_t>(tensor.size(i)) : 0;
    t->strides[i] = 0;
  }
  t->numEl = static_cast<uint32_t>(tensor.numel());
  t->status = Tensor::kUninited;
}

void fillTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Tensors with more than 3 dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = tensor.data_ptr();
  t->rank = static_cast<int8_t>(tensor.dim());
  t->elementSize = tensor.element_size();
  t->elementType = static_cast<uint8_t>(tensor.scalar_type());
  for (int i = 0; i < 3; ++i) {
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

void fillScalarParam(const c10::IValue& ivalue, void* dest) {
  if (ivalue.isInt()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toInt();
  } else if (ivalue.isDouble()) {
    *reinterpret_cast<double*>(dest) = ivalue.toDouble();
  } else if (ivalue.isBool()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toBool() ? 1 : 0;
  } else {
    TORCH_CHECK(
        false, "Unsupported IValue type for kernel param: ", ivalue.tagKind());
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

// Launching a block for fewer elements per thread is counter-productive.
constexpr int32_t kMinElementsPerThread = 4;

// Default SM count when device info is unavailable.
constexpr int32_t kDefaultNumSMs = 100;

// Default blocks per SM when occupancy info is unavailable.
constexpr int32_t kDefaultBlocksPerSM = 4;

int32_t makeGrid(
    const std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM) {
  const int32_t blockSize = WaveConfig::get().blockSize;

  // Compute cost per launch: numElements * unitCost.
  sv.costs.resize(launches.size());
  float totalCost = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    sv.costs[i] = static_cast<float>(launches[i].numElements) *
        launches[i].launch->op->unitCost();
    totalCost += sv.costs[i];
  }

  // Max blocks each launch could use.
  sv.maxBlocks.resize(launches.size());
  const int32_t elementsPerBlock = blockSize * kMinElementsPerThread;
  for (size_t i = 0; i < launches.size(); ++i) {
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
  int32_t targetBlocks =
      sv.isCgGrid ? static_cast<int32_t>(maxBlocks * 0.90f) : maxBlocks;

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
      auto alignedElems = roundUp(elemsPerBlock, elementsPerBlock);
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
    float scale =
        static_cast<float>(targetBlocks) / static_cast<float>(totalAssigned);
    totalAssigned = 0;
    for (size_t i = 0; i < launches.size(); ++i) {
      auto scaled = std::max(
          1, static_cast<int32_t>(sv.numBlocksPerLaunch[i] * scale + 0.5f));
      sv.numBlocksPerLaunch[i] = scaled;
      totalAssigned += scaled;
    }
    if (totalAssigned > maxBlocks) {
      int32_t avg = totalAssigned / static_cast<int32_t>(launches.size());
      int32_t excess = totalAssigned - maxBlocks;
      for (size_t i = 0; i < launches.size() && excess > 0; ++i) {
        if (sv.numBlocksPerLaunch[i] > avg && sv.numBlocksPerLaunch[i] > 1) {
          --sv.numBlocksPerLaunch[i];
          --totalAssigned;
          --excess;
        }
      }
    }
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

Launch::Launch(
    NodeCP standaloneNode,
    const ValueTypes& types,
    WaveGraph& waveGraph)
    : standalone(standaloneNode) {
  auto* meta = Registry::metadata(standaloneNode->target());
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
    int32_t sequenceNumber)
    : kernel_(std::move(kernel)),
      ops_(std::move(ops)),
      ivalueStorage_(std::move(ivalueStorage)),
      sequenceNumber_(sequenceNumber) {}

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
    TORCH_CHECK(
        nodeIt != op.nodeMap().end(), "Standalone node not found in nodeMap");
    standalone = nodeIt->second;
    for (auto& input : launch.standalone->inputs()) {
      actualInputs.push_back(translateId(input.value));
    }
    for (auto* output : launch.standalone->outputs()) {
      actualOutputs.push_back(translateId(output));
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

  // Only compile the kernel if a GPU is available.
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
      fillTensorParam(
          frame.getIValue(actualId).toTensor(), paramBase + elemOffset);
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
          paramBase + launch.scalarOffsets[i]);
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
    const auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      fillTensorParam(ivalue.toTensor(), dest);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      fillScalarParam(ivalue, dest);
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
      TORCH_CHECK(ivalue.isTensor(), "Expected tensor for output param");
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
      // Non-tensor output: write a 64-bit zero placeholder.
      *reinterpret_cast<int64_t*>(dest) = 0;
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
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
    fillScalarParam(*opConstants[idx], dest);
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
      executeNode(descs[i].viewNode, it->second, frame);
      continue;
    }
    if (descs[i].delegated) {
      continue;
    }
    auto actualId = actualOutputs[i];

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
  sv.gridChanged = false;
  sv.isCgGrid = false;
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
            grids[i].singleBlock = wantSingleBlock;
          }
          if (!wantSingleBlock && !projectOp->cgGrid().empty() &&
              WaveConfig::get().isCg.has_value() && *WaveConfig::get().isCg) {
            newGrid = &projectOp->cgGrid();
          }
          if (newGrid && newGrid != grids[i].grid) {
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

        allocateLaunchOutputs(
            data,
            *state.frame,
            *state.valueTypes,
            largestId,
            state.kernelMap,
            idToValue);
        if (!launch.op->barrierCounters().empty()) {
          sv.isCgGrid = true;
        }
        ++kernelIdx;
      } else {
        if (standaloneIdx >= static_cast<int32_t>(sv.standalones.size())) {
          sv.standalones.emplace_back(launch, ops_[i], idToValue);
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
  int32_t numMismatches = 0;
  std::string passedIds;
  int32_t numPassed = 0;
  for (const auto& data : launches) {
    bool nodeChecked = false;
    for (auto actualId : data.actualOutputs) {
      auto refIt = ref->find(actualId);
      if (refIt == ref->end()) {
        continue;
      }
      const auto& actual = frame.getIValue(actualId);
      if (!actual.isTensor() || !refIt->second.isTensor()) {
        continue;
      }
      const auto& actualTensor = actual.toTensor();
      const auto& refTensor = refIt->second.toTensor();
      if (actualTensor.numel() == 0) {
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
          const auto& actual = frame.getIValue(actualId);
          if (actual.isTensor() &&
              tensorsMatch(actual.toTensor(), refIt->second.toTensor())) {
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
        const auto& actual = frame.getIValue(actualId);
        if (!actual.isTensor()) {
          continue;
        }
        if (!tensorsMatch(actual.toTensor(), refIt->second.toTensor())) {
          ++numCorrupted;
          auto limit = WaveConfig::get().tensorPrintElementLimit;
          LOG(ERROR) << "INPUT CORRUPTION: value %" << actualId
                     << " no longer matches reference\n  "
                     << firstDifference(
                            actual.toTensor(), refIt->second.toTensor())
                     << "\n  expected: "
                     << tensorDebugString(refIt->second.toTensor(), limit)
                     << "\n  actual:   "
                     << tensorDebugString(actual.toTensor(), limit);
        }
      }
    }
    // Re-verify previously passed outputs.
    for (auto prevId : state.verifiedIds) {
      auto refIt = ref->find(prevId);
      if (refIt == ref->end() || !refIt->second.isTensor()) {
        continue;
      }
      const auto& actual = frame.getIValue(prevId);
      if (!actual.isTensor()) {
        continue;
      }
      if (!tensorsMatch(actual.toTensor(), refIt->second.toTensor())) {
        ++numCorrupted;
        auto limit = WaveConfig::get().tensorPrintElementLimit;
        LOG(ERROR) << "CORRUPTION: previously passed value %" << prevId
                   << " no longer matches reference\n  "
                   << firstDifference(
                          actual.toTensor(), refIt->second.toTensor())
                   << "\n  expected: "
                   << tensorDebugString(refIt->second.toTensor(), limit)
                   << "\n  actual:   "
                   << tensorDebugString(actual.toTensor(), limit);
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

  if (WaveConfig::get().trace) {
    std::cout << "==== Node " << sequenceNumber_ << std::endl;
  }

  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber_, 0);
  auto& gridChoices = sv0.gridChoices;
  if (gridChoices.empty()) {
    for (auto& op : ops_) {
      gridChoices.push_back({0, false, &op.projectOp()->grid()});
    }
  }

  int32_t blockSize;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    // Re-fetch since the resize above may have invalidated the reference.
    auto& currentGridChoices =
        state.stepVectors.at(sequenceNumber_).at(0).gridChoices;

    {
      Timer t("gather", WaveConfig::get().printTiming);
      gatherLaunches(state, currentGridChoices, stepIdx, sv);
    }
    if (sv.gridChanged) {
      invalidateReusedState(
          state.stepVectors[sequenceNumber_],
          state.pinnedBuffers,
          sequenceNumber_,
          stepIdx);
    }
    if (sv.kernels.empty() && sv.standalones.empty()) {
      break;
    }

    if (sv.kernels.empty()) {
      if (WaveConfig::get().trace) {
        traceStep(stepIdx, sv, currentGridChoices);
      }
      runStandalones(
          sv.standalones,
          state,
          *state.kernelMap,
          *state.standaloneIndices,
          *state.standaloneStats);
      verifyAgainstReference(sv.standalones, frame, state);
      continue;
    }

    // Trace inputs of kernel launches before execution.
    if (!state.traceState.empty()) {
      for (const auto& launch : sv.kernels) {
        traceFrameValues("input", launch.actualInputs, frame, state.traceState);
      }
    }

    {
      Timer t("grid", WaveConfig::get().printTiming);
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
    }

    auto numBlocks = sv.blocks.size();
    auto blockInfoBytes = static_cast<int64_t>(numBlocks) *
        static_cast<int64_t>(sizeof(BlockInfo));

    int64_t totalPinnedBytes;
    int64_t totalAllocBytes;
    uint8_t* pinnedBase;
    uint8_t* deviceBase;
    {
      Timer t("alloc outputs", WaveConfig::get().printTiming);
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
    }

    auto* deviceDebugBase =
        reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    {
      Timer t("fill params", WaveConfig::get().printTiming);
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
    }

    if (WaveConfig::get().trace) {
      traceStep(stepIdx, sv, currentGridChoices);
    }

    state.launchDebugInfos.push_back(
        {reinterpret_cast<DebugInfo*>(pinnedBase + totalPinnedBytes),
         deviceDebugBase,
         static_cast<int32_t>(numBlocks),
         sequenceNumber_,
         stepIdx});

    auto runStepStandalones = [&]() {
      if (!sv.standalones.empty()) {
        runStandalones(
            sv.standalones,
            state,
            *state.kernelMap,
            *state.standaloneIndices,
            *state.standaloneStats);
      }
    };

    {
      Timer t("launch1", WaveConfig::get().printTiming);
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
    }

    if (returnBegin >= 0) {
      processReturnData(sv, frame, pinnedBase);
    }

    // Trace outputs of kernel launches after execution.
    if (!state.traceState.empty()) {
      state.stream->wait();
      for (const auto& launch : sv.kernels) {
        traceFrameValues(
            "output", launch.actualOutputs, frame, state.traceState);
      }
    }

    verifyAgainstReference(sv.standalones, frame, state);
    verifyAgainstReference(sv.kernels, frame, state);
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
      auto launchIdx = sv.launchIndices[active];
      bool hasBarriers = launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
          sv.kernels[launchIdx].launch && sv.kernels[launchIdx].launch->op &&
          !sv.kernels[launchIdx].launch->op->barrierCounters().empty();

      // Set all opcodes to kDebugNoOp on device.
      setOpCodes(deviceBlocks, 0, numBlocks, kDebugNoOp, stream);

      if (hasBarriers) {
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

      // Reset barrier counters on device for the active op.
      if (hasBarriers) {
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
        if (hasBarriers) {
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
    if (sv.isCgGrid) {
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
    } else if (betweenLaunchAndSync) {
      betweenLaunchAndSync();
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
